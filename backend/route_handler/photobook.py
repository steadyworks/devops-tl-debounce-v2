import logging
from datetime import datetime
from typing import Optional, Self
from uuid import UUID

from fastapi import File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dal import (
    DALPages,
    DALPhotobooks,
    DAOPagesUpdate,
    DAOPhotobooksCreate,
    DAOPhotobooksUpdate,
    FilterOp,
    OrderDirection,
    safe_commit,
)
from backend.db.data_models import DAOPhotobooks, PhotobookStatus, UserProvidedOccasion
from backend.db.externals import PhotobooksOverviewResponse
from backend.lib.asset_manager.base import AssetManager
from backend.lib.job_manager.base import JobManager
from backend.lib.job_manager.types import JobQueue
from backend.lib.utils.common import utcnow
from backend.lib.utils.web_requests import save_uploads_to_tempdir
from backend.route_handler.base import RouteHandler
from backend.worker.job_processor.types import AssetCompressUploadInputPayload, JobType

from .base import enforce_response_model, unauthenticated_route
from .page import PagesFullResponse


class UploadedFileInfo(BaseModel):
    filename: str
    storage_key: str


class FailedUploadInfo(BaseModel):
    filename: str
    error: str


class NewPhotobookResponse(BaseModel):
    photobook_id: UUID
    skipped_non_media: list[str]


class PhotobookEditTitleRequest(BaseModel):
    new_title: str


class PhotobooksFullResponse(PhotobooksOverviewResponse):
    pages: list[PagesFullResponse]

    @classmethod
    async def rendered_from_dao(
        cls: type[Self],
        dao: DAOPhotobooks,
        db_session: AsyncSession,
        asset_manager: AssetManager,
    ) -> Self:
        resp = await PhotobooksOverviewResponse.rendered_from_dao(
            dao,
            db_session,
            asset_manager,
        )
        pages = await DALPages.list_all(
            db_session,
            {"photobook_id": (FilterOp.EQ, dao.id)},
            order_by=[("page_number", OrderDirection.ASC)],
        )
        pages_response_full = await PagesFullResponse.rendered_from_daos(
            pages, db_session, asset_manager
        )
        return cls(
            **resp.model_dump(),
            pages=pages_response_full,
        )


class EditPageRequest(BaseModel):
    page_id: UUID
    new_user_message: str


class PhotobookEditPagesRequest(BaseModel):
    edits: list[EditPageRequest]


class PhotobookDeleteResponse(BaseModel):
    success: bool
    error_message: Optional[str] = None


class PhotobookAPIHandler(RouteHandler):
    def register_routes(self) -> None:
        self.route("/api/photobook/new", "photobook_new", ["POST"])
        self.route("/api/photobook/{photobook_id}", "get_photobook_by_id", ["GET"])
        self.route(
            "/api/photobook/{photobook_id}/edit_title", "photobook_edit_title", ["POST"]
        )
        self.route(
            "/api/photobook/{photobook_id}/edit_pages", "photobook_edit_pages", ["POST"]
        )
        self.route("/api/photobook/{photobook_id}/delete", "photobook_delete", ["POST"])

    @staticmethod
    def is_accepted_mime(mime: Optional[str]) -> bool:
        return mime is not None and (
            mime.startswith("image/")
            # or mime.startswith("video/") # FIXME / TODO: only images allowed for now
        )

    @enforce_response_model
    async def photobook_new(
        self,
        request: Request,
        files: list[UploadFile] = File(...),
        user_provided_occasion: UserProvidedOccasion = Form(...),
        user_provided_custom_details: Optional[str] = Form(None),
        user_provided_context: Optional[str] = Form(None),
    ) -> NewPhotobookResponse:
        request_context = await self.get_request_context(request)

        # Filter valid files according to FastAPI reported mime type
        valid_files = [
            file
            for file in files
            if PhotobookAPIHandler.is_accepted_mime(file.content_type)
        ]
        file_names = [file.filename for file in valid_files]
        skipped_files_due_to_mime_validation = [
            file.filename
            for file in files
            if file not in valid_files and file.filename is not None
        ]
        logging.debug(
            {
                "accepted_files": file_names,
                "skipped_non_media": skipped_files_due_to_mime_validation,
            }
        )

        temp_uploads_metadata = await save_uploads_to_tempdir(valid_files)

        # Persist metadata with new photobook DB entry
        async with self.app.new_db_session() as db_session:
            async with safe_commit(
                db_session, context="photobook creation DB write", raise_on_fail=True
            ):
                photobook = await DALPhotobooks.create(
                    db_session,
                    DAOPhotobooksCreate(
                        user_id=request_context.user_id,
                        title=f"New Photobook {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        caption=None,
                        theme=None,
                        status=PhotobookStatus.PENDING,
                        user_provided_occasion=user_provided_occasion,
                        user_provided_occasion_custom_details=user_provided_custom_details,
                        user_provided_context=user_provided_context,
                        thumbnail_asset_id=None,
                        deleted_at=None,
                        status_last_edited_by=None,
                    ),
                )

            # Enqueue local image compression / upload job
            async with JobManager(
                self.app.local_redis_factory, JobQueue.LOCAL_MAIN_TASK_QUEUE
            ) as job_manager:
                await job_manager.enqueue(
                    JobType.LOCAL_ASSET_COMPRESS_UPLOAD,
                    job_payload=AssetCompressUploadInputPayload(
                        root_tempdir=temp_uploads_metadata.root_dir,
                        absolute_media_paths=[
                            file.absolute_path for file in temp_uploads_metadata.files
                        ],
                        originating_photobook_id=photobook.id,
                        user_id=request_context.user_id,
                    ),
                    max_retries=2,
                    db_session=db_session,
                )

        return NewPhotobookResponse(
            photobook_id=photobook.id,
            skipped_non_media=skipped_files_due_to_mime_validation,
        )

    @unauthenticated_route
    @enforce_response_model
    async def get_photobook_by_id(
        self,
        photobook_id: UUID,
    ) -> PhotobooksFullResponse:
        async with self.app.new_db_session() as db_session:
            # Step 1: Fetch photobook
            photobook = await DALPhotobooks.get_by_id(db_session, photobook_id)
            if photobook is None:
                raise HTTPException(status_code=404, detail="Photobook not found")
            return await PhotobooksFullResponse.rendered_from_dao(
                photobook, db_session, self.app.asset_manager
            )

    @enforce_response_model
    async def photobook_edit_title(
        self, photobook_id: UUID, payload: PhotobookEditTitleRequest
    ) -> PhotobooksOverviewResponse:
        async with self.app.new_db_session() as db_session:
            async with safe_commit(db_session):
                photobook = await DALPhotobooks.update_by_id(
                    db_session,
                    photobook_id,
                    DAOPhotobooksUpdate(
                        title=payload.new_title,
                    ),
                )
            return await PhotobooksOverviewResponse.rendered_from_dao(
                photobook, db_session, self.app.asset_manager
            )

    @enforce_response_model
    async def photobook_edit_pages(
        self, photobook_id: UUID, payload: PhotobookEditPagesRequest
    ) -> PhotobooksFullResponse:
        async with self.app.new_db_session() as db_session:
            # 1. Validate photobook exists
            photobook = await DALPhotobooks.get_by_id(db_session, photobook_id)
            if photobook is None:
                raise HTTPException(status_code=404, detail="Photobook not found")

            # 2. Batch apply page updates
            async with safe_commit(db_session):
                update_map = {
                    edit.page_id: DAOPagesUpdate(user_message=edit.new_user_message)
                    for edit in payload.edits
                }
                await DALPages.update_many_by_ids(db_session, update_map)

            # 3. Return updated photobook and its pages
            return await PhotobooksFullResponse.rendered_from_dao(
                photobook, db_session, self.app.asset_manager
            )

    @enforce_response_model
    async def photobook_delete(
        self,
        photobook_id: UUID,
    ) -> PhotobookDeleteResponse:
        async with self.app.new_db_session() as db_session:
            photobook = await DALPhotobooks.get_by_id(db_session, photobook_id)
            if photobook is None:
                return PhotobookDeleteResponse(
                    success=False, error_message="Photobook not found"
                )

            if (
                photobook.status == PhotobookStatus.DELETED
                or photobook.status == PhotobookStatus.PERMANENTLY_DELETED
            ):
                return PhotobookDeleteResponse(
                    success=False, error_message="Photobook already deleted"
                )

            async with safe_commit(db_session):
                await DALPhotobooks.update_by_id(
                    db_session,
                    photobook_id,
                    DAOPhotobooksUpdate(
                        deleted_at=utcnow(), status=PhotobookStatus.DELETED
                    ),
                )

            return PhotobookDeleteResponse(success=True)
