import logging
from typing import Optional
from uuid import UUID

from pydantic import BaseModel

from backend.db.dal import (
    DALPhotobookBookmarks,
    DALPhotobooks,
    DAOPhotobookBookmarksCreate,
    FilterOp,
    OrderDirection,
    safe_commit,
)
from backend.db.data_models import PhotobookStatus
from backend.db.externals import (
    PhotobookBookmarksOverviewResponse,
    PhotobooksOverviewResponse,
)
from backend.route_handler.base import RouteHandler

from .base import enforce_response_model


class UserBookmarkPhotobookInputPayload(BaseModel):
    photobook_id: UUID
    source_analytics: Optional[str] = None


class UserGetPhotobooksResponse(BaseModel):
    photobooks: list[PhotobooksOverviewResponse]


class UserBookmarkPhotobookDeleteResponse(BaseModel):
    success: bool
    error_message: Optional[str] = None


class UserAPIHandler(RouteHandler):
    def register_routes(self) -> None:
        self.route(
            "/api/user/{user_id}/photobooks",
            "user_get_photobooks",
            methods=["GET"],
        )
        self.route(
            "/api/user/{user_id}/photobooks/bookmarks",
            "user_get_bookmarked_photobooks",
            methods=["GET"],
        )
        self.route(
            "/api/user/{user_id}/photobooks/bookmark_new",
            "user_photobook_bookmark_new",
            methods=["POST"],
        )
        self.route(
            "/api/user/{user_id}/photobooks/bookmark_remove/{photobook_id}",
            "user_photobook_bookmark_remove",
            methods=["DELETE"],
        )

    @enforce_response_model
    async def user_get_photobooks(
        self,
        user_id: UUID,
    ) -> UserGetPhotobooksResponse:
        async with self.app.new_db_session() as db_session:
            photobooks = await DALPhotobooks.list_all(
                db_session,
                {
                    "user_id": (FilterOp.EQ, user_id),
                    "status": (
                        FilterOp.NOT_IN,
                        [
                            PhotobookStatus.DELETED,
                            PhotobookStatus.PERMANENTLY_DELETED,
                        ],
                    ),
                },
                order_by=[("updated_at", OrderDirection.DESC)],
            )
            resp = UserGetPhotobooksResponse(
                photobooks=await PhotobooksOverviewResponse.rendered_from_daos(
                    photobooks, db_session, self.app.asset_manager
                )
            )
            return resp

    @enforce_response_model
    async def user_get_bookmarked_photobooks(
        self, user_id: UUID
    ) -> UserGetPhotobooksResponse:
        async with self.app.new_db_session() as db_session:
            photobook_bookmarks = await DALPhotobookBookmarks.list_all(
                db_session,
                {"user_id": (FilterOp.EQ, user_id)},
                order_by=[("created_at", OrderDirection.DESC)],
            )
            photobooks = await DALPhotobooks.list_all(
                db_session,
                filters={
                    "id": (
                        FilterOp.IN,
                        [bookmark.photobook_id for bookmark in photobook_bookmarks],
                    ),
                    "status": (
                        FilterOp.NOT_IN,
                        [
                            PhotobookStatus.DELETED,
                            PhotobookStatus.PERMANENTLY_DELETED,
                        ],
                    ),
                },
            )
            return UserGetPhotobooksResponse(
                photobooks=await PhotobooksOverviewResponse.rendered_from_daos(
                    photobooks, db_session, self.app.asset_manager
                )
            )

    @enforce_response_model
    async def user_photobook_bookmark_new(
        self,
        user_id: UUID,
        payload: UserBookmarkPhotobookInputPayload,
    ) -> PhotobookBookmarksOverviewResponse:
        async with self.app.new_db_session() as db_session:
            async with safe_commit(db_session):
                dao = await DALPhotobookBookmarks.create(
                    db_session,
                    DAOPhotobookBookmarksCreate(
                        user_id=user_id,
                        photobook_id=payload.photobook_id,
                        source=payload.source_analytics,
                    ),
                )
            return PhotobookBookmarksOverviewResponse.from_dao(dao)

    @enforce_response_model
    async def user_photobook_bookmark_remove(
        self,
        user_id: UUID,
        photobook_id: UUID,
    ) -> UserBookmarkPhotobookDeleteResponse:
        async with self.app.new_db_session() as db_session:
            try:
                bookmarks = await DALPhotobookBookmarks.list_all(
                    db_session,
                    filters={
                        "user_id": (FilterOp.EQ, user_id),
                        "photobook_id": (FilterOp.EQ, photobook_id),
                    },
                    limit=1,
                )

                if not bookmarks:
                    return UserBookmarkPhotobookDeleteResponse(
                        success=False,
                        error_message="Bookmark not found.",
                    )

                dao = bookmarks[0]

                async with safe_commit(db_session):
                    await DALPhotobookBookmarks.delete_by_id(db_session, dao.id)

                return UserBookmarkPhotobookDeleteResponse(success=True)
            except Exception as e:
                logging.exception(f"Failed to remove bookmark: {e}")
                return UserBookmarkPhotobookDeleteResponse(
                    success=False,
                    error_message="An unexpected error occurred.",
                )
