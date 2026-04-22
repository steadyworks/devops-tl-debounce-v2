import asyncio
import logging
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal
from uuid import UUID

from backend.db.dal import (
    DALAssets,
    DALPhotobooks,
    DAOAssetsCreate,
    DAOAssetsUpdate,
    DAOPhotobooksUpdate,
    safe_commit,
)
from backend.db.data_models import AssetUploadStatus, PhotobookStatus
from backend.db.session.factory import AsyncSessionFactory
from backend.lib.asset_manager.base import AssetManager, AssetStorageKey
from backend.lib.job_manager.base import JobManager
from backend.lib.job_manager.types import JobQueue
from backend.lib.redis.factory import RedisClientFactory
from backend.lib.types.asset import Asset

from .local import LocalJobProcessor
from .types import (
    AssetCompressUploadInputPayload,
    AssetCompressUploadOutputPayload,
    JobType,
    PhotobookGenerationInputPayload,
)
from .utils.types import CompressionTier
from .utils.vips import ImageProcessingLibrary

MIN_FREE_DISK_BYTES = 100 * 1024 * 1024  # 100MB


@contextmanager
def compression_tier_tempdir(
    tier: CompressionTier,
    root_path: Path,
) -> Generator[Path, None, None]:
    """
    Creates a temporary output subdirectory for a given compression tier,
    and cleans it up on exit.
    """
    tempdir = root_path / f"compressed_{tier.value}_{uuid.uuid4().hex}"
    tempdir.mkdir(parents=True, exist_ok=True)

    try:
        yield tempdir
    finally:
        try:
            shutil.rmtree(tempdir, ignore_errors=True)
        except Exception as e:
            logging.warning(f"[TempDir] Failed to clean up {tempdir}: {e}")


class AssetCompressUploadLocalJobProcessor(
    LocalJobProcessor[AssetCompressUploadInputPayload, AssetCompressUploadOutputPayload]
):
    def __init__(
        self,
        job_id: UUID,
        asset_manager: AssetManager,
        db_session_factory: AsyncSessionFactory,
        remote_redis_client_factory: RedisClientFactory,
    ) -> None:
        self._image_lib = ImageProcessingLibrary(
            max_concurrent=1,
        )
        super().__init__(
            job_id,
            asset_manager,
            db_session_factory,
            remote_redis_client_factory,
        )

    def _get_asset_key_type_by_compression_tier(
        self, compression_tier: CompressionTier
    ) -> Literal["asset_key_original", "asset_key_display", "asset_key_llm"]:
        if compression_tier == CompressionTier.HIGH_END_DISPLAY:
            return "asset_key_display"
        elif compression_tier == CompressionTier.LLM:
            return "asset_key_llm"
        else:
            raise Exception("Only asset_key_display and asset_key_llm supported")

    async def _compress_and_upload_safe(
        self,
        tier: CompressionTier,
        media_paths_asset_uuids_map: dict[Path, UUID],
        root_path: Path,
        originating_photobook_id: UUID,
    ) -> None:
        try:
            with compression_tier_tempdir(tier, root_path) as root_compress_tempdir:
                result = await self._image_lib.compress_by_tier_on_thread(
                    input_paths=list(media_paths_asset_uuids_map.keys()),
                    output_dir=root_compress_tempdir,
                    format="jpeg",
                    tier=tier,
                    strip_metadata=False,
                )

                # Upload assets where compressions are successful
                media_paths_compressed_paths_map: dict[Path, Path] = dict()
                for original_path, (success, compressed_path) in result.items():
                    if success and compressed_path is not None:
                        media_paths_compressed_paths_map[original_path] = (
                            compressed_path
                        )

                await self._upload_to_asset_storage_persisting_metadata(
                    media_paths_asset_uuids_map,
                    media_paths_compressed_paths_map,
                    originating_photobook_id,
                    self._get_asset_key_type_by_compression_tier(tier),
                )
        except Exception:
            logging.warning(
                f"[AssetCompressUploadLocalJobProcessor] Upload failed "
                f"for tier {tier}, photobook_id {originating_photobook_id}"
            )

    async def _upload_to_asset_storage_persisting_metadata(
        self,
        media_paths_asset_uuids_map: dict[Path, UUID],
        media_paths_compressed_paths_map: dict[Path, Path],
        originating_photobook_id: UUID,
        asset_key_type: Literal[
            "asset_key_original", "asset_key_display", "asset_key_llm"
        ],
    ) -> None:
        compressed_paths_orig_paths_map = {
            v: k for k, v in media_paths_compressed_paths_map.items()
        }

        upload_inputs: list[tuple[Path, AssetStorageKey]] = []
        for orig_local_path in media_paths_asset_uuids_map.keys():
            if orig_local_path not in media_paths_compressed_paths_map:
                continue
            compressed_local_path = media_paths_compressed_paths_map[orig_local_path]
            upload_inputs.append(
                (
                    compressed_local_path,
                    self.asset_manager.mint_asset_key(
                        originating_photobook_id, compressed_local_path.name
                    ),
                )
            )
        upload_results = await self.asset_manager.upload_files_batched(upload_inputs)

        dao_updates: dict[UUID, DAOAssetsUpdate] = dict()
        for compressed_local_path, upload_res in upload_results.items():
            if not isinstance(upload_res, Asset):
                continue
            if compressed_local_path not in compressed_paths_orig_paths_map:
                continue
            local_orig_path = compressed_paths_orig_paths_map[compressed_local_path]
            asset_uuid = media_paths_asset_uuids_map.get(local_orig_path)
            if asset_uuid is None:
                continue
            update_obj = None
            if asset_key_type == "asset_key_original":
                update_obj = DAOAssetsUpdate(
                    asset_key_original=upload_res.asset_storage_key
                )
            elif asset_key_type == "asset_key_display":
                update_obj = DAOAssetsUpdate(
                    asset_key_display=upload_res.asset_storage_key
                )
            elif asset_key_type == "asset_key_llm":
                update_obj = DAOAssetsUpdate(asset_key_llm=upload_res.asset_storage_key)
            else:
                pass
            if update_obj is None:
                continue

            dao_updates[asset_uuid] = update_obj

        if not dao_updates:
            logging.warning(
                "[AssetCompressUploadLocalJobProcessor] no asset uploads to persist"
            )
            if asset_key_type == "asset_key_original":
                raise Exception("No original assets were uploaded")

        async with self.db_session_factory.new_session() as db_session:
            async with safe_commit(
                db_session,
                context="persist uploaded asset storage keys",
                raise_on_fail=asset_key_type == "asset_key_original",
            ):
                await DALAssets.update_many_by_ids(db_session, dao_updates)

    def _sanity_check_paths_and_free_storage(
        self, input_payload: AssetCompressUploadInputPayload
    ) -> tuple[bool, str]:  # (should_abort, error_message)
        should_abort, error_message = False, ""
        # Sanity check that all image paths and root dir exists
        missing_paths = [
            p for p in input_payload.absolute_media_paths if not p.is_file()
        ]
        if not input_payload.root_tempdir.is_dir():
            should_abort = True
            error_message = (
                "[AssetCompressUploadLocalJobProcessor] Temp output directory does not exist: "
                f"{input_payload.root_tempdir}"
            )
        if missing_paths:
            should_abort = True
            error_message = (
                "[AssetCompressUploadLocalJobProcessor] Missing input files:"
                f"{', '.join(str(p) for p in missing_paths)}"
            )

        # Sanity check that we have enough spare disk space for compression
        free_bytes = shutil.disk_usage(input_payload.root_tempdir).free
        if free_bytes < MIN_FREE_DISK_BYTES:
            should_abort = True
            error_message = (
                f"[AssetCompressUploadLocalJobProcessor] Not enough free disk space in {input_payload.root_tempdir} "
                f"({free_bytes / (1024**2):.2f} MB available)"
            )
        return should_abort, error_message

    async def process(
        self, input_payload: AssetCompressUploadInputPayload
    ) -> AssetCompressUploadOutputPayload:
        try:
            # 1. Sanity check
            should_abort, error_message = self._sanity_check_paths_and_free_storage(
                input_payload
            )
            if should_abort:
                async with self.db_session_factory.new_session() as db_session:
                    async with safe_commit(
                        db_session,
                        context="upload failed status DB update",
                        raise_on_fail=True,
                    ):
                        await DALPhotobooks.update_by_id(
                            db_session,
                            input_payload.originating_photobook_id,
                            DAOPhotobooksUpdate(
                                status=PhotobookStatus.UPLOAD_FAILED,
                            ),
                        )
                raise FileNotFoundError(error_message)

            # Begin compression and upload
            # 2. Insert initial objects
            media_paths_asset_uuids_map: dict[Path, UUID] = dict()
            async with self.db_session_factory.new_session() as db_session:
                async with safe_commit(
                    db_session,
                    context="photobook asset compression and upload status DB update",
                    raise_on_fail=False,
                ):
                    await DALPhotobooks.update_by_id(
                        db_session,
                        input_payload.originating_photobook_id,
                        DAOPhotobooksUpdate(
                            status=PhotobookStatus.UPLOADING,
                        ),
                    )

                async with safe_commit(
                    db_session,
                    context="initializing asset objects",
                    raise_on_fail=True,
                ):
                    dao_creates: list[DAOAssetsCreate] = []
                    for _idx in range(len(input_payload.absolute_media_paths)):
                        dao_creates.append(
                            DAOAssetsCreate(
                                user_id=input_payload.user_id,
                                asset_key_original=None,
                                asset_key_display=None,
                                asset_key_llm=None,
                                metadata_json=None,
                                original_photobook_id=input_payload.originating_photobook_id,
                                upload_status=AssetUploadStatus.PENDING,
                            )
                        )
                    daos = await DALAssets.create_many(db_session, dao_creates)

                for media_path, dao in zip(input_payload.absolute_media_paths, daos):
                    media_paths_asset_uuids_map[media_path] = dao.id

            # Step 3: launch LLM-level photo quality compression tasks first
            await self._compress_and_upload_safe(
                tier=CompressionTier.LLM,
                media_paths_asset_uuids_map=media_paths_asset_uuids_map,
                root_path=input_payload.root_tempdir,
                originating_photobook_id=input_payload.originating_photobook_id,
            )

            # Step 4: Enqueue job for LLM creation
            async with self.db_session_factory.new_session() as db_session:
                async with JobManager(
                    self.remote_redis_client_factory,
                    JobQueue.REMOTE_MAIN_TASK_QUEUE,
                ) as job_manager:
                    enqueued_remote_job_id = await job_manager.enqueue(
                        job_type=JobType.REMOTE_PHOTOBOOK_GENERATION,
                        job_payload=PhotobookGenerationInputPayload(
                            user_id=input_payload.user_id,
                            originating_photobook_id=input_payload.originating_photobook_id,
                            asset_ids=[
                                asset_id
                                for asset_id in media_paths_asset_uuids_map.values()
                            ],
                        ),
                        max_retries=2,
                        db_session=db_session,
                    )

            # Step 5: In parallel compress, and upload displayed version + original copies
            upload_original_copies_task = asyncio.create_task(
                self._upload_to_asset_storage_persisting_metadata(
                    media_paths_asset_uuids_map=media_paths_asset_uuids_map,
                    media_paths_compressed_paths_map={  # No compression, identical with media_paths_asset_uuids_map
                        k: k for k in media_paths_asset_uuids_map.keys()
                    },
                    originating_photobook_id=input_payload.originating_photobook_id,
                    asset_key_type="asset_key_original",
                )
            )
            highend_display_compress_upload_task = asyncio.create_task(
                self._compress_and_upload_safe(
                    tier=CompressionTier.HIGH_END_DISPLAY,
                    media_paths_asset_uuids_map=media_paths_asset_uuids_map,
                    root_path=input_payload.root_tempdir,
                    originating_photobook_id=input_payload.originating_photobook_id,
                )
            )

            upload_orig, _upload_highend_display = await asyncio.gather(
                upload_original_copies_task,
                highend_display_compress_upload_task,
                return_exceptions=True,
            )
            if isinstance(upload_orig, Exception):
                # Permanent failure
                async with self.db_session_factory.new_session() as db_session:
                    async with safe_commit(
                        db_session,
                        context="upload failed status DB update",
                        raise_on_fail=True,
                    ):
                        await DALPhotobooks.update_by_id(
                            db_session,
                            input_payload.originating_photobook_id,
                            DAOPhotobooksUpdate(
                                status=PhotobookStatus.UPLOAD_FAILED,
                            ),
                        )

            return AssetCompressUploadOutputPayload(
                job_id=self.job_id,
                enqueued_photobook_creation_remote_job_id=enqueued_remote_job_id,
            )
        finally:
            # Step 6: cleanup tempdir
            try:
                shutil.rmtree(input_payload.root_tempdir, ignore_errors=True)
            except Exception as e:
                logging.warning(
                    f"[AssetCompressUploadLocalJobProcessor] Failed to clean up "
                    f"tempdir {input_payload.root_tempdir}: {e}"
                )
