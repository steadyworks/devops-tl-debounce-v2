from typing import Any, Optional
from uuid import UUID

from backend.db.session.factory import AsyncSessionFactory
from backend.lib.asset_manager.base import AssetManager
from backend.lib.redis.factory import RedisClientFactory

from .asset_compress_upload import AssetCompressUploadLocalJobProcessor
from .base import AbstractJobProcessor
from .local import LocalJobProcessor
from .photobook_generation import PhotobookGenerationRemoteJobProcessor
from .remote import RemoteJobProcessor
from .types import JobInputPayload, JobOutputPayload, JobType

# Registry with erased generics
JOB_TYPE_JOB_PROCESSOR_REGISTRY: dict[
    JobType, type[AbstractJobProcessor[Any, JobOutputPayload]]
] = {
    # Local job processors
    JobType.LOCAL_ASSET_COMPRESS_UPLOAD: AssetCompressUploadLocalJobProcessor,
    # Remote job processors
    JobType.REMOTE_PHOTOBOOK_GENERATION: PhotobookGenerationRemoteJobProcessor,
}


class JobProcessorFactory:
    @classmethod
    def new_processor(
        cls,
        job_id: UUID,
        job_type: JobType,
        asset_manager: AssetManager,
        db_session_factory: AsyncSessionFactory,
        remote_redis_client_factory: Optional[RedisClientFactory],
    ) -> AbstractJobProcessor[JobInputPayload, JobOutputPayload]:
        processor_cls = JOB_TYPE_JOB_PROCESSOR_REGISTRY.get(job_type)
        if processor_cls is None:
            raise Exception(f"{job_id} not found")

        if issubclass(processor_cls, RemoteJobProcessor):
            return processor_cls(
                job_id=job_id,
                asset_manager=asset_manager,
                db_session_factory=db_session_factory,
            )
        elif issubclass(processor_cls, LocalJobProcessor):
            assert remote_redis_client_factory is not None
            return processor_cls(
                job_id=job_id,
                asset_manager=asset_manager,
                db_session_factory=db_session_factory,
                remote_redis_client_factory=remote_redis_client_factory,
            )

        raise Exception(
            f"{processor_cls} is not subclass of either RemoteJobProcessor"
            " or LocalJobProcessor"
        )
