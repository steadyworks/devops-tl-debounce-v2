from typing import TypeVar
from uuid import UUID

from backend.db.session.factory import AsyncSessionFactory
from backend.lib.asset_manager.base import AssetManager
from backend.lib.redis.factory import RedisClientFactory

from .base import AbstractJobProcessor
from .types import JobInputPayload, JobOutputPayload

TInputPayload = TypeVar("TInputPayload", bound=JobInputPayload, contravariant=True)
TOutputPayload = TypeVar("TOutputPayload", bound=JobOutputPayload, covariant=True)


class LocalJobProcessor(AbstractJobProcessor[TInputPayload, TOutputPayload]):
    def __init__(
        self,
        job_id: UUID,
        asset_manager: AssetManager,
        db_session_factory: AsyncSessionFactory,
        remote_redis_client_factory: RedisClientFactory,
    ) -> None:
        super().__init__(job_id, asset_manager, db_session_factory)
        self.remote_redis_client_factory = remote_redis_client_factory
