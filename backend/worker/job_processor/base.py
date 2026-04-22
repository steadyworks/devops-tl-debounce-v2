from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from backend.db.session.factory import AsyncSessionFactory
from backend.lib.asset_manager.base import AssetManager

from .types import JobInputPayload, JobOutputPayload

TInputPayload = TypeVar("TInputPayload", bound=JobInputPayload, contravariant=True)
TOutputPayload = TypeVar("TOutputPayload", bound=JobOutputPayload, covariant=True)


class AbstractJobProcessor(Generic[TInputPayload, TOutputPayload], ABC):
    def __init__(
        self,
        job_id: UUID,
        asset_manager: AssetManager,
        db_session_factory: AsyncSessionFactory,
    ) -> None:
        self.job_id = job_id
        self.asset_manager = asset_manager
        self.db_session_factory = db_session_factory

    @abstractmethod
    async def process(self, input_payload: TInputPayload) -> TOutputPayload: ...
