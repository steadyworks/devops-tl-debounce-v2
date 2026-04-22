from typing import Optional

from backend.db.session.factory import AsyncSessionFactory
from backend.lib.job_manager.types import JobQueue
from backend.lib.redis.factory import RedisClientFactory

from .base import AbstractWorkerProcess


class LocalJobWorkerProcess(AbstractWorkerProcess):
    _remote_redis_client_factory: Optional[RedisClientFactory]

    def _get_num_concurrent_worker_tasks(self) -> int:
        # We expect local jobs to be mainly CPU bound, thus limiting concurrency.
        # We will spin up 2 processes for process level crash isolation.
        return 1

    def _get_job_queue(self) -> JobQueue:
        return JobQueue.LOCAL_MAIN_TASK_QUEUE

    def _create_redis_client_factory(self) -> RedisClientFactory:
        return RedisClientFactory.from_local_defaults()

    def _create_db_session_factory(self) -> AsyncSessionFactory:
        return AsyncSessionFactory()

    def _initialize_process_level_resource(self) -> None:
        self._remote_redis_client_factory = RedisClientFactory.from_remote_defaults()
