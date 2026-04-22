from backend.db.session.factory import AsyncSessionFactory
from backend.lib.job_manager.types import JobQueue
from backend.lib.redis.factory import RedisClientFactory

from .base import AbstractWorkerProcess


class RemoteJobWorkerProcess(AbstractWorkerProcess):
    def _get_num_concurrent_worker_tasks(self) -> int:
        return 6  # We expect remote jobs to be mainly IO bound

    def _get_job_queue(self) -> JobQueue:
        return JobQueue.REMOTE_MAIN_TASK_QUEUE

    def _create_redis_client_factory(self) -> RedisClientFactory:
        return RedisClientFactory.from_remote_defaults()

    def _create_db_session_factory(self) -> AsyncSessionFactory:
        return AsyncSessionFactory()
