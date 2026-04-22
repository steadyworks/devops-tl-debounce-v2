from enum import Enum


class JobQueue(Enum):
    # Local queues (same host)
    LOCAL_MAIN_TASK_QUEUE = "local_main_task_queue"

    # Remote queues (global)
    REMOTE_MAIN_TASK_QUEUE = "remote_main_task_queue"
