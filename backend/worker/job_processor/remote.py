from typing import TypeVar

from .base import AbstractJobProcessor
from .types import JobInputPayload, JobOutputPayload

TInputPayload = TypeVar("TInputPayload", bound=JobInputPayload, contravariant=True)
TOutputPayload = TypeVar("TOutputPayload", bound=JobOutputPayload, covariant=True)


class RemoteJobProcessor(AbstractJobProcessor[TInputPayload, TOutputPayload]):
    pass
