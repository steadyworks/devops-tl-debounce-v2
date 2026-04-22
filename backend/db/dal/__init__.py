from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.data_models import (
    DAOAssets,
    DAOJobEvents,
    DAOJobs,
    DAOPages,
    DAOPagesAssetsRel,
    DAOPhotobookBookmarks,
    DAOPhotobooks,
    DAOUsers,
)
from backend.lib.utils.common import get_host_info

from .base import (
    AsyncPostgreSQLDAL,
    FilterOp,
    InvalidFilterFieldError,
    OrderDirection,
    safe_commit,
)
from .schemas import (
    DAOAssetsCreate,
    DAOAssetsUpdate,
    DAOJobEventsCreate,
    DAOJobEventsUpdate,
    DAOJobsCreate,
    DAOJobsUpdate,
    DAOPagesAssetsRelCreate,
    DAOPagesAssetsRelUpdate,
    DAOPagesCreate,
    DAOPagesUpdate,
    DAOPhotobookBookmarksCreate,
    DAOPhotobookBookmarksUpdate,
    DAOPhotobooksCreate,
    DAOPhotobooksUpdate,
    DAOUsersCreate,
    DAOUsersUpdate,
)


class DALAssets(AsyncPostgreSQLDAL[DAOAssets, DAOAssetsCreate, DAOAssetsUpdate]):
    model = DAOAssets


class DALJobs(AsyncPostgreSQLDAL[DAOJobs, DAOJobsCreate, DAOJobsUpdate]):
    model = DAOJobs


class DALJobEvents(
    AsyncPostgreSQLDAL[DAOJobEvents, DAOJobEventsCreate, DAOJobEventsUpdate]
):
    model = DAOJobEvents

    @classmethod
    async def create(
        cls, session: AsyncSession, obj_in: DAOJobEventsCreate
    ) -> DAOJobEvents:
        if obj_in.host is None:
            obj_in.host = get_host_info()
        return await super().create(session, obj_in)


class DALPages(AsyncPostgreSQLDAL[DAOPages, DAOPagesCreate, DAOPagesUpdate]):
    model = DAOPages


class DALPagesAssetsRel(
    AsyncPostgreSQLDAL[
        DAOPagesAssetsRel, DAOPagesAssetsRelCreate, DAOPagesAssetsRelUpdate
    ]
):
    model = DAOPagesAssetsRel


class DALPhotobooks(
    AsyncPostgreSQLDAL[DAOPhotobooks, DAOPhotobooksCreate, DAOPhotobooksUpdate]
):
    model = DAOPhotobooks


class DALUsers(AsyncPostgreSQLDAL[DAOUsers, DAOUsersCreate, DAOUsersUpdate]):
    model = DAOUsers


class DALPhotobookBookmarks(
    AsyncPostgreSQLDAL[
        DAOPhotobookBookmarks, DAOPhotobookBookmarksCreate, DAOPhotobookBookmarksUpdate
    ]
):
    model = DAOPhotobookBookmarks


__all__ = [
    # DALs
    "DALAssets",
    "DALJobs",
    "DALJobEvents",
    "DALPages",
    "DALPagesAssetsRel",
    "DALPhotobooks",
    "DALPhotobookBookmarks",
    # DAL base
    "AsyncPostgreSQLDAL",
    "FilterOp",
    "InvalidFilterFieldError",
    "OrderDirection",
    # ORM objects
    "DAOAssets",
    "DAOJobs",
    "DAOJobEvents",
    "DAOPages",
    "DAOPagesAssetsRel",
    "DAOPhotobooks",
    "DAOUsers",
    "DAOPhotobookBookmarks",
    # Schemas
    "DAOAssetsCreate",
    "DAOAssetsUpdate",
    "DAOJobsCreate",
    "DAOJobsUpdate",
    "DAOJobEventsCreate",
    # "DAOJobEventsUpdate",   # Updating job events is not allowed, as it's an append-only log trail
    "DAOPagesCreate",
    "DAOPagesUpdate",
    "DAOPagesAssetsRelCreate",
    "DAOPagesAssetsRelUpdate",
    "DAOPhotobooksCreate",
    "DAOPhotobooksUpdate",
    "DAOUsersCreate",
    "DAOUsersUpdate",
    "DAOPhotobookBookmarksUpdate",
    "DAOPhotobookBookmarksCreate",
    # Utils
    "safe_commit",
]
