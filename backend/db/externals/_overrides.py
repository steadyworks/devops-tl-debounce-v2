# pyright: reportPrivateUsage=false

from typing import Optional, Self, Sequence

from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.dal import DALAssets
from backend.db.data_models import DAOAssets, DAOPhotobooks
from backend.db.utils.common import retrieve_available_asset_key_in_order_of
from backend.lib.asset_manager.base import AssetManager

from ._generated_DO_NOT_USE import _AssetsOverviewResponse, _PhotobooksOverviewResponse


class AssetsOverviewResponse(_AssetsOverviewResponse):
    asset_key_original: Optional[str] = Field(exclude=True)
    asset_key_display: Optional[str] = Field(exclude=True)
    asset_key_llm: Optional[str] = Field(exclude=True)
    signed_asset_url: str

    @classmethod
    async def rendered_from_dao(
        cls,
        dao: DAOAssets,
        asset_manager: AssetManager,
    ) -> Self:
        signed_url = await asset_manager.generate_signed_url(
            retrieve_available_asset_key_in_order_of(
                dao,
                [
                    "asset_key_display",
                    "asset_key_llm",
                    "asset_key_original",
                ],
            )
        )
        return cls(
            **dao.model_dump(),
            signed_asset_url=signed_url,
        )


class PhotobooksOverviewResponse(_PhotobooksOverviewResponse):
    thumbnail_asset_signed_url: Optional[str]

    @classmethod
    async def rendered_from_dao(
        cls: type[Self],
        dao: DAOPhotobooks,
        db_session: AsyncSession,
        asset_manager: AssetManager,
    ) -> Self:
        thumbnail_signed_url = None
        if dao.thumbnail_asset_id is not None:
            thumbnail_asset = await DALAssets.get_by_id(
                db_session, dao.thumbnail_asset_id
            )
            if thumbnail_asset is not None:
                thumbnail_signed_url = await asset_manager.generate_signed_url(
                    retrieve_available_asset_key_in_order_of(
                        thumbnail_asset,
                        ["asset_key_llm", "asset_key_display", "asset_key_original"],
                    )
                )
        return cls(
            **dao.model_dump(),
            thumbnail_asset_signed_url=thumbnail_signed_url,
        )

    @classmethod
    async def rendered_from_daos(
        cls: type[Self],
        daos: Sequence[DAOPhotobooks],
        db_session: AsyncSession,
        asset_manager: AssetManager,
    ) -> list[Self]:
        # Step 4: Collect all asset_ids used
        thumbnail_asset_ids = [
            dao.thumbnail_asset_id for dao in daos if dao.thumbnail_asset_id is not None
        ]
        thumbnail_asset_list = await DALAssets.get_by_ids(
            db_session, thumbnail_asset_ids
        )
        thumbnail_assets_by_ids = {asset.id: asset for asset in thumbnail_asset_list}

        # Step 5: Generate signed URLs for original asset keys
        uuid_asset_keys_map = {
            asset.id: retrieve_available_asset_key_in_order_of(
                asset,
                [
                    "asset_key_llm",
                    "asset_key_display",
                    "asset_key_original",
                ],
            )
            for asset in thumbnail_asset_list
        }
        signed_urls = await asset_manager.generate_signed_urls_batched(
            list(uuid_asset_keys_map.values())
        )

        rendered_resps: list[Self] = []
        for dao in daos:
            thumbnail_signed_url: Optional[str] = None
            if dao.thumbnail_asset_id is not None:
                thumbnail_asset = thumbnail_assets_by_ids.get(dao.thumbnail_asset_id)
                if thumbnail_asset is not None:
                    thumbnail_signed_url_or_exception = signed_urls.get(
                        uuid_asset_keys_map[thumbnail_asset.id]
                    )
                    if isinstance(thumbnail_signed_url_or_exception, str):
                        thumbnail_signed_url = thumbnail_signed_url_or_exception

            resp = cls(
                **dao.model_dump(),
                thumbnail_asset_signed_url=thumbnail_signed_url,
            )
            rendered_resps.append(resp)
        return rendered_resps
