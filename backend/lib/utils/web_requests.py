import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import List, Optional

from fastapi import UploadFile

from backend.lib.types.asset import Asset

UserOriginalFileName = str


@dataclass
class TempUploadEntry:
    original_filename: UserOriginalFileName
    absolute_path: Path


@dataclass
class TempUploadsMetadata:
    root_dir: Path
    files: list[TempUploadEntry]


async def save_uploads_to_tempdir(
    upload_files: list[UploadFile],
    tmp_root: Path = Path("/tmp"),
) -> TempUploadsMetadata:
    temp_dir = tmp_root / uuid.uuid4().hex

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_entries: list[TempUploadEntry] = []

    for upload_file in upload_files:
        original_name = upload_file.filename or f"unnamed_{uuid.uuid4().hex}.bin"
        ext = Path(original_name).suffix or ".bin"
        safe_name = f"{uuid.uuid4().hex}{ext}"
        temp_path = temp_dir / safe_name
        contents = await upload_file.read()

        def write_bytes(_path: Path, _data: bytes) -> None:
            with open(_path, "wb") as f:
                f.write(_data)

        await asyncio.to_thread(write_bytes, temp_path, contents)

        temp_file_entries.append(
            TempUploadEntry(original_filename=original_name, absolute_path=temp_path)
        )

    return TempUploadsMetadata(root_dir=temp_dir, files=temp_file_entries)


def cleanup_tempdir(temp_dir: Path) -> None:
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logging.warning(f"Failed to cleanup tempdir {temp_dir}: {e}")


class UploadFileTempDirManager_DEPRECATED:
    def __init__(
        self, job_id: str, upload_files: List[UploadFile], tmp_root: Path = Path("/tmp")
    ):
        self.upload_files = upload_files
        self.tmp_root = tmp_root
        self.temp_dir: Path = tmp_root / job_id
        self.managed_assets: list[tuple[UserOriginalFileName, Asset]] = []

    async def __aenter__(self) -> list[tuple[UserOriginalFileName, Asset]]:
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        for upload_file in self.upload_files:
            # Fallbacks for missing filename or content_type
            original_name = upload_file.filename or f"unnamed_{uuid.uuid4().hex}.bin"
            ext = Path(original_name).suffix or ".bin"
            safe_name = f"{uuid.uuid4().hex}{ext}"
            temp_path = self.temp_dir / safe_name
            contents = await upload_file.read()

            def write_bytes_to_file(_path: Path, _data: bytes) -> None:
                with open(_path, "wb") as f:
                    f.write(_data)

            await asyncio.to_thread(write_bytes_to_file, temp_path, contents)
            self.managed_assets.append(
                (
                    original_name,
                    Asset(
                        cached_local_path=temp_path,
                        asset_storage_key=None,
                    ),
                )
            )

        return self.managed_assets

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        return None
