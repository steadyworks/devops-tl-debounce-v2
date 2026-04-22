import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import piexif
from pyvips import Image

from .types import CompressionTier

logging.getLogger("pyvips").setLevel(logging.WARNING)


DEFAULT_TIMEOUT_PER_IMAGE_SECS = 2.5


@dataclass(frozen=True)
class CompressionSettings:
    max_size: int
    quality: int


COMPRESSION_SETTING_PRESETS: dict[CompressionTier, CompressionSettings] = {
    CompressionTier.HIGH_END_DISPLAY: CompressionSettings(max_size=2048, quality=85),
    CompressionTier.MOBILE_DISPLAY: CompressionSettings(max_size=1250, quality=75),
    CompressionTier.LLM: CompressionSettings(max_size=800, quality=75),
    CompressionTier.THUMBNAIL: CompressionSettings(max_size=400, quality=60),
}


class ImageProcessingLibrary:
    def __init__(
        self,
        max_concurrent: int,
        timeout_secs: int | float = DEFAULT_TIMEOUT_PER_IMAGE_SECS,
    ) -> None:
        self._sema = asyncio.Semaphore(max_concurrent)
        self._timeout_secs = timeout_secs

    async def compress_image_on_thread(
        self,
        input_paths: Sequence[str | Path],
        output_dir: str | Path,
        format: Literal["jpeg", "webp"],
        max_size: int,
        quality: int,
        strip_metadata: bool,
        processed_filename_suffix: str,
    ) -> dict[Path, tuple[bool, Optional[Path]]]:
        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"[ImageProcessingLibrary] Output directory does not exist: {output_dir}"
            )

        results: dict[Path, tuple[bool, Optional[Path]]] = {}
        for input_path in input_paths:
            input_path = Path(input_path)
            base_name = input_path.stem
            async with self._sema:
                try:
                    success, output_path = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._compress_image_sync,
                            input_path,
                            output_dir,
                            base_name,
                            format,
                            max_size,
                            quality,
                            strip_metadata,
                            processed_filename_suffix=processed_filename_suffix,
                        ),
                        timeout=self._timeout_secs,
                    )
                    results[input_path] = (success, output_path if success else None)
                except asyncio.TimeoutError:
                    logging.warning(f"[ImageLibrary] Timeout on {input_path}")
                    results[input_path] = (False, None)
                except Exception as e:
                    logging.exception(
                        f"[ImageLibrary] Compression failed for {input_path}: {e}"
                    )
                    results[input_path] = (False, None)

        return results

    async def compress_by_tier_on_thread(
        self,
        input_paths: Sequence[str | Path],
        output_dir: str | Path,
        format: Literal["jpeg", "webp"],
        tier: CompressionTier,
        strip_metadata: bool,
    ) -> dict[Path, tuple[bool, Optional[Path]]]:
        settings = COMPRESSION_SETTING_PRESETS[tier]
        return await self.compress_image_on_thread(
            input_paths=input_paths,
            output_dir=output_dir,
            format=format,
            max_size=settings.max_size,
            quality=settings.quality,
            strip_metadata=strip_metadata,
            processed_filename_suffix=tier.value,
        )

    async def compress_many_tiers(
        self,
        tiers: Sequence[CompressionTier],
        input_paths: Sequence[str | Path],
        output_dir: str | Path,
        format: Literal["jpeg", "webp"] = "jpeg",
        strip_metadata: bool = False,
    ) -> dict[CompressionTier, dict[Path, tuple[bool, Optional[Path]]]]:
        output_dir = Path(output_dir)
        results: dict[CompressionTier, dict[Path, tuple[bool, Optional[Path]]]] = {}

        for tier in tiers:
            try:
                tier_result: dict[
                    Path, tuple[bool, Optional[Path]]
                ] = await self.compress_by_tier_on_thread(
                    tier=tier,
                    input_paths=input_paths,
                    output_dir=output_dir,
                    format=format,
                    strip_metadata=strip_metadata,
                )
            except Exception as e:
                logging.warning(
                    f"[ImageLibrary] Unexpected error on tier {tier.value}: {e}"
                )
                tier_result = {Path(p): (False, None) for p in input_paths}
            results[tier] = tier_result

        return results

    def _compress_image_sync(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        base_name: str,
        format: str,
        max_size: int,
        quality: int,
        strip_metadata: bool,
        processed_filename_suffix: str,
    ) -> tuple[bool, Optional[Path]]:
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        # Normalize format
        format = format.lower()
        normalized_format = "jpeg" if format in {"jpeg", "jpg"} else format
        ext = "jpg" if normalized_format == "jpeg" else normalized_format
        output_path = output_dir / f"{base_name}.{processed_filename_suffix}.{ext}"

        try:
            image = Image.new_from_file(str(input_path), access="sequential")
            image_format = image.format.lower()

            is_jpeg = image_format in {"jpeg", "jpg"}
            is_webp = image_format == "webp"
            format_matches = (normalized_format == "jpeg" and is_jpeg) or (
                normalized_format == "webp" and is_webp
            )

            # Determine if autorotation is needed
            try:
                orientation = image.get("orientation")
                autorot_needed = orientation != 1
            except Exception:
                autorot_needed = False  # Assume safe default

            # Check if compression is needed
            no_resize_needed = max(image.width, image.height) <= max_size

            if (
                no_resize_needed
                and format_matches
                and not strip_metadata
                and not autorot_needed
            ):
                try:
                    shutil.copy2(input_path, output_path)
                    logging.info(
                        f"[ImageLibrary] Skipped compression; copied to {output_path}"
                    )
                    return True, output_path
                except Exception as copy_err:
                    logging.warning(f"[ImageLibrary] Failed to copy: {copy_err}")
                    return False, None

            # Prepare EXIF if needed
            exif_bytes: Optional[bytes] = None
            if normalized_format == "jpeg" and not strip_metadata:
                try:
                    exif_dict = piexif.load(str(input_path))
                    exif_bytes = piexif.dump(exif_dict)
                except Exception as exif_err:
                    logging.debug(f"[ImageLibrary] Could not read EXIF: {exif_err}")

            # Perform autorotation and resizing
            image = image.autorot()
            scale = self._compute_scale(image.width, image.height, max_size)
            if scale < 1.0:
                image = image.resize(scale)

            # Save compressed image
            save_opts = self._get_save_options(
                normalized_format, quality, strip_metadata
            )
            image.write_to_file(str(output_path), **save_opts)

            # Reinsert EXIF
            if exif_bytes:
                try:
                    piexif.insert(exif_bytes, str(output_path))
                except Exception as insert_err:
                    logging.warning(
                        f"[ImageLibrary] Failed to reinsert EXIF: {insert_err}"
                    )

            logging.debug(f"[ImageLibrary] Compressed image written to {output_path}")
            return True, output_path

        except Exception as e:
            logging.warning(f"[ImageLibrary] Compression failed for {input_path}: {e}")
            return False, None

    @staticmethod
    def _compute_scale(width: int, height: int, max_size: int) -> float:
        return min(1.0, max_size / max(width, height))

    @staticmethod
    def _get_save_options(
        format: str, quality: int, strip: bool
    ) -> dict[str, int | bool]:
        opts: dict[str, Any] = {"Q": quality, "strip": strip}
        if format == "jpeg":
            return opts
        elif format == "webp":
            return {**opts, "lossless": False}
        else:
            raise ValueError(f"Unsupported output format: {format}")
