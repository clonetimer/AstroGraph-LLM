from __future__ import annotations

import re
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True)
class ZipExtractionLimits:
    max_selected_members: int = 10_000
    max_single_file_bytes: int = 512 * 1024 * 1024
    max_total_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024
    max_compression_ratio: float = 500.0


class ZipExtractionBudget:
    def __init__(self, limits: ZipExtractionLimits | None = None) -> None:
        self.limits = limits or ZipExtractionLimits()
        self.member_count = 0
        self.total_uncompressed_bytes = 0

    def admit(self, info: zipfile.ZipInfo) -> None:
        self.member_count += 1
        self.total_uncompressed_bytes += int(info.file_size)
        if self.member_count > self.limits.max_selected_members:
            raise ValueError("ZIP selected member count exceeds configured limit")
        if info.file_size > self.limits.max_single_file_bytes:
            raise ValueError(f"ZIP member exceeds single-file limit: {info.filename}")
        if self.total_uncompressed_bytes > self.limits.max_total_uncompressed_bytes:
            raise ValueError("ZIP selected uncompressed size exceeds configured limit")
        if info.file_size > 0:
            compressed = max(1, int(info.compress_size))
            ratio = float(info.file_size) / compressed
            if ratio > self.limits.max_compression_ratio:
                raise ValueError(f"ZIP member compression ratio exceeds limit: {info.filename}")


def normalize_zip_member(info: zipfile.ZipInfo) -> str:
    """Return a safe POSIX member name or raise for unsafe archive metadata."""

    raw = info.filename.replace("\\", "/")
    if "\x00" in raw:
        raise ValueError("ZIP member contains NUL")
    if raw.startswith(("/", "//")) or _WINDOWS_DRIVE.match(raw):
        raise ValueError(f"ZIP member must be relative: {info.filename}")

    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"ZIP member contains unsafe path segment: {info.filename}")

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise ValueError(f"ZIP symbolic-link member is not allowed: {info.filename}")
    return path.as_posix()


def safe_destination(root: Path, relative: str) -> Path:
    root = root.resolve()
    destination = (root / PurePosixPath(relative)).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"ZIP member escapes extraction root: {relative}") from exc
    return destination


def extract_selected_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    output_root: Path,
    relative: str,
    budget: ZipExtractionBudget,
) -> Path:
    budget.admit(info)
    destination = safe_destination(output_root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with archive.open(info) as source, destination.open("wb") as target:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            written += len(block)
            if written > budget.limits.max_single_file_bytes:
                raise ValueError(f"ZIP member exceeds extraction byte limit: {info.filename}")
            target.write(block)
    if written != info.file_size:
        destination.unlink(missing_ok=True)
        raise ValueError(f"ZIP member size mismatch during extraction: {info.filename}")
    return destination
