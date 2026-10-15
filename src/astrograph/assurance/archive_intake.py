"""Read-only ZIP intake scanner for research datasets.

The scanner validates archive identity and structure without deserialising any
pickle/joblib object.  It is an intake gate, not a semantic dataset validator.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Optional


class IntakeStatus(str, Enum):
    PASS = "PASS"
    QUARANTINE_REQUIRED = "QUARANTINE_REQUIRED"
    REJECT = "REJECT"


_PICKLE_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".dill"}
_EXECUTABLE_EXTENSIONS = {".exe", ".dll", ".so", ".dylib", ".bat", ".cmd", ".ps1"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unsafe_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return path.is_absolute() or ".." in path.parts or bool(re.match(r"^[A-Za-z]:", name))


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


@dataclass(frozen=True)
class ArchiveIntakeRecord:
    dataset_id: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    member_count: int
    total_uncompressed_bytes: int
    extension_counts: dict[str, int]
    duplicate_member_names: tuple[str, ...]
    unsafe_member_names: tuple[str, ...]
    symlink_members: tuple[str, ...]
    encrypted_members: tuple[str, ...]
    pickle_members: tuple[str, ...]
    executable_members: tuple[str, ...]
    bad_crc_member: Optional[str]
    max_compression_ratio: float
    central_directory_fingerprint_sha256: str
    status: IntakeStatus
    reasons: tuple[str, ...]
    schema_version: str = "astrograph-dataset-archive-intake.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "archive_path": self.archive_path,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "member_count": self.member_count,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "extension_counts": dict(sorted(self.extension_counts.items())),
            "duplicate_member_names": list(self.duplicate_member_names),
            "unsafe_member_names": list(self.unsafe_member_names),
            "symlink_members": list(self.symlink_members),
            "encrypted_members": list(self.encrypted_members),
            "pickle_members": list(self.pickle_members),
            "executable_members": list(self.executable_members),
            "bad_crc_member": self.bad_crc_member,
            "max_compression_ratio": self.max_compression_ratio,
            "central_directory_fingerprint_sha256": self.central_directory_fingerprint_sha256,
            "status": self.status.value,
            "reasons": list(self.reasons),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def scan_zip_archive(path: Path | str, dataset_id: str, perform_crc_test: bool = True) -> ArchiveIntakeRecord:
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_sha = _sha256_file(archive)
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        bad_crc = handle.testzip() if perform_crc_test else None

    names = [info.filename for info in infos]
    counts = Counter(names)
    duplicates = tuple(sorted(name for name, count in counts.items() if count > 1))
    unsafe = tuple(sorted(name for name in names if _unsafe_name(name)))
    symlinks = tuple(sorted(info.filename for info in infos if _is_symlink(info)))
    encrypted = tuple(sorted(info.filename for info in infos if info.flag_bits & 0x1))
    ext_counts = Counter()
    pickles = []
    executables = []
    max_ratio = 0.0
    metadata = []
    total_uncompressed = 0
    for info in infos:
        ext = PurePosixPath(info.filename).suffix.lower() or "<none>"
        ext_counts[ext] += 1
        if ext in _PICKLE_EXTENSIONS:
            pickles.append(info.filename)
        if ext in _EXECUTABLE_EXTENSIONS:
            executables.append(info.filename)
        total_uncompressed += info.file_size
        ratio = info.file_size / max(info.compress_size, 1)
        max_ratio = max(max_ratio, ratio)
        metadata.append({
            "name": info.filename,
            "crc": info.CRC,
            "compressed": info.compress_size,
            "uncompressed": info.file_size,
            "method": info.compress_type,
            "flags": info.flag_bits,
        })

    reasons = []
    status = IntakeStatus.PASS
    if bad_crc is not None:
        status = IntakeStatus.REJECT
        reasons.append("CRC_FAILURE")
    if unsafe:
        status = IntakeStatus.REJECT
        reasons.append("PATH_TRAVERSAL_OR_ABSOLUTE_MEMBER")
    if symlinks:
        status = IntakeStatus.REJECT
        reasons.append("SYMLINK_MEMBER")
    if encrypted:
        status = IntakeStatus.REJECT
        reasons.append("ENCRYPTED_MEMBER")
    if duplicates:
        status = IntakeStatus.REJECT
        reasons.append("DUPLICATE_MEMBER_NAME")
    if status is not IntakeStatus.REJECT and pickles:
        status = IntakeStatus.QUARANTINE_REQUIRED
        reasons.append("PICKLE_DESERIALISATION_REQUIRES_ISOLATED_CONVERSION")
    if status is not IntakeStatus.REJECT and executables:
        status = IntakeStatus.QUARANTINE_REQUIRED
        reasons.append("EXECUTABLE_MEMBER_REQUIRES_REVIEW")
    if max_ratio > 1000.0 and total_uncompressed > 100 * 1024 * 1024:
        if status is IntakeStatus.PASS:
            status = IntakeStatus.QUARANTINE_REQUIRED
        reasons.append("EXTREME_COMPRESSION_RATIO_REQUIRES_REVIEW")

    return ArchiveIntakeRecord(
        dataset_id=dataset_id,
        archive_path=str(archive),
        archive_sha256=archive_sha,
        archive_size_bytes=archive.stat().st_size,
        member_count=len(infos),
        total_uncompressed_bytes=total_uncompressed,
        extension_counts=dict(ext_counts),
        duplicate_member_names=duplicates,
        unsafe_member_names=unsafe,
        symlink_members=symlinks,
        encrypted_members=encrypted,
        pickle_members=tuple(sorted(pickles)),
        executable_members=tuple(sorted(executables)),
        bad_crc_member=bad_crc,
        max_compression_ratio=round(max_ratio, 6),
        central_directory_fingerprint_sha256=_canonical_sha256(sorted(metadata, key=lambda x: x["name"])),
        status=status,
        reasons=tuple(reasons),
    )
