from __future__ import annotations
import hashlib, os, stat, zipfile
from pathlib import Path
from typing import Iterable, Any
from .protocol import CapabilityProtocolError

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def resolve_scoped(path_value: str, roots: Iterable[Path]) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise CapabilityProtocolError("PATH_REQUIRED", "path must be a non-empty string")
    candidate=Path(path_value).expanduser().resolve()
    root_list=[Path(r).expanduser().resolve() for r in roots]
    if not any(candidate == r or r in candidate.parents for r in root_list):
        raise CapabilityProtocolError("PATH_OUTSIDE_ALLOWED_ROOTS", f"path {candidate} is outside configured roots")
    return candidate

def verify_file(path_value: str, expected_sha256: str, roots: Iterable[Path]) -> dict[str, Any]:
    path=resolve_scoped(path_value,roots)
    if not path.is_file():
        raise CapabilityProtocolError("ARTIFACT_NOT_FOUND", f"artifact not found: {path}")
    actual=sha256_file(path)
    if expected_sha256 and actual.lower()!=expected_sha256.lower():
        raise CapabilityProtocolError("SHA256_MISMATCH", "artifact SHA-256 mismatch", {"expected":expected_sha256,"actual":actual})
    return {"path":str(path),"sha256":actual,"bytes":path.stat().st_size,"verified":True}

def scan_zip(path_value: str, roots: Iterable[Path]) -> dict[str, Any]:
    path=resolve_scoped(path_value,roots)
    if not path.is_file():
        raise CapabilityProtocolError("ARCHIVE_NOT_FOUND", f"archive not found: {path}")
    findings=[]; members=[]; total=0
    with zipfile.ZipFile(path) as zf:
        bad=zf.testzip()
        if bad: findings.append({"severity":"ERROR","code":"CRC_FAILURE","member":bad})
        seen=set()
        for info in zf.infolist():
            name=info.filename; rel=Path(name)
            total += info.file_size
            if name in seen: findings.append({"severity":"ERROR","code":"DUPLICATE_MEMBER","member":name})
            seen.add(name)
            if rel.is_absolute() or ".." in rel.parts: findings.append({"severity":"ERROR","code":"PATH_TRAVERSAL","member":name})
            mode=(info.external_attr>>16)&0o177777
            if stat.S_ISLNK(mode): findings.append({"severity":"ERROR","code":"SYMLINK_MEMBER","member":name})
            lower=name.lower()
            if lower.endswith((".pkl",".pickle",".joblib",".dill")): findings.append({"severity":"WARN","code":"SERIALIZED_PYTHON_OBJECT","member":name})
            if info.flag_bits & 0x1: findings.append({"severity":"ERROR","code":"ENCRYPTED_MEMBER","member":name})
            ratio=(info.file_size/max(info.compress_size,1)) if info.file_size else 1.0
            if ratio>1000: findings.append({"severity":"WARN","code":"EXTREME_COMPRESSION_RATIO","member":name,"ratio":ratio})
            members.append({"name":name,"bytes":info.file_size,"compressed_bytes":info.compress_size})
    return {"path":str(path),"sha256":sha256_file(path),"zip_crc":"PASS" if not any(f["code"]=="CRC_FAILURE" for f in findings) else "FAIL","member_count":len(members),"uncompressed_bytes":total,"findings":findings,"admission":"REJECT" if any(f["severity"]=="ERROR" for f in findings) else ("QUARANTINE_REQUIRED" if any(f["code"]=="SERIALIZED_PYTHON_OBJECT" for f in findings) else "READ_ONLY_ACCEPT")}
