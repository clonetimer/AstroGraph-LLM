"""Fetch only hash-pinned Win7/Python 3.8 build materials.

Build-host utility only. Never bundled into the end-user startup path or used
to download dependencies at runtime. Materials are the audited lockfile list.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit
import urllib.request
import zipfile

from build_win7 import RUNTIME_SHA256, RUNTIME_URL, verified_wheels

MAX_RUNTIME_BYTES = 40 * 1024 * 1024
MAX_WHEEL_BYTES = 30 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pinned(url: str, target: Path, digest: str, *, maximum: int, hosts: set[str]) -> dict:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.password:
        raise ValueError("UNTRUSTED_VENDOR_DOWNLOAD_URL")
    if len(digest) != 64 or not all(ch in "0123456789abcdef" for ch in digest):
        raise ValueError("INVALID_PINNED_DIGEST")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or sha256(target) != digest:
        staging = target.with_name(target.name + ".download")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AstroGraph-Win7-hash-pinned-build/1"})
            with urllib.request.urlopen(request, timeout=120) as response, staging.open("wb") as out:
                if response.geturl() != url:
                    raise ValueError("VENDOR_REDIRECT_NOT_PERMITTED")
                total = 0
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    total += len(block)
                    if total > maximum:
                        raise ValueError("VENDOR_ARTIFACT_TOO_LARGE")
                    out.write(block)
            if sha256(staging) != digest:
                raise ValueError("VENDOR_SHA256_MISMATCH")
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
    return {"filename": target.name, "url": url, "bytes": target.stat().st_size, "sha256": sha256(target)}


def prepare(output_dir: Path, *, lock_path: Path) -> dict:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("python") != "3.8.10" or not isinstance(lock.get("wheels"), list):
        raise ValueError("UNSUPPORTED_LOCKFILE_VERSION")
    runtime = root / "python-3.8.10-embed-amd64.zip"
    runtime_receipt = fetch_pinned(
        RUNTIME_URL, runtime, RUNTIME_SHA256, maximum=MAX_RUNTIME_BYTES,
        hosts={"www.python.org"},
    )
    with zipfile.ZipFile(runtime) as z:
        if z.testzip() is not None or "python.exe" not in z.namelist():
            raise ValueError("INVALID_OFFICIAL_PY38_RUNTIME")
    wheels_dir = root / "wheels-win38"
    wheels_dir.mkdir(exist_ok=True)
    receipts = []
    for item in lock["wheels"]:
        name = item["filename"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("UNSAFE_LOCKFILE_FILENAME")
        receipts.append(fetch_pinned(
            item["url"], wheels_dir / name, item["sha256"], maximum=MAX_WHEEL_BYTES,
            hosts={"files.pythonhosted.org"},
        ))
    verified_wheels(wheels_dir, lock_path)
    receipt = {
        "schema_version": "astrograph-win7-vendor-provenance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "Windows 7 SP1 x64 compatibility trial / Python 3.8.10",
        "runtime": runtime_receipt,
        "wheels": receipts,
        "offline_target_installation": True,
        "native_windows7_acceptance": False,
        "sources": ["python.org", "files.pythonhosted.org"],
        "next_steps": ["offline package build", "Windows Server CI", "native Win7 field test"],
    }
    (root / "win7-vendor-provenance.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "runtime": str(runtime), "wheels_dir": str(wheels_dir),
        "runtime_sha256": runtime_receipt["sha256"], "wheel_count": len(receipts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lockfile", type=Path, default=Path(__file__).with_name("win7_dependencies.lock.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, lock_path=args.lockfile), indent=2))


if __name__ == "__main__":
    main()
