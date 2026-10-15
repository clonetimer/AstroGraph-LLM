"""Offline integrity checker for this source delivery (excluding the manifest itself)."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "DELIVERY_FILES_SHA256.txt"


def main() -> int:
    if not MANIFEST.is_file():
        print("FAIL: missing DELIVERY_FILES_SHA256.txt")
        return 1
    entries = 0
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, sep, name = raw.partition("  ")
        rel = Path(name)
        if not sep or not expected or rel.is_absolute() or ".." in rel.parts:
            print(f"FAIL: invalid manifest entry {name!r}")
            return 1
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            print(f"FAIL: missing or symlink {name}")
            return 1
        if sha256(path.read_bytes()).hexdigest() != expected:
            print(f"FAIL: hash mismatch {name}")
            return 1
        entries += 1
    print(f"PASS: {entries} delivery source file hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
