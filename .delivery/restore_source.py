"""One-time, hash-pinned recovery of the delivered portable source. No network."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import lzma
import os
import subprocess
import tempfile

BASE = "5389bfd78f8a9f62117f8804eab181fda7557046"
DIGEST = "36fccffe42e5e8edade990426b1d8f28ea3e5bef341ea936e5ceb19af2a1b141"
FILES = {
    "docs/superpowers/plans/2026-11-02-win10-portable-workbench.md",
    "docs/superpowers/plans/2026-11-03-portable-closeout.md",
    "docs/superpowers/plans/2026-11-03-portable-v2.md",
    "docs/superpowers/specs/2026-11-02-win10-portable-workbench.md",
    "docs/superpowers/specs/2026-11-03-portable-v2.md",
    "portable/CHECK.bat", "portable/README_zh-CN.md", "portable/START.bat",
    "portable/assets/app.css", "portable/assets/app.js", "portable/assets/index.html",
    "portable/build.py", "portable/check_installation.py",
    "portable/examples/diagnosis_sample.txt", "portable/prepare_vendor.py",
    "portable/requirements.txt", "portable/test_closeout.py",
    "portable/test_installation.py", "portable/test_workbench.py", "portable/workbench.py",
    "src/astrograph/integrations/long_report_analysis.py",
}
PATCH_FILES = {
    "src/astrograph/integrations/external_diagnosis_analysis.py",
    "src/astrograph/integrations/external_diagnosis_report.py",
    "src/astrograph/retrieval/structured_document.py",
}

def restore(root: Path) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_REF") != "refs/heads/feat/portable-v2":
        raise RuntimeError("recovery is confined to feat/portable-v2")
    parts = [root / ".delivery" / f"recovery-{i:02d}.part" for i in range(7)]
    if not any(p.exists() for p in parts):
        print("Ordinary source is already restored.")
        return
    if not all(p.is_file() and p.stat().st_size <= 9000 for p in parts):
        raise RuntimeError("incomplete recovery parts")
    raw = b"".join(p.read_bytes() for p in parts)
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise RuntimeError("recovery SHA-256 mismatch")
    decoder = lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)
    decoded = decoder.decompress(raw, max_length=1024 * 1024)
    if not decoder.eof or decoder.unused_data:
        raise RuntimeError("oversized or trailing recovery content")
    payload = json.loads(decoded)
    if payload["base_commit"] != BASE or set(payload["files"]) != FILES:
        raise RuntimeError("unexpected base or file inventory")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=root, check=True)
    for name, content in payload["files"].items():
        path = root / name
        if not isinstance(content, str) or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("invalid recovery entry")
        if path.exists() and path.read_text("utf-8") != content:
            raise RuntimeError(f"refusing to overwrite existing file: {name}")
    changed = set()
    for line in payload["patch"].splitlines():
        if line.startswith("diff --git "):
            _, _, a, b = line.split()
            if a != "a/" + b[2:] or b[2:] not in PATCH_FILES:
                raise RuntimeError("unexpected patch path")
            changed.add(b[2:])
    if changed != PATCH_FILES:
        raise RuntimeError("incomplete shared-core patch")
    with tempfile.TemporaryDirectory() as directory:
        patch = Path(directory) / "shared-core.patch"
        patch.write_text(payload["patch"], encoding="utf-8")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
    for name, content in payload["files"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
    for path in parts:
        path.unlink()
    print(f"Restored {len(FILES)} ordinary files and {len(PATCH_FILES)} shared-core changes.")

if __name__ == "__main__":
    restore(Path(__file__).resolve().parents[1])
