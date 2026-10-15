"""Verify the full source inventory without modifying the source tree."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import subprocess

IGNORED={"node_modules", ".git", ".claude", ".pytest_cache", "__pycache__", ".tmp_tests", "worktrees", ".venv", "astrograph_core.egg-info"}


def verify(root: Path) -> dict:
    root=root.resolve()
    expected={}
    for line in (root/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest,name=line.split("  ",1)
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in expected:
            raise ValueError("unsafe or duplicate checksum entry")
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError("checksum path escapes source tree")
        expected[name]=digest
    errors=[]
    for name,digest in expected.items():
        path=root/name
        if not path.is_file():errors.append("MISSING: "+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:errors.append("CHANGED: "+name)
    # Match refresh_distribution_metadata and validate_distribution in a working
    # checkout: local ignored datasets/state are not distributable source files.
    candidates = root.rglob("*")
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True, check=False,
        )
        if result.returncode == 0:
            candidates = [root / name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    actual={p.relative_to(root).as_posix() for p in candidates if p.is_file()
            and not IGNORED.intersection(p.relative_to(root).parts)
            and p.suffix not in {".pyc",".pyo"}
            and not p.relative_to(root).as_posix().startswith(("scripts/integration/out/", "runs/"))}
    errors.extend("UNLISTED: "+name for name in sorted(actual-set(expected)-{"SHA256SUMS.txt"}))
    return {"status":"FAIL" if errors else "PASS","verified_files":len(expected),"errors":errors}


if __name__=="__main__":
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    result=verify(p.parse_args().root)
    print(json.dumps(result,ensure_ascii=False,indent=2));sys.exit(0 if result["status"]=="PASS" else 1)
