#!/usr/bin/env python3
"""Freeze the exact Python distribution environment used by a confirmatory AstroGraph client."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from astrograph.evaluation.confirmatory_runtime_identity import build_confirmatory_python_environment_lock

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_confirmatory_python_environment_lock()
    path = args.output.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "lock_sha256": payload["lock_sha256"], "package_count": payload["package_count"]}, ensure_ascii=False, indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
