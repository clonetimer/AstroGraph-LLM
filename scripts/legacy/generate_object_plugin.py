from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# scripts/ 顶层含 _bootstrap.py；本脚本位于 scripts/legacy/ 子目录，
# 运行时 sys.path[0] 是子目录，需先把 scripts/ 顶层加入 sys.path。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from astrograph.plugin_factory import ObjectPluginBuildSpec, ObjectPluginFactoryService


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a controlled object plugin scaffold")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    spec = ObjectPluginBuildSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
    result = ObjectPluginFactoryService().generate(spec, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
