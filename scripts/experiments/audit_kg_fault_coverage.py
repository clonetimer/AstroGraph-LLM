from __future__ import annotations

import argparse
from pathlib import Path

from astrograph.assurance.kg_coverage import write_kg_coverage_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit executable KG/rule fault-universe coverage without reading labels."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = write_kg_coverage_audit(args.project_root, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
