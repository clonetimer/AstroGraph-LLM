#!/usr/bin/env python3
"""AI-Native Phase 5: read-only governance health check runner.

Runs the advisory GovernanceHealthChecker over the repository and writes a
deterministic report into governance/evidence/ai_native/governance_health/.

Usage:
    python scripts/ai_native/run_governance_health_check.py [--output-dir out]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.governance.health_check import GovernanceHealthChecker

DEFAULT_OUTPUT = ROOT / "governance/evidence/ai_native/governance_health"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = GovernanceHealthChecker().run(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"governance_health_{date.today().isoformat()}.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(f"[health] overall={report.overall_status.value} -> {path}")
    for item in report.checks:
        print(f"  {item.check_id}: {item.status.value}")
    return 0 if report.overall_status.value == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
