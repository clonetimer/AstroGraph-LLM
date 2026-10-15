#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.governance.spacecraft_convergence import load_spacecraft_scope_convergence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/release_claims/spacecraft_scope_convergence_r32.v1.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--output", default=str(ROOT / "governance/evidence/research/r32_spacecraft_convergence/PATCH_R32_SPACECRAFT_CONVERGENCE.json"))
    args = ap.parse_args()
    decision = load_spacecraft_scope_convergence(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "software_feature_freeze": decision.software_feature_freeze,
        "diagnostic_external_qualification": decision.diagnostic_external_qualification,
        "stable_engineering_candidate": decision.stable_engineering_candidate,
        "next_phase": decision.next_phase,
        "decision_sha256": decision.decision_sha256,
    }, ensure_ascii=False, indent=2))
    return 0 if decision.stable_engineering_candidate.value == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
