#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.fusion_dependence import FusionEvaluationCase, audit_birrf_leave_branch_out
from astrograph.fusion import BiRRFConfig, FaultRanking


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline independent-group BiRRF leave-branch-out audit; not causal attribution or calibration.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--design", type=Path, default=Path("configs/evaluation/birrf_dependence_r18.v1.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    bundle = _load(args.input)
    design = _load(args.design)
    config_payload = dict(design["baseline_config"])
    config_payload.update(bundle.get("config_overrides", {}))
    config = BiRRFConfig(**config_payload)
    cases = []
    for row in bundle["cases"]:
        cases.append(
            FusionEvaluationCase(
                case_id=row["case_id"],
                group_id=row["group_id"],
                true_fault_id=row["true_fault_id"],
                data_ranking=FaultRanking.model_validate(row["data_ranking"]),
                forward_ranking=FaultRanking.model_validate(row["forward_ranking"]),
                backward_ranking=FaultRanking.model_validate(row["backward_ranking"]),
            )
        )
    report = audit_birrf_leave_branch_out(
        cases,
        config,
        bootstrap_resamples=int(design["bootstrap_resamples"]),
        bootstrap_seed=int(design["bootstrap_seed"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
