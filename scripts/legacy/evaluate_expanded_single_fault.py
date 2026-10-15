#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from astrograph.diagnosis.expanded_single_fault import ExpandedSingleFaultEngine
from astrograph.diagnosis.telemetry_contract import is_prohibited
from astrograph.diagnosis.unified import UnifiedDiagnosisService


def _case_root(case_id: str, roots: list[Path]) -> Path:
    for root in roots:
        candidate = root / case_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(case_id)


def evaluate(*, truth_path: Path, public_roots: list[Path]) -> dict:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))["cases"]
    engine = ExpandedSingleFaultEngine()
    service = UnifiedDiagnosisService()
    results = []
    leakage = []
    for item in truth:
        root = _case_root(item["case_id"], public_roots)
        nominal_path = root / "nominal_trace.csv"
        fault_path = root / "fault_trace.csv"
        nominal = pd.read_csv(nominal_path, low_memory=False)
        fault = pd.read_csv(fault_path, low_memory=False)
        prohibited = sorted({column for column in nominal.columns.tolist() + fault.columns.tolist() if is_prohibited(column)})
        if prohibited:
            leakage.append({"case_id": item["case_id"], "columns": prohibited})
        direct = engine.diagnose_frames(nominal=nominal, effect=fault)
        expected_class = engine.mode_to_group.get(item["physical_mode"], item["physical_mode"])
        unified_a = service.diagnose_paths(
            case_id="anonymous-a",
            nominal_path=nominal_path,
            telemetry_path=fault_path,
        )
        unified_b = service.diagnose_paths(
            case_id="anonymous-b",
            nominal_path=nominal_path,
            telemetry_path=fault_path,
        )
        class_match = direct.observation_class == expected_class
        exact_expected = item["physical_mode"] not in engine.mode_to_group
        exact_match = direct.primary_mode == item["physical_mode"] if exact_expected else direct.primary_mode is None
        candidate_complete = (
            set(direct.candidate_modes) == set(engine.groups[expected_class]["members"])
            if not exact_expected else direct.candidate_modes == [item["physical_mode"]]
        )
        invariant = (
            unified_a.status == unified_b.status
            and unified_a.primary_effect_id == unified_b.primary_effect_id
            and unified_a.conclusion_effect_ids == unified_b.conclusion_effect_ids
            and unified_a.resolution_basis == unified_b.resolution_basis
        )
        labels_used = any(record.raw_result and record.raw_result.get("labels_used") for record in unified_a.engine_runs)
        truth_used = any(record.raw_result and record.raw_result.get("truth_used") for record in unified_a.engine_runs)
        results.append({
            "case_id": item["case_id"],
            "split": item["split"],
            "formal": bool(item["acceptance_valid"]),
            "physical_mode": item["physical_mode"],
            "expected_observation_class": expected_class,
            "observed_class": direct.observation_class,
            "engine_status": direct.status,
            "unified_status": unified_a.status.value,
            "class_match": class_match,
            "exact_mode_expected": exact_expected,
            "exact_mode_match": exact_match,
            "candidate_complete": candidate_complete,
            "case_id_permutation_invariant": invariant,
            "labels_used": bool(labels_used or unified_a.labels_used),
            "truth_used": bool(truth_used or unified_a.truth_used),
            "nearest_mode": direct.nearest_mode,
            "cosine_distance": direct.cosine_distance,
        })
    def count(where):
        subset = [r for r in results if where(r)]
        return {
            "count": len(subset),
            "class_matches": sum(r["class_match"] for r in subset),
            "exact_mode_expected": sum(r["exact_mode_expected"] for r in subset),
            "exact_mode_matches": sum(r["exact_mode_match"] for r in subset if r["exact_mode_expected"]),
            "equivalence_cases": sum(not r["exact_mode_expected"] for r in subset),
            "candidate_complete": sum(r["candidate_complete"] for r in subset),
        }
    formal = count(lambda r: r["formal"])
    research = count(lambda r: not r["formal"])
    development = count(lambda r: r["split"] == "development")
    holdout = count(lambda r: r["split"] == "holdout")
    errors = [r for r in results if not (r["class_match"] and r["exact_mode_match"] and r["candidate_complete"])]
    pass_gate = (
        formal["count"] == 132
        and formal["class_matches"] == formal["count"]
        and formal["exact_mode_matches"] == formal["exact_mode_expected"]
        and formal["candidate_complete"] == formal["count"]
        and holdout["class_matches"] == holdout["count"]
        and not leakage
        and all(r["case_id_permutation_invariant"] for r in results)
        and not any(r["labels_used"] or r["truth_used"] for r in results)
    )
    return {
        "schema_version": "astrograph-expanded-single-fault-evaluation.v1",
        "status": "PASS" if pass_gate else "FAIL",
        "formal": formal,
        "research_only": research,
        "development": development,
        "holdout": holdout,
        "engine_status_counts": dict(sorted(Counter(r["engine_status"] for r in results).items())),
        "unified_status_counts": dict(sorted(Counter(r["unified_status"] for r in results).items())),
        "equivalence_group_count": len(engine.groups),
        "unique_mode_count": len(engine.known_modes),
        "leakage_findings": leakage,
        "errors": errors,
        "results": results,
        "claim_boundary": (
            "Formal metrics include only ACCEPTANCE_VALID cases. Research-only cases are reported separately. "
            "A correct equivalence-class result is not counted as an exact root-mode diagnosis."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(truth_path=args.truth, public_roots=args.public_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "formal", "holdout", "engine_status_counts")}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
