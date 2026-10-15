#!/usr/bin/env python3
"""Validate P-015 candidate-2 authorization without converting PENDING into approval."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


PREAPPROVAL_REQUEST_SHA256 = (
    "c8df97cd2a172d76da1b14b61c688793415b09db8e30d560bef94c903c97fc99"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(root: Path) -> dict:
    protocol = root / "protocols/acceptance"
    request_path = protocol / "p015_candidate2_authorization_request.json"
    request = _read(request_path)
    Draft202012Validator(
        _read(protocol / "candidate2_authorization.schema.json")
    ).validate(request)
    source_path = root / request["approval"]["source_attestation_path"]
    source_exists = source_path.is_file()
    source = _read(source_path) if source_exists else {}
    if source_exists:
        Draft202012Validator(
            _read(protocol / "candidate2_timer_attestation.schema.json")
        ).validate(source)
    expected_declarations = {
        "g1_1_artifacts_unchanged": True,
        "known_mechanisms": request["denominators"]["known_mechanisms"],
        "open_set_mechanisms": request["denominators"]["open_set_mechanisms"],
        "classes_not_merged_or_expanded": True,
        "thresholds_and_denominators_unchanged": True,
        "development_seed": request["candidate"]["development_seed"],
        "calibration_seed": request["candidate"]["calibration_seed"],
        "exhausted_validation_seed": request["candidate"]["exhausted_validation_seed"],
        "new_single_use_validation_seed": request["candidate"][
            "new_single_use_validation_seed"
        ],
        "formal_hidden_truth_used": request["candidate"]["formal_hidden_truth_used"],
        "model_path": request["candidate"]["model_path"],
        "single_l2_head_only": True,
        "ensemble_fallback_or_legacy_router_allowed": False,
        "maximum_candidates": request["candidate"]["maximum_candidates"],
        "maximum_runtime_seconds": request["candidate"]["maximum_runtime_seconds"],
        "run_failures_count_as_metric_failures": True,
        "promotion_preapproved": False,
        "independent_review_required": True,
    }
    checks = {
        "all_bound_artifact_hashes_match": all(
            (root / item["path"]).is_file()
            and _sha(root / item["path"]) == item["sha256"]
            for item in request["bound_artifacts"]
        ),
        "candidate_ordinal_within_budget": (
            request["candidate"]["ordinal"] == 2
            and request["candidate"]["maximum_candidates"] == 3
        ),
        "old_validation_seed_exhausted": (
            request["candidate"]["exhausted_validation_seed"] == 20280820
        ),
        "new_validation_seed_predeclared": (
            request["candidate"]["new_single_use_validation_seed"] == 20290820
            and request["candidate"]["new_seed_frozen_before_candidate_implementation"]
        ),
        "thresholds_unchanged": request["unchanged_thresholds"]
        == {
            "detection_accuracy": 0.98,
            "classification_accuracy": 0.9,
            "localization_accuracy": 0.95,
            "reasoning_correctness": 0.9,
            "known_class_retention": 0.98,
            "open_set_rejection": 0.98,
            "open_set_alpha": 0.02,
        },
        "denominators_unchanged": all(
            request["denominators"][key] == value
            for key, value in {
                "normal": 240,
                "known": 720,
                "open_set": 720,
                "total": 1680,
            }.items()
        ),
        "independent_evaluator_is_timer": request["requested_reviewer"] == "Timer",
        "approval_present": request["approval"]["status"] == "APPROVED",
        "approval_reviewer_matches": request["approval"]["reviewer"] == "Timer",
        "source_attestation_hash_matches": (
            source_exists
            and _sha(source_path) == request["approval"]["source_attestation_sha256"]
        ),
        "source_attestation_binds_preapproval_request": (
            source.get("authorization_request_preapproval_sha256")
            == PREAPPROVAL_REQUEST_SHA256
        ),
        "source_attestation_identity_matches": (
            source.get("request_id") == request["request_id"]
            and source.get("reviewer") == request["requested_reviewer"]
            and source.get("status") == "APPROVED"
        ),
        "approval_attestation_is_verbatim_source": (
            request["approval"]["attestation_text_zh"]
            == source.get("verbatim_attestation_text_zh")
        ),
        "approval_declarations_cover_frozen_request": (
            source.get("declarations") == expected_declarations
        ),
        "approval_timestamp_present": bool(request["approval"]["reviewed_at_utc"]),
    }
    technical_keys = {
        "all_bound_artifact_hashes_match",
        "candidate_ordinal_within_budget",
        "old_validation_seed_exhausted",
        "new_validation_seed_predeclared",
        "thresholds_unchanged",
        "denominators_unchanged",
        "independent_evaluator_is_timer",
    }
    technical = all(checks[key] for key in technical_keys)
    approved = technical and all(checks.values())
    return {
        "schema_version": "astrograph-candidate-authorization-gate.v1",
        "request_id": request["request_id"],
        "technical_status": "PASS" if technical else "FAIL",
        "status": (
            "PASS" if approved else "BLOCKED_APPROVAL" if technical else "FAIL"
        ),
        "checks": checks,
        "claim_boundary": (
            "Validates authorization integrity only. PASS permits candidate-2 implementation but "
            "does not promote a model or establish T-310, T-311, G3 or G4."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.project_root.resolve())
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["status"] in {"PASS", "BLOCKED_APPROVAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
