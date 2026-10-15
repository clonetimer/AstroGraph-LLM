#!/usr/bin/env python3
"""Derive immutable outcome attribution from one completed A10 report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.run_ops_sat_compact_bundle_comparator import _digest, summarize_attributable_outcomes


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not already exist")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != "astrograph-ops-sat-compact-bundle-comparator.v1":
        parser.error("unsupported A10 report schema")
    content = dict(report)
    claimed_digest = content.pop("report_sha256", None)
    if not isinstance(claimed_digest, str) or _digest(content) != claimed_digest:
        raise ValueError("A10 report content digest mismatch")
    summary = report["a10_compact_bundle"]
    a8 = report["baseline_arms"]["A8_ACTIVE_TELEMETRY"]
    derived = {
        "schema_version": "astrograph-ops-sat-compact-bundle-outcome-audit.v1",
        "source_report": {
            "path": str(args.report.resolve()),
            "file_sha256": _sha_file(args.report),
            "content_sha256": claimed_digest,
        },
        "attributable_outcome_audit": summarize_attributable_outcomes(report["cases"]),
        "comparison_to_a8": {
            "accuracy_delta_a10_minus_a8": (
                summary["accuracy_with_abstention_as_incorrect"]
                - a8["accuracy_with_abstention_as_incorrect"]
            ),
            "healthy_false_alarm_delta_a10_minus_a8": (
                summary["healthy_false_alarm_count"] - a8["healthy_false_alarm_count"]
            ),
            "abstain_or_ambiguous_delta_a10_minus_a8": (
                summary["abstain_or_ambiguous_count"] - a8["abstain_or_ambiguous_count"]
            ),
            "cost_comparability": "MECHANISM_ONLY_NOT_EQUAL_COMPUTE_OR_LLM_BUDGET",
        },
        "claim_boundary": (
            "Deterministic post-hoc classification of already recorded A10 outcomes. "
            "It does not add cases, rerun the model, establish significance, or create formal evidence."
        ),
    }
    derived["report_sha256"] = _digest(derived)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(derived, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(derived, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
