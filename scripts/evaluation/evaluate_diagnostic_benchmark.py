#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from astrograph.evaluation.diagnostic_regression import (
    BenchmarkCase,
    EvidenceClass,
    evaluate_diagnostic_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an R36 diagnostic benchmark case artifact")
    parser.add_argument("--version", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument(
        "--evidence-class",
        required=True,
        choices=[
            EvidenceClass.DEVELOPMENT.value,
            EvidenceClass.FIXTURE_ONLY_NOT_QUALIFICATION.value,
        ],
        help="This standalone CLI cannot issue or verify QUALIFICATION authorization.",
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = args.cases.read_bytes()
    payload = json.loads(raw)
    case_payloads = payload["cases"] if isinstance(payload, dict) else payload
    cases = [BenchmarkCase.model_validate(item) for item in case_payloads]
    snapshot = evaluate_diagnostic_benchmark(
        astrograph_version=args.version,
        benchmark_id=args.benchmark_id,
        dataset_id=args.dataset_id,
        protocol_id=args.protocol_id,
        cohort_id=args.cohort_id,
        source_artifact_sha256=hashlib.sha256(raw).hexdigest(),
        evidence_class=EvidenceClass(args.evidence_class),
        cases=cases,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
