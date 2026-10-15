#!/usr/bin/env python3
"""Run the externally pinned AstroGraph governance gate against a candidate tree."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_PROJECT_ROOT = SCRIPT_ROOT.parent
sys.path.insert(0, str(TRUSTED_PROJECT_ROOT / "src"))

from astrograph.governance import (  # noqa: E402
    HiddenTruthAccessStatus,
    IsolatedGateEvaluator,
    ImprovementContract,
    decide_release,
    frozen_artifact_hash,
    load_json_yaml,
    reserve_gate_query,
    sha256_file,
    verify_evidence_artifacts,
    verify_gate_query_receipt,
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-state-dir", type=Path, required=True)
    parser.add_argument(
        "--hidden-truth-access-status",
        choices=[item.value for item in HiddenTruthAccessStatus],
        default=HiddenTruthAccessStatus.NOT_APPLICABLE.value,
    )
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    output_dir = args.output_dir.resolve()
    state_dir = args.gate_state_dir.resolve()
    if _is_within(output_dir, candidate_root):
        raise SystemExit("output directory must be outside the candidate source tree")
    if _is_within(state_dir, candidate_root):
        raise SystemExit("gate state directory must be outside the candidate source tree")
    if sha256_file(args.contract) != args.expected_contract_sha256:
        raise SystemExit("externally pinned contract hash does not match")

    contract = load_json_yaml(args.contract, ImprovementContract)
    receipt, receipt_source = reserve_gate_query(
        contract,
        contract_sha256=args.expected_contract_sha256,
        candidate_id=args.candidate_id,
        state_dir=state_dir,
    )
    evaluator = IsolatedGateEvaluator(Path(__file__))
    evidence = evaluator.run(
        contract_path=args.contract,
        candidate_root=candidate_root,
        expected_contract_sha256=args.expected_contract_sha256,
        candidate_id=args.candidate_id,
        output_dir=output_dir,
        gate_query_receipt=receipt,
        gate_query_receipt_source=receipt_source,
        hidden_truth_access_status=HiddenTruthAccessStatus(args.hidden_truth_access_status),
    )

    artifact_errors = verify_evidence_artifacts(evidence, output_dir)
    receipt_errors = verify_gate_query_receipt(
        evidence,
        receipt_path=receipt_source,
        contract=contract,
        expected_contract_sha256=args.expected_contract_sha256,
    )
    decision = decide_release(
        contract,
        evidence,
        expected_contract_sha256=args.expected_contract_sha256,
        expected_evaluator_sha256=frozen_artifact_hash(contract, role="formal-gate-entrypoint"),
        expected_evaluator_module_sha256=frozen_artifact_hash(contract, role="isolated-evaluator"),
        evidence_artifact_errors=artifact_errors,
        query_receipt_errors=receipt_errors,
    )
    decision_path = output_dir / "release_decision.json"
    decision_path.write_text(decision.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if decision.status.value == "ACCEPT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
