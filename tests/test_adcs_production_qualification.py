import json
from pathlib import Path

from scripts.data_driven.qualify_adcs_production import ROOT, evaluate


def test_adcs_p2_production_gate_uses_complete_repository_evidence() -> None:
    report = evaluate()
    assert report["status"] == "BLOCKED"
    assert report["evidence_status"] == "COMPLETE"
    assert report["checks"]["metrics_evidence_present"] is True
    assert report["checks"]["fivefold_protocol"] is True
    assert report["checks"]["localization_macro_f1"] is True
    assert report["checks"]["runtime_provider_present"] is True
    assert report["checks"]["hash_pinned_model_package_present"] is True
    assert report["checks"]["exact_mapping_coverage"] is True
    assert report["runtime"]["exact_mapping_count"] == 14
    assert report["runtime"]["dataset_class_count"] == 14
    assert report["runtime"]["registry_promoted"] is False
    assert report["runtime"]["chain_statuses"] == {
        "rule": "SUCCEEDED",
        "llm": "SUCCEEDED",
    }
    assert report["checks"]["candidate_runtime_chain_statuses"] is True
    assert report["missing_artifacts"] == {}


def test_adcs_p2_production_gate_fails_closed_on_missing_runtime_evidence(
    tmp_path: Path,
) -> None:
    policy_path = ROOT / "configs/data_driven/adcs_production_qualification.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    required_configs = (
        policy_path.relative_to(ROOT),
        Path("configs/datasets/adcs_v5_2_2/fault_class_map.json"),
        Path("configs/datasets/adcs_v5_2_2/diagnosis_hierarchy.json"),
    )
    for relative_path in required_configs:
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative_path).read_bytes())

    provider_path = tmp_path / policy["runtime_gates"]["provider_path"]
    provider_path.parent.mkdir(parents=True, exist_ok=True)
    provider_path.write_text("# presence-only qualification fixture\n", encoding="utf-8")

    report = evaluate(tmp_path)
    assert report["status"] == "BLOCKED"
    assert report["evidence_status"] == "MISSING_REQUIRED_ARTIFACTS"
    assert report["checks"]["metrics_evidence_present"] is False
    assert report["checks"]["fivefold_protocol"] is False
    assert report["checks"]["localization_macro_f1"] is False
    assert report["checks"]["runtime_provider_present"] is True
    assert report["checks"]["hash_pinned_model_package_present"] is False
    assert report["checks"]["exact_mapping_coverage"] is True
    assert report["runtime"]["exact_mapping_count"] == 14
    assert report["runtime"]["dataset_class_count"] == 14
    assert report["runtime"]["registry_promoted"] is False
    assert report["runtime"]["chain_statuses"] == {
        "rule": "MISSING",
        "llm": "MISSING",
    }
    assert report["checks"]["candidate_runtime_chain_statuses"] is False
    assert set(report["missing_artifacts"]) == {
        "metrics_evidence",
        "model_package",
        "model_receipt",
        "runtime_chain_evidence",
    }
