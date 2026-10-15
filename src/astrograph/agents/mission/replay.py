"""Offline dataset replay tasks exposed through Mission without masquerading as online diagnosis.

post48.dev18 extends the replay contract so ADCS and PHM use the same Mission
``task=replay`` entry point while preserving their different retrospective
workers and claim boundaries.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReplayManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["astrograph-replay-manifest.v1"] = "astrograph-replay-manifest.v1"
    dataset_kind: Literal["phm_v6_retrospective", "adcs_v5_2_2_retrospective"]
    model_strategy: Literal["promoted_extra_trees", "adaptive", "legacy_lr"] = "promoted_extra_trees"
    adcs_cohort: Literal["development", "used-test-regression"] = "development"
    skip_llm: bool = True
    claim_boundary: str = (
        "Offline retrospective replay. Evaluation may read labels and activation boundaries and must not be presented as online diagnosis."
    )

    @model_validator(mode="after")
    def dataset_specific_contract(self):
        # ADCS replay has a frozen native model/worker and therefore does not use
        # the PHM model_strategy switch.  Keeping the field in v1 preserves old
        # manifests while making the non-applicability explicit in outputs.
        if self.dataset_kind == "adcs_v5_2_2_retrospective" and not self.skip_llm:
            raise ValueError("ADCS retrospective replay has no LLM execution path; skip_llm must remain true")
        return self


def _metric(result: dict, *names: str):
    for name in names:
        value = result.get(name)
        if value is not None:
            return value
    return None


def _run_phm(*, project_root: Path, workspace_root: Path, dataset_root: Path,
             manifest: ReplayManifest, run_subprocess) -> dict:
    marker = dataset_root / "sample_index.csv"
    if not marker.is_file():
        raise ValueError("PHM replay dataset root must contain sample_index.csv")
    out = workspace_root / "dataset-replay"
    args = [
        str(project_root / "scripts/integration/run_phm_v60_regression.py"),
        "--dataset-root", str(dataset_root), "--output-dir", str(out),
        "--model-strategy", manifest.model_strategy,
    ]
    if manifest.skip_llm:
        args.append("--skip-llm")
    run_subprocess(args)
    report_path = out / "phm_v60_regression_report.json"
    if not report_path.is_file():
        raise RuntimeError("PHM replay worker did not write report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = report.get("three_stage_result", {})
    chains = {}
    for key, value in report.get("chains", {}).items():
        payload = value.get("output_payload", {}) if isinstance(value, dict) else {}
        chains[key] = {
            "status": value.get("status") if isinstance(value, dict) else None,
            "decision_record": payload.get("decision_record") or (value.get("decision") if isinstance(value, dict) else None),
            "structured_diagnosis_output": payload.get("structured_diagnosis_output"),
            "diagnosis_policy_decision": payload.get("diagnosis_policy_decision"),
        }
    return {
        "status": "SUCCEEDED",
        "dataset_kind": manifest.dataset_kind,
        "dataset_source_kind": "DIRECTORY",
        "model_strategy": report.get("model_strategy"),
        "selected_chain_sample": report.get("selected_chain_sample"),
        "metrics": {
            "detection": _metric(result, "detection", "detection_metrics"),
            "localization": _metric(result, "localization", "localization_metrics"),
            "classification": _metric(result, "classification_predicted_component", "classification_metrics_deployed"),
            "end_to_end": _metric(result, "end_to_end_18_class", "end_to_end_metrics"),
        },
        "split_audit": report.get("split_audit"),
        "chains": chains,
        "report_path": str(report_path),
        "semantic_inference_sha256": report.get("semantic_inference_sha256"),
        "offline_evaluation_truth_used": True,
        "online_diagnosis_claim_allowed": False,
        "claim_boundary": manifest.claim_boundary,
    }


def _run_adcs(*, project_root: Path, workspace_root: Path, dataset_zip: Path,
              manifest: ReplayManifest, run_subprocess) -> dict:
    if dataset_zip.suffix.lower() != ".zip" or not dataset_zip.is_file():
        raise ValueError("ADCS replay requires a dataset zip file")
    out = workspace_root / "dataset-replay"
    args = [
        str(project_root / "scripts/evaluate_three_task_agent_zip.py"),
        "--dataset-zip", str(dataset_zip),
        "--project-root", str(project_root),
        "--output", str(out),
        "--cohort", manifest.adcs_cohort,
    ]
    run_subprocess(args)
    report_path = out / "report.json"
    if not report_path.is_file():
        raise RuntimeError("ADCS replay worker did not write report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stable_keys = (
        "schema_version", "windows", "active_fault_windows", "normal_windows",
        "active_head_outputs_available", "model_heads", "branches", "label_universe",
        "claim_boundary", "cohort", "semantic_inference_sha256",
        "execution_status_counts", "agent_status_counts", "tool_calls", "eager_tool_calls",
        "model_weights_changed", "fusion_weights_changed", "net_changes",
    )
    semantic_summary = {key: report.get(key) for key in stable_keys}
    return {
        "status": "SUCCEEDED",
        "dataset_kind": manifest.dataset_kind,
        "dataset_source_kind": "ZIP",
        "cohort": manifest.adcs_cohort,
        "semantic_summary": semantic_summary,
        "semantic_inference_sha256": report.get("semantic_inference_sha256"),
        "raw_inference_sha256": report.get("inference_sha256"),
        "report_path": str(report_path),
        "offline_evaluation_truth_used": True,
        "online_diagnosis_claim_allowed": False,
        "claim_boundary": manifest.claim_boundary,
    }


def run_replay(*, project_root: Path, workspace_root: Path,
               manifest: ReplayManifest, run_subprocess,
               dataset_root: Path | None = None, dataset_zip: Path | None = None) -> dict:
    """Execute a dataset-specific retrospective worker behind one Mission contract.

    The two source forms are intentionally distinct: PHM's worker consumes an
    extracted directory, while the frozen ADCS replay worker consumes the
    original ZIP so its dataset identity and label-join boundary remain intact.
    """
    if manifest.dataset_kind == "phm_v6_retrospective":
        if dataset_root is None or dataset_zip is not None:
            raise ValueError("PHM replay requires dataset_root and does not accept dataset_zip")
        return _run_phm(project_root=project_root, workspace_root=workspace_root,
                        dataset_root=dataset_root, manifest=manifest, run_subprocess=run_subprocess)
    if manifest.dataset_kind == "adcs_v5_2_2_retrospective":
        if dataset_zip is None or dataset_root is not None:
            raise ValueError("ADCS replay requires dataset_zip and does not accept dataset_root")
        return _run_adcs(project_root=project_root, workspace_root=workspace_root,
                         dataset_zip=dataset_zip, manifest=manifest, run_subprocess=run_subprocess)
    raise ValueError("unsupported replay dataset kind")
