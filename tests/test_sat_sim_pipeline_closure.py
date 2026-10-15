from __future__ import annotations

import errno
import os
import stat
import json
from pathlib import Path

import pytest

from astrograph.integrations import SatSimPipelineClosureRequest, SatSimPipelineClosureWorkflow


@pytest.fixture()
def evidence_root() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "sat_sim_closure"


def _load(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def test_rw_jam_vertical_slice_is_review_only(evidence_root: Path, tmp_path: Path) -> None:
    runtime = _load(evidence_root, "runtime_rw_jam.json")
    mapping = _load(evidence_root, "mapping_bundle.json")
    result = SatSimPipelineClosureWorkflow().invoke(
        SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping, persist_dir=str(tmp_path))
    )
    assert result.fault_id == "ADCS_RW_JAM"
    assert result.fusion["top_fault_id"] == "ADCS_RW_JAM"
    assert result.disposition == "REVIEW_REQUIRED"
    assert result.formal_diagnosis_publishable is False
    assert result.formal_model_binding_enabled is False
    assert "FORMAL_MODEL_BINDING_DISABLED" in result.blockers
    assert result.external_stage_outputs["ml_model"]["status"] == "CANDIDATE_SCREENING_ONLY"
    assert result.data_driven_prediction["output_semantics"] == "NONNEGATIVE_SCORES"
    assert result.data_driven_prediction["class_value_semantics"] == "NORMALIZED_RANKING_MASS"
    assert result.score_semantics_audit["probability_threshold_eligible"] is False
    assert "NORMALIZED_RANKING_MASS_NOT_PROBABILITY" in result.score_semantics_audit["warnings"]
    assert result.fusion["fused_score_semantics"] == "RECIPROCAL_RANK_SCORE_NOT_PROBABILITY"
    assert result.fusion["normal_probability"] is None
    assert result.kg_projection["status"] == "LABEL_CONDITIONED_AUDIT_ONLY"
    assert result.kg_projection["contributes_to_fusion"] is False
    assert "LABEL_CONDITIONED_KG_PROJECTION_EXCLUDED_FROM_FUSION" in result.blockers
    for ranking_name in ("forward_ranking", "backward_ranking"):
        assert all(
            candidate["contributes_to_fusion"] is False
            for candidate in result.kg_projection[ranking_name]["candidates"]
        )
    for candidate in result.fusion["candidates"]:
        assert set(candidate["reciprocal_contributions"]) <= {"DATA_DRIVEN"}
    assert (tmp_path / "sat_sim_pipeline_closure.json").is_file()
    assert (tmp_path / "sat_sim_pipeline_report.md").is_file()



def test_rw_jam_persistence_survives_directory_fsync_ebadf(
    evidence_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PIPELINE_RUNTIME core diagnosis must not fail on Windows-style directory fsync EBADF."""
    import astrograph.assurance.artifact_paths as artifact_paths

    runtime = _load(evidence_root, "runtime_rw_jam.json")
    mapping = _load(evidence_root, "mapping_bundle.json")
    real_fsync = artifact_paths.os.fsync

    def fsync_with_directory_ebadf(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EBADF, "Bad file descriptor")
        real_fsync(fd)

    monkeypatch.setattr(artifact_paths.os, "fsync", fsync_with_directory_ebadf)
    result = SatSimPipelineClosureWorkflow().invoke(
        SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping, persist_dir=str(tmp_path))
    )
    assert result.fusion["top_fault_id"] == "ADCS_RW_JAM"
    assert (tmp_path / "sat_sim_pipeline_closure.json").is_file()

def test_leakage_guard_is_mandatory(evidence_root: Path) -> None:
    runtime = _load(evidence_root, "runtime_rw_jam.json")
    mapping = _load(evidence_root, "mapping_bundle.json")
    runtime["local_outputs"]["feature_vector"]["leakage_guard"]["simulator_truth_excluded"] = False
    with pytest.raises(ValueError, match="leakage"):
        SatSimPipelineClosureWorkflow().invoke(
            SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping)
        )


def test_runtime_and_mapping_fault_identity_must_match(evidence_root: Path) -> None:
    runtime = _load(evidence_root, "runtime_rw_jam.json")
    mapping = _load(evidence_root, "mapping_bundle.json")
    with pytest.raises(ValueError, match="requested fault"):
        SatSimPipelineClosureRequest(
            runtime=runtime,
            mapping_bundle=mapping,
            requested_fault_id="ADCS_RW_FRICTION_INCREASE",
        )


def test_runtime_rejects_internally_inconsistent_fault_identity(evidence_root: Path) -> None:
    runtime = _load(evidence_root, "runtime_rw_jam.json")
    mapping = _load(evidence_root, "mapping_bundle.json")
    runtime["local_outputs"]["diagnostic_validation"]["fault_id"] = (
        "ADCS_RW_FRICTION_INCREASE"
    )
    with pytest.raises(ValueError, match="fault identity mismatch"):
        SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping)


def test_api_exposes_review_only_sat_sim_closure(evidence_root: Path) -> None:
    from fastapi.testclient import TestClient
    from astrograph.api import create_app

    runtime = _load(evidence_root, "runtime_rw_jam.json")
    mapping = _load(evidence_root, "mapping_bundle.json")
    response = TestClient(create_app()).post(
        "/v1/integrations/sat-sim/pipeline-diagnose",
        json={"runtime": runtime, "mapping_bundle": mapping},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["fusion"]["top_fault_id"] == "ADCS_RW_JAM"
    assert payload["formal_diagnosis_publishable"] is False
    assert payload["disposition"] == "REVIEW_REQUIRED"


def test_rw_jam_persistence_survives_regular_file_fsync_ebadf(
    evidence_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PIPELINE_RUNTIME persistence must survive a live-file CRT fsync EBADF."""
    import astrograph.assurance.artifact_paths as artifact_paths

    runtime = _load(evidence_root, "runtime_rw_jam.json")
    mapping = _load(evidence_root, "mapping_bundle.json")
    real_fsync = artifact_paths.os.fsync

    def fsync_with_regular_ebadf(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EBADF, "Bad file descriptor")
        real_fsync(fd)

    monkeypatch.setattr(artifact_paths.os, "fsync", fsync_with_regular_ebadf)
    result = SatSimPipelineClosureWorkflow().invoke(
        SatSimPipelineClosureRequest(runtime=runtime, mapping_bundle=mapping, persist_dir=str(tmp_path))
    )
    assert result.fusion["top_fault_id"] == "ADCS_RW_JAM"
    assert json.loads((tmp_path / "sat_sim_pipeline_closure.json").read_text(encoding="utf-8"))["fault_id"] == "ADCS_RW_JAM"
