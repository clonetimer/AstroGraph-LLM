from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from astrograph.evaluation.rw_active_discovery import (
    RWActiveDiscoveryComparisonService,
    RWActiveDiscoveryPilotReport,
    RWActiveDiscoveryPilotService,
)
from astrograph.integrations.dor_active_discovery import (
    DORActiveDiscoveryArtifactBinding,
    DORActiveDiscoveryAuditRequest,
    DORActiveDiscoveryAuditService,
    DORActiveDiscoveryStatus,
)
from astrograph.integrations.dor_runtime import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DOR_PYTHON = Path(os.environ.get("ASTROGRAPH_DOR_PYTHON", "__optional_dor_python_not_found__"))
DOR_CLI = Path(os.environ.get("ASTROGRAPH_DOR_EXECUTABLE", "__optional_dor_not_found__"))
requires_dor_r5 = pytest.mark.skipif(
    not DOR_PYTHON.is_file() or not DOR_CLI.is_file(),
    reason="optional DOR R5 runtime is not configured",
)
RUNNER = ROOT / "scripts" / "integration" / "run_dor_r5_rw_pilot.py"


@pytest.fixture(scope="module")
def report() -> RWActiveDiscoveryPilotReport:
    if not DOR_PYTHON.is_file() or not DOR_CLI.is_file():
        pytest.skip("optional DOR R5 runtime is not configured")
    return RWActiveDiscoveryPilotService().run(
        python_executable=str(DOR_PYTHON),
        runner_path=RUNNER,
    )


def test_real_isgn_pilot_is_multiseed_truth_quarantined_and_three_structure(report):
    assert report.runtime["isgn_version"] == "2.4.0+timer.1"
    assert report.aggregate.episode_count == 9
    assert report.aggregate.evaluation_seed_count == 3
    assert report.aggregate.mean_final_entropy_bits < report.aggregate.mean_initial_entropy_bits
    assert all(not row.rank_input_truth_fields for row in report.episodes)
    assert report.provenance.adcs_v5_2_2_test_partition_accessed is False
    assert report.provenance.phm_v6_0_test_partition_accessed is False
    assert report.provenance.dataset_metric_claimed is False


def test_pilot_representative_payload_passes_real_dor_advisory_firewall(report, tmp_path):
    payload = tmp_path / "payload.json"
    payload.write_text(
        report.representative_advisory_payload.model_dump_json(exclude_none=True),
        encoding="utf-8",
    )
    audit = DORActiveDiscoveryAuditService().audit(
        DORActiveDiscoveryAuditRequest(
            input_artifact=DORActiveDiscoveryArtifactBinding(
                path=str(payload),
                sha256=sha256_file(payload),
            ),
            executable_path=str(DOR_CLI),
        )
    )
    assert audit.status == DORActiveDiscoveryStatus.ADVISORY_ONLY
    assert audit.eligible_for_offline_planning is True
    assert audit.execution_allowed is False
    assert audit.active_graph_modified is False
    assert audit.birrf_branch_count_delta == 0


def test_report_rejects_dataset_metric_privilege(report):
    raw = json.loads(report.model_dump_json())
    raw["provenance"]["dataset_metric_claimed"] = True
    with pytest.raises(ValueError, match="cannot access dataset test partitions"):
        RWActiveDiscoveryPilotReport.model_validate(raw)


@requires_dor_r5
def test_equal_budget_comparison_applies_stop_gate_without_online_privilege():
    comparison = RWActiveDiscoveryComparisonService().run(
        python_executable=str(DOR_PYTHON),
        runner_path=RUNNER,
        evaluation_seed_count=2,
    )
    assert len(comparison.points) == 9
    assert comparison.stop_gate_status == "STOP_NO_CLEAR_BENEFIT"
    assert comparison.experiments_to_entropy_target == {
        "active": 1,
        "fixed": 1,
        "random": 1,
    }
    assert comparison.online_integration_allowed is False
    assert comparison.kg_update_allowed is False
    assert comparison.birrf_integration_allowed is False


def test_comparison_report_rejects_online_privilege():
    from astrograph.evaluation.rw_active_discovery import RWActiveDiscoveryComparisonReport

    with pytest.raises(ValueError):
        RWActiveDiscoveryComparisonReport.model_validate(
            {
                "schema_version": "astrograph-dor-r5-rw-comparison.v1",
                "points": [],
                "entropy_target_bits": 0.05,
                "experiments_to_entropy_target": {},
                "stop_gate_status": "ADVANCE",
                "reasons": ["invalid privilege probe"],
                "online_integration_allowed": True,
            }
        )
