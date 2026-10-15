from __future__ import annotations

from astrograph.diagnosis.unified import (
    EXPANDED_SINGLE_FAULT_ENGINE_ID,
    _fuse_paired,
)
from astrograph.diagnosis.unified_models import EngineExecutionStatus, EngineRunRecord, UnifiedDiagnosisStatus


def _record(status: str, *, detected: list[str], primary: str | None) -> EngineRunRecord:
    return EngineRunRecord(
        engine_id=EXPANDED_SINGLE_FAULT_ENGINE_ID,
        eligible=True,
        execution_status=EngineExecutionStatus.EXECUTED,
        status=status,
        detected_effect_ids=detected,
        primary_effect_id=primary,
    )


def test_unified_fusion_accepts_identifiable_expanded_mode() -> None:
    status, conclusions, basis = _fuse_paired([_record("DIAGNOSED", detected=["mode_a"], primary="mode_a")])
    assert status == UnifiedDiagnosisStatus.DIAGNOSED
    assert conclusions == ["mode_a"]
    assert basis == "EXPANDED_SINGLE_FAULT_IDENTIFIABLE"


def test_unified_fusion_preserves_equivalence_candidates() -> None:
    status, conclusions, basis = _fuse_paired([
        _record("OBSERVATIONALLY_EQUIVALENT", detected=["mode_a", "mode_b"], primary=None)
    ])
    assert status == UnifiedDiagnosisStatus.AMBIGUOUS
    assert conclusions == ["mode_a", "mode_b"]
    assert basis == "OBSERVATIONALLY_EQUIVALENT_TELEMETRY"
