from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.data_driven.sota_comparison import (
    SOTA_FAMILIES,
    validate_domain_sota_comparison_audit,
)


def test_domain_sota_family_set_is_frozen() -> None:
    assert SOTA_FAMILIES == ("inception_time", "timesnet", "modern_tcn")


def test_domain_sota_audit_rejects_tampering(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({
        "status": "PASS_DOMAIN_SOTA_FAIR_COMPARISON",
        "anti_suppression_checks": {"all": True},
        "audit_sha256": "0" * 64,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_domain_sota_comparison_audit(path)


def test_governed_domain_sota_audit_is_current() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = validate_domain_sota_comparison_audit(
        root / "governance/evidence/research/data_driven_runs/"
        "domain_sota_group_cv_20260901/fairness_audit.json"
    )
    assert payload["status"] == "PASS_DOMAIN_SOTA_FAIR_COMPARISON"
    assert set(payload["results"]["adcs.v5_2_2"]) == {
        "inception_time", "timesnet", "modern_tcn", "residual_tcn",
    }
    assert set(payload["results"]["phm.v6_0"]) == {
        "astrograph_specialist", "inception_time", "timesnet", "modern_tcn",
    }
    assert len(payload["training_receipts"]["adcs"]) == 20
    assert len(payload["training_receipts"]["phm"]) == 40
