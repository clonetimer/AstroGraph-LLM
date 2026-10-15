from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.data_driven.mechanism_features import (
    compose_columns,
    load_feature_bank,
    profile_columns,
    require_available,
)

ROOT = Path(__file__).parents[1]
BANK = ROOT / "configs/data_driven/mechanism_feature_banks.json"


def test_project_feature_bank_profiles_are_deterministic_and_nonduplicating() -> None:
    bank = load_feature_bank(BANK)
    additions = profile_columns(bank, "adcs_v5_2_2", "full")
    combined = compose_columns(["base.a", additions[0]], additions)

    assert len(additions) == 20
    assert len(combined) == 21
    assert combined[0] == "base.a"
    assert len(profile_columns(bank, "phm_v6_0", "mechanism_required")) == 19


def test_feature_bank_fails_closed_on_unknown_profile_and_missing_column() -> None:
    bank = load_feature_bank(BANK)
    with pytest.raises(ValueError, match="unknown mechanism feature profile"):
        profile_columns(bank, "adcs_v5_2_2", "unknown")
    with pytest.raises(ValueError, match="are missing"):
        require_available(["present", "missing"], ["present"])


def test_feature_bank_rejects_duplicate_family_columns(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({
        "schema_version": "astrograph-mechanism-feature-banks.v1",
        "datasets": {"dataset": {
            "families": {"bad": ["same", "same"]},
            "profiles": {"bad": ["bad"]},
        }},
    }))
    with pytest.raises(ValueError, match="duplicate column"):
        load_feature_bank(path)
