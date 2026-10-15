import pytest

from astrograph.evaluation.data_exposure import admit_dataset_use, dataset_use, guard_development


def test_seen_competition_tune_is_blocked():
    with pytest.raises(ValueError, match="R21_SEEN_EVALUATION_REUSE_FOR_DEVELOPMENT_BLOCKED"):
        admit_dataset_use("DXC09_COMPETITION", "SEEN_EVALUATION", "TUNE")


def test_ops_external_fit_is_blocked():
    with pytest.raises(ValueError, match="R21_SEEN_EVALUATION_REUSE_FOR_DEVELOPMENT_BLOCKED"):
        admit_dataset_use("OPS_SAT", "EXTERNAL_VALIDATION", "FIT")


def test_esa_confirmatory_select_is_blocked():
    with pytest.raises(ValueError, match="R21_SEEN_EVALUATION_REUSE_FOR_DEVELOPMENT_BLOCKED"):
        admit_dataset_use("ESA_MISSION1", "CONFIRMATORY_EVALUATION", "SELECT")


def test_readonly_decompose_and_development_fit_are_allowed():
    assert admit_dataset_use("DXC09_COMPETITION", "SEEN_EVALUATION", "DECOMPOSE")["exposure_status"] == "SEEN_EVALUATION_READ_ONLY"
    assert admit_dataset_use("DXC09_SAMPLE", "DEVELOPMENT", "FIT")["exposure_status"] == "DEVELOPMENT_PERMITTED"


def test_shared_development_api_cannot_inherit_readonly_permission():
    with dataset_use("DXC09_COMPETITION", "SEEN_EVALUATION", "DECOMPOSE"):
        with pytest.raises(ValueError, match="R21_SEEN_EVALUATION_REUSE_FOR_DEVELOPMENT_BLOCKED"):
            guard_development("FIT")


def test_future_holdout_is_always_blocked():
    with pytest.raises(ValueError, match="R21_FUTURE_HOLDOUT_ACCESS_BLOCKED"):
        admit_dataset_use("FUTURE_HOLDOUT", "RESERVED", "AUDIT")
