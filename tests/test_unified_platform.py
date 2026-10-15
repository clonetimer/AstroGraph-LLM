"""AI-Native Phase 3: unified platform model skeleton tests.
Verifies the shared-backbone / per-dataset-adapter contract without training:
forward shapes per dataset, adapter routing, shared-parameter accounting,
channel mismatch fail-closed, and unknown adapter rejection.
"""

from __future__ import annotations

import pytest

from astrograph.data_driven.unified_platform import (
    DatasetTaskSpec,
    make_unified_platform_model,
)

ADCS = DatasetTaskSpec(channels=43, components=5, classes=14, window_size=20)
PHM = DatasetTaskSpec(channels=96, components=13, classes=18, window_size=60)


def test_unified_platform_preserves_per_dataset_task_shapes():
    torch = pytest.importorskip("torch")
    model = make_unified_platform_model({"adcs": ADCS, "phm": PHM})
    adcs_out = model(torch.randn(3, 20, 43), "adcs")
    phm_out = model(torch.randn(2, 60, 96), "phm")
    assert [tuple(item.shape) for item in adcs_out] == [(3, 2), (3, 5), (3, 14)]
    assert [tuple(item.shape) for item in phm_out] == [(2, 2), (2, 13), (2, 18)]


def test_shared_backbone_parameters_are_accounted():
    torch = pytest.importorskip("torch")
    model = make_unified_platform_model({"adcs": ADCS, "phm": PHM})
    shared = model.shared_parameter_count()
    assert shared > 0
    # Shared params are exactly blocks + pooling; adapters are excluded.
    from torch import nn
    adapters_total = sum(
        parameter.numel()
        for adapter in model.adapters.values()
        for parameter in adapter.parameters()
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    assert shared + adapters_total == total


def test_receptive_field_covers_longest_window():
    torch = pytest.importorskip("torch")
    model = make_unified_platform_model({"adcs": ADCS, "phm": PHM})
    assert model.receptive_field >= 60


def test_channel_mismatch_fails_closed():
    torch = pytest.importorskip("torch")
    model = make_unified_platform_model({"adcs": ADCS})
    with pytest.raises(ValueError):
        model(torch.randn(2, 20, 44), "adcs")


def test_unknown_dataset_adapter_rejected():
    torch = pytest.importorskip("torch")
    model = make_unified_platform_model({"adcs": ADCS})
    with pytest.raises(KeyError):
        model(torch.randn(2, 20, 43), "unknown")


def test_empty_specs_rejected():
    with pytest.raises(ValueError):
        make_unified_platform_model({})


def test_preregistration_doc_is_present():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "docs" / "AI_NATIVE_UNIFIED_BASELINE_PREREGISTRATION.md"
    assert path.is_file()
    content = path.read_text(encoding="utf-8")
    assert "止损门" in content
    assert "0.8974" in content
