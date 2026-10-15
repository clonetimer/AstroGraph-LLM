from __future__ import annotations

import pytest

from astrograph.data_driven.sota_backbones import (
    SOTA_MODEL_FAMILIES,
    make_sota_multitask_model,
)


@pytest.mark.parametrize("family", SOTA_MODEL_FAMILIES)
@pytest.mark.parametrize("window", [20, 60])
def test_sota_backbones_preserve_common_three_task_contract(family: str, window: int) -> None:
    import torch

    torch.manual_seed(7)
    model = make_sota_multitask_model(
        family, channels=12, components=5, classes=14, window_size=window
    )
    model.eval()
    value = torch.randn(3, window, 12)
    with torch.no_grad():
        detection, localization, classification = model(value)
    assert detection.shape == (3, 2)
    assert localization.shape == (3, 5)
    assert classification.shape == (3, 14)
    assert all(torch.isfinite(item).all() for item in (detection, localization, classification))


def test_unknown_sota_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown SOTA model family"):
        make_sota_multitask_model(
            "marketing_sota", channels=4, components=2, classes=3, window_size=20
        )
