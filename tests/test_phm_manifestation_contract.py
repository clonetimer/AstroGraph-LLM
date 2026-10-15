from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.data_driven.compare_phm_causal_windows import supervision_active


def test_manifestation_contract_does_not_label_latent_interval_observable() -> None:
    record = SimpleNamespace(
        sample_id="fault",
        condition_id="eps_battery_capacity_loss",
        fault_manifestation_time_s=105.0,
    )
    times = np.arange(100.0, 108.0)
    injected = np.ones(len(times), dtype=bool)
    manifested = supervision_active(record, times, injected, "manifestation_aware")
    assert manifested.tolist() == [False, False, False, False, False, True, True, True]


def test_manifestation_contract_keeps_nominal_false() -> None:
    record = SimpleNamespace(
        sample_id="nominal", condition_id="nominal", fault_manifestation_time_s=np.nan,
    )
    result = supervision_active(
        record, np.arange(4, dtype=float), np.zeros(4, dtype=bool), "manifestation_aware"
    )
    assert not result.any()
