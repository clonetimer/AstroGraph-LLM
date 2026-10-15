from __future__ import annotations

import numpy as np
import pytest

from astrograph.evaluation.data_exposure import dataset_use
from astrograph.evaluation.esa_high_resolution import (
    ESAHighResolutionConfig,
    fit_delta_reference,
    group_alarm_intervals,
    score_high_resolution_transients,
)


def _reference():
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT", group_ids=["esa-synth-fit"]):
        return fit_delta_reference({
            "channel_41": [np.zeros(40)],
            "channel_42": [np.zeros(40)],
            "channel_43": [np.zeros(40)],
        })


def test_short_multichannel_transient_survives_raw_grid_and_is_grouped():
    reference = _reference()
    time_ns = np.arange(100, dtype=np.int64) * 1_000_000_000
    values = {name: np.zeros(100) for name in ("channel_41", "channel_42", "channel_43")}
    for array in values.values():
        array[50:52] = 1.0
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["esa-synth-score"]):
        score = score_high_resolution_transients(
            reference=reference,
            time_ns=time_ns,
            channel_values=values,
            config=ESAHighResolutionConfig(
                robust_z_threshold=8.0,
                minimum_coincident_channels=3,
                alarm_group_gap_s=2.0,
            ),
        )
    assert [i for i, value in enumerate(score.alarm) if value] == [50, 52]
    assert score.score_semantics.endswith("NOT_PROBABILITY")
    assert group_alarm_intervals(score, gap_s=2.0) == ((50_000_000_000, 52_000_000_000),)


def test_high_resolution_channel_contract_fails_closed():
    with dataset_use("SYNTHETIC_PROTOCOL", "MODEL_TRAIN", "FIT", group_ids=["esa-synth-fit-2"]):
        reference = fit_delta_reference({
            "channel_41": [[0.0, 0.0, 0.0]],
            "channel_42": [[0.0, 0.0, 0.0]],
        })
    with dataset_use("SYNTHETIC_PROTOCOL", "DEVELOPMENT_SELECTION", "OPTIMIZE", group_ids=["esa-synth-score-2"]):
        with pytest.raises(ValueError, match="CHANNEL_SET_MISMATCH"):
            score_high_resolution_transients(
                reference=reference,
                time_ns=[0, 1, 2],
                channel_values={"channel_41": [0.0, 0.0, 1.0]},
                config=ESAHighResolutionConfig(robust_z_threshold=8.0, minimum_coincident_channels=1),
            )


def test_seen_esa_confirmatory_partition_cannot_regenerate_high_resolution_predictions():
    reference = _reference()
    with dataset_use("ESA_MISSION1", "CONFIRMATORY_EVALUATION", "AUDIT", group_ids=["id_107"]):
        with pytest.raises(ValueError, match="SEEN_EVALUATION_PREDICTION_REGENERATION_BLOCKED"):
            score_high_resolution_transients(
                reference=reference,
                time_ns=[0, 1, 2],
                channel_values={item.channel_id: [0.0, 0.0, 0.0] for item in reference.channels},
                config=ESAHighResolutionConfig(robust_z_threshold=8.0, minimum_coincident_channels=1),
            )
