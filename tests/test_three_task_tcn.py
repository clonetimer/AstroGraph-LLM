from __future__ import annotations

import pytest
import numpy as np

from astrograph.data_driven.three_task_tcn import (
    hierarchy_js_loss,
    make_causal_multiview_detector_tcn,
    make_decoupled_tcn,
    make_hierarchy_event_tcn,
    make_mechanism_gated_gat_tcn,
    receptive_field,
    task_gradient_statistics,
    trajectory_uniform_weights,
    mean_plus_cvar,
)
from astrograph.data_driven.progressive_hierarchy import (
    make_progressive_hierarchical_gat_tcn,
)
from scripts.data_driven.compare_phm_causal_windows import (
    normalize_detection_scores,
    summary_features,
)


def test_receptive_field_matches_current_and_phm_window() -> None:
    assert receptive_field((1, 2, 4)) == 29
    assert receptive_field((1, 2, 4, 8)) == 61


def test_trajectory_uniform_weights_emphasize_nominal_without_window_bias() -> None:
    sample_ids = ["eps:fault:mc_00"] * 2 + ["eps:full:nominal:mc_00"] * 4
    values = trajectory_uniform_weights(sample_ids, nominal_multiplier=3.0)
    assert sum(values[:2]) == pytest.approx(1.0)
    assert sum(values[2:]) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="positive"):
        trajectory_uniform_weights(sample_ids, nominal_multiplier=0.0)


def test_mean_plus_cvar_penalizes_upper_tail_and_validates_contract() -> None:
    torch = pytest.importorskip("torch")
    losses = torch.tensor([1.0, 1.0, 2.0, 6.0])
    assert float(mean_plus_cvar(losses)) == pytest.approx(2.5)
    assert float(mean_plus_cvar(losses, cvar_weight=0.5, cvar_fraction=0.25)) == pytest.approx(5.5)
    with pytest.raises(ValueError, match="valid range"):
        mean_plus_cvar(losses, cvar_weight=0.5, cvar_fraction=0.0)


def test_hierarchy_event_tcn_preserves_three_task_contract() -> None:
    torch = pytest.importorskip("torch")
    adcs_model = make_hierarchy_event_tcn(43, 5, 14, window_size=20)
    phm_model = make_hierarchy_event_tcn(96, 13, 18, window_size=60)
    assert adcs_model.receptive_field == 29
    assert phm_model.receptive_field == 61
    assert [tuple(value.shape) for value in adcs_model(torch.randn(3, 20, 43))] == [
        (3, 2), (3, 5), (3, 14)
    ]
    assert [tuple(value.shape) for value in phm_model(torch.randn(2, 60, 96))] == [
        (2, 2), (2, 13), (2, 18)
    ]


def test_causal_multiview_detector_preserves_three_head_contract() -> None:
    torch = pytest.importorskip("torch")
    model = make_causal_multiview_detector_tcn(12, 13, 17, window_size=60)
    outputs = model(torch.randn(3, 60, 12))
    assert [tuple(value.shape) for value in outputs] == [(3, 2), (3, 13), (3, 17)]
    assert model.anchor_length == 15
    assert model.branching_mode == "detection_specialist_absolute_delta_anchor_statistics"


def test_change_summary_extends_legacy_without_nonfinite_values() -> None:
    rng = np.random.default_rng(17)
    values = rng.normal(size=(4, 60, 6)).astype(np.float32)
    legacy = summary_features(values, profile="legacy_summary")
    change = summary_features(values, profile="change_summary")
    assert legacy.shape == (4, 24)
    assert change.shape == (4, 48)
    assert np.allclose(change[:, :legacy.shape[1]], legacy)
    assert np.isfinite(change).all()


def test_initial_logit_score_normalization_is_group_local_and_causal() -> None:
    scores = np.asarray([0.2, 0.25, 0.8, 0.7, 0.7, 0.7], dtype=float)
    groups = ["a", "a", "a", "b", "b", "b"]
    times = np.asarray([59.0, 64.0, 90.0, 59.0, 64.0, 90.0])
    normalized = normalize_detection_scores(
        scores, groups, times, profile="initial_logit_delta", baseline_duration_s=5.0
    )
    assert np.allclose(normalized[[0, 1, 3, 4]], 0.5, atol=0.04)
    assert normalized[2] > 0.9
    assert abs(normalized[5] - 0.5) < 1e-6

    changed_future = scores.copy()
    changed_future[[2, 5]] = [0.99, 0.01]
    replay = normalize_detection_scores(
        changed_future, groups, times, profile="initial_logit_delta", baseline_duration_s=5.0
    )
    assert np.allclose(normalized[[0, 1, 3, 4]], replay[[0, 1, 3, 4]])


def test_hierarchy_js_loss_is_small_when_heads_agree() -> None:
    torch = pytest.importorskip("torch")
    component = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    classes = torch.tensor([[8.0, -8.0], [-8.0, 8.0]])
    active = torch.tensor([True, True])
    aligned = hierarchy_js_loss(component, classes, [0, 1], active)
    wrong = hierarchy_js_loss(component, classes, [1, 0], active)
    assert float(aligned) < 1e-5
    assert float(wrong) > 0.5


def test_hierarchy_js_loss_ignores_inactive_rows() -> None:
    torch = pytest.importorskip("torch")
    component = torch.randn(2, 2, requires_grad=True)
    classes = torch.randn(2, 3, requires_grad=True)
    loss = hierarchy_js_loss(component, classes, [0, 0, 1], torch.tensor([False, False]))
    assert float(loss.detach()) == 0.0
    loss.backward()


def test_adcs_validation_only_training_never_scores_test(monkeypatch) -> None:
    pytest.importorskip("torch")
    from scripts.data_driven import compare_adcs_deep_encoders as deep
    from scripts.integration import run_adcs_v522_regression as adcs

    class_names = sorted(adcs.FAULT_TO_COMPONENT)
    component_names = list(adcs.COMPONENT_NAMES)
    class_index = np.arange(28) % len(class_names)
    detection = np.asarray([0, 1] * 14, dtype=np.int64)
    isolation = np.where(detection == 1, class_index, -1).astype(np.int64)
    component_lookup = {
        name: index for index, name in enumerate(component_names)
    }
    component = np.asarray([
        component_lookup[adcs.FAULT_TO_COMPONENT[class_names[value]]] if active else -1
        for active, value in zip(detection, class_index, strict=True)
    ], dtype=np.int64)

    def window(seed: int) -> adcs.WindowSet:
        rng = np.random.default_rng(seed)
        return adcs.WindowSet(
            x=rng.normal(size=(28, adcs.WINDOW, len(adcs.FEATURE_COLUMNS))).astype(np.float32),
            detection=detection,
            isolation=isolation,
            trajectory_ids=[f"sample-{index}" for index in range(28)],
            end_times_s=np.arange(28, dtype=float),
            onset_times_s=np.zeros(28, dtype=float),
            component=component,
        )

    train, validation, hidden_test = window(1), window(2), window(3)
    scored: list[int] = []

    def detector_score(_train, data, _seed):
        scored.append(id(data))
        return np.asarray(data.detection, dtype=float)

    monkeypatch.setattr(deep, "logistic_detection_scores", detector_score)
    result = deep.train_model(
        "residual_tcn", train, validation, hidden_test,
        class_names, component_names, epochs=1, seed=7, batch_size=14,
        evaluate_test_split=False,
    )
    assert scored == [id(validation)]
    assert result["evaluation_scope"] == "validation_only; test scores were not computed"
    assert "detection" not in result


def test_trajectory_relative_detector_features_remove_constant_operating_offset() -> None:
    from scripts.data_driven import compare_adcs_deep_encoders as deep
    from scripts.integration import run_adcs_v522_regression as adcs

    x = np.arange(4 * 20 * 3, dtype=np.float32).reshape(4, 20, 3)
    base = adcs.WindowSet(
        x=x, detection=np.asarray([0, 1, 0, 1]),
        isolation=np.asarray([-1, 0, -1, 0]),
        trajectory_ids=["a", "a", "b", "b"],
        end_times_s=np.asarray([20.0, 40.0, 20.0, 40.0]),
        onset_times_s=np.asarray([30.0] * 4),
        component=np.asarray([-1, 0, -1, 0]),
    )
    shifted = adcs.WindowSet(
        x=x + np.asarray([[[100.0]], [[100.0]], [[-50.0]], [[-50.0]]]),
        detection=base.detection, isolation=base.isolation,
        trajectory_ids=base.trajectory_ids, end_times_s=base.end_times_s,
        onset_times_s=base.onset_times_s, component=base.component,
    )
    assert np.allclose(
        deep.trajectory_relative_features(base),
        deep.trajectory_relative_features(shifted),
    )


def test_task_gradient_statistics_detect_opposing_gradients() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    first = parameter[0] + parameter[1]
    second = -parameter[0] - 0.5 * parameter[1]
    statistics = task_gradient_statistics([first, second], [parameter])
    assert statistics["pairwise_cosine"]["0-1"] < 0.0
    assert statistics["conflict_fraction"] == 1.0


@pytest.mark.parametrize("kind", [
    "pooled_task_adapter_tcn",
    "late_fork_tcn",
    "tied_late_fork_tcn",
    "capacity_matched_shared_tcn",
])
def test_decoupled_variants_preserve_three_head_contract(kind: str) -> None:
    torch = pytest.importorskip("torch")
    model = make_decoupled_tcn(kind, 43, 5, 14)
    outputs = model(torch.randn(3, 20, 43))
    assert [tuple(value.shape) for value in outputs] == [(3, 2), (3, 5), (3, 14)]
    assert model.receptive_field == 29


def test_late_fork_task_losses_only_update_their_private_branch() -> None:
    torch = pytest.importorskip("torch")
    model = make_decoupled_tcn("late_fork_tcn", 8, 3, 4, width=16)
    detection, localization, classification = model(torch.randn(4, 20, 8))
    detection.sum().backward()
    assert any(parameter.grad is not None for parameter in model.detection_branch.parameters())
    assert all(parameter.grad is None for parameter in model.diagnosis_branch.parameters())

    model.zero_grad(set_to_none=True)
    detection, localization, classification = model(torch.randn(4, 20, 8))
    (localization.sum() + classification.sum()).backward()
    assert all(parameter.grad is None for parameter in model.detection_branch.parameters())
    assert any(parameter.grad is not None for parameter in model.diagnosis_branch.parameters())


def test_tied_fork_reuses_the_same_final_block() -> None:
    pytest.importorskip("torch")
    model = make_decoupled_tcn("tied_late_fork_tcn", 8, 3, 4, width=16)
    assert model.detection_branch is model.diagnosis_branch


def test_pooled_adapters_receive_only_their_task_gradients() -> None:
    torch = pytest.importorskip("torch")
    model = make_decoupled_tcn("pooled_task_adapter_tcn", 8, 3, 4, width=16, adapter_width=4)
    detection, localization, classification = model(torch.randn(4, 20, 8))
    detection.sum().backward()
    assert any(parameter.grad is not None for parameter in model.detection_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.diagnosis_adapter.parameters())

    model.zero_grad(set_to_none=True)
    detection, localization, classification = model(torch.randn(4, 20, 8))
    (localization.sum() + classification.sum()).backward()
    assert all(parameter.grad is None for parameter in model.detection_adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.diagnosis_adapter.parameters())


def _small_gat_model():
    return make_mechanism_gated_gat_tcn(
        6, 3, 4,
        subsystem_groups=[[0, 1], [2, 3], [4, 5]],
        component_group=[0, 1, 2],
        class_component=[0, 1, 1, 2],
        graph_edges=[(0, 1), (1, 0), (1, 2), (2, 1)],
        temporal_width=16,
        node_width=8,
        attention_heads=2,
    )


def test_mechanism_gated_gat_preserves_contract_and_sparse_mask() -> None:
    torch = pytest.importorskip("torch")
    model = _small_gat_model()
    outputs = model(torch.randn(5, 20, 6))
    assert [tuple(value.shape) for value in outputs] == [(5, 2), (5, 3), (5, 4)]
    mask = model.graph_attention.attention_mask.cpu().numpy()
    assert mask.tolist() == [
        [True, True, False],
        [True, True, True],
        [False, True, True],
    ]
    diagnostics = model.graph_diagnostics()
    assert 0.09 < diagnostics["message_gate"] < 0.11
    assert diagnostics["self_loops"] == 3


def test_mechanism_gated_gat_keeps_graph_out_of_detection_path() -> None:
    torch = pytest.importorskip("torch")
    model = _small_gat_model()
    detection, localization, classification = model(torch.randn(4, 20, 6))
    detection.sum().backward()
    assert any(parameter.grad is not None for parameter in model.global_blocks.parameters())
    assert all(parameter.grad is None for parameter in model.local_blocks.parameters())
    assert all(parameter.grad is None for parameter in model.graph_attention.parameters())

    model.zero_grad(set_to_none=True)
    detection, localization, classification = model(torch.randn(4, 20, 6))
    (localization.sum() + classification.sum()).backward()
    assert any(parameter.grad is not None for parameter in model.global_blocks.parameters())
    assert any(parameter.grad is not None for parameter in model.local_blocks.parameters())
    assert any(parameter.grad is not None for parameter in model.graph_attention.parameters())


def _small_progressive_model():
    return make_progressive_hierarchical_gat_tcn(
        8, 3, 4,
        parameter_groups=[[0, 1], [2], [3, 4], [5], [6, 7]],
        parameter_part=[0, 0, 1, 2, 3],
        part_component=[0, 1, 1, 2],
        component_module=[0, 1, 1],
        module_subsystem=[0, 0],
        class_component=[0, 1, 1, 2],
        temporal_width=16,
        node_width=8,
        attention_heads=2,
    )


def test_progressive_hierarchy_preserves_three_task_contract() -> None:
    torch = pytest.importorskip("torch")
    model = _small_progressive_model()
    outputs = model(torch.randn(5, 20, 8))
    assert [tuple(value.shape) for value in outputs] == [(5, 2), (5, 3), (5, 4)]
    diagnostics = model.graph_diagnostics()
    assert diagnostics["hierarchy_levels"] == [
        "parameter", "part", "component", "functional_module", "subsystem",
        "whole_spacecraft",
    ]
    assert diagnostics["level_counts"] == [5, 4, 3, 2, 1, 1]
    assert diagnostics["detection_uses_graph"] is False


def test_progressive_hierarchy_keeps_graph_out_of_detection_gradient() -> None:
    torch = pytest.importorskip("torch")
    model = _small_progressive_model()
    detection, _, _ = model(torch.randn(4, 20, 8))
    detection.sum().backward()
    assert any(parameter.grad is not None for parameter in model.global_blocks.parameters())
    assert all(parameter.grad is None for parameter in model.parameter_blocks.parameters())
    assert all(parameter.grad is None for parameter in model.upward.parameters())


def test_progressive_hierarchy_rejects_missing_parent() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="empty parent"):
        make_progressive_hierarchical_gat_tcn(
            3, 2, 2,
            parameter_groups=[[0], [1], [2]], parameter_part=[0, 2, 2],
            part_component=[0, 1, 1], component_module=[0, 0],
            module_subsystem=[0], class_component=[0, 1],
            temporal_width=16, node_width=8, attention_heads=2,
        )


def test_gru_inception_specialist_separates_detection_and_diagnosis_gradients() -> None:
    torch = pytest.importorskip("torch")
    from scripts.data_driven import compare_adcs_deep_encoders as deep

    model = deep.make_model("gru_inception_specialist", 8, 3, 4, specialist_seed=17)
    detection, localization, classification = model(torch.randn(3, 20, 8))
    assert [tuple(item.shape) for item in (detection, localization, classification)] == [
        (3, 2), (3, 3), (3, 4),
    ]
    detection.sum().backward()
    assert any(parameter.grad is not None for parameter in model.detection_expert.parameters())
    assert all(parameter.grad is None for parameter in model.diagnosis_expert.parameters())

    model.zero_grad(set_to_none=True)
    _, localization, classification = model(torch.randn(3, 20, 8))
    (localization.sum() + classification.sum()).backward()
    assert all(parameter.grad is None for parameter in model.detection_expert.parameters())
    assert any(parameter.grad is not None for parameter in model.diagnosis_expert.parameters())


def test_gru_inception_specialist_matches_standalone_initialization() -> None:
    torch = pytest.importorskip("torch")
    from scripts.data_driven import compare_adcs_deep_encoders as deep

    seed = 23
    torch.manual_seed(seed)
    gru = deep.make_model("gru", 8, 3, 4)
    torch.manual_seed(seed)
    inception = deep.make_model("inception_time", 8, 3, 4)
    specialist = deep.make_model(
        "gru_inception_specialist", 8, 3, 4, specialist_seed=seed,
    )
    for expected, observed in (
        (gru.state_dict(), specialist.detection_expert.state_dict()),
        (inception.state_dict(), specialist.diagnosis_expert.state_dict()),
    ):
        assert expected.keys() == observed.keys()
        assert all(torch.equal(expected[key], observed[key]) for key in expected)
