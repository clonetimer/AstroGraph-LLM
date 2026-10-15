from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from astrograph.physics import default_dynamics_contracts
from astrograph.semantics import SamplingProfile, SemanticSignal, SemanticTelemetryFrame
from astrograph.state_estimation import (
    GrayBoxStateEstimator,
    ProductionEstimatorRegistry,
    UncertaintyCalibrationProfile,
)
from astrograph.state_estimation.gray_box import StateEstimationError


ROOT = Path(__file__).resolve().parents[1]


def _calibration_module():
    path = ROOT / "scripts/acceptance/calibrate_gray_box_uncertainty.py"
    spec = importlib.util.spec_from_file_location("calibrate_gray_box_uncertainty", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(
    *,
    archetype: str,
    timestamps: np.ndarray,
    signals: list[tuple[str, str, str, list[float]]],
) -> SemanticTelemetryFrame:
    semantic_signals = [
        SemanticSignal(
            semantic_channel_id=f"asset.component.{index}",
            asset_instance_id="asset-under-test",
            component_archetype_id=archetype,
            port_role_id=port_role,
            quantity_kind_id=quantity_kind,
            canonical_unit=unit,
            direction=1,
            values=values,
            missing_mask=[False] * len(values),
        )
        for index, (port_role, quantity_kind, unit, values) in enumerate(signals)
    ]
    payload = {
        "timestamps": timestamps.tolist(),
        "signals": [item.model_dump(mode="json") for item in semantic_signals],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    period = float(np.median(np.diff(timestamps)))
    return SemanticTelemetryFrame(
        frame_id=digest[:16],
        semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=timestamps.tolist(),
        signals=semantic_signals,
        sampling=SamplingProfile(
            row_count=len(timestamps),
            nominal_period_s=period,
            minimum_period_s=float(np.min(np.diff(timestamps))),
            maximum_period_s=float(np.max(np.diff(timestamps))),
            irregular=False,
            missing_value_count=0,
        ),
        semantic_data_sha256=digest,
    )


def _parameter(result, parameter_id: str) -> float:
    return next(item.value for item in result.parameters if item.parameter_id == parameter_id)


def test_storage_balance_recovers_bounded_capacity_without_truth_or_labels() -> None:
    timestamps = np.arange(0.0, 240.0, 1.0)
    # Alternating charge/discharge excitation so the storage v2 efficiency branch
    # (charge FIXED 0.98, discharge LEARNABLE) is identifiable.  The state
    # integrates P*eff over the step; charge uses charge_eff, discharge divides.
    power = np.asarray([35.0 if (i // 30) % 2 == 0 else -35.0 for i in range(len(timestamps))])
    capacity = 20_000.0
    charge_eff = 0.98
    state = np.empty_like(timestamps)
    state[0] = 0.25
    for index, dt in enumerate(np.diff(timestamps)):
        p = 0.5 * (power[index] + power[index + 1])
        eff = charge_eff if p >= 0.0 else 1.0 / 0.97
        state[index + 1] = state[index] + dt * p * eff / capacity
    frame = _frame(
        archetype="storage.energy_buffer",
        timestamps=timestamps,
        signals=[
            ("state.energy_fraction", "energy.state_of_charge", "1", state.tolist()),
            ("power.exchange", "electrical.active_power", "W", power.tolist()),
        ],
    )
    contract = default_dynamics_contracts()["storage.energy_buffer"]
    # A2 (2026-08-04): the storage efficiency path is NOT_CALIBRATED (the frozen
    # profile excludes storage because the efficiency branch does not pass the
    # noise-aware conservation gate), so the estimator runs uncalibrated.
    result = GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract)

    assert _parameter(result, "energy_capacity") == pytest.approx(capacity, rel=1e-8)
    # Efficiency branch reconstructs with the identified discharge_eff; allow the
    # small numerical slack from the piecewise efficiency model (was <1e-10 under
    # the single-slope model).
    assert result.residuals[0].root_mean_square < 1e-2
    assert {item.residual_id for item in result.residuals} == {
        "state_reconstruction",
        "balance_equation",
        "balance_innovation_whitened",
    }
    assert result.receipt.labels_used is False and result.receipt.truth_used is False
    assert result.receipt.residual_model == "NONE"
    # Honest uncalibrated position for the storage efficiency path.
    assert result.uncertainty.calibrated is False
    assert result.uncertainty.method == "linearized_residual_covariance"
    assert result.receipt.uncertainty_calibration_version == "NOT_CALIBRATED"
    assert result.receipt.conservation_calibration_version == "NOT_CALIBRATED"
    assert result.conservation_assessment.method == "UNCALIBRATED_RAW_ONLY"
    assert result.conservation_assessment.raw_violation_count == 0
    assert result.conservation_assessment.p_value_upper_confidence == 1.0
    assert result.conservation_assessment.statistically_significant is False
    assert result.receipt.uncertainty_calibration_sha256 == "NONE"


def test_storage_parameter_recovery_is_sampling_rate_invariant() -> None:
    contract = default_dynamics_contracts()["storage.energy_buffer"]
    estimates: list[float] = []
    for step in (0.5, 1.0, 2.0):
        timestamps = np.arange(0.0, 240.0 + step, step)
        # Alternating charge/discharge so the storage v2 efficiency branch is
        # identifiable across every sampling rate.
        power = np.asarray([35.0 if (i // 30) % 2 == 0 else -35.0 for i in range(len(timestamps))])
        capacity = 15_000.0
        charge_eff = 0.98
        state = np.empty_like(timestamps)
        state[0] = 0.2
        for index, dt in enumerate(np.diff(timestamps)):
            p = 0.5 * (power[index] + power[index + 1])
            eff = charge_eff if p >= 0.0 else 1.0 / 0.97
            state[index + 1] = state[index] + dt * p * eff / capacity
        frame = _frame(
            archetype="storage.energy_buffer",
            timestamps=timestamps,
            signals=[
                ("state.energy_fraction", "energy.state_of_charge", "1", state.tolist()),
                ("power.exchange", "electrical.active_power", "W", power.tolist()),
            ],
        )
        estimates.append(
            _parameter(
                GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract),
                "energy_capacity",
            )
        )

    assert estimates == pytest.approx([15_000.0, 15_000.0, 15_000.0], rel=1e-8)


def test_thermal_balance_recovers_capacity_and_conductance_on_new_input_profile() -> None:
    timestamps = np.arange(0.0, 160.0, 1.0)
    heat = 18.0 + 7.0 * np.sin(timestamps / 11.0) + 3.0 * np.cos(timestamps / 5.0)
    ambient = 292.0 + 0.8 * np.sin(timestamps / 37.0)
    capacity = 750.0
    conductance = 2.5
    temperature = np.empty_like(timestamps)
    temperature[0] = 294.0
    for index, dt in enumerate(np.diff(timestamps)):
        heat_mid = 0.5 * (heat[index] + heat[index + 1])
        ambient_mid = 0.5 * (ambient[index] + ambient[index + 1])
        left = capacity / dt + conductance / 2.0
        right = (
            temperature[index] * (capacity / dt - conductance / 2.0)
            + heat_mid
            + conductance * ambient_mid
        )
        temperature[index + 1] = right / left
    frame = _frame(
        archetype="thermal.lumped_capacity",
        timestamps=timestamps,
        signals=[
            ("state.temperature", "thermal.temperature", "K", temperature.tolist()),
            ("heat.exchange", "electrical.active_power", "W", heat.tolist()),
            ("context.ambient_temperature", "thermal.temperature", "K", ambient.tolist()),
        ],
    )
    contract = default_dynamics_contracts()["thermal.lumped_capacity"]
    result = GrayBoxStateEstimator().estimate(frame, contract)

    assert _parameter(result, "thermal_capacity") == pytest.approx(capacity, rel=1e-7)
    assert _parameter(result, "thermal_conductance") == pytest.approx(conductance, rel=1e-7)
    assert result.residuals[0].root_mean_square < 1e-8
    assert result.constraint_violations == []


def test_estimator_fails_closed_when_required_physical_input_is_missing() -> None:
    timestamps = np.arange(0.0, 12.0, 1.0)
    frame = _frame(
        archetype="storage.energy_buffer",
        timestamps=timestamps,
        signals=[
            (
                "state.energy_fraction",
                "energy.state_of_charge",
                "1",
                np.linspace(0.2, 0.3, len(timestamps)).tolist(),
            )
        ],
    )
    with pytest.raises(StateEstimationError, match="power.exchange"):
        GrayBoxStateEstimator().estimate(
            frame, default_dynamics_contracts()["storage.energy_buffer"]
        )


def test_production_registry_rejects_ensembles_and_hidden_fallbacks() -> None:
    registry = ProductionEstimatorRegistry()
    registry.register("storage.energy_buffer", GrayBoxStateEstimator())
    assert registry.manifest() == {
        "storage.energy_buffer": "astrograph.gray-box.bounded-balance"
    }
    with pytest.raises(ValueError, match="already registered"):
        registry.register("storage.energy_buffer", GrayBoxStateEstimator())


def test_frozen_production_manifest_has_one_slot_and_no_residual_or_fallback() -> None:
    manifest = json.loads(
        (ROOT / "configs/state_estimation/production_estimators.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["production_slot"] == "component_state_estimator"
    assert manifest["estimator_id"] == "astrograph.gray-box.bounded-balance"
    assert manifest["residual_model"] == "NONE"
    assert manifest["estimator_version"] == "1.2.0"
    assert manifest["uncertainty_calibration_resource"].endswith(
        "gray_box_uncertainty_calibration.v1.json"
    )
    calibration_path = (
        ROOT / "src/astrograph/resources/gray_box_uncertainty_calibration.v1.json"
    )
    assert manifest["uncertainty_calibration_sha256"] == hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()
    assert manifest["fallback_estimators"] == []
    assert manifest["ensemble_members"] == []
    archetypes = [item["component_archetype_id"] for item in manifest["bindings"]]
    assert len(archetypes) == len(set(archetypes)) == 2


def test_failed_conformal_conservation_experiment_remains_non_passing() -> None:
    evidence = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/gray_box_uncertainty_calibration_validation.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["estimator_version"] == "1.2.0"
    assert evidence["noise_aware_conservation_status"] == "FAIL"
    assert evidence["t306_status"] == "INCOMPLETE"
    assert evidence["validation"]["thermal.lumped_capacity"][
        "noise_aware_false_positive_rate"
    ] > 0.05
    assert evidence["equation_violation_validation"]["thermal.lumped_capacity"][
        "detection_rate"
    ] < 0.95


def test_frozen_gls_calibration_passes_noise_aware_gate_without_erasing_raw_failures() -> None:
    profile_path = ROOT / "src/astrograph/resources/gray_box_uncertainty_calibration.v1.json"
    profile = UncertaintyCalibrationProfile.model_validate_json(
        profile_path.read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (
            ROOT
            / "governance/evidence/0.7.1-extension/gray_box_conservation_gls_validation.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert profile.estimator_version == "1.2.0"
    assert profile.version == "1.2.0"
    assert profile.formal_hidden_truth_used is False
    assert profile.synthetic_reference_used is True
    for calibration in profile.archetypes:
        assert calibration.conservation_method == (
            "GLS_CHANGE_POINT_PARAMETRIC_BOOTSTRAP"
        )
        assert calibration.conservation_false_alarm_rate == 0.05
        assert calibration.conservation_bootstrap_replicates == 256
        assert calibration.conservation_mc_confidence_level == 0.95
    assert evidence["formal_hidden_truth_used"] is False
    assert evidence["uncertainty_calibration_status"] == "PASS"
    assert evidence["residual_whiteness_status"] == "PASS"
    assert evidence["noise_aware_conservation_status"] == "PASS"
    assert evidence["t306_status"] == "PASS"
    for result in evidence["validation"].values():
        assert result["state_simultaneous_coverage"] >= 0.90
        assert result["parameter_joint_coverage"] >= 0.90
        assert result["median_absolute_whitened_innovation_lag1_correlation"] <= 0.2
        assert result["noise_aware_false_positive_rate"] <= 0.05
        assert result["raw_violation_receipts_preserved"] is True
        assert result["median_raw_conservation_violation_fraction"] > 0.0
        assert result["raw_conservation_status"] == "FAIL"
    for result in evidence["equation_violation_validation"].values():
        assert result["total_detection_or_fail_closed_rate"] >= 0.95
        assert result["successful_assessment_fraction"] >= 0.80
        assert result["raw_violation_receipts_preserved"] is True


def test_frozen_calibration_is_reproducible_from_declared_seed() -> None:
    generated_profile, generated_evidence = _calibration_module().build_calibration_bundle(
        20260802
    )
    frozen_profile = UncertaintyCalibrationProfile.model_validate_json(
        (
            ROOT / "src/astrograph/resources/gray_box_uncertainty_calibration.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert generated_profile.random_seed == frozen_profile.random_seed == 20260802
    assert generated_profile.estimator_version == frozen_profile.estimator_version
    # Profile-level reproducibility: every frozen archetype (thermal + dual_node;
    # storage is deliberately excluded under A2) regenerates byte-identical from
    # the declared seed.
    assert {a.component_archetype_id for a in generated_profile.archetypes} == {
        a.component_archetype_id for a in frozen_profile.archetypes
    }
    for generated, frozen in zip(
        generated_profile.archetypes, frozen_profile.archetypes, strict=True
    ):
        assert generated.component_archetype_id == frozen.component_archetype_id
        assert generated.calibration_split_sha256 == frozen.calibration_split_sha256
        assert generated.state_interval_multiplier == pytest.approx(
            frozen.state_interval_multiplier, rel=1e-10
        )
        assert generated.parameter_interval_multiplier == pytest.approx(
            frozen.parameter_interval_multiplier, rel=1e-10
        )
    assert generated_evidence["residual_whiteness_status"] == "PASS"
    # A2 (2026-08-04): storage is deliberately excluded from the frozen
    # calibration (its efficiency branch does not pass the noise-aware gate),
    # so T306 is INCOMPLETE by honest design, not a regression.  The old
    # gray_box_conservation_gls_validation.v1.json records the historical
    # storage single-slope T-306 evidence and is not regenerated here.
    assert generated_evidence["t306_status"] == "INCOMPLETE"
    assert frozen_profile.for_archetype("storage.energy_buffer") is None
    assert frozen_profile.for_archetype("thermal.dual_node") is not None
