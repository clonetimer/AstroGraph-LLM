"""2.C state-estimation completeness audit (2026-08-04).

Locks in (a) the five governed GLS-identifiable dynamics contract families and
(b) the gap-closure behaviour.  Gaps 1 (storage charge/discharge efficiency)
and 2 (dual-node thermal) are CLOSED in the estimator: each is verified on
synthetic generator-matching data to sub-percent parameter recovery.  Gap 3
(calibration coverage for the loop-13 contracts) stays a documented exemption.

The closures were implemented as additive contract families (storage v2
efficiency params, thermal.dual_node archetype) so legacy contracts keep their
behaviour and the frozen-split feature surface changes only when the mainline
switches to the new families.
"""

from __future__ import annotations

import inspect

import pytest

from astrograph.data_driven.diagnosis_features import PHYSICAL_NOMINAL, _gray_box_estimate
from astrograph.physics.archetypes import default_dynamics_contracts

EXPECTED_FAMILIES = {
    "storage.energy_buffer": "bounded_integral_balance",
    "thermal.lumped_capacity": "first_order_thermal_balance",
    "thermal.dual_node": "dual_node_thermal_balance",
    "actuator.controlled_effort": "first_order_actuator_lag",
    "actuator.spinning_rotor": "inertial_spin_balance",
}


def test_five_governed_contract_families_exist() -> None:
    contracts = default_dynamics_contracts()
    assert set(contracts) == set(EXPECTED_FAMILIES)
    for archetype, family in EXPECTED_FAMILIES.items():
        assert contracts[archetype].equation_family == family
        assert contracts[archetype].observability.minimum_samples >= 8
        assert contracts[archetype].conservation_laws, f"{archetype} has no conservation law"


def test_gap1_storage_efficiency_is_identifiable_on_charge_discharge() -> None:
    """Gap-1 closure: the storage v2 contract carries charge (FIXED) and
    discharge (LEARNABLE) efficiency; the estimator recovers C and
    discharge_eff from charge+discharge excitation.  Verified on synthetic
    generator-matching data to sub-percent error."""
    import numpy as np
    from astrograph.physics.archetypes import default_dynamics_contracts
    from astrograph.state_estimation.gray_box import GrayBoxStateEstimator
    from astrograph.semantics import SemanticTelemetryFrame, SemanticSignal, SamplingProfile

    contract = default_dynamics_contracts()["storage.energy_buffer"]
    assert "discharge_efficiency" in [p.parameter_id for p in contract.parameters]
    assert "charge_efficiency" in [p.parameter_id for p in contract.parameters]

    c_j = 160.0 * 3600.0
    cap = 1.0 / c_j
    charge_eff, dis_eff = 0.98, 0.97
    n = 6000
    t = np.arange(n, dtype=float)
    soc = np.empty(n)
    soc[0] = 0.62
    power = np.empty(n)
    for i in range(n):
        power[i] = 35.0 if (i // 600) % 2 == 0 else -35.0
    for i in range(n - 1):
        p = power[i]
        soc[i + 1] = np.clip(
            soc[i] + (p * charge_eff * cap if p >= 0 else p / dis_eff * cap), 0.0, 1.0
        )
    signals = [
        SemanticSignal(semantic_channel_id="a1", asset_instance_id="x",
            component_archetype_id="storage.energy_buffer", port_role_id="state.energy_fraction",
            quantity_kind_id="energy.state_of_charge", canonical_unit="1", direction=1,
            values=soc.tolist(), missing_mask=[False] * n),
        SemanticSignal(semantic_channel_id="a2", asset_instance_id="x",
            component_archetype_id="storage.energy_buffer", port_role_id="power.exchange",
            quantity_kind_id="electrical.active_power", canonical_unit="W", direction=1,
            values=power.tolist(), missing_mask=[False] * n),
    ]
    frame = SemanticTelemetryFrame(frame_id="t", semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=t.tolist(), signals=signals,
        sampling=SamplingProfile(row_count=n, nominal_period_s=1.0, irregular=False, missing_value_count=0),
        semantic_data_sha256="0" * 64, labels_used=False, truth_used=False, claim_boundary="t")
    result = GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract)
    params = {p.parameter_id: p.value for p in result.parameters}
    assert abs(params["energy_capacity"] / c_j - 1.0) < 0.01
    assert abs(params["discharge_efficiency"] / dis_eff - 1.0) < 0.01


def test_gap2_dual_node_thermal_is_identifiable() -> None:
    """Gap-2 closure: the dual_node thermal archetype recovers bus/battery
    capacities, bus-battery conductance and radiator conductance from the
    generator's two-node physics, to sub-percent error (was -11.2% under the
    single-node model)."""
    import numpy as np
    from astrograph.physics.archetypes import default_dynamics_contracts
    from astrograph.state_estimation.gray_box import GrayBoxStateEstimator
    from astrograph.semantics import SemanticTelemetryFrame, SemanticSignal, SamplingProfile

    contract = default_dynamics_contracts()["thermal.dual_node"]
    c_bus, c_batt = 18000.0, 9000.0
    g_rad, g_cond = 0.9, 0.45
    sink = -35.0 + 273.15
    n = 3000
    t = np.arange(n, dtype=float)
    power = np.empty(n)
    for i in range(n):
        power[i] = 40.0 if (i // 300) % 2 == 0 else 20.0
    t_bus = np.empty(n)
    t_batt = np.empty(n)
    t_bus[0] = 293.15
    t_batt[0] = 291.15
    for i in range(n - 1):
        t_bus[i + 1] = t_bus[i] + (power[i] - g_rad * (t_bus[i] - sink) - g_cond * (t_bus[i] - t_batt[i])) / c_bus
        t_batt[i + 1] = t_batt[i] + (g_cond * (t_bus[i] - t_batt[i])) / c_batt
    signals = [
        SemanticSignal(semantic_channel_id="b1", asset_instance_id="x",
            component_archetype_id="thermal.dual_node", port_role_id="state.temperature",
            quantity_kind_id="thermal.temperature", canonical_unit="K", direction=1,
            values=t_bus.tolist(), missing_mask=[False] * n),
        SemanticSignal(semantic_channel_id="b2", asset_instance_id="x",
            component_archetype_id="thermal.dual_node", port_role_id="state.temperature_secondary",
            quantity_kind_id="thermal.temperature", canonical_unit="K", direction=1,
            values=t_batt.tolist(), missing_mask=[False] * n),
        SemanticSignal(semantic_channel_id="h", asset_instance_id="x",
            component_archetype_id="thermal.dual_node", port_role_id="heat.exchange",
            quantity_kind_id="electrical.active_power", canonical_unit="W", direction=1,
            values=power.tolist(), missing_mask=[False] * n),
        SemanticSignal(semantic_channel_id="a", asset_instance_id="x",
            component_archetype_id="thermal.dual_node", port_role_id="context.ambient_temperature",
            quantity_kind_id="thermal.temperature", canonical_unit="K", direction=1,
            values=[sink] * n, missing_mask=[False] * n),
    ]
    frame = SemanticTelemetryFrame(frame_id="t", semantic_schema_id="astrograph.core.physical-quantities.v1",
        timestamps_s=t.tolist(), signals=signals,
        sampling=SamplingProfile(row_count=n, nominal_period_s=1.0, irregular=False, missing_value_count=0),
        semantic_data_sha256="0" * 64, labels_used=False, truth_used=False, claim_boundary="t")
    result = GrayBoxStateEstimator(use_frozen_calibration=False).estimate(frame, contract)
    params = {p.parameter_id: p.value for p in result.parameters}
    assert abs(params["thermal_capacity_bus"] / c_bus - 1.0) < 0.01
    assert abs(params["thermal_capacity_battery"] / c_batt - 1.0) < 0.01
    assert abs(params["thermal_conductance"] / g_cond - 1.0) < 0.01
    assert abs(params["radiator_conductance"] / g_rad - 1.0) < 0.01


def test_gap3_calibration_exemption_is_documented() -> None:
    """The estimator runs use_frozen_calibration=False as a deliberate S0-era
    exemption; the frozen calibration profile carries conservation scores for
    ONLY storage.energy_buffer / thermal.lumped_capacity, so the loop-13
    contracts (spinning_rotor/controlled_effort/queue/converter) have NO
    per-archetype calibration data and fall back to UNcalibrated even under
    use_frozen_calibration=True (the estimator identity DOES match, so no
    identity error is raised).  The exemption must stay documented as a
    pre-production decision point, not silently flipped."""
    source = inspect.getsource(_gray_box_estimate)
    assert "use_frozen_calibration=False" in source
    assert "calibration research exemption" in source
    # precise root cause: per-archetype coverage gap, not an identity mismatch.
    assert "per-archetype" in source
    assert "calibration coverage" in source
    assert "identity" in source
    assert "production" in source.lower()
