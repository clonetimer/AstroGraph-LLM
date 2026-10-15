from __future__ import annotations

from astrograph.semantics import (
    DIMENSIONLESS,
    ENERGY,
    MOMENTUM,
    MOMENT_OF_INERTIA,
    THERMAL_CAPACITY,
    THERMAL_CONDUCTANCE,
    TIME,
)

from .contracts import (
    ConservationLaw,
    DynamicsContract,
    EventMode,
    InputSpec,
    ObservabilitySpec,
    ParameterRole,
    ParameterSpec,
    StateSpec,
)


def _legacy_storage_contract() -> DynamicsContract:
    """Legacy single-slope energy-buffer contract (v1, no efficiency terms).

    Retained for the frozen generic-diagnosis lineage (ADR-004 ①) which was
    calibrated against the pre-efficiency storage model.  The mainline storage
    v1.1 contract (with efficiency) is returned by ``default_dynamics_contracts``;
    this legacy variant is constructible for regression tests only and is NOT
    part of the active estimator dispatch surface.
    """
    return DynamicsContract(
        contract_id="astrograph.physics.storage.energy-buffer.v1",
        version="1.0.0",
        component_archetype_id="storage.energy_buffer",
        equation_family="bounded_integral_balance",
        states=[StateSpec(state_id="stored_fraction", quantity_kind_id="energy.state_of_charge", canonical_unit="1", lower_bound=0.0, upper_bound=1.0, observation_port_role_id="state.energy_fraction")],
        inputs=[InputSpec(input_id="net_power", port_role_id="power.exchange", quantity_kind_id="electrical.active_power", canonical_unit="W")],
        parameters=[ParameterSpec(parameter_id="energy_capacity", dimension=ENERGY, canonical_unit="J", role=ParameterRole.LEARNABLE, lower_bound=10.0, upper_bound=1.0e12, initial_value=1.0e6)],
        conservation_laws=[ConservationLaw(law_id="stored_energy_balance", state_id="stored_fraction", input_ids=["net_power"], residual_tolerance=1.0e-4, description="Change in stored fraction equals integrated signed power divided by capacity.")],
        event_modes=[
            EventMode(event_id="lower_saturation", guard_expression="stored_fraction <= 0", transition="clamp_to_lower_bound"),
            EventMode(event_id="upper_saturation", guard_expression="stored_fraction >= 1", transition="clamp_to_upper_bound"),
        ],
        observability=ObservabilitySpec(observed_state_ids=["stored_fraction"], minimum_samples=8, minimum_input_variance=1.0e-12),
        stability_statement="The bounded integral state remains in [0, 1] under explicit saturation events.",
    )


def _storage_contract() -> DynamicsContract:
    """Energy-buffer integral balance with explicit charge/discharge efficiency.

    The generator's battery model applies ``charge_efficiency`` (P >= 0) or
    ``1/discharge_efficiency`` (P < 0) to the integrated power, so the estimator
    must carry both terms to avoid the -3% net-pool bias (gap-1 closure).  The
    parameters are optional-declared on the contract; ``_estimate_integral_balance``
    enables the efficiency branch only when BOTH efficiency parameters exist, so
    legacy contracts without them keep the old single-slope behaviour (regression).
    """
    return DynamicsContract(
        contract_id="astrograph.physics.storage.energy-buffer.v1",
        version="1.1.0",
        component_archetype_id="storage.energy_buffer",
        equation_family="bounded_integral_balance",
        states=[StateSpec(state_id="stored_fraction", quantity_kind_id="energy.state_of_charge", canonical_unit="1", lower_bound=0.0, upper_bound=1.0, observation_port_role_id="state.energy_fraction")],
        inputs=[InputSpec(input_id="net_power", port_role_id="power.exchange", quantity_kind_id="electrical.active_power", canonical_unit="W")],
        parameters=[
            ParameterSpec(parameter_id="energy_capacity", dimension=ENERGY, canonical_unit="J", role=ParameterRole.LEARNABLE, lower_bound=10.0, upper_bound=1.0e12, initial_value=1.0e6),
            # charge_efficiency is a FIXED physical constant (coulombic charging
            # efficiency, near 1).  It is not identifiable from a single charge
            # slope jointly with capacity, so it stays FIXED and the estimator
            # uses its initial_value as the known constant.
            ParameterSpec(parameter_id="charge_efficiency", dimension=DIMENSIONLESS, canonical_unit="1", role=ParameterRole.FIXED, lower_bound=0.5, upper_bound=1.0, initial_value=0.98),
            # discharge_efficiency is the gap-1 target: LEARNABLE, recovered from
            # the discharge-slope-to-charge-slope ratio.
            ParameterSpec(parameter_id="discharge_efficiency", dimension=DIMENSIONLESS, canonical_unit="1", role=ParameterRole.LEARNABLE, lower_bound=0.5, upper_bound=1.0, initial_value=1.0),
        ],
        conservation_laws=[ConservationLaw(law_id="stored_energy_balance", state_id="stored_fraction", input_ids=["net_power"], residual_tolerance=1.0e-4, description="Change in stored fraction equals integrated signed power (charge/discharge-efficiency-adjusted) divided by capacity.")],
        event_modes=[
            EventMode(event_id="lower_saturation", guard_expression="stored_fraction <= 0", transition="clamp_to_lower_bound"),
            EventMode(event_id="upper_saturation", guard_expression="stored_fraction >= 1", transition="clamp_to_upper_bound"),
        ],
        observability=ObservabilitySpec(observed_state_ids=["stored_fraction"], minimum_samples=8, minimum_input_variance=1.0e-12),
        stability_statement="The bounded integral state remains in [0, 1] under explicit saturation events.",
    )


def _thermal_contract() -> DynamicsContract:
    return DynamicsContract(
        contract_id="astrograph.physics.thermal.lumped-capacity.v1",
        version="1.0.0",
        component_archetype_id="thermal.lumped_capacity",
        equation_family="first_order_thermal_balance",
        states=[StateSpec(state_id="temperature", quantity_kind_id="thermal.temperature", canonical_unit="K", lower_bound=1.0, upper_bound=5000.0, observation_port_role_id="state.temperature")],
        inputs=[
            InputSpec(input_id="net_heat", port_role_id="heat.exchange", quantity_kind_id="electrical.active_power", canonical_unit="W"),
            InputSpec(input_id="ambient_temperature", port_role_id="context.ambient_temperature", quantity_kind_id="thermal.temperature", canonical_unit="K"),
        ],
        parameters=[
            ParameterSpec(parameter_id="thermal_capacity", dimension=THERMAL_CAPACITY, canonical_unit="J/K", role=ParameterRole.LEARNABLE, lower_bound=1.0, upper_bound=1.0e12, initial_value=1.0e4),
            ParameterSpec(parameter_id="thermal_conductance", dimension=THERMAL_CONDUCTANCE, canonical_unit="W/K", role=ParameterRole.LEARNABLE, lower_bound=1.0e-9, upper_bound=1.0e9, initial_value=1.0),
        ],
        conservation_laws=[ConservationLaw(law_id="thermal_energy_balance", state_id="temperature", input_ids=["net_heat", "ambient_temperature"], residual_tolerance=1.0e-3, description="Thermal storage rate equals applied heat minus conductive loss to ambient.")],
        observability=ObservabilitySpec(observed_state_ids=["temperature"], minimum_samples=12, minimum_input_variance=1.0e-12),
        stability_statement="Positive thermal capacity and conductance produce an asymptotically stable first-order response for bounded inputs.",
    )


def _dual_node_thermal_contract() -> DynamicsContract:
    """Dual-node (bus + battery) thermal balance — gap-2 closure.

    Generator physics (subsystem_thermal_basic_lumped)::
        C_bus·dT_bus/dt   = P_internal + P_solar + P_heater
                            − P_radiator(T_bus) − G_cond·(T_bus − T_batt)
        C_batt·dT_batt/dt = P_batt_internal + G_cond·(T_bus − T_batt)

    ``P_radiator`` is the Stefan–Boltzmann nonlinearity; we linearise it around
    the observed state as an effective ``radiator_conductance`` so the whole
    family stays in the shared GLS machinery.  The bus and battery nodes use
    DISTINCT port roles (``state.temperature`` / ``state.temperature_secondary``)
    so the estimator can resolve each node unambiguously.
    """
    return DynamicsContract(
        contract_id="astrograph.physics.thermal.dual-node.v1",
        version="1.0.0",
        component_archetype_id="thermal.dual_node",
        equation_family="dual_node_thermal_balance",
        states=[
            StateSpec(state_id="bus_temperature", quantity_kind_id="thermal.temperature", canonical_unit="K", lower_bound=1.0, upper_bound=5000.0, observation_port_role_id="state.temperature"),
            StateSpec(state_id="battery_temperature", quantity_kind_id="thermal.temperature", canonical_unit="K", lower_bound=1.0, upper_bound=5000.0, observation_port_role_id="state.temperature_secondary"),
        ],
        inputs=[
            InputSpec(input_id="net_heat", port_role_id="heat.exchange", quantity_kind_id="electrical.active_power", canonical_unit="W"),
            InputSpec(input_id="ambient_temperature", port_role_id="context.ambient_temperature", quantity_kind_id="thermal.temperature", canonical_unit="K"),
        ],
        parameters=[
            ParameterSpec(parameter_id="thermal_capacity_bus", dimension=THERMAL_CAPACITY, canonical_unit="J/K", role=ParameterRole.LEARNABLE, lower_bound=1.0, upper_bound=1.0e12, initial_value=1.0e4),
            ParameterSpec(parameter_id="thermal_capacity_battery", dimension=THERMAL_CAPACITY, canonical_unit="J/K", role=ParameterRole.LEARNABLE, lower_bound=1.0, upper_bound=1.0e12, initial_value=1.0e4),
            ParameterSpec(parameter_id="thermal_conductance", dimension=THERMAL_CONDUCTANCE, canonical_unit="W/K", role=ParameterRole.LEARNABLE, lower_bound=1.0e-9, upper_bound=1.0e9, initial_value=1.0),
            ParameterSpec(parameter_id="radiator_conductance", dimension=THERMAL_CONDUCTANCE, canonical_unit="W/K", role=ParameterRole.LEARNABLE, lower_bound=1.0e-9, upper_bound=1.0e9, initial_value=1.0),
        ],
        conservation_laws=[
            ConservationLaw(law_id="bus_energy_balance", state_id="bus_temperature", input_ids=["net_heat", "ambient_temperature"], residual_tolerance=1.0e-3, description="Bus storage rate equals applied heat minus radiator and bus-battery conduction."),
            ConservationLaw(law_id="battery_energy_balance", state_id="battery_temperature", input_ids=["ambient_temperature"], residual_tolerance=1.0e-3, description="Battery storage rate equals bus-battery conduction."),
        ],
        observability=ObservabilitySpec(observed_state_ids=["bus_temperature", "battery_temperature"], minimum_samples=12, minimum_input_variance=1.0e-12),
        stability_statement="Positive thermal capacities and conductances produce a stable two-node response for bounded inputs.",
    )


def _actuator_contract() -> DynamicsContract:
    """First-order actuator lag: d(response)/dt = (gain·command − response)/τ.

    Linearised as d(response)/dt = (gain/τ)·command − (1/τ)·response, so the
    coefficients come from the same GLS machinery as the balance families.
    """
    return DynamicsContract(
        contract_id="astrograph.physics.actuator.controlled-effort.v1",
        version="1.0.0",
        component_archetype_id="actuator.controlled_effort",
        equation_family="first_order_actuator_lag",
        states=[StateSpec(state_id="actuator_response", quantity_kind_id="mechanical.torque", canonical_unit="N*m", lower_bound=-1.0e12, upper_bound=1.0e12, observation_port_role_id="response.effort")],
        inputs=[InputSpec(input_id="command_torque", port_role_id="command.effort", quantity_kind_id="mechanical.torque", canonical_unit="N*m")],
        parameters=[
            ParameterSpec(parameter_id="actuator_gain", dimension=DIMENSIONLESS, canonical_unit="1", role=ParameterRole.LEARNABLE, lower_bound=0.0, upper_bound=10.0, initial_value=1.0),
            ParameterSpec(parameter_id="actuator_time_constant", dimension=TIME, canonical_unit="s", role=ParameterRole.LEARNABLE, lower_bound=1.0e-3, upper_bound=1.0e3, initial_value=1.0),
        ],
        conservation_laws=[ConservationLaw(law_id="command_response_balance", state_id="actuator_response", input_ids=["command_torque"], residual_tolerance=1.0e-3, description="Response rate equals gain times command minus response, divided by the time constant.")],
        event_modes=[
            EventMode(event_id="response_saturation", guard_expression="abs(actuator_response) >= max_torque", transition="clamp_to_saturation"),
        ],
        observability=ObservabilitySpec(observed_state_ids=["actuator_response"], minimum_samples=12, minimum_input_variance=1.0e-12),
        stability_statement="Positive time constant produces an asymptotically stable first-order lag for bounded commands.",
    )


def _inertial_spin_contract() -> DynamicsContract:
    """Reaction-wheel speed-inertia balance: J·dω/dt = T − b·ω.

    A spinning rotor is not a controllable-effort actuator (command==response
    degenerates for its torque channel), so it gets its own archetype.  The
    equation linearises as dω/dt = (1/J)·T − (b/J)·ω, which lets the shared GLS
    machinery recover the moment of inertia and the viscous damping coefficient.
    """
    return DynamicsContract(
        contract_id="astrograph.physics.actuator.spinning-rotor.v1",
        version="1.0.0",
        component_archetype_id="actuator.spinning_rotor",
        equation_family="inertial_spin_balance",
        states=[StateSpec(state_id="angular_rate", quantity_kind_id="mechanical.angular_rate", canonical_unit="rad/s", lower_bound=-1.0e5, upper_bound=1.0e5, observation_port_role_id="state.rate")],
        inputs=[InputSpec(input_id="command_torque", port_role_id="command.effort", quantity_kind_id="mechanical.torque", canonical_unit="N*m")],
        parameters=[
            ParameterSpec(parameter_id="moment_of_inertia", dimension=MOMENT_OF_INERTIA, canonical_unit="kg*m^2", role=ParameterRole.LEARNABLE, lower_bound=1.0e-6, upper_bound=1.0e3, initial_value=0.1),
            ParameterSpec(parameter_id="damping_coefficient", dimension=MOMENTUM, canonical_unit="N*m*s", role=ParameterRole.LEARNABLE, lower_bound=0.0, upper_bound=1.0e-3, initial_value=1.0e-5),
        ],
        conservation_laws=[ConservationLaw(law_id="angular_momentum_balance", state_id="angular_rate", input_ids=["command_torque"], residual_tolerance=1.0e-3, description="Wheel spin-rate change equals (applied torque minus viscous damping) divided by the moment of inertia.")],
        event_modes=[
            EventMode(event_id="speed_saturation", guard_expression="abs(angular_rate) >= max_speed", transition="clamp_to_saturation"),
        ],
        observability=ObservabilitySpec(observed_state_ids=["angular_rate"], minimum_samples=12, minimum_input_variance=1.0e-12),
        stability_statement="Positive moment of inertia and non-negative damping produce an asymptotically stable first-order spin response for bounded torque commands.",
    )


def default_dynamics_contracts() -> dict[str, DynamicsContract]:
    contracts = [
        _storage_contract(),
        _thermal_contract(),
        _dual_node_thermal_contract(),
        _actuator_contract(),
        _inertial_spin_contract(),
    ]
    return {item.component_archetype_id: item for item in contracts}


__all__ = ["default_dynamics_contracts", "legacy_storage_contract"]


def legacy_storage_contract() -> DynamicsContract:
    """Legacy single-slope storage contract (v1.0.0) for frozen regression."""
    return _legacy_storage_contract()
