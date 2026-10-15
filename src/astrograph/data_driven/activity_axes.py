from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .campaign import IndependentRunCampaignSpec


class ActivityAxis(str, Enum):
    SEED = "seed"
    SEVERITY = "severity"
    INITIAL_STATE = "initial_state"
    OPERATING_CONDITION = "operating_condition"
    FAULT_TIMING = "fault_timing"
    NOISE = "noise"
    DROPOUT = "dropout"
    SATELLITE_INSTANCE = "satellite_instance"
    FLIGHT_DATA = "flight_data"


class ActivityAxisGateStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"


class RuntimeActivityAxisCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis: ActivityAxis
    controllable: bool = False
    recorded_in_run_manifest: bool = False
    capability_domain: Literal["simulation_runtime", "dataset_intake"] = "simulation_runtime"
    accepted_cli_args: list[str] = Field(default_factory=list)
    evidence: str = Field(min_length=1)
    claim_boundary: str = "Controllable means the registered runtime accepts the axis before execution."

    @field_validator("accepted_cli_args")
    @classmethod
    def unique_cli_args(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("accepted_cli_args must be unique")
        return value


class RuntimeActivityAxisCapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-runtime-activity-axis-capability-manifest.v1"
    manifest_id: str = Field(min_length=1)
    runtime_id: str = Field(min_length=1)
    runtime_script_name: str = Field(min_length=1)
    capabilities: list[RuntimeActivityAxisCapability]
    claim_boundary: str = (
        "The manifest describes which activity axes are currently controllable by the registered simulator "
        "interface. Recorded metadata is not evidence that an axis can be varied."
    )

    @field_validator("capabilities")
    @classmethod
    def unique_axes(cls, value: list[RuntimeActivityAxisCapability]) -> list[RuntimeActivityAxisCapability]:
        axes = [item.axis for item in value]
        if len(axes) != len(set(axes)):
            raise ValueError("activity axis capabilities must be unique by axis")
        return value

    @property
    def supported_axes(self) -> list[ActivityAxis]:
        return sorted((item.axis for item in self.capabilities if item.controllable), key=lambda item: item.value)

    @property
    def unsupported_axes(self) -> list[ActivityAxis]:
        supported = set(self.supported_axes)
        return sorted((axis for axis in ActivityAxis if axis not in supported), key=lambda item: item.value)


class ActivityAxisGateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-activity-axis-gate-decision.v1"
    status: ActivityAxisGateStatus
    requested_axes: list[ActivityAxis]
    accepted_axes: list[ActivityAxis]
    blocked_axes: list[ActivityAxis]
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    manifest_id: str
    runtime_id: str
    claim_boundary: str = (
        "ACCEPTED only means the requested activity axes are declared controllable by the registered runtime "
        "interface. It does not prove telemetry content changes until execution artifacts are hashed."
    )


class ActivityAxisGate:
    def decide(
        self,
        manifest: RuntimeActivityAxisCapabilityManifest,
        *,
        requested_axes: list[ActivityAxis],
    ) -> ActivityAxisGateDecision:
        requested = sorted(set(requested_axes), key=lambda item: item.value)
        supported = set(manifest.supported_axes)
        blocked = sorted((axis for axis in requested if axis not in supported), key=lambda item: item.value)
        accepted = sorted((axis for axis in requested if axis in supported), key=lambda item: item.value)
        blocking = [f"UNSUPPORTED_ACTIVITY_AXIS:{axis.value}" for axis in blocked]
        warnings: list[str] = []
        if not requested:
            warnings.append("NO_ACTIVITY_AXIS_REQUESTED")
        return ActivityAxisGateDecision(
            status=ActivityAxisGateStatus.BLOCKED if blocking else ActivityAxisGateStatus.ACCEPTED,
            requested_axes=requested,
            accepted_axes=accepted,
            blocked_axes=blocked,
            blocking_reasons=blocking,
            warnings=warnings,
            manifest_id=manifest.manifest_id,
            runtime_id=manifest.runtime_id,
        )


def requested_axes_from_campaign_spec(spec: IndependentRunCampaignSpec) -> list[ActivityAxis]:
    axes: set[ActivityAxis] = set()
    if len(spec.seeds) > 1:
        axes.add(ActivityAxis.SEED)
    initial_states = {item.initial_state_id for item in spec.variants if item.initial_state_id is not None}
    operating_conditions = {
        item.operating_condition_id for item in spec.variants if item.operating_condition_id is not None
    }
    severities = {item.fault_severity_id for item in spec.variants if item.fault_severity_id is not None}
    satellites = {item.satellite_instance_id for item in spec.variants if item.satellite_instance_id is not None}
    if len(initial_states) > 1:
        axes.add(ActivityAxis.INITIAL_STATE)
    if len(operating_conditions) > 1:
        axes.add(ActivityAxis.OPERATING_CONDITION)
    if len(severities) > 1:
        axes.add(ActivityAxis.SEVERITY)
    if len(satellites) > 1:
        axes.add(ActivityAxis.SATELLITE_INSTANCE)
    cli_tokens = [token for variant in spec.variants for token in variant.cli_args]
    cli_text = " ".join(cli_tokens).lower()
    if (
        "--fault-start-s" in cli_text
        or "--fault-duration-s" in cli_text
        or "fault_start" in cli_text
        or "fault_duration" in cli_text
    ):
        axes.add(ActivityAxis.FAULT_TIMING)
    if "noise" in cli_text:
        axes.add(ActivityAxis.NOISE)
    if "dropout" in cli_text:
        axes.add(ActivityAxis.DROPOUT)
    if "dataset-root" in cli_text or "flight-data" in cli_text or "telemetry.csv" in cli_text:
        axes.add(ActivityAxis.FLIGHT_DATA)
    return sorted(axes, key=lambda item: item.value)


def decide_campaign_activity_axes(
    spec: IndependentRunCampaignSpec,
    manifest: RuntimeActivityAxisCapabilityManifest,
) -> ActivityAxisGateDecision:
    return ActivityAxisGate().decide(manifest, requested_axes=requested_axes_from_campaign_spec(spec))


def default_v0522_activity_axis_manifest() -> RuntimeActivityAxisCapabilityManifest:
    return RuntimeActivityAxisCapabilityManifest(
        manifest_id="RUNTIME-ACTIVITY-AXES-V0529-SEED-TIMING-ALL-FAULT-SEVERITY-NOISE-DROPOUT-OPERATING-INITIAL-SATELLITE-FLIGHT-DATA",
        runtime_id="phm-whole-spacecraft-minimal-runtime-v1.1",
        runtime_script_name="run_whole_spacecraft_all_faults.py",
        capabilities=[
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.SEED,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--seed"],
                evidence="Existing campaign runner passes --seed and run_manifest records seed.",
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.FAULT_TIMING,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--fault-start-s", "--fault-duration-s"],
                evidence="Top-level runtime accepts --fault-start-s and --fault-duration-s and records activity_axes.fault_timing plus event start_s/duration_s.",
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.SEVERITY,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--fault-severity"],
                evidence=(
                    "Top-level runtime accepts --fault-severity for mapped fault effects and records "
                    "activity_axes.severity plus applied event/runtime parameter overrides. V0.5.28 maps every "
                    "registered all_fault_scenario_catalog single scenario and composite scenario to "
                    "low/high severity telemetry-changing controls."
                ),
                claim_boundary=(
                    "Severity controllability is validated for the registered synthetic catalog only; it is not "
                    "flight-calibrated severity or diagnostic correctness."
                ),
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.INITIAL_STATE,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--initial-state"],
                evidence=(
                    "Top-level runtime accepts --initial-state, applies registered initial SOC/temperature/"
                    "pointing-error parameter overrides before execution, and records activity_axes.initial_state."
                ),
                claim_boundary=(
                    "Initial-state controllability is limited to registered low_soc, nominal_state, and high_soc "
                    "parameter maps; it is not an arbitrary state-vector loader."
                ),
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.OPERATING_CONDITION,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--operating-condition"],
                evidence=(
                    "Top-level runtime accepts --operating-condition, applies registered load/window parameter "
                    "overrides before execution, and records activity_axes.operating_condition."
                ),
                claim_boundary=(
                    "Operating-condition controllability is limited to registered low_load, nominal_load, and "
                    "high_load parameter maps; it is not a general environment generator."
                ),
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.NOISE,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--sensor-noise-scale"],
                evidence=(
                    "Top-level runtime accepts --sensor-noise-scale, passes it into run_case as "
                    "sensor_noise_scale, and records activity_axes.noise in run_manifest."
                ),
                claim_boundary=(
                    "Noise controllability means all registered VirtualSensorSuite channel noise_std values are "
                    "scaled before observation; it is not packet dropout or missing-data injection."
                ),
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.DROPOUT,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--sensor-dropout-rate"],
                evidence=(
                    "Top-level runtime accepts --sensor-dropout-rate, applies deterministic missing-cell "
                    "injection to sensor.* telemetry fields after observation, and records activity_axes.dropout."
                ),
                claim_boundary=(
                    "Dropout controllability means sensor telemetry cells are blanked before CSV export; truth, "
                    "status, counter, event, and label fields are not dropped."
                ),
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.SATELLITE_INSTANCE,
                controllable=True,
                recorded_in_run_manifest=True,
                accepted_cli_args=["--satellite-instance"],
                evidence=(
                    "Top-level runtime accepts --satellite-instance, applies registered platform parameter "
                    "overrides before execution, and records activity_axes.satellite_instance."
                ),
                claim_boundary=(
                    "Satellite-instance controllability is limited to registered small_bus, baseline_bus, and "
                    "large_bus parameter maps; it is not multi-spacecraft formation simulation."
                ),
            ),
            RuntimeActivityAxisCapability(
                axis=ActivityAxis.FLIGHT_DATA,
                controllable=True,
                recorded_in_run_manifest=True,
                capability_domain="dataset_intake",
                accepted_cli_args=["--dataset-root", "--dataset-id"],
                evidence=(
                    "Dataset onboarding accepts --dataset-root/--dataset-id, reads telemetry.csv, records file "
                    "hash/schema/label/axis coverage, and fail-closes unknown or unmappable datasets before "
                    "diagnosis release."
                ),
                claim_boundary=(
                    "Flight-data controllability means deterministic dataset intake and capability gating. It "
                    "does not mean the synthetic simulator can replay arbitrary flight data, and it does not "
                    "permit released diagnosis without evidence-policy approval."
                ),
            ),
        ],
        claim_boundary=(
            "The manifest describes which activity axes are currently controllable by the registered simulator "
            "or deterministic dataset-intake interface. Recorded metadata is not evidence that an axis improves "
            "diagnostic correctness."
        ),
    )


def manifest_sha256(manifest: RuntimeActivityAxisCapabilityManifest) -> str:
    return hashlib.sha256(
        manifest.model_dump_json(exclude_none=False).encode("utf-8")
    ).hexdigest()


def load_activity_axis_manifest(path: str | Path) -> RuntimeActivityAxisCapabilityManifest:
    return RuntimeActivityAxisCapabilityManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_activity_axis_manifest(manifest: RuntimeActivityAxisCapabilityManifest, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
