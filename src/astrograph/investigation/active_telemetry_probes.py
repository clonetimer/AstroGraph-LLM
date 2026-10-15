"""Opt-in, shadow-only inspection of revalidated runtime telemetry."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import pandas as pd

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.evidence.configured_runtime import ConfiguredChannelMapEntry
from astrograph.investigation.diagnostic_probes import (
    DiagnosticProbeCatalog,
    DiagnosticProbeExecutor,
    DiagnosticProbeKind,
    DiagnosticProbeResult,
    DiagnosticProbeSpec,
)
from astrograph.semantics import is_prohibited_field


_ADMISSION_KIND = "ADMITTED_RUNTIME_TELEMETRY_V1"
_PAIRED_BASIS = "PAIRED_NOMINAL_RESIDUAL_V1"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def is_active_telemetry_probe(spec: DiagnosticProbeSpec) -> bool:
    return (
        spec.bound_inputs.get("admission_kind") == _ADMISSION_KIND
        or spec.probe_id.startswith("PROBE-TELEMETRY-")
    )


def is_admitted_telemetry_probe(spec: DiagnosticProbeSpec, catalog: DiagnosticProbeCatalog) -> bool:
    if not is_active_telemetry_probe(spec):
        return False
    inputs = spec.bound_inputs
    provenance = {key: inputs.get(key) for key in ("before", "after", "admission_kind")}
    paired = inputs.get("measurement_basis") == _PAIRED_BASIS
    expected_keys = {
        "before", "after", "admission_kind", "semantic_channel_id",
        "source_sha256", "mapping_sha256",
    }
    if paired:
        expected_keys.update({"measurement_basis", "nominal_sha256"})
    nominal_sha = inputs.get("nominal_sha256")
    digest_payload = {
        "episode_id": spec.episode_id,
        "channel": inputs.get("semantic_channel_id"),
        "source_sha256": inputs.get("source_sha256"),
        "mapping_sha256": inputs.get("mapping_sha256"),
        "bound_inputs": provenance,
        "unit": spec.unit,
        "target_fault_ids": list(spec.target_fault_ids),
    }
    if paired:
        digest_payload.update({"measurement_basis": _PAIRED_BASIS, "nominal_sha256": nominal_sha})
    return (
        catalog.telemetry_admission_status == "ADMITTED_RUNTIME_TELEMETRY"
        and set(inputs) == expected_keys
        and inputs.get("admission_kind") == _ADMISSION_KIND
        and spec.kind == DiagnosticProbeKind.PRE_POST_CHANGE
        and spec.episode_id == catalog.episode_id
        and spec.probe_id == f"PROBE-TELEMETRY-{spec.source_artifact_sha256[:12].upper()}"
        and isinstance(inputs.get("source_sha256"), str)
        and isinstance(inputs.get("mapping_sha256"), str)
        and isinstance(inputs.get("semantic_channel_id"), str)
        and all(
            isinstance(inputs.get(key), list)
            and 2 <= len(inputs[key]) <= 32
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in inputs[key])
            for key in ("before", "after")
        )
        and all(
            len(inputs[key]) == 64 and all(char in "0123456789abcdef" for char in inputs[key])
            for key in ("source_sha256", "mapping_sha256")
        )
        and (not paired or (
            isinstance(nominal_sha, str) and len(nominal_sha) == 64
            and all(char in "0123456789abcdef" for char in nominal_sha)
        ))
        and spec.source_artifact_sha256 == _digest(digest_payload)
    )


class ActiveTelemetryProbeExecutor(DiagnosticProbeExecutor):
    """Require admission before labeling an observation as newly inspected data."""

    def execute(self, *, catalog: DiagnosticProbeCatalog, probe_id: str) -> DiagnosticProbeResult:
        spec = catalog.get(probe_id)
        if not is_active_telemetry_probe(spec):
            return super().execute(catalog=catalog, probe_id=probe_id)
        if not is_admitted_telemetry_probe(spec, catalog):
            raise ValueError("raw telemetry probe requires an admitted runtime telemetry catalog")
        result = super().execute(catalog=catalog, probe_id=probe_id)
        result_hash = _digest({
            "base_result_sha256": result.result_sha256,
            "creates_new_raw_observation": True,
            "algorithm_version": "active-telemetry-probe-executor.v1",
        })
        return result.model_copy(update={
            "observation_id": f"PROBEOBS-{result_hash[:16].upper()}",
            "result_sha256": result_hash,
            "creates_new_raw_observation": True,
            "claim_boundary": (
                (
                    "Newly inspected current-minus-paired-normal time-local telemetry residual; "
                    "both source hashes are bound to the admitted probe. "
                    if spec.bound_inputs.get("measurement_basis") == _PAIRED_BASIS else
                    "Newly inspected, host-admitted runtime telemetry. "
                )
                + "Not a newly acquired sensor sample. "
                "Shadow observation only; it cannot alter formal evidence, release, or control authorization."
            ),
        })


def build_admitted_telemetry_change_probe_catalog(
    *, bundle: EpisodeEvidenceBundle, mapped_frame: pd.DataFrame,
    approved_channels: list[ConfiguredChannelMapEntry], source_sha256: str,
    mapping_sha256: str, max_probes: int = 8, samples_per_side: int = 8,
    priority_fault_ids: tuple[str, ...] = (),
    paired_nominal_mapped_frame: pd.DataFrame | None = None,
    nominal_sha256: str | None = None,
) -> DiagnosticProbeCatalog:
    """Bind time-local samples, optionally as a paired-normal residual, after revalidation."""
    if not all(
        isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
        for value in (source_sha256, mapping_sha256)
    ):
        raise ValueError("telemetry source and mapping hashes must be SHA-256 digests")
    if not 1 <= max_probes <= 16 or not 2 <= samples_per_side <= 32:
        raise ValueError("telemetry probe budgets are outside the bounded range")
    paired = paired_nominal_mapped_frame is not None
    if paired != (nominal_sha256 is not None):
        raise ValueError("paired nominal frame and source hash must be supplied together")
    if paired and (
        not isinstance(nominal_sha256, str) or len(nominal_sha256) != 64
        or any(char not in "0123456789abcdef" for char in nominal_sha256)
    ):
        raise ValueError("paired nominal source hash must be a SHA-256 digest")
    if "time_s" not in mapped_frame or mapped_frame.empty:
        raise ValueError("admitted telemetry frame requires a time axis and samples")
    time = pd.to_numeric(mapped_frame["time_s"], errors="coerce")
    if (
        time.isna().any() or not all(math.isfinite(float(value)) for value in time)
        or not time.is_monotonic_increasing or not time.is_unique
    ):
        raise ValueError("admitted telemetry time axis must be finite and strictly increasing")
    selected = mapped_frame.loc[time.between(bundle.episode.start_s, bundle.episode.end_s, inclusive="both")]
    nominal_selected = None
    if paired:
        assert paired_nominal_mapped_frame is not None
        if "time_s" not in paired_nominal_mapped_frame:
            raise ValueError("paired nominal telemetry requires a time axis")
        nominal_time = pd.to_numeric(paired_nominal_mapped_frame["time_s"], errors="coerce")
        if (
            nominal_time.isna().any()
            or not all(math.isfinite(float(value)) for value in nominal_time)
            or not nominal_time.is_monotonic_increasing or not nominal_time.is_unique
        ):
            raise ValueError("paired nominal time axis must be finite and strictly increasing")
        nominal_selected = paired_nominal_mapped_frame.loc[
            nominal_time.between(bundle.episode.start_s, bundle.episode.end_s, inclusive="both")
        ]
        if not time.loc[selected.index].reset_index(drop=True).equals(
            nominal_time.loc[nominal_selected.index].reset_index(drop=True)
        ):
            raise ValueError("paired telemetry time axes must align exactly")
    if len(selected) < 2 * samples_per_side:
        return DiagnosticProbeCatalog(
            episode_id=bundle.episode.episode_id, probes=(),
            telemetry_admission_status="ADMITTED_RUNTIME_TELEMETRY",
        )
    feature_channels = {
        item.feature_definition_id: item.source_field_template for item in bundle.feature_definitions
    }
    target_faults_by_channel: dict[str, set[str]] = {}
    for definition in bundle.evidence_definitions:
        channel_id = feature_channels.get(definition.feature_definition_id)
        if channel_id is not None:
            target_faults_by_channel.setdefault(channel_id, set()).add(definition.target_fault_id)
    registered_fault_ids = {item.target_fault_id for item in bundle.evidence_definitions}
    priority = set(priority_fault_ids) & registered_fault_ids

    def channel_order(channel: ConfiguredChannelMapEntry) -> tuple[int, int, str]:
        targets = target_faults_by_channel.get(channel.semantic_channel_id, set())
        overlap = targets & priority
        if not priority:
            tier = 0
        elif overlap and overlap != priority:
            tier = 0  # Separates at least one, but not all, current candidates.
        elif overlap:
            tier = 1
        else:
            tier = 2
        return tier, -len(overlap), channel.semantic_channel_id

    probes: list[DiagnosticProbeSpec] = []
    for channel in sorted(approved_channels, key=channel_order):
        if len(probes) >= max_probes:
            break
        channel_id = channel.semantic_channel_id
        if (
            is_prohibited_field(channel_id) or is_prohibited_field(channel.source_field)
            or channel_id == "time_s" or channel_id not in selected
        ):
            continue
        values = pd.to_numeric(selected[channel_id], errors="coerce")
        if paired:
            assert nominal_selected is not None
            if channel_id not in nominal_selected:
                continue
            nominal_values = pd.to_numeric(nominal_selected[channel_id], errors="coerce")
            values = values.reset_index(drop=True) - nominal_values.reset_index(drop=True)
        before = values.iloc[:samples_per_side]
        after = values.iloc[-samples_per_side:]
        if before.isna().any() or after.isna().any():
            continue
        before_values = [float(value) for value in before]
        after_values = [float(value) for value in after]
        if not all(math.isfinite(value) for value in before_values + after_values):
            continue
        bound_inputs = {
            "before": before_values, "after": after_values,
            "admission_kind": _ADMISSION_KIND, "semantic_channel_id": channel_id,
            "source_sha256": source_sha256, "mapping_sha256": mapping_sha256,
        }
        if paired:
            bound_inputs.update({
                "measurement_basis": _PAIRED_BASIS,
                "nominal_sha256": nominal_sha256,
            })
        provenance = {key: bound_inputs[key] for key in ("before", "after", "admission_kind")}
        target_fault_ids = tuple(sorted(target_faults_by_channel.get(channel_id, set())))
        source_hash = _digest({
            "episode_id": bundle.episode.episode_id, "channel": channel_id,
            "source_sha256": source_sha256, "mapping_sha256": mapping_sha256,
            "bound_inputs": provenance,
            "unit": channel.canonical_unit,
            "target_fault_ids": list(target_fault_ids),
            **({"measurement_basis": _PAIRED_BASIS, "nominal_sha256": nominal_sha256} if paired else {}),
        })
        probes.append(DiagnosticProbeSpec(
            probe_id=f"PROBE-TELEMETRY-{source_hash[:12].upper()}",
            episode_id=bundle.episode.episode_id, kind=DiagnosticProbeKind.PRE_POST_CHANGE,
            title=(
                f"Inspect paired-normal residual change in {channel_id}" if paired
                else f"Inspect temporal change in {channel_id}"
            ),
            objective=(
                "Compare fixed early and late current-minus-paired-normal telemetry residuals; "
                "do not change formal evidence."
                if paired else
                "Compare fixed early and late admitted telemetry samples without changing formal evidence."
            ),
            target_fault_ids=target_fault_ids,
            unit=channel.canonical_unit, bound_inputs=bound_inputs,
            source_artifact_sha256=source_hash,
        ))
    return DiagnosticProbeCatalog(
        episode_id=bundle.episode.episode_id, probes=tuple(probes),
        telemetry_admission_status="ADMITTED_RUNTIME_TELEMETRY",
        claim_boundary=(
            "Host-admitted telemetry samples are newly inspected by the LLM investigation, not newly acquired "
            "from a sensor. Probe results remain shadow observations and cannot modify formal evidence or release."
        ),
    )
