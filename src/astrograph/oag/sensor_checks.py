"""On-demand, same-window sensor integrity checks for a paired ADCS adapter.

Only read recorded measurements; no fault labels, predictions, path names or
future history are inputs. Generic integrity/interval residual operations are
kept separate from the versioned research evidence rules that interpret them.
Missing or inapplicable observations stay missing (never replaced by zero).
"""
from __future__ import annotations
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd

from astrograph.evidence.executable import FeatureResult, stable_id
from .contracts import content_hash
from .evidence_tools import EvidenceCheckTool

PACKAGE_ID = "adcs-v522-on-demand-sensor-evidence"
VERSION = "paired-sensor-integrity.v1.1"
# Numeric zero test only; NOT a fault probability or a flight health threshold.
ZERO_ATOL = 1e-12
METRICS = {
    "gyro_zero_axes": "1",
    "gyro_reference_excited_axes": "1",
    "imu_all_output_zero": "1",
    "star_output_frozen": "1",
    "star_quality_valid": "1",
    "gyro_aligned_bias_norm": "rad/s",
    "gyro_aligned_spread_ratio": "1",
    "actuator_response_consistent": "1",
}


def feature_id(key: str) -> str:
    return "FD-SENSOR-CHECK-" + key.upper().replace("_", "-") + "-V1"


def statistic_fields() -> list[str]:
    return ["derived.adcs.sensor_check." + k for k in METRICS]


def deferred_sensor_fields(source: pd.DataFrame, nominal: pd.DataFrame):
    """Declare unavailable output columns without executing or fabricating checks."""
    out, ref = source.copy(), nominal.copy()
    for field in statistic_fields():
        out[field] = np.nan
        ref[field] = np.nan
    return out, ref


def channel_zero_count(values: np.ndarray, *, atol: float = ZERO_ATOL) -> int:
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[0] < 8 or not np.isfinite(a).all() or not 0 <= atol < float('inf'):
        raise ValueError("eight finite vector observations and a finite nonnegative tolerance required")
    return int(np.count_nonzero(np.max(np.abs(a), axis=0) <= atol))


def quaternion_spread(values: np.ndarray) -> float:
    """Maximum chord distance from the first orientation, invariant to q/-q."""
    q = np.asarray(values, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or len(q) < 8 or not np.isfinite(q).all():
        raise ValueError("finite quaternion observations required")
    norms = np.linalg.norm(q, axis=1)
    if np.any(np.abs(norms - 1) > .001):
        raise ValueError("invalid quaternion norm")
    q = q / norms[:, None]
    q = q * np.where(q @ q[0] < 0, -1., 1.)[:, None]
    return float(np.max(np.linalg.norm(q - q[0], axis=1)))


def sensor_statistics(source: pd.DataFrame, nominal: pd.DataFrame) -> dict[str, float]:
    if len(source) < 8 or len(source) != len(nominal):
        raise ValueError("eight aligned paired samples required")
    times = source.time_s.to_numpy(float)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0) or not np.array_equal(times, nominal.time_s):
        raise ValueError("nonfinite, unordered or unaligned time axes")
    result = {}
    def read(fields):
        if not all(f in source and f in nominal for f in fields):
            return None
        a, b = source[fields].to_numpy(float), nominal[fields].to_numpy(float)
        return (a, b) if np.isfinite(a).all() and np.isfinite(b).all() else None
    gyro = read([f"adcs.sensor.gyro_measured_rad_s_{axis}" for axis in "xyz"])
    accel = read([f"adcs.sensor.accel_measured_m_s2_{axis}" for axis in "xyz"])
    if gyro is not None:
        g, ng = gyro
        result["gyro_zero_axes"] = float(channel_zero_count(g))
        result["gyro_reference_excited_axes"] = float(3 - channel_zero_count(ng))
        if accel is not None:
            result["imu_all_output_zero"] = float(channel_zero_count(g) == 3 and channel_zero_count(accel[0]) == 3)
    quality = read(["adcs.star_tracker.valid", "adcs.star_tracker.solution_fresh"])
    if quality is not None:
        result["star_quality_valid"] = float(np.all(quality[0] == 1) and np.all(quality[1] == 1))
    attitude = read(["adcs.star_tracker.q" + str(j) for j in range(4)])
    if attitude is not None:
        try:
            spread, nspread = [quaternion_spread(q) for q in attitude]
        except ValueError:
            pass
        else:
            # No excitation in the reference is UNKNOWN, not a false healthy flag.
            if nspread > ZERO_ATOL and quality is not None and np.all(quality[0][:, 0] == 1) and np.all(quality[1] == 1):
                result["star_output_frozen"] = float(spread <= ZERO_ATOL)
    # Preserve the existing post47 independent-actuator-mismatch guard.
    # A stable innovation can be a downstream control/actuator symptom; it
    # is not specific gyro support while command execution independently fails.
    response = read([field for j in range(4) for field in
                    (f"adcs.rw.command_torque_nm_{j}", f"adcs.control.applied_torque_nm_{j}")])
    if response is not None:
        a, b = response
        mismatch = np.abs((a[:, 0::2]-a[:, 1::2])-(b[:, 0::2]-b[:, 1::2]))
        bounds = np.maximum(2e-6, .05*np.abs(a[:, 0::2]))
        result["actuator_response_consistent"] = float(not np.any(np.count_nonzero(mismatch > bounds, axis=0) >= 4))
    innovation = read([f"adcs.innovation.star_gyro_rate_residual_rad_s_{axis}" for axis in "xyz"])
    if gyro is not None and innovation is not None and result.get("star_quality_valid") == 1:
        g, ng = gyro
        a, b = innovation
        # Match the recorder's interval quaternion rate to trapezoidal gyro.
        delta = (a[1:] + .5*np.diff(g, axis=0)) - (b[1:] + .5*np.diff(ng, axis=0))
        mean = delta.mean(axis=0)
        amplitude = float(np.linalg.norm(mean))
        spread = float(np.sqrt(np.mean(np.sum((delta-mean)**2, axis=1))))
        result["gyro_aligned_bias_norm"] = amplitude
        result["gyro_aligned_spread_ratio"] = spread / max(amplitude, ZERO_ATOL)
    return result


def sensor_check_tools(data, *, source: pd.DataFrame, nominal: pd.DataFrame):
    """Bind source hashes before registering one bounded, trusted read-only check.

    All arithmetic is deferred until the Agent selects this tool. The result
    contains numerical features only; existing evaluate_evidence applies rules.
    """
    from .native_bridge import project_registered_telemetry
    import json
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parents[3] /
        'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text(encoding="utf-8"))['canonical_sha256']
    source = project_registered_telemetry(source, schema_sha256=schema)
    nominal = project_registered_telemetry(nominal, schema_sha256=schema)
    ep = data.bundle.episode
    source = source.loc[source.time_s.between(ep.start_s, ep.end_s)].copy()
    nominal = nominal.loc[nominal.time_s.between(ep.start_s, ep.end_s)].copy()
    receipt = data.three_task_result.evidence_projection if data.three_task_result else None
    def frame_hash(frame):
        return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    if receipt is None or (frame_hash(source), frame_hash(nominal)) != (receipt.source_window_sha256, receipt.nominal_window_sha256):
        raise ValueError("sensor checks are not bound to the admitted source/reference window")
    by_id = {d.feature_definition_id: d for d in data.bundle.feature_definitions}
    selected = {feature_id(k): k for k in METRICS if feature_id(k) in by_id}
    if not selected:
        return ()
    old = {f.feature_definition_id: f for f in data.bundle.feature_results}
    digest = content_hash({"source": receipt.source_window_sha256,
                           "reference": receipt.nominal_window_sha256, "algorithm": VERSION})
    @lru_cache(maxsize=1)
    def calculate():
        return sensor_statistics(source, nominal)
    def handler(requested):
        if not set(requested) <= set(selected):
            raise ValueError("requested feature outside sensor check contract")
        values = calculate()
        results = []
        for fid in requested:
            key = selected[fid]
            if key not in values:
                continue
            d = by_id[fid]
            if d.algorithm_version != VERSION or d.unit != METRICS[key]:
                raise ValueError("sensor operator version/unit differs from published definition")
            results.append(FeatureResult(
                feature_result_id=stable_id('FR-SENSOR', {'episode': ep.episode_id, 'feature': fid, 'input': digest}),
                feature_definition_id=fid, run_id=ep.run_id, paired_nominal_run_id=ep.paired_nominal_run_id,
                episode_id=ep.episode_id, resolved_field=d.source_field_template,
                component_instance=old[fid].component_instance if fid in old else data.bundle.analysis_component_instance,
                value=values[key], unit=d.unit, window_start_s=ep.start_s, window_end_s=ep.end_s,
                sample_count=len(source), algorithm_version=VERSION, input_data_sha256=digest, quality_flags=[]))
        return results
    return (EvidenceCheckTool(tool_id='telemetry.paired_sensor_integrity', tool_version=VERSION,
            feature_definition_ids=frozenset(selected), input_hashes=frozenset({digest}), handler=handler),)
