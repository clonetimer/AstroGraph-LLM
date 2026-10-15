"""Paired, causal-window RW statistics; no labels, model predictions or fault IDs."""
from __future__ import annotations
import numpy as np
import pandas as pd

PACKAGE_ID = 'adcs-v522-paired-rw-window'
METRICS = {
    'command_rms': 'N*m',
    'nominal_command_rms': 'N*m',
    'transfer_gain': '1',
    'nominal_transfer_gain': '1',
    'gain_distance_from_one': '1',
    'transfer_fit_error': '1',
    'nominal_transfer_fit_error': '1',
    'speed_absmax': 'rad/s',
    'nominal_speed_absmin': 'rad/s',
    'speed_offset_absmean': 'rad/s',
    'speed_offset_std': 'rad/s',
    'paired_command_absmax': 'N*m',
}

def statistic_fields():
    return [f'derived.rw.window.{j}.{k}' for j in range(4) for k in METRICS]

def derive_window_statistics(source: pd.DataFrame, nominal: pd.DataFrame):
    """Return derived frames; caller supplies precisely the admitted model window.

    Each statistic describes this entire window and is repeated only to let
    the existing scalar feature/evidence operators consume it. It is NOT a
    per-sample measurement or a continuous time series.
    """
    if len(source) < 8 or len(source) != len(nominal):
        raise ValueError('paired RW statistics require at least eight aligned observations')
    if not np.array_equal(source.time_s.to_numpy(), nominal.time_s.to_numpy()):
        raise ValueError('RW statistic time axes differ')
    out = source.copy()
    ref = nominal.copy()
    for j in range(4):
        fields = [f'adcs.rw.command_torque_nm_{j}', f'adcs.control.applied_torque_nm_{j}',
                  f'adcs.rw.measured_speed_rad_s_{j}']
        a = source[fields].to_numpy(float)
        b = nominal[fields].to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('non-finite required RW observation')
        c, applied, speed = a.T
        nc, napplied, nspeed = b.T
        def transfer(cmd, measured):
            rms = float(np.sqrt(np.mean(cmd**2)))
            gain = float(np.dot(cmd, measured) / max(np.dot(cmd, cmd), 1e-20))
            error = float(np.sqrt(np.mean((measured - gain*cmd)**2)) / max(rms, 1e-12))
            return rms, gain, error
        rms, gain, error = transfer(c, applied)
        nrms, ngain, nerror = transfer(nc, napplied)
        delta = speed - nspeed
        values = [rms, nrms, gain, ngain, abs(gain-1), error, nerror,
                  float(np.max(abs(speed))), float(np.min(abs(nspeed))),
                  float(np.mean(abs(delta))), float(np.std(delta)), float(np.max(abs(c-nc)))]
        nominal_values = [nrms, nrms, ngain, ngain, abs(ngain-1), nerror, nerror,
                          float(np.max(abs(nspeed))), float(np.min(abs(nspeed))), 0., 0., 0.]
        for metric, value, base in zip(METRICS, values, nominal_values, strict=True):
            out[f'derived.rw.window.{j}.{metric}'] = value
            ref[f'derived.rw.window.{j}.{metric}'] = base
    return out, ref


TERMINAL_PACKAGE_ID = 'adcs-v522-paired-rw-terminal8'

def derive_terminal_statistics(source: pd.DataFrame, nominal: pd.DataFrame):
    """Fixed causal persistence window; raw model input remains unchanged.

    Eight is the existing minimum observation count, frozen before evaluation.
    This is not change-point detection and does not estimate fault onset.
    Statistics are named terminal8 so their support is not mistaken for 20 points.
    """
    if len(source) != len(nominal) or not np.array_equal(source.time_s, nominal.time_s):
        raise ValueError('RW statistic time axes differ')
    if len(source) < 8 or not np.all(np.diff(source.time_s.to_numpy(float)) > 0):
        raise ValueError('terminal statistics require eight ordered observations')
    tail, ref_tail = derive_window_statistics(source.iloc[-8:], nominal.iloc[-8:])
    out, ref = source.copy(), nominal.copy()
    for field in statistic_fields():
        target = field.replace('.window.', '.terminal8.')
        out[target], ref[target] = float(tail[field].iloc[-1]), float(ref_tail[field].iloc[-1])
    return out, ref
