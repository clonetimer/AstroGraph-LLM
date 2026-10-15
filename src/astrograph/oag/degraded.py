"""Observation-only fallback. Never repairs, resamples or invokes the model."""
from __future__ import annotations
from .diagnostic_scope import diagnostic_scope
import hashlib
import numpy as np
import pandas as pd
from .window_features import derive_window_statistics, derive_terminal_statistics, TERMINAL_PACKAGE_ID
from .native_bridge import project_registered_telemetry


def mechanism_only_fallback(*, builder, event, source, nominal, schema_sha256, window_length=20):
    """Provide bounded RW hypotheses after explicit packet loss, never a class.

    Only complete paired observations enter descriptive mechanism statistics.
    The original timestamps and missing timestamps remain in the receipt.
    This does not make the fixed-rate model compatible with missing data.
    """
    source = project_registered_telemetry(source, schema_sha256=schema_sha256).iloc[-window_length:]
    nominal = project_registered_telemetry(nominal, schema_sha256=schema_sha256)
    paired = nominal.loc[nominal.time_s.between(source.time_s.iloc[0], source.time_s.iloc[-1])]
    if len(source) != window_length or not np.array_equal(source.time_s, paired.time_s):
        raise ValueError('degraded evidence requires exact paired time alignment')
    times = source.time_s.to_numpy(float)
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('degraded evidence requires ordered finite timestamps')
    fields = [f'{stem}_{j}' for j in range(4) for stem in (
        'adcs.rw.command_torque_nm', 'adcs.control.applied_torque_nm', 'adcs.rw.measured_speed_rad_s')]
    valid = np.isfinite(source[fields].to_numpy(float)).all(axis=1) & np.isfinite(paired[fields].to_numpy(float)).all(axis=1)
    if valid.all() or valid.sum() < max(8, int(np.ceil(.9*window_length))):
        raise ValueError('fallback requires missing RW observations and at least 90 percent valid pairs')
    # Retain gaps, do not present these rows to the fixed-rate model.
    derive = derive_terminal_statistics if builder.package.package_id == TERMINAL_PACKAGE_ID else derive_window_statistics
    s,n = derive(source.iloc[np.flatnonzero(valid)], paired.iloc[np.flatnonzero(valid)])
    bundle,gate = builder.build_from_frames(event, source=s, nominal=n)
    if not gate.passed:
        raise ValueError('degraded mechanism evidence failed its quality gate')
    ids = sorted({f.target_fault_id for f in bundle.evidence_facts
                  if f.status.value == 'SATISFIED' and f.target_fault_id.startswith('ADCS_RW_')})
    return {'schema_version':'astrograph-degraded-mechanism-evidence.v1',
        'status':'MECHANISM_CANDIDATES_ONLY' if 0 < len(ids) <= 3 else 'INSUFFICIENT_EVIDENCE',
        'candidate_fault_ids':ids if len(ids)<=3 else [], 'confirmed_fault_id':None,
        'diagnostic_scopes':[diagnostic_scope(fid) for fid in ids] if len(ids)<=3 else [],
        'model_invoked':False, 'normality_established':False, 'resampling_performed':False,
        'window_start_s':float(times[0]), 'window_end_s':float(times[-1]),
        'valid_observations':int(valid.sum()), 'scheduled_observations':window_length,
        'missing_times_s':times[~valid].tolist(), 'observed_times_s':times[valid].tolist(),
        'source_sha256':hashlib.sha256(source.to_csv(index=False).encode()).hexdigest(),
        'nominal_sha256':hashlib.sha256(paired.to_csv(index=False).encode()).hexdigest(),
        'evidence_package_sha256':builder.package_sha256,
        'evidence_facts':[f.model_dump(mode='json') for f in bundle.evidence_facts],
        'required_next_observations':['Acquire a complete fixed-rate model window',
            'Validate independent wheel speed and actuator electrical/drive measurements'],
        'claim_boundary':'Descriptive statistics on observed pairs with gaps; no model class, normality or motor root-cause confirmation.'}
