"""Simple ESA screening baseline: isolated observation input, never fault evidence."""
from __future__ import annotations
import numpy as np

def fit_reference(nominal_training_values, *, relative_floor=1e-6):
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT')
    values=np.asarray(nominal_training_values,dtype=float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('ESA_TRAINING_OBSERVATIONS_UNADMITTED')
    center=float(np.median(values));scale=max(float(1.4826*np.median(np.abs(values-center))),abs(center)*relative_floor,1e-9)
    return {'center':center,'scale':scale}

def _score_core(payload,reference):
    """Truth-free ESA score core; governance wrappers decide which data may call it."""
    if set(payload)!={'time_ns','value','observed','channel'}:raise ValueError('ESA_TRUTH_OR_UNADMITTED_FIELD_IN_MODEL_INPUT')
    values=np.asarray(payload['value'],dtype=float);observed=np.asarray(payload['observed'],dtype=bool);times=np.asarray(payload['time_ns'])
    if values.ndim!=1 or values.shape!=observed.shape or values.shape!=times.shape:raise ValueError('ESA_OBSERVATION_SHAPE_DRIFT')
    if not np.isfinite(values[observed]).all() or np.any(np.diff(times)<=0):raise ValueError('ESA_OBSERVATION_IDENTITY_DRIFT')
    center=float(reference['center']);scale=float(reference['scale'])
    if not (np.isfinite(center) and np.isfinite(scale) and scale>0):raise ValueError('ESA_REFERENCE_IDENTITY_DRIFT')
    scores=np.zeros_like(values);scores[observed]=np.abs(values[observed]-center)/scale
    return scores

def score(payload,reference):
    from .data_exposure import guard_prediction_generation
    guard_prediction_generation()
    return _score_core(payload,reference)
