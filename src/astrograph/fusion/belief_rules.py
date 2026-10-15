"""Small RIMER-style belief-rule engine; belief degrees are not probabilities.

ER combines activated rules, not independent telemetry sources. The caller must
build one dependence-aware antecedent packet and preserve missing information.
Reference: Yang et al. (2006), DOI 10.1109/TSMCA.2005.851270, section III.
"""
from __future__ import annotations

import math


def er_aggregate(weights: list[float], beliefs: list[list[float]]) -> dict:
    """Analytic form of recursive ER, retaining incomplete consequent mass."""
    if not weights or len(weights) != len(beliefs):
        raise ValueError('one belief vector per activation weight is required')
    n = len(beliefs[0])
    if n < 2 or any(len(b) != n for b in beliefs):
        raise ValueError('at least two consistent consequent grades required')
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-10):
        raise ValueError('activation weights must sum to one')
    if any(any(not math.isfinite(x) or x < 0 for x in b) or sum(b) > 1+1e-10 for b in beliefs):
        raise ValueError('belief vectors must be finite nonnegative subdistributions')
    residual = [1-w*sum(b) for w, b in zip(weights, beliefs, strict=True)]
    bprod = math.prod(residual)
    cprod = math.prod(1-w for w in weights)
    products = [math.prod(w*b[j]+h for w,b,h in zip(weights,beliefs,residual,strict=True)) for j in range(n)]
    denominator = sum(products)-(n-1)*bprod-cprod
    if denominator <= 1e-14:
        return {'beliefs': [0.]*n, 'ignorance': 1., 'status': 'UNRESOLVED'}
    out = [max(0., (v-bprod)/denominator) for v in products]
    unknown = max(0., (bprod-cprod)/denominator)
    return {'beliefs': out, 'ignorance': unknown, 'status': 'OK'}


def matching_degrees(value: float) -> list[float]:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('antecedent must be finite in [0,1]')
    return [max(0.,1-2*value), 1-abs(2*value-1), max(0.,2*value-1)]


MECHANISM_STATES = ('UNSUPPORTED', 'SUPPORTED', 'UNKNOWN', 'NONSPECIFIC')


def activation(data_support: float, mechanism_state: str) -> list[float]:
    if mechanism_state not in MECHANISM_STATES:
        raise ValueError('unknown mechanism state')
    levels = matching_degrees(data_support)
    return [levels[i] if mechanism_state == state else 0.
            for state in MECHANISM_STATES for i in range(3)]


def infer(data_support: float, mechanism_state: str, rules: list[list[float]]) -> dict:
    if len(rules) != 12:
        raise ValueError('this small BRB requires exactly twelve rules')
    weights = activation(data_support, mechanism_state)
    result = er_aggregate(weights, rules)
    return {**result, 'activation_weights': weights,
            'score_semantics': 'BELIEF_DEGREE_NOT_CALIBRATED_POSTERIOR',
            'source_independence_assumed': False}
