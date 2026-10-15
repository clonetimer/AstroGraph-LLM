"""R9 host-reviewed contextual applicability for material causal references.

The structured context is an expert *scope annotation*, never an LLM fact,
a flight observation or proof of causal correctness. Optional legacy contexts
are unknown rather than globally applicable. Categories and units are closed
vocabularies to avoid silent comparisons of incomparable measurements.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

SCHEMA = 'astrograph-reviewed-material-applicability.v1'
_IDENTIFIER = re.compile(r'^[A-Z][A-Z0-9_]{1,63}$')
_ENV_KEYS = frozenset({'THERMAL_REGIME', 'PRESSURE_REGIME', 'ILLUMINATION', 'ORBITAL_PHASE'})
# Only explicitly supported conversions may be compared. Other units fail closed.
_UNITS = {'K': ('K', 1.0, 0.0), 'DEGC': ('K', 1.0, 273.15),
          'V': ('V', 1.0, 0.0), 'MV': ('V', 0.001, 0.0),
          'A': ('A', 1.0, 0.0), 'MA': ('A', 0.001, 0.0),
          'RPM': ('RPM', 1.0, 0.0), 'RAD_S': ('RAD_S', 1.0, 0.0)}
_MAX_RANGE_ITEMS = 12


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError('invalid reviewed applicability ' + label)
    return value


def normalize_reviewed_applicability(value: Mapping[str, Any], *, equipment_scope: str) -> dict[str, Any]:
    """Validate only expert-owned context; never accept it from graph extraction.

    Empty/wildcard scopes are not represented as universal assertions. The
    optional dimensions describe narrow predicates, not a verified flight state.
    """
    if not isinstance(value, Mapping) or set(value) != {
        'schema_version', 'equipment_model', 'operating_modes', 'environment',
        'parameter_ranges', 'review_basis_ref',
    }:
        raise ValueError('invalid reviewed applicability schema keys')
    if value['schema_version'] != SCHEMA:
        raise ValueError('invalid reviewed applicability version')
    model = _identifier(value['equipment_model'], 'equipment_model')
    if model != equipment_scope:
        raise ValueError('reviewed applicability equipment scope mismatch')
    basis = value['review_basis_ref']
    if not isinstance(basis, str) or not 8 <= len(basis.strip()) <= 300:
        raise ValueError('reviewed applicability independent basis required')
    modes = value['operating_modes']
    if not isinstance(modes, list) or not 1 <= len(modes) <= 12:
        raise ValueError('reviewed applicability mode set required')
    mode_set = [_identifier(mode, 'operating_mode') for mode in modes]
    if len(set(mode_set)) != len(mode_set):
        raise ValueError('duplicate operating_mode')
    env = value['environment']
    if not isinstance(env, dict) or not set(env).issubset(_ENV_KEYS) or len(env) > 4:
        raise ValueError('unsupported applicability environment dimension')
    checked_env = {_identifier(k, 'environment key'): _identifier(v, 'environment value')
                   for k, v in env.items()}
    ranges = value['parameter_ranges']
    if not isinstance(ranges, list) or len(ranges) > _MAX_RANGE_ITEMS:
        raise ValueError('invalid reviewed physical parameter bounds')
    canonical = []
    keys = set()
    for item in ranges:
        if (not isinstance(item, dict) or set(item) not in (
                {'name', 'unit', 'min', 'max'},
                {'name', 'unit', 'min', 'max', 'source_unit'})):
            raise ValueError('invalid physical parameter range shape')
        name = _identifier(item['name'], 'parameter name')
        unit = item['unit']
        if not isinstance(unit, str) or unit not in _UNITS:
            raise ValueError('unsupported or ambiguous physical parameter unit')
        low, high = item['min'], item['max']
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in (low, high)) or low > high:
            raise ValueError('invalid/nonfinite physical parameter range')
        base, scale, offset = _UNITS[unit]
        source_unit = item.get('source_unit', unit)
        if source_unit not in _UNITS or _UNITS[source_unit][0] != base:
            raise ValueError('invalid physical range source unit')
        if 'source_unit' in item and unit != base:
            raise ValueError('noncanonical stored physical unit')
        minimum, maximum = low * scale + offset, high * scale + offset
        if not all(math.isfinite(v) for v in (minimum, maximum)):
            raise ValueError('physical parameter conversion overflow')
        if name in keys:
            raise ValueError('duplicate physical parameter constraint')
        keys.add(name)
        canonical.append({'name': name, 'unit': base, 'min': round(minimum, 10),
                          'max': round(maximum, 10), 'source_unit': source_unit})
    return {'schema_version': SCHEMA, 'equipment_model': model,
            'operating_modes': sorted(mode_set), 'environment': dict(sorted(checked_env.items())),
            'parameter_ranges': sorted(canonical, key=lambda item: item['name']),
            'review_basis_ref': basis.strip()}


def assess_contexts(edges: Sequence[Mapping[str, Any]], *, equipment_scope: str) -> dict[str, Any]:
    """Find contradictions in explicit reviewed constraints, not hypothetical flight facts.

    A missing context is UNKNOWN, never a match or a conflict. Model scope
    mismatches and malformed annotations fail closed; disjoint ranges/modes
    block the *whole path*. A consistent intersection is not causal proof.
    """
    reasons: set[str] = set()
    normalized = []
    unknown_edge_ids = []
    for edge in edges:
        value = edge.get('reviewed_applicability')
        if value is None:
            unknown_edge_ids.append(edge.get('edge_id') or edge.get('proposal_id') or 'UNKNOWN')
            continue
        try:
            normalized.append(normalize_reviewed_applicability(value, equipment_scope=equipment_scope))
        except (ValueError, TypeError, OverflowError):
            reasons.add('INVALID_REVIEWED_APPLICABILITY')
    if reasons:
        return {'status': 'REJECTED_INVALID_CONTEXT', 'conflict_reasons': sorted(reasons),
                'unknown_edge_ids': unknown_edge_ids, 'reviewed_context_count': len(normalized),
                'flight_applicability_verified': False}
    modes: set[str] | None = None
    env: dict[str, str] = {}
    ranges: dict[str, dict[str, Any]] = {}
    for item in normalized:
        available = set(item['operating_modes'])
        modes = available if modes is None else modes & available
        if not modes:
            reasons.add('OPERATING_MODE_DISJOINT')
        for key, val in item['environment'].items():
            if key in env and env[key] != val:
                reasons.add('ENVIRONMENT_CONFLICT:' + key)
            else:
                env[key] = val
        for bound in item['parameter_ranges']:
            name = bound['name']
            previous = ranges.get(name)
            if previous is None:
                ranges[name] = {k: bound[k] for k in ('unit', 'min', 'max')}
            elif previous['unit'] != bound['unit']:
                reasons.add('PARAMETER_DIMENSION_CONFLICT:' + name)
            else:
                previous['min'] = max(previous['min'], bound['min'])
                previous['max'] = min(previous['max'], bound['max'])
                if previous['min'] > previous['max']:
                    reasons.add('PHYSICAL_RANGE_DISJOINT:' + name)
    return {'status': ('REJECTED_INCOMPATIBLE_CONTEXT' if reasons else
                        'INCOMPLETE_REVIEWED_CONTEXT' if unknown_edge_ids else
                        'REVIEWED_CONSTRAINTS_MUTUALLY_COMPATIBLE_NOT_FLIGHT_VERIFIED'),
            'conflict_reasons': sorted(reasons), 'unknown_edge_ids': unknown_edge_ids,
            'reviewed_context_count': len(normalized),
            'intersection': {'operating_modes': sorted(modes) if modes is not None else [],
                             'environment': dict(sorted(env.items())),
                             'parameter_ranges': dict(sorted(ranges.items()))},
            'flight_applicability_verified': False}


def contexts_provably_disjoint(left: Mapping[str, Any], right: Mapping[str, Any], *, equipment_scope: str) -> bool:
    """Different approved contexts cannot directly contradict each other.

    UNKNOWN does not establish disjointness: the old conservative conflict
    behavior remains for legacy unannotated claims.
    """
    if left.get('reviewed_applicability') is None or right.get('reviewed_applicability') is None:
        return False
    return assess_contexts((left, right), equipment_scope=equipment_scope)['status'] == 'REJECTED_INCOMPATIBLE_CONTEXT'
