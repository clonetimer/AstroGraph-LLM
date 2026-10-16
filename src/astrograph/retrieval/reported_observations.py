"""Bounded, non-voting intake for numbered human-reported fault symptoms.

This is NOT a Chinese NER, telemetry parser, fault classifier, or NLI engine.
No time synchronisation, root cause, equipment identity, or quantitative
threshold is inferred from prose. The raw numbered sentence is the evidence.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

SCHEMA_VERSION = 'astrograph-reported-observation-ledger.v1'
_NUMBERED = re.compile(r'^\s*(?:\d{1,2}\s*\\?[.．、)）]|\(\d{1,2}\)|（\d{1,2}）)\s*(\S.*)$')
_TEMPERATURE = re.compile(r'(?<!\d)\d+(?:\.\d+)?\s*(?:°\s*[Cc]|℃|摄氏度)')
_DIP_COUNT = re.compile(r'(?<!\d)\d+(?:\.\d+)?\s*次\s*(?:跌落|下降|下跌)')


def review_numbered_observations(source_text: str, *, max_items: int = 12) -> dict[str, Any]:
    """Preserve user assertions without converting them to instrument evidence.

    Only explicit, line-start, numbered observations are admitted. Other task
    prose is NOT implicitly classified as an observation. `max_items` is host
    bounded and truncation is surfaced rather than silently treating the first
    samples as the whole report.
    """
    if not isinstance(source_text, str):
        raise TypeError('observation input must be text')
    if type(max_items) is not int or not 1 <= max_items <= 12:
        raise ValueError('invalid reported observation bound')
    # Do not put long untrusted content in the diagnostic LLM payload.
    input_truncated = len(source_text) > 12000
    source_text = source_text[:12000]
    rows: list[dict[str, Any]] = []
    found = 0
    for line_number, line in enumerate(source_text.splitlines(), 1):
        match = _NUMBERED.match(line)
        if not match:
            continue
        statement = match.group(1).strip()
        if not statement:
            continue
        found += 1
        if len(rows) >= max_items:
            continue
        # Keep original spans only; do not assign channel names, per-channel
        # event counts, synchronism or physical meaning from lexical overlap.
        statement_truncated = len(statement) > 600
        statement = statement[:600]
        digest = hashlib.sha256(statement.encode('utf-8')).hexdigest()
        scoped_id = hashlib.sha256(f'{line_number}\0{statement}'.encode('utf-8')).hexdigest()
        rows.append({
            'evidence_id': 'USR-' + scoped_id[:16].upper(),
            'source_locator': f'user_text#line:{line_number}',
            'source_sha256': digest,
            'reported_text': statement,
            'statement_truncated': statement_truncated,
            'temperature_literals': _TEMPERATURE.findall(statement)[:12],
            'event_count_literals': _DIP_COUNT.findall(statement)[:12],
            'source_type': 'USER_REPORTED_UNVERIFIED',
            'instrument_observation_verified': False,
            'independent_time_alignment_verified': False,
            'physical_cause_verified': False,
            'diagnostic_vote_allowed': False,
        })
    return {
        'schema_version': SCHEMA_VERSION,
        'status': 'USER_REPORT_ONLY' if rows else 'NO_NUMBERED_OBSERVATIONS',
        'observations': rows,
        'matched_numbered_lines': found,
        'truncated': input_truncated or found > max_items or any(r['statement_truncated'] for r in rows),
        'source_input_truncated': input_truncated,
        'no_frozen_fault_candidate_generated': True,
        'no_cross_line_timing_or_causality_inferred': True,
        'missing_numeric_baselines_not_filled': True,
        'automatic_control_authorized': False,
        'interpretation_boundary': (
            'User narrative is a set of unverified assertions, not time-aligned '
            'telemetry or ground-truth fault labels; matching two events in one '
            'sentence does not establish simultaneity or independent event counts.'
        ),
    }
