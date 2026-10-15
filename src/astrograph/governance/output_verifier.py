from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field


class VerificationVerdict(str, Enum):
    PASS = 'PASS'
    FAIL = 'FAIL'
    WARNING = 'WARNING'


class OutputVerificationRecord(BaseModel):
    """One independent deterministic verification of a generated (LLM) output."""
    model_config = ConfigDict(extra='forbid')
    schema_version: str = 'astrograph-output-verification.v1'
    verification_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    verdict: VerificationVerdict
    checks: list[dict] = Field(default_factory=list)
    verified_at: str = Field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S+08:00'))


def verify_hypothesis_set(output: dict[str, Any], *, candidate_universe: Sequence[str]) -> OutputVerificationRecord:
    """Independently verify a generated hypothesis set against the frozen candidate universe.

    Deterministic checks (no LLM self-attestation):
    1. Every SUPPORTED hypothesis target_fault_id must be in the candidate universe.
    2. SUPPORTED hypotheses must carry at least one supporting evidence fact id.
    3. A rejected/contradicted hypothesis must not also be SUPPORTED.
    """
    checks: list[dict] = []
    hypotheses = output.get('hypotheses') or []
    universe = set(candidate_universe)
    supported = [h for h in hypotheses if h.get('status') == 'SUPPORTED']
    # Check 1: candidate universe membership.
    unknown = sorted({h.get('target_fault_id') for h in supported} - universe)
    checks.append({'check': 'candidate_universe_membership', 'passed': not unknown, 'detail': 'unknown=' + str(unknown) if unknown else 'all supported in universe'})
    # Check 2: supporting evidence required.
    no_evidence = [h.get('hypothesis_id') or h.get('target_fault_id') for h in supported if not (h.get('supporting_evidence_fact_ids') or h.get('supporting_evidence'))]
    checks.append({'check': 'supporting_evidence_present', 'passed': not no_evidence, 'detail': 'no_evidence=' + str(no_evidence) if no_evidence else 'all supported carry evidence'})
    # Check 3: no status contradiction within one hypothesis.
    contradictory = [h.get('hypothesis_id') for h in hypotheses if h.get('status') in {'CONTRADICTED', 'REJECTED'} and h.get('supporting_evidence_fact_ids')]
    checks.append({'check': 'no_status_contradiction', 'passed': not contradictory, 'detail': 'contradictory=' + str(contradictory) if contradictory else 'no contradiction'})
    failed = [c for c in checks if not c['passed']]
    verdict = VerificationVerdict.FAIL if failed else VerificationVerdict.PASS
    return OutputVerificationRecord(
        verification_id='VER-' + time.strftime('%Y%m%d%H%M%S'),
        artifact_type='hypothesis_set',
        artifact_id=str(output.get('episode_id') or output.get('hypothesis_set_id') or 'unknown'),
        verdict=verdict,
        checks=checks,
    )


__all__ = ['VerificationVerdict', 'OutputVerificationRecord', 'verify_hypothesis_set']