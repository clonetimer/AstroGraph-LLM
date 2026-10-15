"""Conservative report checks; absence of flags is not semantic proof."""
from __future__ import annotations
import re
from .contracts import OAGExplanation
from .assessment import assess_summary,validate_assessment


def audit_report(explanation: OAGExplanation, summary: dict, documents: list) -> dict:
    from .runtime import _validate_explanation
    issues=[]
    try:
        _validate_explanation(explanation,summary,documents)
        if explanation.assessment is None:issues.append('ASSESSMENT_MISSING')
        else:validate_assessment(explanation.assessment,assess_summary(summary))
    except ValueError: issues.append('EVIDENCE_CONTRACT_VIOLATION')
    if explanation.candidate_fault_id != summary['suspected_fault_id']:
        issues.append('DECISION_NOT_PRESERVED')
    if not explanation.limitations:issues.append('LIMITATIONS_MISSING')
    for i,claim in enumerate(explanation.claims):
        if summary['suspected_fault_id'] is not None and re.search(
                r'\bNO_REGISTERED_FAULT\b|\bno (?:registered )?fault hypothesis.*(?:supported|selected)|没有可选的故障假设|无任何受支持的故障候选',
                claim.text,re.I):
            issues.append(f'CLAIM_{i}_PRIMARY_CONCLUSION_CONTRADICTION')
        if claim.kind != 'HYPOTHESIS' and re.search(r'确诊|已证明因果|百分之百|definitively confirmed|proven root cause',claim.text,re.I):
            issues.append(f'CLAIM_{i}_UNSUPPORTED_CERTAINTY')
        if claim.kind=='OBSERVATION' and re.search(r'执行关机|发送指令|立即切换|command the spacecraft',claim.text,re.I):
            issues.append(f'CLAIM_{i}_UNAUTHORIZED_ACTION')
    prose = [(f'CLAIM_{i}', c.text) for i,c in enumerate(explanation.claims)]
    prose += [(f'LIMITATION_{i}', text) for i,text in enumerate(explanation.limitations)]
    unmet_definitions = any(c.get('missing_required_definition_ids') for c in summary.get('candidates', []))
    for location,text in prose:
        if re.search(r'\bBiRRF(?:\s+\w+){0,3}\s+(?:are|as)\s+(?:(?:uncalibrated|calibrated)\s+)?probabilities\b',text,re.I):
            issues.append(f'{location}_RANK_SCORE_MISDESCRIBED_AS_PROBABILITY')
        if unmet_definitions and re.search(r'missing required (?:evidence )?definitions|lacks? (?:the )?required definition',text,re.I):
            issues.append(f'{location}_UNMET_CONDITION_MISDESCRIBED_AS_ABSENT_DEFINITION')
    return {'schema_version':'astrograph-report-audit.v1','status':'FLAGGED' if issues else 'NO_DETECTED_CONTRACT_VIOLATION',
        'issues':issues,'independent_semantic_review':'REQUIRED_FOR_QUALITY_CLAIM',
        'claim_boundary':'Deterministic citation/assessment/limited phrase checks only; not full entailment or diagnosis accuracy.'}
