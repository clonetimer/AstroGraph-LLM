"""Fail-closed extractive audit for material-report model claims.

This is deliberately NOT an NLI/entailment classifier. Matching an exact
source passage establishes quotation provenance only, not causal truth.
Unmatched model statements are retained as hashes in the public report, with
raw text still available in the private knowledge-enrichment artifact.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
import unicodedata
from typing import Any, Mapping

SCHEMA_VERSION = 'astrograph-material-claim-audit.v1'
# Explicit labels the host permits before an *entire* source quotation.
_PREFIXES = ('上传资料片段:', '文献摘要片段:', '日志原文:', '原文摘录:',
             'Reference excerpt:', 'Source excerpt:', 'Log excerpt:')
_SOURCE_KINDS = {
    'LOCAL_KNOWLEDGE', 'LITERATURE', 'ONTOLOGY_CANDIDATE',
    'MATERIAL_LOG_CUE', 'MATERIAL_REFERENCE',
}


def _norm(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def _sources(knowledge: Mapping[str, Any], reasoning: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}

    def add(kind: str, row: Mapping[str, Any], key: str, text_key: str, status: str) -> None:
        sid = row.get(key)
        quote = row.get(text_key)
        if not isinstance(sid, str) or not sid or not isinstance(quote, str) or not quote.strip():
            return
        item = {'quote': quote, 'status': status}
        index = (kind, sid)
        # Duplicate IDs with conflicting underlying text are not trustworthy.
        if index in result and result[index].get('quote') != quote:
            result[index] = {'quote': '', 'status': 'AMBIGUOUS_DUPLICATE_CITATION'}
        elif index not in result:
            result[index] = item

    for row in (knowledge.get('local_retrieval') or {}).get('hits') or []:
        if isinstance(row, Mapping):
            add('LOCAL_KNOWLEDGE', row, 'knowledge_id', 'quote', 'UPLOADED_REFERENCE')
    for row in (knowledge.get('paperseek') or {}).get('documents') or []:
        if isinstance(row, Mapping):
            add('LITERATURE', row, 'document_id', 'text', 'BIBLIOGRAPHIC_RECORD_NOT_CURRENT_OBSERVATION')
    for row in (knowledge.get('anything2ontology') or {}).get('review_material') or []:
        if isinstance(row, Mapping):
            add('ONTOLOGY_CANDIDATE', row, 'ontology_id', 'quote', 'UNREVIEWED_ONTOLOGY_PROPOSAL')
    for candidate in reasoning.get('candidates') or []:
        if not isinstance(candidate, Mapping):
            continue
        for row in (candidate.get('log_evidence_matrix') or {}).get('events') or []:
            if isinstance(row, Mapping):
                add('MATERIAL_LOG_CUE', row, 'evidence_id', 'excerpt', 'UNVERIFIED_TEXT_CUE')
        for row in (candidate.get('reference_retrieval') or {}).get('hits') or []:
            if isinstance(row, Mapping):
                add('MATERIAL_REFERENCE', row, 'knowledge_id', 'quote',
                    str(row.get('knowledge_validation_status') or 'UNREVIEWED_REFERENCE'))
    return result


def audit_material_claims(knowledge: Mapping[str, Any], reasoning: Mapping[str, Any],
                          explanation: Mapping[str, Any], *, limit: int = 12) -> dict[str, Any]:
    """Return publishable *quotations* and per-claim opaque audit receipts.

    A cited hypothesis, case, graph path, or numerical diagnosis is never
    promoted to an extractive factual statement. Does not execute code, run
    model inference, or depend on the model-supplied description of sources.
    """
    if type(limit) is not int or not 0 <= limit <= 12:
        raise ValueError('invalid claim audit bound')
    sources = _sources(knowledge, reasoning)
    visible = []
    receipts = []
    totals: Counter[str] = Counter()
    for ordinal, claim in enumerate((explanation.get('claims') or [])[:limit], 1):
        if not isinstance(claim, Mapping):
            continue
        kind = str(claim.get('kind') or '')
        raw = str(claim.get('text') or '')[:4000]
        citations = claim.get('citation_ids') or []
        citations = citations if isinstance(citations, list) else []
        identifiers = [x for x in citations if isinstance(x, str) and x]
        status = 'INFERENCE_OR_NONEXTRACTIVE_WITHHELD'
        source_statuses = []
        if kind in _SOURCE_KINDS and len(identifiers) == 1:
            source = sources.get((kind, identifiers[0]))
            if source and source['quote']:
                quotation = _norm(source['quote'])
                published = _norm(raw)
                exact = published == quotation or any(
                    published == _norm(prefix) + ' ' + quotation or
                    published == _norm(prefix) + quotation
                    for prefix in _PREFIXES
                )
                if exact:
                    status = 'EXTRACTIVE_MATCH_NOT_SEMANTIC_ENTAILMENT'
                    source_statuses = [source['status']]
                    visible.append({
                        'kind': kind,
                        'text': raw,
                        'citation_ids': identifiers,
                        'claim_validation': status,
                        'source_validation': source['status'],
                        'verified_current_root_cause': False,
                        'diagnostic_vote_allowed': False,
                    })
            elif source and source['status'] == 'AMBIGUOUS_DUPLICATE_CITATION':
                status = 'DUPLICATE_SOURCE_ID_CONFLICT_WITHHELD'
            else:
                status = 'CITATION_SOURCE_TEXT_MISSING_WITHHELD'
        elif kind in _SOURCE_KINDS and not identifiers:
            status = 'NO_CITATION_WITHHELD'
        elif kind == 'HYPOTHESIS':
            status = 'HYPOTHESIS_NOT_VALIDATED_WITHHELD'
        elif kind in {'GRAPH_ASSOCIATION', 'CONFIRMED_CASE'}:
            status = 'ASSOCIATION_OR_HISTORY_NOT_CURRENT_CAUSE_WITHHELD'
        elif kind == 'OBSERVATION':
            status = 'NO_INDEPENDENT_RAW_OBSERVATION_PROOF_WITHHELD'
        totals[status] += 1
        receipts.append({
            'index': ordinal, 'kind': kind,
            'claim_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
            'citation_ids': identifiers[:16], 'status': status,
            'source_statuses': source_statuses,
        })
    return {
        'schema_version': SCHEMA_VERSION,
        'status': 'EXTRACTIVE_ONLY_NOT_SEMANTIC_VERIFICATION',
        'claim_receipts': receipts, 'status_counts': dict(sorted(totals.items())),
        'published_extracts': visible,
        'withheld_claim_count': len(receipts) - len(visible),
        'natural_language_entailment_verified': False,
        'root_cause_verified': False,
        'automatic_control_authorized': False,
    }
