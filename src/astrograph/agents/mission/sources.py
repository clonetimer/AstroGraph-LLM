"""Citable views of host facts, distinct from opaque graph/query/lineage IDs.

These are deterministic report projections, not new evidence or LLM summaries.
The full core artifact and JSON pointers remain the source of truth.
"""
from __future__ import annotations
import copy
import json
from typing import Any
from .storage import canonical, sha


def normalize_sources(sources: list[dict]) -> list[dict]:
    """Require a stable citation identity and text (JSON objects use JSON, never repr).

    A caller must not reuse an ID for different texts. Multiple projected passages
    are produced later by ContextBuilder and share the original source identity.
    """
    seen = {}
    output = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError('SOURCE_MUST_BE_OBJECT')
        item = copy.deepcopy(source)
        identity = item.get('citation_id')
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError('SOURCE_CITATION_ID_REQUIRED')
        text = item.get('text', '')
        if isinstance(text, (dict, list)):
            item['original_value_sha256'] = sha(text)
            text = canonical(text).decode('utf-8')
            item['text_format'] = 'CANONICAL_JSON'
        if not isinstance(text, str):
            raise ValueError('SOURCE_TEXT_MUST_BE_STRING_OR_JSON_CONTAINER')
        if identity in seen and seen[identity] != text:
            raise ValueError('SOURCE_ID_COLLISION')
        seen[identity] = text
        item['text'] = text
        output.append(item)
    return output


def diagnostic_sources(core: dict, *, exploration: bool = False) -> list[dict]:
    """Short, source-bound facts, with an indispensable diagnostic boundary anchor.

    Every non-anchor fact is separately budgetable; omitted facts remain in core.
    No candidate is dropped from the mandatory diagnostic context itself.
    """
    diagnosis = core.get('diagnosis', {}) if isinstance(core, dict) else {}
    core_digest = sha(core)
    root = ('TASK-' if exploration else 'DIAGNOSIS-') + core_digest[:20]
    fields = ('status', 'detected', 'selected_fault_id', 'candidate_fault_ids',
              'requires_discriminating_evidence', 'knowledge_verified',
              'root_cause_confirmed', 'working_hypothesis_is_confirmed',
              'score_semantics', 'calibrated_probability', 'control_actions_authorized')
    boundary = {k: diagnosis[k] for k in fields if k in diagnosis}
    boundary['projection_role'] = 'KERNEL_REPORT_NOT_INDEPENDENT_CONFIRMATION'
    items = []
    def add(label, value, pointer, required=False):
        items.append({'citation_id':root+':'+label, 'text':canonical(value).decode('utf-8'),
                      'core_artifact':'core-result.json', 'core_sha256':core_digest,
                      'source_json_pointer':pointer, 'source_role':'HOST_CORE_REPORT',
                      'required_for_grounding':required,
                      'diagnostic_vote_allowed':False})
    add('boundary', boundary, '/diagnosis', True)
    # Complete individual fields; never arbitrary character slices of an object.
    explanation = core.get('explanation', {})
    if isinstance(explanation, dict) and 'conclusion' in explanation:
        add('conclusion', {'conclusion':explanation['conclusion']}, '/explanation/conclusion')
    if 'claim_boundary' in diagnosis:
        add('claim-boundary', {'claim_boundary':diagnosis['claim_boundary']}, '/diagnosis/claim_boundary')
    for index, candidate in enumerate(diagnosis.get('candidate_explanations', [])):
        if not isinstance(candidate, dict):
            continue
        keys = ('fault_id','selected','knowledge_state','supporting_evidence_ids',
                'contradicting_evidence_ids','missing_required_definition_ids')
        add('candidate-'+str(index+1), {k:candidate[k] for k in keys if k in candidate},
            '/diagnosis/candidate_explanations/'+str(index))
    if exploration:
        for key in ('planning_trace','evidence_revisions'):
            values = core.get(key, [])
            if isinstance(values, list):
                for index, value in enumerate(values):
                    add(key+'-'+str(index+1), {key:value}, '/'+key+'/'+str(index))
        if 'stopping_reason' in core:
            add('stopping-reason', {'stopping_reason':core['stopping_reason']}, '/stopping_reason')
    return items


def visible_sources(payload: dict) -> list[dict]:
    return [source for key in ('sources_untrusted', 'retrieved_context_untrusted')
            for source in payload.get(key, []) if isinstance(source, dict) and source.get('text')]


def visible_ids(payload: dict) -> list[str]:
    return sorted({source['citation_id'] for source in visible_sources(payload)})
