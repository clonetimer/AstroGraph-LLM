import pytest
from pydantic import ValidationError

from astrograph.diagnosis import build_structured_diagnosis_output


def test_final_structured_output_synthesizes_and_hashes_all_stages():
    payload = {
        'simulation_event': {'run_id': 'RUN-1'},
        'data_gate_report': {'passed': True},
        'episode_evidence_bundle': {
            'episode': {'episode_id': 'EP-1'},
            'evidence_facts': [
                {'evidence_fact_id': 'EF-1', 'status': 'SATISFIED'},
                {'evidence_fact_id': 'EF-2', 'status': 'CONTRADICTED'},
            ],
        },
        'soft_evidence': {
            'bundle_sha256': 'a' * 64, 'calibrated_fact_count': 2,
            'unavailable_fact_count': 1, 'calibration_policy_id': 'P-1',
            'calibration_policy_sha256': 'b' * 64,
        },
        'soft_graph_shadow': {
            'inference_id': 'SGI-1', 'result_sha256': 'c' * 64,
            'suggested_action': 'ABSTAIN', 'candidate_fault_ids': [],
            'reason_codes': ['CONFLICT'], 'runtime_effect_applied': False,
            'fault_assessments': [],
        },
        'candidate_fusion': {'birrf_result': {'top_fault_id': 'F-1', 'candidates': [], 'conflict_flags': []}},
        'document_claim_shadow': {
            'status': 'SHADOW_ZERO_VOTE', 'execution_mode': 'SHADOW_ONLY',
            'claim_trace_sha256': '8' * 64, 'mapping_bundle_sha256': '9' * 64,
            'candidate_fusion_modified': False,
            'reason_codes': ['NO_FORMALLY_ELIGIBLE_DOCUMENT_CLAIM_VOTES'],
            'ranking_build': {
                'ranking': {
                    'ranking_id': 'RANK-DOC-1', 'source': 'KG_FORWARD',
                    'candidates': [
                        {'fault_id': 'F-1', 'contributes_to_fusion': False}
                    ],
                    'kg_fault_coverage': {'supported_fault_ids': []},
                },
                'consumed_claim_ids': [], 'consumed_mapping_ids': [],
                'ignored_ineligible_claim_ids': ['CLAIM-1'],
            },
        },
        'pre_reasoning_retrieval': {
            'status': 'SUFFICIENT', 'package_id': 'R-1',
            'local_hits': [{
                'chunk_id': 'chunk:1', 'source_id': 'document:1',
                'source_uri': 'repo://document.md', 'source_sha256': '1' * 64,
                'chunk_sha256': '2' * 64, 'knowledge_version': 'v1',
                'rank': 1, 'lane': 'LOCAL_DOCUMENT', 'source_locators': ['lines:1-4'],
            }],
            'ontology_paths': [{
                'seed_concept_id': 'mechanism:m1', 'target_concept_id': 'effect:e1',
                'lane': 'ONTOLOGY', 'hops': [{
                    'relation_id': 'edge:1', 'source_concept_id': 'mechanism:m1',
                    'relation_type': 'MANIFESTS_AS', 'target_concept_id': 'effect:e1',
                    'source_id': 'source:tbox', 'source_sha256': '3' * 64,
                    'knowledge_version': 'graph:v1',
                }],
            }],
            'ontology_seed_concept_ids': ['mechanism:m1'],
            'ontology_asset_audits': [{
                'retrieval_only': True, 'diagnostic_evidence_allowed': False,
                'runtime_vote_allowed': False, 'active_graph_modified': False,
            }],
            'governance_filtered_chunk_count': 2,
            'document_access_reason_counts': {'DESIGN_HUMAN_REVIEW_REQUIRED': 2},
            'experience_hit_count': 1,
            'experience_source_ids': ['experience:E-1'],
            'knowledge_update_candidates': [{'active_graph_modified': False}],
            'active_graph_modified': False,
        },
        'hypothesis_set': {'hypotheses': [{'hypothesis_id': 'H-1', 'target_fault_id': 'F-1', 'status': 'SUPPORTED'}]},
        'critique_result': {'blocking': False, 'findings': []},
        'deterministic_core_result': {'status': 'PASS'},
        'decision_record': {'episode_id': 'EP-1', 'disposition': 'ACCEPTED', 'accepted_fault_id': 'F-1'},
        'diagnosis_policy_decision': {'status': 'WITHHELD', 'blocking_reasons': ['E0'], 'warnings': []},
        'verification_plan': {'verification_plan_id': 'VP-1', 'tasks': [{}]},
        'episode_evidence_graph': {'nodes': [], 'edges': []},
        'dataset_profile_receipt': {'profile_id': 'PROFILE-1'},
        'diagnosis_execution_plan': {'enabled_branches': ['KG_FORWARD']},
        'component_spatiotemporal_fault_graph': {'graph_id': 'STG-1'},
        'spatiotemporal_template_matches': [{'template_id': 'T-1'}],
        'paperseek_fallback_decision': {'status': 'NOT_REQUIRED'},
        'agent_trace': {'trace_id': 'TRACE-1'},
        'reasoner_mode': 'llm',
        'model_id': 'MODEL-1',
        'workflow_version': 'WF-1',
        'llm_runtime_snapshot': {'status': 'LLM_BACKEND'},
        'runtime_stage_artifacts': {
            'online.evidence.v1': {
                'content_sha256': 'd' * 64,
                'semantic_sha256': 'e' * 64,
            },
        },
    }
    result = build_structured_diagnosis_output(payload)
    assert result.final_decision['accepted_fault_id'] == 'F-1'
    assert result.output_sha256 == result.compute_sha256()
    assert result.evidence_summary['fact_status_counts'] == {'SATISFIED': 1, 'CONTRADICTED': 1}
    assert result.knowledge_and_retrieval_summary['experience_hit_count'] == 1
    assert result.knowledge_and_retrieval_summary['local_hit_references'][0][
        'chunk_sha256'
    ] == '2' * 64
    assert result.knowledge_and_retrieval_summary['ontology_path_references'][0][
        'hops'
    ][0]['relation_type'] == 'MANIFESTS_AS'
    assert result.knowledge_and_retrieval_summary['ontology_seed_concept_ids'] == [
        'mechanism:m1'
    ]
    assert result.knowledge_and_retrieval_summary['diagnostic_effect_claimed'] is False
    document_shadow = result.knowledge_and_retrieval_summary[
        'document_graph_diagnosis_shadow'
    ]
    assert document_shadow['status'] == 'SHADOW_ZERO_VOTE'
    assert document_shadow['ignored_ineligible_claim_count'] == 1
    assert document_shadow['contributing_fault_ids'] == []
    assert document_shadow['candidate_fusion_modified'] is False
    assert result.knowledge_and_retrieval_summary['path_use_attribution'] == (
        'CONTEXT_AVAILABLE_NOT_CAUSAL_ATTRIBUTION'
    )
    assert 'snippet' not in result.knowledge_and_retrieval_summary[
        'local_hit_references'
    ][0]
    assert result.safety_and_release_summary['policy_status'] == 'WITHHELD'
    assert result.uncertainty_and_soft_graph_summary['suggested_action'] == 'ABSTAIN'
    assert result.uncertainty_and_soft_graph_summary['runtime_effect_applied'] is False
    assert result.verification_summary['task_count'] == 1
    assert 'decision_record' in result.lineage_sha256
    assert 'soft_evidence' in result.lineage_sha256
    assert 'soft_graph_shadow' in result.lineage_sha256
    assert 'document_claim_shadow' in result.lineage_sha256
    assert result.execution_context_summary['dataset_profile_receipt']['profile_id'] == 'PROFILE-1'
    assert result.spatiotemporal_summary['component_spatiotemporal_fault_graph']['graph_id'] == 'STG-1'
    assert result.trace_and_artifact_summary['runtime_stage_artifact_sha256'][
        'online.evidence.v1'
    ]['content_sha256'] == 'd' * 64
    for key in (
        'dataset_profile_receipt', 'diagnosis_execution_plan',
        'component_spatiotemporal_fault_graph', 'spatiotemporal_template_matches',
        'paperseek_fallback_decision', 'agent_trace', 'llm_runtime_snapshot',
    ):
        assert key in result.lineage_sha256

    tampered = result.model_dump(mode='json')
    tampered['final_decision']['accepted_fault_id'] = 'F-TAMPERED'
    with pytest.raises(ValidationError, match='output_sha256 mismatch'):
        type(result).model_validate(tampered)


def test_structured_output_rejects_noncontiguous_ontology_trace():
    payload = {
        'simulation_event': {'run_id': 'RUN-1'},
        'data_gate_report': {'passed': True},
        'episode_evidence_bundle': {'episode': {'episode_id': 'EP-1'}, 'evidence_facts': []},
        'hypothesis_set': {'hypotheses': []},
        'decision_record': {'episode_id': 'EP-1', 'disposition': 'WITHHELD'},
        'pre_reasoning_retrieval': {
            'ontology_paths': [{
                'seed_concept_id': 'mechanism:m1', 'target_concept_id': 'effect:e1',
                'hops': [{
                    'source_concept_id': 'mechanism:wrong',
                    'target_concept_id': 'effect:e1',
                }],
            }],
        },
    }
    with pytest.raises(ValueError, match='not contiguous'):
        build_structured_diagnosis_output(payload)


def test_structured_output_rejects_document_shadow_runtime_effect():
    payload = {
        'simulation_event': {'run_id': 'RUN-1'},
        'data_gate_report': {'passed': True},
        'episode_evidence_bundle': {
            'episode': {'episode_id': 'EP-1'}, 'evidence_facts': []
        },
        'hypothesis_set': {'hypotheses': []},
        'decision_record': {'episode_id': 'EP-1', 'disposition': 'WITHHELD'},
        'document_claim_shadow': {
            'status': 'SHADOW_RANKING_READY',
            'execution_mode': 'SHADOW_ONLY',
            'candidate_fusion_modified': True,
        },
    }
    with pytest.raises(ValueError, match='must not modify candidate fusion'):
        build_structured_diagnosis_output(payload)
