"""R11-B: only frozen, same-task reviewed materials can enter LLM tooling."""
from __future__ import annotations
from copy import deepcopy

from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService,
    bind_frozen_material_review,
)
from astrograph.tools.material_review import TOOL_IDS, MaterialToolPlan
from astrograph.integrations.platform_knowledge import PlatformKnowledgeExplanation


def sample_context():
    return {"candidates": [
        {"fault_id":"PUMP_FAULT_A", "multi_hop":{"causal_chain_review":{"chains":[], "blocked":[]}}},
        {"fault_id":"PUMP_FAULT_B", "multi_hop":{"causal_chain_review":{"chains":[], "blocked":[]}}},
    ], "competing_hypothesis_review":{"status":"COMPARE_ONLY"}}


class TwoPhaseSpy:
    def __init__(self, actions):
        self.actions=actions; self.payloads=[]; self.last_call_metadata={}
    def generate_structured(self, *, agent_id, output_model, payload):
        self.payloads.append((agent_id,payload))
        if agent_id == "platform-knowledge-material-tool-selector":
            return MaterialToolPlan.model_validate({"actions":self.actions})
        return PlatformKnowledgeExplanation(candidate_fault_id=None,abstained=True,
                                            summary="Independent confirmation is unavailable.",claims=[])


def freeze(*, group='CASE-ONE'):
    return {'candidate_fault_id':'PUMP_FAULT_A',
            'candidate_fault_ids':['PUMP_FAULT_A','PUMP_FAULT_B'],
            'material_source_group_id':group,'evidence':[]}


def scoped(*, group='CASE-ONE'):
    data=sample_context();data['source_group_id']=group
    return data


def test_matching_source_and_candidate_set_is_approved():
    context,receipt=bind_frozen_material_review(freeze(),scoped())
    assert context['source_group_id']=='CASE-ONE'
    assert receipt['status']=='BOUND_FROZEN_DIAGNOSIS'
    assert receipt['source_group_checked'] is True
    assert receipt['diagnostic_vote_allowed'] is False


def test_other_run_cannot_launder_approved_material_into_model(tmp_path):
    spy=TwoPhaseSpy([{'tool_id':TOOL_IDS[1], 'fault_id':'PUMP_FAULT_A'}])
    service=PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(a2o_enabled=False, material_tool_calls_enabled=True),llm_backend=spy)
    output=service.enrich(task='Review frozen result', diagnosis_snapshot=freeze(),
             document_paths=[], output_root=tmp_path/'blocked', material_review_context=scoped(group='CASE-OTHER'))
    assert output['material_binding']['status']=='DENIED_TASK_BINDING'
    assert output['material_binding']['reason_code']=='MATERIAL_SOURCE_GROUP_MISMATCH'
    assert [agent for agent,_ in spy.payloads]==['platform-knowledge-explanation']
    model=spy.payloads[-1][1]['material_reasoning']
    assert model['reviewed_causal_chains'] is None
    assert model['candidates']==[]
    assert model['tool_investigation']['status']=='DENIED_TASK_BINDING'
    assert output['diagnostic_candidate_can_be_overridden'] is False


def test_other_fault_or_unbound_candidate_list_is_denied():
    wrong=deepcopy(scoped());wrong['candidates'][1]['fault_id']='FOREIGN-FAULT'
    _,receipt=bind_frozen_material_review(freeze(),wrong)
    assert receipt['reason_code']=='FROZEN_CANDIDATES_MISMATCH'
    older={'candidate_fault_id':'PUMP_FAULT_A','evidence':[]}
    _,receipt=bind_frozen_material_review(older,sample_context())
    assert receipt['reason_code']=='FROZEN_CANDIDATES_NOT_BOUND'


def test_public_case_without_review_keeps_model_in_single_call(tmp_path):
    spy=TwoPhaseSpy([{'tool_id':TOOL_IDS[0]}])
    service=PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(a2o_enabled=False,material_tool_calls_enabled=True),llm_backend=spy)
    output=service.enrich(task='Summarize uploaded material',
        diagnosis_snapshot={'candidate_fault_id':'CASE-A','evidence':[]},
        document_paths=[], output_root=tmp_path/'public-case')
    assert output['material_binding']['status']=='NOT_AVAILABLE_NO_APPROVED_MATERIAL_REVIEW'
    assert [agent for agent,_ in spy.payloads]==['platform-knowledge-explanation']
    assert spy.payloads[-1][1]['material_reasoning']['tool_investigation']['status']== 'NOT_AVAILABLE_NO_APPROVED_MATERIAL_REVIEW'


def test_source_group_missing_on_either_side_is_denied():
    old=scoped();old.pop('source_group_id')
    _,receipt=bind_frozen_material_review(freeze(),old)
    assert receipt['reason_code']=='MATERIAL_SOURCE_GROUP_MISMATCH'
    _,receipt=bind_frozen_material_review({'candidate_fault_id':'PUMP_FAULT_A',
      'candidate_fault_ids':['PUMP_FAULT_A','PUMP_FAULT_B'],'evidence':[]},scoped())
    assert receipt['reason_code']=='MATERIAL_SOURCE_GROUP_MISMATCH'
