import pytest
from astrograph.oag.kernel import EvidenceDiagnosisKernel
from astrograph.oag.contracts import OAGInput
from astrograph.oag.runtime import build_csv_oag_runtime, OAG_PIPELINE_ID
from astrograph.runtime import RuntimeRequest
from astrograph.fusion.decision import FusionDecisionService


def test_observed_counter_can_correct_data_top(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=2)
    r=EvidenceDiagnosisKernel(graph).evaluate(data)
    assert r['diagnosis']['selected_fault_id']=='B'
    assert r['diagnosis']['status']=='SUPPORTED_SINGLETON'
    assert r['diagnosis']['decision_changed_data_top']
    assert 'DATA_TOP_REJECTED_BY_OBSERVED_COUNTEREVIDENCE' in r['diagnosis']['reason_codes']
    a=next(c for c in r['diagnosis']['candidate_explanations'] if c['fault_id']=='A')
    assert a['contradicting_evidence_ids'] and a['source_ranks']


def test_knowledge_does_not_delete_an_uncovered_data_top(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=0,data_ids=('UNCOVERED','B'))
    d=EvidenceDiagnosisKernel(graph).evaluate(data)['diagnosis']
    assert d['selected_fault_id'] is None
    assert set(d['candidate_fault_ids'])=={'UNCOVERED','B'}


def test_no_positive_knowledge_keeps_model_only_not_fake_support(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=0,counter=0,data_ids=('UNCOVERED','B'))
    d=EvidenceDiagnosisKernel(graph).evaluate(data)['diagnosis']
    assert d['status']=='MODEL_ONLY' and d['selected_fault_id']=='UNCOVERED'
    assert not d['knowledge_verified'] and d['calibrated_probability'] is None
    assert 'KG_COVERAGE_ABSENT_NOT_NEGATIVE_EVIDENCE' in d['reason_codes']


def test_unresolved_disagreement_not_forced_into_singleton(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=0)
    d=EvidenceDiagnosisKernel(graph).evaluate(data)['diagnosis']
    assert d['status']=='CANDIDATE_SET' and d['selected_fault_id'] is None
    assert set(d['candidate_fault_ids'])=={'A','B'}


def test_knowledge_branch_can_supply_candidate_without_model(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=0,data_ids=())
    d=EvidenceDiagnosisKernel(graph).evaluate(data)['diagnosis']
    assert d['selected_fault_id']=='B'


def test_no_sources_is_not_normal(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=None,b=None,counter=None,data_ids=())
    d=EvidenceDiagnosisKernel(graph).evaluate(data)['diagnosis']
    assert d['status']=='INSUFFICIENT_EVIDENCE' and d['selected_fault_id'] is None


def test_complete_candidate_table_is_required(make_diagnostic_case):
    from astrograph.fusion import CandidateFusionPackage
    data,graph=make_diagnostic_case(a=0,b=2,counter=0)
    result=EvidenceDiagnosisKernel(graph).evaluate(data)
    package=CandidateFusionPackage.model_validate(result['fusion'])
    summary={**result['summary'],'candidates':result['summary']['candidates'][:1]}
    with pytest.raises(ValueError,match='complete'):
        FusionDecisionService().decide(bundle=data.bundle,package=package,summary=summary)


def test_new_policy_is_wired_into_existing_oag_runtime(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=2)
    runtime=build_csv_oag_runtime(graph)
    out=runtime.execute(RuntimeRequest(request_id='r',pipeline_id=OAG_PIPELINE_ID,payload=data.model_dump(mode='json')))
    assert out.status.value=='SUCCEEDED',out.error
    assert out.output['summary']['diagnostic_decision']['selected_fault_id']=='B'
    assert out.output['fusion']['metadata']['fusion_controls_llm_candidate_order_only'] is False


def test_legacy_policy_is_still_available(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=2)
    legacy=OAGInput.model_validate({**data.model_dump(),'decision_policy':'legacy'})
    out=build_csv_oag_runtime(graph).execute(RuntimeRequest(request_id='r',pipeline_id=OAG_PIPELINE_ID,payload=legacy.model_dump(mode='json')))
    assert out.status.value=='SUCCEEDED'
    assert 'diagnostic_decision' not in out.output['summary']
    assert out.output['fusion']['metadata']['fusion_controls_llm_candidate_order_only'] is True


def test_detector_disagreement_not_silently_overridden(make_diagnostic_case):
    from astrograph.fusion import CandidateFusionPackage
    data,graph=make_diagnostic_case(a=0,b=2,counter=2)
    r=EvidenceDiagnosisKernel(graph).evaluate(data)
    d=FusionDecisionService().decide(bundle=data.bundle,package=CandidateFusionPackage.model_validate(r['fusion']),summary=r['summary'],detected=False)
    assert d['status']=='CANDIDATE_SET' and d['selected_fault_id'] is None
    assert d['detector_mechanism_disagreement']


def test_no_detector_trigger_is_not_independent_health_confirmation(make_diagnostic_case):
    from astrograph.fusion import CandidateFusionPackage
    data,graph=make_diagnostic_case(a=0,b=0,counter=0)
    r=EvidenceDiagnosisKernel(graph).evaluate(data)
    d=FusionDecisionService().decide(bundle=data.bundle,package=CandidateFusionPackage.model_validate(r['fusion']),summary=r['summary'],detected=False)
    assert d['status']=='NO_DETECTED_ANOMALY'
    assert not d['root_cause_confirmed']


def test_knowledge_snapshot_change_is_rejected(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=2)
    kernel=EvidenceDiagnosisKernel(graph)
    with (graph/'edges.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='release changed'):kernel.evaluate(data)


def test_absent_entire_executable_knowledge_keeps_data_branch(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=0,counter=0)
    d=data.model_dump();d['bundle']['evidence_definitions']=[];d['bundle']['evidence_facts']=[]
    out=EvidenceDiagnosisKernel(graph).evaluate(OAGInput.model_validate(d))
    assert out['diagnosis']['status']=='MODEL_ONLY'
    assert out['diagnosis']['selected_fault_id']=='A'


@pytest.mark.parametrize('change',[{'window_end_s':3},{'window_start_s':-0.1},{'unit':'A'},{'episode_id':'another'}])
def test_unadmitted_initial_feature_context_is_rejected(make_diagnostic_case,change):
    data,graph=make_diagnostic_case(a=2,b=0,counter=0)
    raw=data.model_dump();raw['bundle']['feature_results'][0].update(change)
    with pytest.raises(ValueError):EvidenceDiagnosisKernel(graph).evaluate(OAGInput.model_validate(raw))


def test_fabricated_satisfied_status_is_not_trusted(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=0,counter=0)
    raw=data.model_dump();raw['bundle']['evidence_facts'][0]['status']='SATISFIED'
    with pytest.raises(ValueError,match='executable condition'):
        EvidenceDiagnosisKernel(graph).evaluate(OAGInput.model_validate(raw))


def test_modified_fact_threshold_is_not_trusted(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=2,b=0,counter=0)
    raw=data.model_dump();raw['bundle']['evidence_facts'][0]['threshold_value']=0
    with pytest.raises(ValueError,match='threshold'):
        EvidenceDiagnosisKernel(graph).evaluate(OAGInput.model_validate(raw))


def test_working_preference_does_not_claim_verified_singleton(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=2,counter=0)
    result=EvidenceDiagnosisKernel(graph).evaluate(data)
    d=result['diagnosis']
    assert d['selected_fault_id'] is None and d['status']=='CANDIDATE_SET'
    assert d['working_hypothesis_fault_id']=='A'
    assert not d['working_hypothesis_is_confirmed'] and not d['knowledge_verified']
    assert d['requires_discriminating_evidence']
    assert set(d['candidate_fault_ids'])=={'A','B'}
    assert '不是知识已确认' in result['explanation']['conclusion']


def test_existing_runtime_also_rejects_fabricated_status(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=0,counter=0)
    raw=data.model_dump();raw['bundle']['evidence_facts'][0]['status']='SATISFIED'
    out=build_csv_oag_runtime(graph).execute(RuntimeRequest(request_id='bad-fact',pipeline_id=OAG_PIPELINE_ID,payload=raw))
    assert out.status.value=='FAILED'
