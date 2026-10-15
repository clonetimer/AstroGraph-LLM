from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from astrograph.oag.window_features import derive_window_statistics
from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
from astrograph.simulation import SimulationRunCompleted
from astrograph.oag.native_bridge import write_package_graph
from astrograph.oag.contracts import OAGInput
from astrograph.oag.runtime import build_csv_oag_runtime, OAG_PIPELINE_ID
from astrograph.runtime import RuntimeRequest
from astrograph.data_driven.models import DataDrivenPrediction, ClassProbability

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'configs/datasets/adcs_v5_2_2/runtime_window_evidence.json'


def frames():
    columns = json.loads((PACKAGE.parent/'telemetry_projection.json').read_text(encoding="utf-8"))['canonical_columns']
    f = pd.DataFrame(0., index=range(20), columns=columns)
    f['time_s'] = np.arange(20, dtype=float)
    for j in range(4):
        f[f'adcs.rw.command_torque_nm_{j}'] = np.linspace(.001, .002, 20)
        f[f'adcs.control.applied_torque_nm_{j}'] = f[f'adcs.rw.command_torque_nm_{j}']
        f[f'adcs.rw.measured_speed_rad_s_{j}'] = np.linspace(15, 16, 20)
    return f.copy(), f


def bundle(source, nominal):
    s, n = derive_window_statistics(source, nominal)
    b, gate = ConfiguredRuntimeEvidenceBuilder(PACKAGE).build_from_frames(
        SimulationRunCompleted(event_id='EV-test',run_id='opaque',run_dir='opaque',
                               nominal_run_dir='reference',runtime_id='unit-test'),source=s,nominal=n)
    assert gate.passed
    return b.model_copy(update={'evaluation_metadata': {}})


def satisfied(b):
    return {e.target_fault_id for e in b.evidence_facts if e.status.value=='SATISFIED'}


@pytest.mark.parametrize('wheel', range(4))
@pytest.mark.parametrize('kind', ['jam', 'torque', 'bias'])
def test_physical_mechanisms_on_each_wheel(kind, wheel):
    s,n=frames()
    if kind=='jam':
        s[f'adcs.control.applied_torque_nm_{wheel}']=0.
        s[f'adcs.rw.measured_speed_rad_s_{wheel}']=.2
    elif kind=='torque':s[f'adcs.control.applied_torque_nm_{wheel}']*=.5
    else:s[f'adcs.rw.measured_speed_rad_s_{wheel}']+=20.
    targets={'jam':'ADCS_RW_JAM','torque':'ADCS_RW_TORQUE_AUTHORITY_LOSS','bias':'ADCS_RW_SPEED_SENSOR_BIAS'}
    assert satisfied(bundle(s,n))=={targets[kind]}


def test_nominal_shared_saturation_is_not_partial_authority_loss():
    s,n=frames()
    for j in range(4):
        s[f'adcs.control.applied_torque_nm_{j}']*=.5
        n[f'adcs.control.applied_torque_nm_{j}']*=.5
    assert not satisfied(bundle(s,n))


def test_no_excitation_or_missing_reference_cannot_establish_gain_fault():
    s,n=frames()
    for j in range(4):
        s[f'adcs.rw.command_torque_nm_{j}']=0.
        n[f'adcs.rw.command_torque_nm_{j}']=0.
    assert not satisfied(bundle(s,n))
    n.loc[3,'adcs.rw.measured_speed_rad_s_0']=np.nan
    with pytest.raises(ValueError,match='non-finite'):derive_window_statistics(s,n)


def test_pair_alignment_and_inputs_are_preserved():
    s,n=frames();before=s.copy(deep=True)
    derive_window_statistics(s,n)
    pd.testing.assert_frame_equal(s,before)
    n.time_s+=.001
    with pytest.raises(ValueError,match='time axes'):derive_window_statistics(s,n)


def execute(tmp_path, *, supported_ties, double=False):
    s,n=frames();s['adcs.rw.measured_speed_rad_s_0']+=20.
    if double:s['adcs.control.applied_torque_nm_1']*=.5
    b=bundle(s,n);faults=['ADCS_STAR_SOLUTION_BIAS','ADCS_RW_SPEED_SENSOR_BIAS','ADCS_RW_JAM','ADCS_RW_TORQUE_AUTHORITY_LOSS']
    pred=DataDrivenPrediction(prediction_id='p',episode_id=b.episode.episode_id,model_id='test',model_version='1',
        feature_schema_id='test',input_data_sha256='a'*64,
        classes=[ClassProbability(class_id=f,rank=i+1,probability=1. if i==0 else 0.) for i,f in enumerate(faults)])
    data=OAGInput(bundle=b,diagnostic_ranking=pred.to_fault_ranking(faults),registered_fault_ids=faults,
                  resolve_supported_ties=supported_ties)
    graph=write_package_graph(PACKAGE,tmp_path/'graph')
    result=build_csv_oag_runtime(graph).execute(RuntimeRequest(request_id='unit',pipeline_id=OAG_PIPELINE_ID,payload=data.model_dump(mode='json')))
    assert result.status.value=='SUCCEEDED'
    return result.output['summary']


def test_unique_supported_exact_tie_can_be_resolved_with_audit(tmp_path):
    s=execute(tmp_path,supported_ties=True)
    assert s['suspected_fault_id']=='ADCS_RW_SPEED_SENSOR_BIAS'
    assert s['tie_resolution']['ranking_modified'] is False
    assert len(s['tie_resolution']['tied_fault_ids'])==2
    assert s['evidence'][0]['conjunctive_observations']


def test_legacy_tie_policy_still_abstains(tmp_path):
    s=execute(tmp_path,supported_ties=False)
    assert s['suspected_fault_id'] is None and s['status']=='AMBIGUOUS'


def test_multiple_supported_tied_candidates_still_abstain(tmp_path):
    s=execute(tmp_path,supported_ties=True,double=True)
    assert s['suspected_fault_id'] is None
    assert s['tie_resolution'] is None


def test_evaluation_does_not_count_abstention_as_normal():
    from scripts.analyze_diagnostic_layers import layer_decisions, ABSTAIN, NORMAL, FAILED
    row={'status':'SUCCEEDED','summary':{'three_task_results':{'detection':{'detected':True},
         'classification':[{'fault_id':'MODEL_FAULT'}]},'candidates':[],'suspected_fault_id':None}}
    assert layer_decisions(row)['FUSION']==ABSTAIN
    assert layer_decisions({**row,'paired_reference_identical':True})['FUSION']==NORMAL
    assert layer_decisions({'status':'FAILED'})['FUSION']==FAILED


@pytest.mark.parametrize(('domain','profile','expected'), [
    ('adcs',None,'adcs_v5_2_2/runtime_window_integrated_evidence.json'),
    ('adcs','legacy','adcs_v5_2_2/runtime_evidence.json'),
    ('phm',None,'phm_v6_0/runtime_evidence.json'),
])
def test_native_cli_uses_same_profile_for_evidence_and_graph(tmp_path,monkeypatch,domain,profile,expected):
    from types import SimpleNamespace
    from astrograph.oag import cli,native_bridge
    from astrograph.integrations import adcs_three_task_torch,phm_three_task_torch
    for module,name in [(adcs_three_task_torch,'adcs'),(phm_three_task_torch,'phm')]:
        monkeypatch.setattr(module,f'load_{name}_three_task_torch_package',lambda *a,**kw:object())
        monkeypatch.setattr(module,f'build_registry_from_{name}_three_task_provider',lambda p:object())
    selected={}
    def bridge(**kwargs):
        selected['evidence']=kwargs['builder'].package_sha256
        return SimpleNamespace(model_dump=lambda **kw:{'prepared':True})
    def graph(path,output):
        selected['graph']=ConfiguredRuntimeEvidenceBuilder(path).package_sha256
    monkeypatch.setattr(native_bridge,'from_native_frames',bridge)
    monkeypatch.setattr(native_bridge,'write_package_graph',graph)
    source=tmp_path/'telemetry.csv';source.write_text('time_s\n0\n')
    model=tmp_path/'model.pt';model.write_bytes(b'test')
    args=['prepare-native','--domain',domain,'--source',str(source),'--nominal',str(source),
          '--model',str(model),'--project-root',str(ROOT),'--output',str(tmp_path/'input.json'),
          '--graph-output',str(tmp_path/'graph')]
    if profile:args+=['--evidence-profile',profile]
    assert cli.main(args)==0
    assert selected['evidence']==selected['graph']==ConfiguredRuntimeEvidenceBuilder(ROOT/'configs/datasets'/expected).package_sha256


def test_native_cli_rejects_rw_profile_for_phm_before_loading_inputs(tmp_path,capsys):
    from astrograph.oag.cli import main
    assert main(['prepare-native','--domain','phm','--evidence-profile','rw-window',
                 '--source','absent','--nominal','absent','--model','absent',
                 '--output',str(tmp_path/'input.json')])==2
    assert 'requires the ADCS domain' in capsys.readouterr().err


def test_integrated_profile_preserves_non_rw_conditions_exactly():
    old=json.loads((PACKAGE.parent/'runtime_evidence.json').read_text(encoding="utf-8"))
    new=json.loads((PACKAGE.parent/'runtime_window_integrated_evidence.json').read_text(encoding="utf-8"))
    expected=[d for d in old['evidence_definitions'] if not d['target_fault_id'].startswith('ADCS_RW_')]
    assert [d for d in new['evidence_definitions'] if not d['target_fault_id'].startswith('ADCS_RW_')]==expected
    ConfiguredRuntimeEvidenceBuilder(PACKAGE.parent/'runtime_window_integrated_evidence.json')


@pytest.mark.parametrize('double',[False,True])
def test_unsupported_top_cannot_hide_lower_ranked_supported_mechanisms(double):
    from types import SimpleNamespace as NS
    from astrograph.oag.summary import build_diagnostic_summary
    s,n=frames();s['adcs.rw.measured_speed_rad_s_0']+=20.
    if double:s['adcs.control.applied_torque_nm_1']*=.5
    b=bundle(s,n)
    ids=['UNSUPPORTED','ADCS_RW_SPEED_SENSOR_BIAS','ADCS_RW_TORQUE_AUTHORITY_LOSS']
    candidates=[NS(fault_id=f,final_rank=i+1,fused_score=1/(i+1),source_ranks={},
                   reciprocal_contributions={},dependency_adjusted=False,supporting_evidence_fact_ids=[])
                for i,f in enumerate(ids)]
    fused=NS(candidates=candidates,top_fault_id='UNSUPPORTED',blocking_conflicts=[],open_set_actions=[],quality_warnings=[],fusion_id='test')
    pkg=NS(episode_id=b.episode.episode_id,birrf_result=fused,fusion_id='test')
    q=NS(truncated=False,query_plan_id='test',rows=[NS(path_entity_ids=[f],path_relation_types=[]) for f in ids])
    result=build_diagnostic_summary(b,pkg,[q],top_k=1,resolve_unsupported_top=True)
    assert len(result['candidates'])==3
    if double:assert result['suspected_fault_id'] is None
    else:
        assert result['suspected_fault_id']=='ADCS_RW_SPEED_SENSOR_BIAS'
        assert result['tie_resolution']['selected_original_rank']==2
        assert result['tie_resolution']['ranking_modified'] is False


def test_report_prompt_distinguishes_rule_coverage_from_displayed_facts(tmp_path):
    from tests.test_design_diagnostic_harness import Backend,execute,prediction_input
    backend=Backend();data=prediction_input()
    result=execute(tmp_path,data,llm_backend=backend)
    assert result['status']=='SUCCEEDED'
    payload=backend.calls[0]
    assert payload['evidence_inventory']['evaluated_fact_count']==len(data.bundle.evidence_facts)
    assert payload['citation_ids_by_kind']['KNOWLEDGE']==[]
    assert set(payload['citation_ids_by_kind']['OBSERVATION'])=={e['evidence_id'] for e in payload['summary']['evidence']}
    assert 'MUST_NOT_REACH_PROMPT' not in json.dumps(payload)


def test_report_audit_flags_no_fault_text_despite_supported_decision():
    from astrograph.oag.contracts import OAGExplanation
    from astrograph.oag.report_audit import audit_report
    summary={'suspected_fault_id':'F','candidates':[],'evidence':[]}
    e=OAGExplanation(candidate_fault_id='F',abstained=False,claims=[{'kind':'HYPOTHESIS','text':'NO_REGISTERED_FAULT','citation_ids':[]}],limitations=['No causal identification.'])
    assert 'CLAIM_0_PRIMARY_CONCLUSION_CONTRADICTION' in audit_report(e,summary,[])['issues']


def test_report_audit_distinguishes_rank_scores_and_unmet_conditions():
    from astrograph.oag.contracts import OAGExplanation
    from astrograph.oag.report_audit import audit_report
    summary={'suspected_fault_id':None,'candidates':[{'missing_required_definition_ids':['ED-present-but-unmet']}],'evidence':[]}
    e=OAGExplanation(abstained=True,limitations=['BiRRF scores are uncalibrated probabilities.','Missing required evidence definitions.'])
    issues=audit_report(e,summary,[])['issues']
    assert 'LIMITATION_0_RANK_SCORE_MISDESCRIBED_AS_PROBABILITY' in issues
    assert 'LIMITATION_1_UNMET_CONDITION_MISDESCRIBED_AS_ABSENT_DEFINITION' in issues
    good=e.model_copy(update={'limitations':['BiRRF scores are reciprocal-rank scores, not probabilities.','Required conditions are not satisfied.']})
    assert not any('MISDESCRIBED' in i for i in audit_report(good,summary,[])['issues'])
