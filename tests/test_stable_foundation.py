import json
import pytest
from astrograph.oag.contracts import content_hash
from astrograph.oag.timeline import build_evidence_timeline
from astrograph.training.source_tools import SourceTaskRequest,source_target,checked_source_result


def request():
    return SourceTaskRequest(task='ONTOLOGY_EXTRACTION',component='wheel',document_id='D1',
        text="ParameterEffect('torque', 'multiply', 0.25, 'motor failure reduces authority')")


def test_checked_adapter_rejects_hallucinations_and_exposes_fallback():
    r=request();answer=source_target(r)
    good=checked_source_result(r,json.dumps(answer));assert good['model_output_accepted']
    wrong=json.loads(json.dumps(answer));wrong['triples'][0]['object']='invented'
    bad=checked_source_result(r,json.dumps(wrong))
    assert bad['fallback_used'] and bad['output']==answer
    assert not bad['diagnostic_vote_allowed'] and not bad['active_kg_modified']


@pytest.mark.parametrize('text',["print('secret')", "ParameterEffect('x','add',dangerous(),'y')", "ParameterEffect('x','unknown',1,'y')"])
def test_source_scope_never_executes_or_accepts_unknown_operations(text):
    with pytest.raises(ValueError):source_target(request().model_copy(update={'text':text}))


def window(end,fault=None,detected=False):
    summary={'window':{'start_s':end-19,'end_s':end},'component':'ADCS','graph_identity':{'id':'frozen'},
        'suspected_fault_id':fault,'three_task_results':{'detection':{'detected':detected}}}
    summary['summary_sha256']=content_hash(summary)
    return {'endpoint_s':end,'status':'SUCCEEDED','summary':summary}


def test_timeline_retains_old_fault_without_relabeling_current_or_dropping_gaps():
    result=build_evidence_timeline([window(20,'JAM',True),{'endpoint_s':40,'status':'FAILED','error':'missing'},window(60,None,True)])
    assert result['scheduled_windows']==3 and result['unavailable_windows']==1
    assert result['current']['candidate'] is None
    assert result['current']['state']=='UNCONFIRMED_MODEL_ALERT'
    assert result['historical_candidates'][0]['fault_id']=='JAM'
    assert result['historical_candidates'][0]['last_supported_s']==20
    assert result['additional_fusion_votes']==0


@pytest.mark.parametrize('kind',['hash','order','future','identity'])
def test_timeline_rejects_unbound_or_mixed_evidence(kind):
    rows=[window(20),window(40)]
    if kind=='hash':rows[1]['summary']['suspected_fault_id']='FAKE'
    if kind=='order':rows.reverse()
    if kind in {'future','identity'}:
        s=rows[1]['summary'];s.pop('summary_sha256')
        if kind=='future':s['window']['end_s']=50
        else:s['component']='EPS'
        s['summary_sha256']=content_hash(s)
    with pytest.raises(ValueError):build_evidence_timeline(rows)


def test_report_audit_catches_valid_citation_with_unsupported_certainty():
    from astrograph.oag.report_audit import audit_report
    from astrograph.oag.assessment import assess_summary
    from astrograph.oag.contracts import OAGExplanation
    s={'suspected_fault_id':None,'status':'INSUFFICIENT_EVIDENCE','candidates':[],'evidence':[{'evidence_id':'E1'}]}
    e=OAGExplanation(abstained=True,assessment=assess_summary(s),limitations=['Simulation only'],claims=[{'kind':'OBSERVATION','text':'已经确诊故障','citation_ids':['E1']}])
    audit=audit_report(e,s,[])
    assert audit['status']=='FLAGGED' and 'CLAIM_0_UNSUPPORTED_CERTAINTY' in audit['issues']


def test_registered_superset_projection_preserves_values_and_rejects_unknown_layout():
    from pathlib import Path
    import pandas as pd
    from astrograph.oag.native_bridge import project_registered_telemetry
    r=Path(__file__).resolve().parents[1]
    c=json.loads((r/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text(encoding="utf-8"))
    # Exact original schema passes through without inventing values.
    frame=pd.DataFrame([{name:float('nan') for name in c['canonical_columns']}])
    assert project_registered_telemetry(frame,schema_sha256=c['canonical_sha256']).equals(frame)
    frame['evaluation_fault_label']=0
    with pytest.raises(ValueError,match='unregistered'):project_registered_telemetry(frame,schema_sha256=c['canonical_sha256'])


def test_uploaded_document_section_title_is_searchable(tmp_path):
    from astrograph.integrations.platform_knowledge import UploadedDocumentKnowledgeIndex
    p=tmp_path/'material.md';p.write_text('# ADCS_RW_TORQUE_AUTHORITY_LOSS\n\nScaling actuator command by a fixed factor.\n')
    index=UploadedDocumentKnowledgeIndex();chunks,_=index.build([p])
    assert index.retrieve('ADCS_RW_TORQUE_AUTHORITY_LOSS',chunks)


def test_identical_paired_telemetry_cannot_publish_supported_anomaly(tmp_path):
    from tests.test_design_diagnostic_harness import prediction_input,GRAPH
    from astrograph.oag.contracts import OAGInput
    from astrograph.oag.three_task import EvidenceProjectionReceipt
    from astrograph.oag.harness import execute_harness
    d=prediction_input();body=d.model_dump(mode='json');heads=body['three_task_result']
    heads['evidence_projection']=EvidenceProjectionReceipt(source_window_sha256='1'*64,nominal_window_sha256='1'*64,
        model_input_sha256=heads['input_data_sha256'],evidence_bundle_sha256=content_hash(body['bundle']),
        feature_input_hashes={f.feature_result_id:f.input_data_sha256 for f in d.bundle.feature_results},
        model_artifact_sha256='3'*64,evidence_package_sha256='4'*64,
        window_start_s=heads['window_start_s'],window_end_s=heads['window_end_s']).model_dump(mode='json')
    data=OAGInput.model_validate(body)
    result=execute_harness(data,graph_dir=GRAPH,output_root=tmp_path,request_id='identical-pair')
    assert result['status']=='SUCCEEDED'
    assert result['output']['summary']['suspected_fault_id'] is None
    assert 'IDENTICAL_PAIRED_TELEMETRY_NO_DEVIATION' in result['output']['summary']['blocking_conflicts']


def test_review_assessment_is_required_in_the_model_schema():
    from astrograph.oag.contracts import ReviewedOAGExplanation
    assert 'assessment' in ReviewedOAGExplanation.model_json_schema()['required']
    with pytest.raises(ValueError):ReviewedOAGExplanation(abstained=True)


def test_harness_does_not_replay_under_changed_implementation(tmp_path,monkeypatch):
    import astrograph.oag.harness as h
    from tests.test_design_diagnostic_harness import prediction_input,GRAPH
    data=prediction_input()
    h.execute_harness(data,graph_dir=GRAPH,output_root=tmp_path,request_id='code-identity')
    monkeypatch.setattr(h,'implementation_sha256',lambda:'changed-code')
    with pytest.raises(ValueError):h.execute_harness(data,graph_dir=GRAPH,output_root=tmp_path,request_id='code-identity')


def test_report_audit_flags_text_negating_the_selected_candidate():
    from astrograph.oag.report_audit import audit_report
    from astrograph.oag.assessment import assess_summary
    from astrograph.oag.contracts import OAGExplanation
    s={'suspected_fault_id':'FAULT','status':'READY_FOR_EXPLANATION','candidates':[],'evidence':[]}
    e=OAGExplanation(candidate_fault_id='FAULT',abstained=False,assessment=assess_summary(s),limitations=['No causal identification'],claims=[{'kind':'HYPOTHESIS','text':'No registered fault hypothesis is supported by sufficient evidence to be selected as the primary conclusion.','citation_ids':[]}])
    assert 'CLAIM_0_PRIMARY_CONCLUSION_CONTRADICTION' in audit_report(e,s,[])['issues']
