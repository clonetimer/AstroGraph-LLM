import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from astrograph.agents.mission.documents import clean_text,DocumentTools
from astrograph.agents.mission.models import AgentSettings,AgentRequest
from astrograph.agents.mission.storage import Workspace,file_sha
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.training import TrainingManifest,validate_dataset,train_candidate,infer_candidate
from astrograph.agents.mission.planner import Planner
from astrograph.oag.contracts import OntologyProposal

ROOT=Path(__file__).resolve().parents[3]

@pytest.mark.parametrize('text',['V < -0.05 V; not failed.','| torque | -1e-3 N m |','无故障；角速度不大于 0.01 rad/s。','$$ x = -a+b $$','a\n\na\n'])
def test_cleaner_preserves_semantics_and_duplicates(text):
    assert clean_text(text)['text']==text

def test_cleaner_records_modifications():
    out=clean_text('a  \r\nb\x00\r\n')
    assert out['text']=='a\nb\n' and out['changes'] and not out['semantic_rewrite_performed']

def docs(tmp_path,text):
    ws=Workspace(tmp_path/'runs',[tmp_path],'doc');ws.root.mkdir(parents=True)
    p=tmp_path/'a.md';p.write_text(text)
    tool=DocumentTools(ws,AgentSettings(read_roots=[str(tmp_path)]))
    tool.clean(tool.convert([str(p)])['documents'])
    return tool

def test_rag_sources_unique_and_not_diagnostic_votes(tmp_path):
    tool=docs(tmp_path,'# Gyro\nGyro bias requires residual verification.\n\nGyro bias requires residual verification.\n')
    result=tool.retrieve('gyro bias')
    assert result['retrieval_mode']=='BM25' and len(result['hits'])==1
    hit=result['hits'][0]
    assert hit['source_locator'] and hit['source_sha256'] and not hit['diagnostic_vote_allowed']
    assert hit['review_status']=='UNREVIEWED_CONTEXT'

def test_missing_dense_provider_explicitly_reports_lexical_fallback(tmp_path):
    tool=docs(tmp_path,'# Gyro\nGyro bias occurs.');tool.settings=tool.settings.model_copy(update={'embedding_model_path':str(tmp_path/'missing')})
    result=tool.retrieve('gyro')
    assert result['retrieval_mode']=='BM25' and result['warnings']

def test_extraction_is_source_quoted_review_only_graph(tmp_path):
    tool=docs(tmp_path,'Gyro bias affects attitude residual.')
    source=tool.documents[0]['document_id']
    class Backend:
        def generate_structured(self,**kw):return OntologyProposal(triples=[{'subject':'gyro_bias','predicate':'affects','object':'attitude_residual','source_document_id':source,'source_quote':'Gyro bias affects attitude residual.'}])
    tool.planner=Planner(Backend(),mode='llm_assisted')
    out=tool.build_graph()
    assert out['status']=='REVIEW_REQUIRED' and not out['active_kg_modified'] and out['edges']

@pytest.mark.parametrize('field,value',[('source_quote','invented observation'),('source_document_id','UNKNOWN')])
def test_unattributed_graph_relation_rejected(tmp_path,field,value):
    tool=docs(tmp_path,'Gyro bias affects attitude residual.')
    triple={'subject':'gyro_bias','predicate':'affects','object':'attitude_residual','source_document_id':tool.documents[0]['document_id'],'source_quote':'Gyro bias affects attitude residual.'};triple[field]=value
    class Backend:
        def generate_structured(self,**kw):return OntologyProposal(triples=[triple])
    tool.planner=Planner(Backend(),mode='llm_assisted')
    with pytest.raises(ValueError,match='quotation'):tool.build_graph()

def make_dataset(tmp_path, *, overlap=False):
    rng=np.random.default_rng(7);arrays={}
    for prefix,n in [('train',24),('validation',12)]:
        labels=np.arange(n)%3;a=(labels>0).astype('int64');f=labels.astype('int64')-1
        x=rng.normal(size=(n,8,2)).astype('float32')*.1
        x[labels==1,:,0]+=2;x[labels==2,:,1]-=2
        arrays.update({prefix+'_x':x,prefix+'_anomaly':a,prefix+'_fault':f,prefix+'_component':f.copy(),
                       prefix+'_group':np.array([('same' if overlap else prefix)+str(i//3) for i in range(n)])})
    path=tmp_path/'data.npz';np.savez(path,**arrays)
    manifest=TrainingManifest(data_path=str(path),data_sha256=file_sha(path),channel_names=['sensor.x','sensor.y'],
        channel_units=['1','1'],fault_ids=['F1','F2'],component_ids=['C1','C2'],fault_to_component={'F1':'C1','F2':'C2'},epochs=2,width=4,batch_size=12)
    mp=tmp_path/'manifest.json';mp.write_text(manifest.model_dump_json())
    return mp,manifest,arrays

def test_group_leakage_rejected(tmp_path):
    p,m,arrays=make_dataset(tmp_path,overlap=True)
    with pytest.raises(ValueError,match='leakage'):validate_dataset(m,Path(m.data_path))

def test_dataset_tampering_rejected(tmp_path):
    p,m,arrays=make_dataset(tmp_path)
    with Path(m.data_path).open('ab') as f:f.write(b'x')
    with pytest.raises(ValueError,match='hash mismatch'):validate_dataset(m,Path(m.data_path))

@pytest.mark.parametrize('channel',['truth.fault','label','target.class','fault_id'])
def test_target_channels_forbidden(tmp_path,channel):
    p,m,_=make_dataset(tmp_path)
    with pytest.raises(ValueError):TrainingManifest.model_validate({**m.model_dump(),'channel_names':[channel,'sensor.y']})

def test_training_and_candidate_inference_execute_real_torch(tmp_path):
    p,m,arrays=make_dataset(tmp_path)
    report=train_candidate(p,tmp_path/'model')
    assert report['weights_changed'] and report['group_overlap']==0 and not report['active_model_modified']
    assert set(report['validation_metrics'])=={'detection','localization','classification'}
    assert not report['test_partition_used']
    features=tmp_path/'features.npz';np.savez(features,x=arrays['validation_x'],channel_names=np.array(m.channel_names),channel_units=np.array(m.channel_units))
    pred=infer_candidate(tmp_path/'model/manifest.json',features)
    assert len(pred['heads']['detection']['probabilities'])==12
    assert pred['score_semantics']=='UNCALIBRATED_SOFTMAX'
    assert not pred['fusion_or_diagnostic_admission_performed']
    np.savez(features,x=arrays['validation_x'],channel_names=np.array(m.channel_names[::-1]),channel_units=np.array(m.channel_units))
    with pytest.raises(ValueError,match='channel order'):infer_candidate(tmp_path/'model/manifest.json',features)

def test_training_is_approval_bound_and_actual_worker_runs(tmp_path):
    p,m,_=make_dataset(tmp_path)
    req=AgentRequest(task_id='train',task='train_model',artifacts={'training_manifest':str(p)})
    s=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/'runs'))
    with pytest.raises(PermissionError,match='TRAINING_APPROVAL'):MissionAgent(s).run(req)
    s=s.model_copy(update={'approved_training_manifest_sha256':[file_sha(p)]})
    out=MissionAgent(s).run(req)
    assert out.status=='SUCCEEDED',out.model_dump()
    assert out.output['task_output']['weights_changed']
    assert not out.active_model_modified

def test_research_without_sources_cannot_report_diagnosis(tmp_path):
    s=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/'runs'))
    out=MissionAgent(s).run(AgentRequest(task_id='research',task='research'))
    assert out.status=='PARTIAL' and not out.output['task_output']['diagnostic_conclusion_generated']

def test_graph_builder_unconfigured_is_not_reported_as_success(tmp_path):
    p=tmp_path/'text.md';p.write_text('Gyro is a sensor.')
    s=AgentSettings(project_root=str(ROOT),read_roots=[str(tmp_path)],output_root=str(tmp_path/'runs'))
    out=MissionAgent(s).run(AgentRequest(task_id='build',task='build_knowledge',documents=[str(p)]))
    assert out.status=='FAILED' and 'ONTOLOGY_EXTRACTOR_NOT_CONFIGURED' in out.output['message']
