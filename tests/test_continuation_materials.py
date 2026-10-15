from pathlib import Path
import csv,json,sys
import pytest
from astrograph.training.review_pack import ReviewDraft,write_review_pack,export_reviewed
from astrograph.oag.contracts import content_hash,OAGInput
from astrograph.oag.three_task import EvidenceProjectionReceipt
from astrograph.integrations.platform_knowledge import Anything2OntologyCLIAdapter,PlatformKnowledgeConfig
from tests.test_design_diagnostic_harness import prediction_input


def draft():
    return ReviewDraft(example_id='D1',task='RETRIEVAL_QUERY',source_group='source-one',
        source_artifact_sha256='1'*64,source_url='https://example.org/source',split='train',
        input={'source_ids':['S1'],'symptom':'wheel speed decreases'},
        candidate_output={'query':'wheel speed decreases','source_ids':['S1']},candidate_author='model-draft')


def reviewed_file(path,change):
    with path.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f));fields=list(r[0])
    change(r)
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r)


def test_drafts_are_not_training_approvals(tmp_path):
    write_review_pack([draft()],tmp_path/'pack')
    with pytest.raises(ValueError,match='no approved'):export_reviewed(tmp_path/'pack/pack.json',tmp_path/'pack/review.csv',tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_review_export_validates_source_and_named_correction(tmp_path):
    write_review_pack([draft()],tmp_path/'pack');csvpath=tmp_path/'pack/review.csv'
    reviewed_file(csvpath,lambda rows:rows[0].update(review_status='APPROVED',reviewer='test-reviewer',review_comment='Test fixture review'))
    receipt=export_reviewed(tmp_path/'pack/pack.json',csvpath,tmp_path/'approved');assert receipt['approved_examples']==1
    reviewed_file(csvpath,lambda rows:rows[0].update(corrected_output_json=json.dumps({'query':'invented','source_ids':['UNKNOWN']})))
    with pytest.raises(ValueError):export_reviewed(tmp_path/'pack/pack.json',csvpath,tmp_path/'invalid')


@pytest.mark.parametrize('field,value',[('frozen_split','test'),('input_sha256','0'*64),('pack_sha256','0'*64),('reviewer',''),('review_comment','')])
def test_review_cannot_change_identity_or_omit_accountability(tmp_path,field,value):
    write_review_pack([draft()],tmp_path/'pack');p=tmp_path/'pack/review.csv'
    reviewed_file(p,lambda rows:rows[0].update(review_status='APPROVED',reviewer='test-reviewer',review_comment='Test fixture',**{field:value}) if field not in {'reviewer','review_comment'} else rows[0].update({'review_status':'APPROVED','reviewer':'test-reviewer','review_comment':'Test fixture',field:value}))
    with pytest.raises(ValueError):export_reviewed(tmp_path/'pack/pack.json',p,tmp_path/'out')


def test_review_pack_rejects_source_alias_split_leak(tmp_path):
    x=draft();y=x.model_copy(update={'example_id':'D2','source_group':'renamed','split':'test'})
    with pytest.raises(ValueError,match='cross'):write_review_pack([x,y],tmp_path/'pack')


def test_a2o_explicit_python_is_pinned(tmp_path,monkeypatch):
    source=tmp_path/'upstream';(source/'src/anything2markdown').mkdir(parents=True);(source/'src/anything2markdown/cli.py').write_text('')
    adapter=Anything2OntologyCLIAdapter(source_root=source,python_executable=sys.executable)
    command,mode=adapter._resolve_stage('anything2md','anything2markdown.cli',['run'])
    assert command[0]==str(Path(sys.executable).absolute()) and mode=='SOURCE_TREE_MODULE'
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_A2O_PYTHON',sys.executable)
    assert PlatformKnowledgeConfig.from_environment().a2o_python_executable==sys.executable
    with pytest.raises(ValueError):Anything2OntologyCLIAdapter(python_executable=tmp_path/'missing')


def test_native_projection_preserves_distinct_hashes_and_detects_tampering():
    data=prediction_input();body=data.model_dump(mode='json');heads=body['three_task_result'];heads['input_data_sha256']='9'*64
    heads['evidence_projection']=EvidenceProjectionReceipt(source_window_sha256='1'*64,nominal_window_sha256='2'*64,
        model_input_sha256='9'*64,evidence_bundle_sha256=content_hash(body['bundle']),
        feature_input_hashes={f.feature_result_id:f.input_data_sha256 for f in data.bundle.feature_results},
        model_artifact_sha256='3'*64,evidence_package_sha256='4'*64,
        window_start_s=heads['window_start_s'],window_end_s=heads['window_end_s']).model_dump(mode='json')
    valid=OAGInput.model_validate(body);assert valid.three_task_result.input_data_sha256=='9'*64
    body['three_task_result']['evidence_projection']['feature_input_hashes']={}
    with pytest.raises(ValueError,match='feature input hashes'):OAGInput.model_validate(body)


def test_registered_model_faults_without_rule_coverage_remain_unexplained(tmp_path):
    from astrograph.oag.harness import execute_harness
    from tests.test_design_diagnostic_harness import GRAPH,Backend
    data=prediction_input();body=data.model_dump(mode='json');heads=body['three_task_result']
    ranking=body['diagnostic_ranking'];candidate=dict(ranking['candidates'][0]);candidate.update(fault_id='MODEL_ONLY_FAULT',rank=len(ranking['candidates'])+1,score=.01)
    ranking['candidates'].append(candidate);heads['classification_ranking']=ranking
    body['registered_fault_ids']=sorted({d.target_fault_id for d in data.bundle.evidence_definitions}|{'MODEL_ONLY_FAULT'})
    data=OAGInput.model_validate(body)
    r=execute_harness(data,graph_dir=GRAPH,output_root=tmp_path,request_id='model-only',llm_backend=Backend())
    assert r['status']=='SUCCEEDED'
    assert 'MODEL_ONLY_FAULT' in {c['fault_id'] for c in r['output']['summary']['three_task_results']['classification']}
    for c in r['output']['summary']['candidates']:
        if c['fault_id']=='MODEL_ONLY_FAULT':assert not c['eligible_for_explanation']


def test_native_package_graph_uses_only_configured_definitions(tmp_path):
    from astrograph.oag.native_bridge import write_package_graph
    from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
    root=Path(__file__).resolve().parents[1];package=root/'configs/datasets/adcs_v5_2_2/runtime_evidence.json'
    graph=write_package_graph(package,tmp_path/'graph');builder=ConfiguredRuntimeEvidenceBuilder(package)
    with (graph/'edges.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(builder.package.evidence_definitions)
    assert {x['target_id'] for x in rows}=={d.evidence_definition_id for d in builder.package.evidence_definitions}
    assert json.loads((graph/'provenance.json').read_text(encoding="utf-8"))['active_kg_modified'] is False
    with pytest.raises(FileExistsError):write_package_graph(package,graph)


def test_external_ontology_context_is_review_only_and_replay_bound(tmp_path):
    from astrograph.oag.harness import execute_harness
    from tests.test_design_diagnostic_harness import GRAPH,Backend
    ctx={'status':'REVIEW_REQUIRED','active_kg_modified':False,'candidate_material':[]}
    r=execute_harness(prediction_input(),graph_dir=GRAPH,output_root=tmp_path,request_id='a2o-context',llm_backend=Backend(),ontology_context=ctx)
    assert r['status']=='SUCCEEDED' and r['output']['ontology']['toolchain']==ctx
    with pytest.raises(ValueError,match='different inputs or configuration'):
        execute_harness(prediction_input(),graph_dir=GRAPH,output_root=tmp_path,request_id='a2o-context',llm_backend=Backend(),ontology_context={**ctx,'receipt_sha256':'new'})
    invalid=execute_harness(prediction_input(),graph_dir=GRAPH,output_root=tmp_path,request_id='a2o-invalid',llm_backend=Backend(),ontology_context={**ctx,'active_kg_modified':True})
    assert invalid['status']=='FAILED'
