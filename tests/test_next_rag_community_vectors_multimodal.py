"""Acceptance guards for source-grounded global retrieval and optional modalities."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from astrograph.integrations.platform_knowledge import _LocalChunk, UploadedDocumentKnowledgeIndex
from astrograph.retrieval.approved_community_search import (
    build_approved_community_reports,global_community_search,
)
from astrograph.retrieval.persistent_vectors import (
    PersistentVectorCache, identity, semantic_rank_with_cache, model_revision_sha256,
)
from astrograph.retrieval.advanced_multimodal import (
    describe_diagram_local, ocr_scanned_pdf,
)

SCOPE = 'SAT-X-ADCS-V1'
SHA = 'a'*64


def edge(*, scope=SCOPE, polarity='AFFIRMATIVE', reviewer='expert',
         text='轮速增高伴随姿控误差', predicate='MANIFESTS_AS'):
    return {
        'proposal_id':'MGR-REFERENCE-001', 'source_group_id':'verified-source',
        'equipment_scope':scope,
        'review_status':'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
        'source':{'entity_id':'MECH-1','name':'飞轮摩擦增大','entity_type':'FAILURE_MECHANISM'},
        'target':{'entity_id':'SYM-1','name':'轮速增高','entity_type':'SYMPTOM'},
        'relation_type':predicate,'polarity':polarity,'evidence_quote':text,
        'provenance':{'source_sha256':SHA,'filename':'ADCS-doc.md','source_locators':['line:22']},
        'reviewer':reviewer,'truth_source_ref':'independent-QA-reference-001',
    }


def localchunk(id='a',text='reaction wheel speed high',safe=True,approved=False):
    return _LocalChunk(chunk_id=id,source_id='doc',filename='a.md',source_sha256=SHA,
        text=text, section_path=('ADCS',),source_locators=('line:5',),
        review_status=('EXPERT_APPROVED_REFERENCE' if approved else 'HUMAN_REVIEWED')
            if safe else 'MACHINE_VALIDATED',
        content_safety_status=('REVIEWED' if approved else 'PASS')
            if safe else 'REVIEW_REQUIRED',modalities=('TEXT',))


class FakeModel:
    def __init__(self): self.batches=[]
    def encode(self,texts,normalize_embeddings=True):
        self.batches.append(list(texts))
        return np.asarray([[1.,0.] if 'wheel' in t.lower() else [0.,1.] for t in texts],dtype=np.float32)


def test_global_community_requires_reviewed_scoped_grounding():
    item=edge()
    idx=build_approved_community_reports(reviewed_edges=[item],equipment_scope=SCOPE,snapshot=2)
    assert idx['status']=='SUCCEEDED' and idx['community_count']==1
    report=idx['reports'][0]
    assert report['summary_type']=='EXTRACTIVE_CITED_CLAIMS_NOT_MODEL_INFERRED'
    assert report['points'][0]['source_sha256']==SHA
    found=global_community_search(idx,'飞轮摩擦增大 轮速增高')
    assert found['status']=='EVIDENCE_POINTERS_FOUND'
    assert found['evidence_points'][0]['evidence_quote']==item['evidence_quote']
    assert found['current_root_cause_proven'] is False
    assert global_community_search(idx,'天王星 蓝牙传感器')['status']=='ABSTAIN_NO_RELEVANT_APPROVED_REPORT'
    assert not build_approved_community_reports(reviewed_edges=[item],equipment_scope='DIFFERENT',snapshot=2)['reports']


def test_conflicting_and_unapproved_edges_excluded_from_global_index():
    assert not build_approved_community_reports(reviewed_edges=[edge(polarity='NEGATED')],
            equipment_scope=SCOPE,snapshot=2)['reports']
    other=edge(polarity='NEGATED')
    item=edge()
    both=build_approved_community_reports(reviewed_edges=[item,other],equipment_scope=SCOPE,snapshot=3)
    assert both['status']=='NO_APPROVED_KNOWLEDGE'
    assert both['excluded_edge_count']==2
    no_reviewer=edge(reviewer='')
    assert not build_approved_community_reports(reviewed_edges=[no_reviewer],equipment_scope=SCOPE,snapshot=2)['reports']
    assert build_approved_community_reports(reviewed_edges=[item],equipment_scope='UNKNOWN',snapshot=0)['status'].startswith('ABSTAIN')


def test_scope_model_revision_persistence_and_unsafe_chunk_exclusion(tmp_path):
    modeldir=tmp_path/'model';modeldir.mkdir()
    (modeldir/'weights.bin').write_bytes(b'weights-v1')
    db=tmp_path/'vector_cache'/'local.sqlite3'
    safe=localchunk('safe','reaction wheel speed high',approved=True)
    unsafe=localchunk('unsafe','unverified wheel hallucination',safe=False)
    m=FakeModel()
    result,receipt=semantic_rank_with_cache(['wheel friction'],[safe,unsafe],model=m,
         model_path=modeldir,scope=SCOPE,cache_path=db)
    assert result==[['safe']]
    assert receipt['persistence']=='SQLITE_EXACT' and receipt['source_vectors_created']==1
    # Second task recomputes the query but NOT the cached source vector.
    again=FakeModel()
    _,second=semantic_rank_with_cache(['wheel torque'],[safe,unsafe],model=again,
         model_path=modeldir,scope=SCOPE,cache_path=db)
    assert second['source_vectors_reused']==1 and second['source_vectors_created']==0
    assert again.batches==[['wheel torque']]
    # A different model's file bytes must never reuse previous vectors.
    (modeldir/'weights.bin').write_bytes(b'weights-v2')
    _,updated=semantic_rank_with_cache(['wheel torque'],[safe],model=FakeModel(),
         model_path=modeldir,scope=SCOPE,cache_path=db)
    assert updated['source_vectors_created']==1
    _,other=semantic_rank_with_cache(['wheel torque'],[safe],model=FakeModel(),
         model_path=modeldir,scope='SAT-Y-ADCS-V1',cache_path=db)
    assert other['source_vectors_created']==1
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE vectors SET vector=x'FFFFFFFF' WHERE equipment_scope=?", (SCOPE,))
    with pytest.raises(ValueError,match='integrity'):
        PersistentVectorCache(db).get_many(scope=SCOPE,
            revision=model_revision_sha256(modeldir), identities=[identity(safe)])


def test_persistent_vectors_never_require_raw_source_or_network(tmp_path):
    modeldir=tmp_path/'model';modeldir.mkdir();(modeldir/'weights.bin').write_text('model')
    obj=localchunk(approved=True)
    store=PersistentVectorCache(tmp_path/'cache.db')
    sha=model_revision_sha256(modeldir)
    key=identity(obj)
    assert store.put_many(scope=SCOPE,revision=sha,vectors={key:np.asarray([1.,0.],dtype='f4')})==1
    assert store.get_many(scope=SCOPE,revision=sha,identities=[key])[key][0]==1.
    assert store.get_many(scope='DIFFERENT',revision=sha,identities=[key])=={}
    with pytest.raises(ValueError):store.get_many(scope='UNKNOWN',revision=sha,identities=[key])
    assert store.delete_scope(SCOPE)==1
    assert store.get_many(scope=SCOPE,revision=sha,identities=[key])=={}
    assert 'reaction wheel' not in store.path.read_bytes().decode('utf-8',errors='ignore')


def _scanned_pdf(tmp_path):
    from PIL import Image,ImageDraw,ImageFont
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    pic=Image.new('RGB',(1050,350),'white')
    draw=ImageDraw.Draw(pic)
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',58)
    except OSError: font=ImageFont.load_default()
    draw.text((60,95),'WHEEL ERROR CODE 2026',font=font,fill='black')
    pdf=tmp_path/'scan.pdf'
    doc=canvas.Canvas(str(pdf),pagesize=(525,175))
    doc.drawImage(ImageReader(pic),0,0,width=525,height=175)
    doc.showPage();doc.save()
    return pdf


@pytest.mark.skipif(not all(shutil.which(x) for x in ('tesseract','pdfinfo','pdftoppm','pdftotext')),
                    reason='local OCR executables absent')
def test_scanned_pdf_ocr_needs_independent_review(tmp_path, monkeypatch):
    pdf=_scanned_pdf(tmp_path)
    receipt=ocr_scanned_pdf(pdf,languages='eng')
    assert receipt['status']=='OCR_UNREVIEWED' and receipt['recognized_pages']==1
    assert receipt['source_sha256']==hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert 'WHEEL' in receipt['elements'][0].text_representation
    assert not receipt['elements'][0].representation_reviewed
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_OCR_ENGINE','tesseract')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_OCR_LANGUAGES','eng')
    chunks,records=UploadedDocumentKnowledgeIndex().build([pdf])
    assert records[0]['parser']=='UNREVIEWED_TESSERACT_OFFLINE_LOCAL'
    assert chunks and all(c.content_safety_status=='REVIEW_REQUIRED' for c in chunks)
    from astrograph.integrations.material_hierarchical_rag import HierarchicalMaterialIndex
    index=HierarchicalMaterialIndex(chunks)
    assert index.retrieve(base_query='WHEEL',fault_id='TEST_FAULT',fault_name='WHEEL')['hits']==[]
    from astrograph.integrations.result_log_review import read_lines
    lines=read_lines(pdf,'references')
    assert 'OCR_PENDING_INDEPENDENT_REVIEW' in lines[0].text
    assert records[0]['status']=='UNREVIEWED_OCR_CANDIDATES'
    assert records[0]['review_required'] is True
    # The upload preflight must not describe placeholder OCR as extracted evidence.
    import json as _json
    from astrograph.integrations.result_log_review import validate_review_inputs
    base=Path(__file__).resolve().parents[1]/'examples'/'result_log_review'
    (tmp_path/'diagnosis_result.json').write_bytes((base/'diagnosis_result.json').read_bytes())
    (tmp_path/'operations.log').write_bytes((base/'operations.log').read_bytes())
    manifest={'schema_version':'astrograph-result-log-review.v1',
              'diagnosis_results':['diagnosis_result.json'],
              'log_reports':['operations.log'],'references':['scan.pdf'],
              'equipment_scope':SCOPE}
    (tmp_path/'analysis_input.json').write_text(_json.dumps(manifest),encoding='utf-8')
    preflight=validate_review_inputs(tmp_path,tmp_path/'analysis_input.json')
    ref=next(x for x in preflight['files'] if x['role']=='references')
    assert ref['parser_status']=='OCR_PENDING_INDEPENDENT_REVIEW_NO_TRUSTED_TEXT'
    assert ref['parsed_segment_count']==0
    with pytest.raises(ValueError):read_lines(pdf,'diagnosis_results')


def test_local_visual_model_validates_json_and_loopback(monkeypatch):
    class FakeResponse:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,limit):
            return json.dumps({'choices':[{'message':{'content':json.dumps({
                'summary':'Detected axis label RPM','observations':['RPM axis visible']})}}]}).encode()
    class FakeOpener:
        def open(self,*args,**kwargs):return FakeResponse()
    monkeypatch.setattr('astrograph.retrieval.advanced_multimodal.build_opener',lambda *args:FakeOpener())
    png=b'\x89PNG\r\n\x1a\n'+b'A'*80
    found=describe_diagram_local(png,endpoint='http://127.0.0.1:8001/v1/chat/completions',model='vision-local')
    assert found['reviewed'] is False and not found['eligible_for_causes']
    with pytest.raises(ValueError):
        describe_diagram_local(png,endpoint='https://external.example/v1/chat/completions',model='vision-local')
    with pytest.raises(ValueError):
        describe_diagram_local(png,endpoint='http://127.0.0.1:8001/v1/other',model='vision-local')


def test_optional_llm_community_requires_verbatim_quotes():
    from astrograph.retrieval.approved_community_search import llm_extract_community_reports
    index=build_approved_community_reports(reviewed_edges=[edge()],equipment_scope=SCOPE,snapshot=3)
    class FakeLLM:
        def __init__(self, quote):self.quote=quote
        def generate_structured(self, **kwargs):
            assert kwargs['payload']['approved_evidence_only'][0]['edge_id']=='MGR-REFERENCE-001'
            return {'selections':[{'edge_id':'MGR-REFERENCE-001','quote':self.quote}]}
    trusted=llm_extract_community_reports(index,backend=FakeLLM('轮速增高伴随姿控误差'))
    assert trusted['llm_summary_status']=='SUCCEEDED'
    assert trusted['reports'][0]['llm_selected_edge_ids']==['MGR-REFERENCE-001']
    invented=llm_extract_community_reports(index,backend=FakeLLM('未经证实的模型幻想'))
    assert invented['llm_summary_status']=='NO_VALID_SUMMARIES'
    assert invented['reports'][0]['llm_summary_status']=='ABSTAIN_NO_VALID_CITED_QUOTES'
    assert 'llm_selected_edge_ids' not in invented['reports'][0]


def test_optional_qdrant_contract_blocks_stale_or_non_session_vectors(tmp_path,monkeypatch):
    """Simulated service contract; real Qdrant I/O needs deployment acceptance."""
    import sys
    import types
    from astrograph.retrieval.qdrant_vectors import qdrant_rank
    modeldir=tmp_path/'model';modeldir.mkdir();(modeldir/'weights.bin').write_text('weights')
    class Entry:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    fake_types=types.SimpleNamespace(
        VectorParams=Entry, Distance=types.SimpleNamespace(COSINE='Cosine'),
        PointStruct=Entry,FieldCondition=Entry,MatchValue=Entry,HasIdCondition=Entry,
        Filter=Entry)
    fake_module=types.ModuleType('qdrant_client')
    fake_module.QdrantClient=lambda **kwargs:None
    fake_module.models=fake_types
    monkeypatch.setitem(sys.modules,'qdrant_client',fake_module)
    class FakeClient:
        def __init__(self):self.rows=[];self.filter=None
        def collection_exists(self,c):return bool(self.rows)
        def create_collection(self,**kwargs):self.params=kwargs['vectors_config']
        def get_collection(self,c):return Entry(config=Entry(params=Entry(vectors=self.params)))
        def upsert(self,**kwargs):self.rows+=kwargs['points']
        def query_points(self,**kwargs):
            self.filter=kwargs['query_filter']
            # Real filtering is service-side; the application also performs
            # the final membership check for a source outside this session.
            return Entry(points=[Entry(id='00000000-0000-0000-0000-000000000000')]+
                         [Entry(id=p.id) for p in self.rows])
    client=FakeClient()
    good=localchunk('good','reaction wheel speed high',approved=True)
    unsafe=localchunk('unsafe','unverified reaction wheel',safe=False)
    ids, receipt=qdrant_rank(queries=['wheel'],chunks=[good,unsafe],model=FakeModel(),
         model_path=modeldir,scope=SCOPE,url='http://127.0.0.1:6333',
         collection='astrograph_material_embeddings',client=client)
    assert ids==[['good']] and receipt['persistence']=='QDRANT_ANN_WITH_EPHEMERAL_SESSION_SEARCH'
    assert len(client.rows)==1
    assert client.rows[0].payload['equipment_scope']==SCOPE
    restriction=client.filter.must
    assert any(getattr(x,'has_id',None) is not None and len(x.has_id)==1 for x in restriction)
    with pytest.raises(ValueError,match='HTTPS'):
        qdrant_rank(queries=['wheel'],chunks=[good],model=FakeModel(),model_path=modeldir,
           scope=SCOPE,url='http://qdrant.example.net:6333',
           collection='astrograph_material_embeddings',client=client)


def test_material_main_report_exposes_global_community_policy(tmp_path,monkeypatch):
    from astrograph.integrations.upload_report import UploadReportJobStore
    base=Path(__file__).resolve().parents[1]/'examples'/'result_log_review'
    store=UploadReportJobStore(tmp_path/'runs')
    job=store.create('test source grounded community index')['job_id']
    for file in base.iterdir():
        if file.is_file(): store.put_file(job,file.name,file.read_bytes())
    assert store.preflight(job)['can_start']
    outcome=store.run(job)
    assert outcome['status']=='SUCCEEDED'
    reasoning=outcome['analysis']['material_reasoning']
    assert reasoning['global_community_index']['status'] in {'ABSTAIN_EQUIPMENT_SCOPE_REQUIRED','NO_APPROVED_KNOWLEDGE'}
    assert not reasoning['global_community_retrieval']['evidence_points']
    assert reasoning['retrieval_architecture']['global_community_summaries_implemented']=='EXTRACTIVE_REVIEWED_EDGE_COMMUNITY_REPORTS'
    assert all('community_retrieval' in c for c in reasoning['candidates'])
    assert '社区' in store.report_markdown_path(job).read_text(encoding='utf-8')


def test_global_community_search_finds_late_edge_without_citing_irrelevant_preview(monkeypatch):
    # All 35 edges are in one connected community, but the sought relation is
    # far outside the default 16-point preview. No first-edge substitution.
    edges=[]
    for i in range(35):
        x=edge(text=f'ordinary generic system evidence {i}')
        x['proposal_id']=f'MGR-{i:04d}'
        x['source']={'entity_id':f'CHAIN-{i}', 'name':f'generic component {i}'}
        x['target']={'entity_id':f'CHAIN-{i+1}', 'name':f'generic component {i+1}'}
        if i==34:
            x['source']['name']='飞轮动量卸载'
            x['target']['name']='轮速饱和告警'
            x['evidence_quote']='飞轮动量卸载相关轮速饱和告警，需核查控制模式。'
        edges.append(x)
    # Simulate a large connected community (one of the documented fallback
    # partition modes); Louvain may subdivide a chain into smaller clusters.
    from astrograph.retrieval import approved_community_search as community_mod
    monkeypatch.setattr(community_mod, '_communities',
        lambda records: (community_mod._components(records),'CONNECTED_COMPONENTS_TEST'))
    idx=build_approved_community_reports(reviewed_edges=edges,equipment_scope=SCOPE,
        snapshot=5,max_points=16)
    assert idx['preview_truncated']
    assert idx['searchable_edge_count']==35
    found=global_community_search(idx,'飞轮动量卸载 轮速饱和告警')
    assert found['status']=='EVIDENCE_POINTERS_FOUND'
    assert found['evidence_points'][0]['edge_id']=='MGR-0034'
    assert all('飞轮' in p['evidence_quote'] or '飞轮' in p['subject']
               for p in found['evidence_points'])
    # No match is reported when only unrelated generic edges exist.
    assert global_community_search(idx,'液氦燃料电池')['status']=='ABSTAIN_NO_RELEVANT_APPROVED_REPORT'


def test_qdrant_second_run_reuses_existing_vectors_without_reencoding_sources(tmp_path, monkeypatch):
    import sys,types
    from astrograph.retrieval.qdrant_vectors import qdrant_rank
    modeldir=tmp_path/'model';modeldir.mkdir();(modeldir/'weights.bin').write_text('weights')
    class Entry:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    fake_types=types.SimpleNamespace(
        VectorParams=Entry,Distance=types.SimpleNamespace(COSINE='Cosine'),
        PointStruct=Entry, FieldCondition=Entry, MatchValue=Entry,
        HasIdCondition=Entry, Filter=Entry)
    fake_module=types.ModuleType('qdrant_client')
    fake_module.QdrantClient=lambda **kwargs:None
    fake_module.models=fake_types
    monkeypatch.setitem(sys.modules,'qdrant_client',fake_module)
    class FakeClient:
        def __init__(self):self.rows={};self.dimension=None;self.upserts=0
        def collection_exists(self,c):return self.dimension is not None
        def create_collection(self,**kwargs):self.dimension=kwargs['vectors_config'].size
        def get_collection(self,c):return Entry(config=Entry(params=Entry(
            vectors=Entry(size=self.dimension))))
        def upsert(self,**kwargs):
            self.upserts+=1
            self.rows.update({p.id:p for p in kwargs['points']})
        def retrieve(self,**kwargs):
            return [Entry(id=p.id,payload=p.payload) for uid,p in self.rows.items()
                    if uid in kwargs['ids']]
        def query_points(self,**kwargs):
            return Entry(points=[Entry(id=uid) for uid in self.rows
                if uid in kwargs['query_filter'].must[-1].has_id])
    client=FakeClient()
    doc=localchunk('a','reaction wheel speed high',approved=True)
    first=FakeModel()
    _,status=qdrant_rank(queries=['wheel'],chunks=[doc],model=first,
        model_path=modeldir,scope=SCOPE,url='http://127.0.0.1:6333',
        collection='astrograph_material_embeddings',client=client)
    assert status['source_vectors_created']==1 and client.upserts==1
    second=FakeModel()
    ranks,status=qdrant_rank(queries=['wheel torque'],chunks=[doc],model=second,
        model_path=modeldir,scope=SCOPE,url='http://127.0.0.1:6333',
        collection='astrograph_material_embeddings',client=client)
    assert ranks==[['a']]
    assert status['source_vectors_reused']==1 and status['source_vectors_created']==0
    assert client.upserts==1 and second.batches==[['wheel torque']]
    # Reject a server-side record where someone has changed the provenance.
    point=next(iter(client.rows.values()))
    point.payload['equipment_scope']='other-spacecraft'
    with pytest.raises(ValueError,match='provenance'):
        qdrant_rank(queries=['wheel'],chunks=[doc],model=FakeModel(),
            model_path=modeldir,scope=SCOPE,url='http://127.0.0.1:6333',
            collection='astrograph_material_embeddings',client=client)


def test_native_upload_embeddings_are_transient_even_with_cache_enabled(tmp_path):
    modeldir=tmp_path/'model';modeldir.mkdir();(modeldir/'weights.bin').write_text('v1')
    db=tmp_path/'private-cache.sqlite'
    uploaded=localchunk('upload','reaction wheel speed high')
    vectors, receipt=semantic_rank_with_cache(['wheel'],[uploaded],model=FakeModel(),
        model_path=modeldir,scope=SCOPE,cache_path=db)
    assert vectors==[['upload']]
    assert receipt['ephemeral_uploaded_vectors']==1
    assert receipt['source_vectors_persisted']==0
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]==0
    # A second task is allowed to retrieve its own upload while no old vector
    # can be silently reused from the first job.
    _,again=semantic_rank_with_cache(['wheel'],[uploaded],model=FakeModel(),
        model_path=modeldir,scope=SCOPE,cache_path=db)
    assert again['source_vectors_reused']==0 and again['source_vectors_created']==1


def test_cache_symlink_is_rejected_before_path_resolution(tmp_path):
    target=tmp_path/'actual.sqlite'
    link=tmp_path/'redirect.sqlite'
    link.symlink_to(target)
    with pytest.raises(ValueError,match='symlink'):
        PersistentVectorCache(link)


def test_qdrant_never_upserts_unapproved_session_upload(tmp_path, monkeypatch):
    import sys, types
    from astrograph.retrieval.qdrant_vectors import qdrant_rank
    modeldir=tmp_path/'model';modeldir.mkdir();(modeldir/'weights.bin').write_text('weights')
    class Entry:
        def __init__(self, **kwargs):self.__dict__.update(kwargs)
    mod=types.ModuleType('qdrant_client')
    mod.models=types.SimpleNamespace(VectorParams=Entry, Distance=types.SimpleNamespace(COSINE='Cosine'),
        PointStruct=Entry,FieldCondition=Entry,MatchValue=Entry,HasIdCondition=Entry,Filter=Entry)
    mod.QdrantClient=lambda **kwargs:None
    monkeypatch.setitem(sys.modules, 'qdrant_client', mod)
    class FakeClient:
        def __init__(self):self.upserts=0;self.queries=0;self.created=0
        def collection_exists(self,name):return False
        def create_collection(self,**kwargs):self.created+=1
        def upsert(self,**kwargs):self.upserts+=1
        def query_points(self,**kwargs):self.queries+=1
    client=FakeClient()
    ranks,receipt=qdrant_rank(queries=['wheel'], chunks=[localchunk('uploaded','reaction wheel')],
        model=FakeModel(),model_path=modeldir,scope=SCOPE,
        url='http://127.0.0.1:6333',collection='astrograph_material_embeddings',client=client)
    assert ranks==[['uploaded']]
    assert receipt['ephemeral_uploaded_vectors']==1
    assert receipt['indexed_reviewed_fragments']==0
    assert client.upserts==client.queries==client.created==0
