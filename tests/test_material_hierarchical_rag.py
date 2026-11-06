"""Contract tests for hierarchical multi-channel RAG and semantic safety."""
import hashlib

from astrograph.integrations.platform_knowledge import _LocalChunk, UploadedDocumentKnowledgeIndex
from astrograph.integrations.material_hierarchical_rag import (
    HierarchicalMaterialIndex, RetrievalBudgets,
)
from astrograph.retrieval.spacecraft_terminology import expand_spacecraft_terms
from astrograph.retrieval.structured_document import (
    DocumentElement, DocumentElementKind, SectionFirstDocumentIngestor,
)


def chunk(i, text, section='Document', source='manual', kind='TEXT', safe=True):
    return _LocalChunk(
        chunk_id=f'{source}:{i:04d}', source_id=source,
        filename=f'{source}.md', source_sha256='a'*64,
        text=text, section_path=(section,), source_locators=(f'line:{i}',),
        review_status='HUMAN_REVIEWED' if safe else 'MACHINE_VALIDATED',
        content_safety_status='PASS' if safe else 'REVIEW_REQUIRED',
        modalities=(kind,),
    )


def search(chunks, name, *, fid='NONSTANDARD_FAULT', references=(), graph=(), scope=None):
    return HierarchicalMaterialIndex(chunks).retrieve(
        base_query=f'{fid} {name}', fault_id=fid, fault_name=name,
        reference_terms=references, approved_edges=graph, equipment_scope=scope,
    )


def test_bilingual_alias_covers_cross_language_semantic_gap_without_llm_guessing():
    good=chunk(1,'Reaction wheel bearing friction increases wheel motor current.', section='ADCS')
    other=chunk(2,'Communication downlink antenna pointing update.', section='COMM', source='other')
    result=search([good,other], '反作用飞轮轴承摩擦')
    assert result['status']=='SUCCEEDED'
    assert result['hits'][0]['knowledge_id']==good.chunk_id
    assert any(item['role']=='CURATED_SYNONYM' for item in result['query_plan'])
    assert result['terminology']['matches']
    assert result['causal_proof'] is False
    assert expand_spacecraft_terms('unregistered hypothetical coil fault')['unknown_vocabulary_not_inferred']


def test_late_section_retrievable_in_long_document_not_only_first_page():
    many=[chunk(i,'routine software service and housekeeping telemetry.',
                section=f'chapter {i}', source='1000page') for i in range(1, 120)]
    target=chunk(121,'The bus undervoltage condition can occur after power converter shutdown.',
                 section='Appendix critical anomalies', source='1000page')
    result=search(many+[target], '母线欠压')
    assert result['hits'] and result['hits'][0]['knowledge_id']==target.chunk_id
    stages=result['retrieval_trace']['stages']
    assert [x['stage'] for x in stages] == [
        'PARALLEL_LEXICAL_AND_SEMANTIC_PROBES','DOCUMENT_SELECTION',
        'SECTION_SELECTION','FINE_CITATION_RERANK_AND_DIVERSIFY',
    ]
    assert target.source_locators[0] in result['hits'][0]['source_locators']
    assert any(x['section_path']==['Appendix critical anomalies']
               for x in stages[2]['selected'])


def test_unverified_image_formula_table_excluded_even_if_keyword_matches():
    approved=chunk(1,'Star tracker calibration and pointing error checks.',safe=True)
    unsafe=[chunk(i, 'star tracker calibration failed ![model caption]',
                  section=f'figure{i}', kind=kind, safe=False)
            for i,kind in enumerate(('IMAGE','TABLE','FORMULA'), start=2)]
    result=search(unsafe+[approved], 'star tracker calibration')
    assert len(result['hits']) == 1
    assert result['hits'][0]['knowledge_id']==approved.chunk_id
    assert result['retrieval_trace']['unsafe_representation_excluded_count']==3
    assert result['retrieval_trace']['excluded_modalities']=={'FORMULA':1,'IMAGE':1,'TABLE':1}
    assert not result['automatic_control_authorized']
    unavailable=search(unsafe, 'star tracker calibration')
    assert unavailable['status']=='NO_ELIGIBLE_REPRESENTATIONS'
    assert unavailable['hits']==[]


def test_unknown_query_abstains_not_a_fake_reference():
    result=search([chunk(1,'battery thermal regulation and internal heater.')],
                  '完全不存在的未登记机械故障',fid='NOT-IN-REFERENCES-92012')
    assert result['status']=='NO_RELEVANT_PASSAGES'
    assert not result['hits']
    assert result['terminology']['unknown_vocabulary_not_inferred']
    assert not result['causal_proof']


def test_direct_approved_graph_neighbor_is_query_hint_not_root_cause_vote():
    edge={
        'proposal_id':'MGR-REVIEW-001', 'equipment_scope':'SAT_X_ADCS_V1',
        'snapshot':3, 'review_status':'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
        'source': {'entity_id':'f1','name':'SAMPLE_FAULT','entity_type':'FAULT_MODE'},
        'target': {'entity_id':'m1','name':'thermoelastic deformation','entity_type':'FAILURE_MECHANISM'},
        'relation_type':'ASSOCIATED_WITH', 'polarity':'AFFIRMATIVE',
        'provenance':{'source_sha256':'a'*64,'source_locators':['line:1']},
        'evidence_quote':'SAMPLE_FAULT associated with thermoelastic deformation',
        'reviewer':'reviewer', 'truth_source_ref':'independent-source',
    }
    hits=search([chunk(1,'Thermoelastic deformation can cause misalignment.')],
                'SAMPLE_FAULT',fid='SAMPLE_FAULT',graph=[edge],scope='SAT_X_ADCS_V1')
    assert 'APPROVED_GRAPH_HINTS' in hits['methods']
    assert any(p['role']=='APPROVED_GRAPH_QUERY_HINT' for p in hits['query_plan'])
    assert hits['hits'][0]['knowledge_validation_status']=='UNREVIEWED_UPLOADED_REFERENCE'
    assert not hits['causal_proof']
    negative={**edge,'polarity':'NEGATIVE'}
    blocked=search([chunk(1,'Thermoelastic deformation can cause misalignment.')],
                'SAMPLE_FAULT',fid='SAMPLE_FAULT',graph=[negative],scope='SAT_X_ADCS_V1')
    assert 'APPROVED_GRAPH_HINTS' not in blocked['methods']


def test_contextual_chunker_keeps_table_and_formula_atomic_and_text_exact():
    ingestor=SectionFirstDocumentIngestor(maximum_section_units=20,
        maximum_chunk_chars=300,preserve_modality_boundaries=True,split_long_text=True)
    paragraph='The star tracker is checked after thermal drift. ' * 100
    elements=[
        DocumentElement(element_id='p',kind=DocumentElementKind.TEXT,
            section_path=['ADCS','Checks'],text_representation=paragraph,
            source_locator='lines:20-28',representation_method='markdown-native',
            representation_reviewed=True),
        DocumentElement(element_id='table',kind=DocumentElementKind.TABLE,
            section_path=['ADCS','Checks'],text_representation='| signal | threshold |\n| -- | -- |\n| s1 | 1mA |',
            source_locator='lines:29-33',representation_method='markdown-native',
            representation_reviewed=True),
        DocumentElement(element_id='eq',kind=DocumentElementKind.FORMULA,
            section_path=['ADCS','Checks'],text_representation='$$ r(t) = y(t) - y_{expected}(t) $$',
            source_locator='lines:34-36',representation_method='markdown-native',
            representation_reviewed=True),
    ]
    result=ingestor.ingest_elements(source_id='doc:one',source_uri='file://one',
        source_sha256='a'*64,knowledge_version='v1',elements=elements)
    paragraphs=[x for x in result.elements if x.kind==DocumentElementKind.TEXT]
    assert len(paragraphs)>1 and ''.join(e.text_representation for e in paragraphs)==paragraph
    assert all('#chars:' in e.source_locator for e in paragraphs)
    assert any(c.modalities==['TABLE'] and '| signal | threshold |' in c.text for c in result.chunks)
    assert any(c.modalities==['FORMULA'] and 'r(t)' in c.text for c in result.chunks)
    assert all(len(c.text)<=300 for c in result.chunks)
    assert all(c.content_safety_status=='PASS' for c in result.chunks)


def test_oversized_atomic_image_never_silently_truncated():
    image=DocumentElement(element_id='figure',kind=DocumentElementKind.IMAGE,
        section_path=['Diagram'],text_representation='unreviewed_caption'*100,
        source_locator='line:10',asset_uri='figure.png',representation_method='generated-caption',
        representation_reviewed=False)
    ingestion=SectionFirstDocumentIngestor(maximum_chunk_chars=256,split_long_text=True,
        preserve_modality_boundaries=True).ingest_elements(source_id='doc:figure',
        source_uri='figure.md',source_sha256='b'*64,knowledge_version='v1',elements=[image])
    assert len(ingestion.chunks)==1
    assert ingestion.chunks[0].content_safety_status=='REVIEW_REQUIRED'
    assert len(ingestion.chunks[0].text)==len(image.text_representation)
    assert 'OVERSIZED_ATOMIC_IMAGE_REVIEW_REQUIRED:line:10' in ingestion.warnings


def test_material_ingest_separates_table_and_text_and_never_trusts_alt(tmp_path):
    (tmp_path/'plot.png').write_bytes(b'not-real-png')
    text='''# Thermal system\n\nThe temperature may exceed limit.\n\n| signal | threshold |\n| --- | --- |\n| radiator | 340K |\n\n![Thermal curve confirms failure](plot.png)\n'''
    p=tmp_path/'note.md';p.write_text(text)
    chunks,receipts=UploadedDocumentKnowledgeIndex().build([p])
    assert receipts[0]['status']=='PARSED'
    assert len(chunks)==3
    assert {c.modalities[0] for c in chunks}=={'TEXT','TABLE','IMAGE'}
    assert next(c for c in chunks if c.modalities==('IMAGE',)).content_safety_status=='REVIEW_REQUIRED'
    result=search(chunks,'temperature thermal curve confirms failure')
    assert all('Thermal curve confirms failure' not in h['quote'] for h in result['hits'])
    assert result['retrieval_trace']['unsafe_representation_excluded_count']==1


def test_large_table_search_returns_contiguous_original_rows_with_header_as_metadata():
    header='| Fault code | Limit | Corrective consideration |\n'
    sep='| --- | --- | --- |\n'
    rows=[f'| E-{i:04d} | {i} mA | check harness channel {i} |\n' for i in range(1, 180)]
    table=header+sep+''.join(rows)
    result=search([chunk(1,table,kind='TABLE')], 'E-0175 harness channel 175', fid='E-0175')
    assert result['status']=='SUCCEEDED'
    assert result['hits'][0]['table_row_window']
    assert 'E-0175' in result['hits'][0]['quote']
    assert result['hits'][0]['quote'] in table
    assert result['hits'][0]['table_header_context'] in table
    assert any('#table-lines:' in x for x in result['hits'][0]['source_locators'])
    assert len(result['hits'][0]['quote'])<=1200


def test_independently_reviewed_glossary_is_scope_bound_and_digest_pinned(tmp_path, monkeypatch):
    import json
    from astrograph.retrieval.spacecraft_terminology import load_pinned_domain_lexicon
    document={
        'schema_version':'astrograph-reviewed-mission-glossary.v1',
        'equipment_scope':'SAT-A_ADCS-V2',
        'review':{'status':'EXPERT_APPROVED','reviewer':'independent-mission-reviewer',
                  'truth_source_ref':'approved-device-terminology-v5'},
        'concepts':[{'id':'tacho_sensor','aliases':['测速脉冲输入','tachometer pulse input']}]}
    path=tmp_path/'reviewed.json'
    raw=json.dumps(document,ensure_ascii=False).encode('utf-8')
    path.write_bytes(raw)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_LEXICON_PATH',str(path))
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_LEXICON_SHA256',hashlib.sha256(raw).hexdigest())
    result=search([chunk(1,'Tachometer pulse input is missing.')], '测速脉冲输入',fid='SENSOR-DROPOUT')
    # No equipment scope -> glossary ignored, no invented match.
    assert result['status']=='NO_RELEVANT_PASSAGES'
    matching=HierarchicalMaterialIndex([chunk(1,'Tachometer pulse input is missing.')]).retrieve(
        base_query='测速脉冲输入',fault_id='SENSOR-DROPOUT',fault_name='测速脉冲输入',
        equipment_scope='SAT-A_ADCS-V2')
    assert matching['hits']
    assert matching['terminology']['mission_glossary']['status']=='EXPERT_REVIEWED_PINNED'
    wrong_scope=HierarchicalMaterialIndex([chunk(1,'Tachometer pulse input is missing.')]).retrieve(
        base_query='测速脉冲输入',fault_id='SENSOR-DROPOUT',fault_name='测速脉冲输入',
        equipment_scope='SAT-B_ADCS-V1')
    assert wrong_scope['status']=='NO_RELEVANT_PASSAGES'
    path.write_text('tampered',encoding='utf-8')
    import pytest
    with pytest.raises(ValueError,match='digest mismatch'):
        load_pinned_domain_lexicon('SAT-A_ADCS-V2')


def test_pdf_raster_figures_are_disclosed_but_unreviewed(monkeypatch,tmp_path):
    import subprocess
    from astrograph.retrieval.structured_document import SectionFirstDocumentIngestor
    file=tmp_path/'example.pdf';file.write_bytes(b'%PDF fake; mocked external utilities')
    class Completed:
        def __init__(self,stdout): self.stdout=stdout
    def run(args,**kwargs):
        if 'pdfinfo' in args[0]: return Completed('Pages:          1\n')
        if 'pdftotext' in args[0]: return Completed(b'Power converter efficiency is important.\f')
        if 'pdfimages' in args[0]: return Completed('page   num  type   width\n   1     0 image    200  300 rgb')
        raise AssertionError(args)
    monkeypatch.setattr('astrograph.retrieval.structured_document.shutil.which',lambda command:command)
    monkeypatch.setattr('astrograph.retrieval.structured_document.subprocess.run',run)
    elements,warnings,audit=SectionFirstDocumentIngestor().parse_pdf_layout(file)
    images=[x for x in elements if x.kind==DocumentElementKind.IMAGE]
    assert len(images)==1
    assert not images[0].representation_reviewed
    assert images[0].source_locator=='pdf#page:1:image:0'
    assert any('PDF_IMAGE_REQUIRES_VISUAL_REVIEW' in w for w in warnings)
    assert audit.page_count==1


def test_excessively_long_corpus_fails_closed_as_partial_not_false_absence():
    indexed=[chunk(i,'nominal housekeeping and status messages',section='routine')
             for i in range(6000)]
    # The only relevant material exceeds this task-level bounded index.
    out=search(indexed+[chunk(6001,'unexpected payload coupler resonance',section='Appendix')],
               'payload coupler resonance',fid='COUPLER-RESONANCE')
    assert out['status']=='PARTIAL_CORPUS_LIMIT_REVIEW_REQUIRED'
    assert out['retrieval_trace']['index_truncated']
    assert out['retrieval_trace']['unindexed_chunk_count']==1
    assert out['retrieval_trace']['corpus_completeness']=='PARTIAL_UNINDEXED_CONTENT_REQUIRES_REVIEW'
    assert not out['causal_proof']
