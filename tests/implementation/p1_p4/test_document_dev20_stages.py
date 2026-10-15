from pathlib import Path
from types import SimpleNamespace

from astrograph.agents.mission.document_quality import document_provider_quality_receipt
from astrograph.agents.mission.literature_context import build_literature_context_assessment


def test_stage86_external_quality_is_not_claimed_from_adapter_presence(tmp_path:Path):
    s=SimpleNamespace(allow_mineru=True,mineru_executable=str(tmp_path/'missing-mineru'),allow_a2o=True,a2o_source_root=str(tmp_path/'missing-a2o'))
    receipt=document_provider_quality_receipt(s)
    assert receipt['native_text']['status']=='AVAILABLE'
    assert receipt['mineru']['status']=='BLOCKED_EXTERNAL_PROVIDER'
    assert receipt['anything2ontology']['status']=='BLOCKED_EXTERNAL_PROVIDER'
    assert receipt['quality_claims']['real_complex_pdf_quality_validated'] is False
    assert receipt['diagnostic_vote_allowed'] is False


def test_stage87_retrieval_relevance_does_not_become_diagnostic_support():
    core={'diagnosis':{'selected_fault_id':'F1','candidate_fault_ids':['F1','F2']},'status':'SUCCEEDED'}
    attachment=build_literature_context_assessment(core,[{'citation_id':'DOC:1','source_sha256':'a'*64,'text':'F1 is mentioned'}])
    assert attachment['status']=='RETRIEVED_CONTEXT_ONLY'
    assert attachment['assessments'][0]['stance']=='UNASSESSED_CONTEXT'
    assert attachment['diagnostic_vote_allowed'] is False
    assert attachment['core_diagnosis_modified'] is False


def test_stage87_reviewed_stance_can_be_explained_but_still_nonvoting():
    core={'diagnosis':{'selected_fault_id':'F1','candidate_fault_ids':['F1','F2']},'status':'SUCCEEDED'}
    attachment=build_literature_context_assessment(core,[{'citation_id':'CLAIM:1','source_sha256':'b'*64,
        'reviewed_claim_polarity':'CONTRADICTS','reviewed_target_fault_id':'F1'}])
    assert attachment['status']=='REVIEWED_STANCE_CONTEXT_AVAILABLE'
    assert attachment['assessments'][0]['stance']=='CONTRADICTS'
    assert attachment['diagnostic_vote_allowed'] is False
