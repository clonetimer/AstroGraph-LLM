from __future__ import annotations
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import pytest
from astrograph.lifecycle import DocumentProposal,ExpertFeedback,KnowledgeFeedbackStore,LifecyclePolicy

def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def test_document_library_requires_expert_review_and_hash_binding(tmp_path:Path):
    doc=tmp_path/'manual.md';doc.write_text('Voltage anomaly requires independent evidence.',encoding='utf-8')
    policy=LifecyclePolicy(approved_reviewers=('expert-a',),permitted_document_roots=(str(tmp_path),))
    store=KnowledgeFeedbackStore(tmp_path/'lifecycle.sqlite',policy)
    prop=DocumentProposal(document_id='DOC-1',source_path=str(doc),source_sha256=h(doc),title='manual',media_type='md',parser='UTF8_NATIVE_TEXT',proposer_actor='agent',proposed_at=datetime.now(timezone.utc))
    staged=store.stage_document(prop,event_id='stage-1')
    assert staged['library_visible'] is False and store.approved_documents()==[]
    approved=store.review_document(document_id='DOC-1',decision='APPROVE',reviewer='expert-a',reason='verified source',event_id='review-1')
    assert approved['library_visible'] is True and len(store.approved_documents())==1
    doc.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):
        store.approved_documents()

def test_feedback_creates_only_offline_improvement_candidates(tmp_path:Path):
    policy=LifecyclePolicy(approved_reviewers=('expert-a',),permitted_document_roots=(str(tmp_path),))
    store=KnowledgeFeedbackStore(tmp_path/'lifecycle.sqlite',policy)
    fb=ExpertFeedback(feedback_id='FB-1',task_id='T-1',result_sha256='a'*64,feedback_type='DIAGNOSIS_CORRECTION',reviewer='expert-a',reviewed_at=datetime.now(timezone.utc),truth_source_ref='expert://case-1',verified_fault_id='FAULT-B',target_layers=('planning','knowledge','model'),diagnosis_error_summary='A was wrong',recommended_change='evaluate B-discriminating evidence first',evidence_refs=('E-1',))
    receipt=store.submit_feedback(fb,event_id='feedback-1')
    assert len(receipt['candidate_ids'])==3 and receipt['production_modified'] is False
    cid=receipt['candidate_ids'][0]
    review=store.review_candidate(candidate_id=cid,decision='APPROVE_FOR_OFFLINE_EVALUATION',reviewer='expert-a',reason='run ablation',event_id='candidate-review-1')
    assert review['status']=='APPROVED_FOR_OFFLINE_EVALUATION'
    assert review['active_kg_modified'] is False and review['active_model_modified'] is False
    export=store.export_candidate(cid)
    assert export['active_release_allowed'] is False

def test_unapproved_reviewer_cannot_submit_feedback(tmp_path:Path):
    store=KnowledgeFeedbackStore(tmp_path/'x.sqlite',LifecyclePolicy(approved_reviewers=('expert-a',),permitted_document_roots=(str(tmp_path),)))
    fb=ExpertFeedback(feedback_id='FB-2',task_id='T',result_sha256='b'*64,feedback_type='TOOL_CORRECTION',reviewer='outsider',reviewed_at=datetime.now(timezone.utc),truth_source_ref='expert://x',target_layers=('tool',),diagnosis_error_summary='bad tool',recommended_change='evaluate another tool')
    with pytest.raises(PermissionError):store.submit_feedback(fb,event_id='x')

def test_knowledge_proposal_requires_expert_review_and_never_writes_active_graph(tmp_path:Path):
    from astrograph.lifecycle import KnowledgeProposal
    from astrograph.agents.mission.storage import sha
    policy=LifecyclePolicy(approved_reviewers=('expert-a',),permitted_document_roots=(str(tmp_path),))
    store=KnowledgeFeedbackStore(tmp_path/'knowledge.sqlite',policy)
    candidate={'status':'REVIEW_REQUIRED','active_kg_modified':False,'edges':[{'subject':'a','predicate':'causes','object':'b'}]}
    proposal=KnowledgeProposal(proposal_id='KP-1',task_id='build-1',candidate_sha256=sha(candidate),candidate_payload=candidate,
        source_bindings=(),proposer_actor='agent',proposed_at=datetime.now(timezone.utc))
    staged=store.stage_knowledge_proposal(proposal,event_id='kp-stage')
    assert staged['requires_expert_review'] is True and staged['active_kg_write_allowed'] is False
    reviewed=store.review_knowledge_proposal(proposal_id='KP-1',decision='APPROVE_FOR_OFFLINE_EVALUATION',reviewer='expert-a',reason='evaluate',event_id='kp-review')
    assert reviewed['status']=='APPROVED_FOR_OFFLINE_EVALUATION'
    assert reviewed['active_kg_modified'] is False and reviewed['active_kg_write_allowed'] is False
    exported=store.export_knowledge_proposal('KP-1')
    assert exported['next_step']=='OFFLINE_GRAPH_REGRESSION_AND_EXISTING_KNOWLEDGE_GOVERNANCE'
