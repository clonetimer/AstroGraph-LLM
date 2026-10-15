"""Expert-governed document and improvement lifecycle.

This module stages durable proposals but never mutates the active KG, production
models, tool permissions, or confirmed memory automatically.  Expert decisions
can admit documents into the local library or approve an improvement candidate
for *offline evaluation* only.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _json(value) -> str:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def _utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None: raise ValueError("timezone-aware time required")
    return value.astimezone(timezone.utc).isoformat()


class Strict(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)


class LifecyclePolicy(Strict):
    schema_version: Literal["astrograph-lifecycle-policy.v1"]="astrograph-lifecycle-policy.v1"
    approved_reviewers: tuple[str,...]=Field(min_length=1)
    permitted_document_roots: tuple[str,...]=Field(min_length=1)


class DocumentProposal(Strict):
    schema_version: Literal["astrograph-document-proposal.v1"]="astrograph-document-proposal.v1"
    document_id: str=Field(min_length=1)
    source_path: str=Field(min_length=1)
    source_sha256: str=Field(pattern=r"^[0-9a-f]{64}$")
    title: str=Field(min_length=1,max_length=500)
    media_type: str=Field(min_length=1,max_length=200)
    parser: str=Field(min_length=1,max_length=200)
    proposer_actor: str=Field(min_length=1)
    proposed_at: datetime
    metadata: dict=Field(default_factory=dict)

    @model_validator(mode="after")
    def aware(self): _utc(self.proposed_at); return self


TargetLayer=Literal["planning","memory","tool","knowledge","model","execution"]


class ExpertFeedback(Strict):
    schema_version: Literal["astrograph-expert-feedback.v1"]="astrograph-expert-feedback.v1"
    feedback_id: str=Field(min_length=1)
    task_id: str=Field(min_length=1)
    result_sha256: str=Field(pattern=r"^[0-9a-f]{64}$")
    feedback_type: Literal["DIAGNOSIS_CORRECTION","INVESTIGATION_CORRECTION","KNOWLEDGE_CORRECTION","TOOL_CORRECTION"]
    reviewer: str=Field(min_length=1)
    reviewed_at: datetime
    truth_source_ref: str=Field(min_length=1)
    verified_fault_id: str | None=None
    target_layers: tuple[TargetLayer,...]=Field(min_length=1)
    diagnosis_error_summary: str=Field(min_length=1,max_length=4000)
    recommended_change: str=Field(min_length=1,max_length=4000)
    evidence_refs: tuple[str,...]=Field(default_factory=tuple)

    @field_validator("target_layers")
    @classmethod
    def unique_layers(cls,value):
        if len(value)!=len(set(value)): raise ValueError("target_layers must be unique")
        return value

    @model_validator(mode="after")
    def validate_feedback(self):
        _utc(self.reviewed_at)
        if self.feedback_type=="DIAGNOSIS_CORRECTION" and not self.verified_fault_id:
            raise ValueError("diagnosis correction requires verified_fault_id")
        return self




class KnowledgeProposal(Strict):
    schema_version: Literal["astrograph-knowledge-proposal.v1"]="astrograph-knowledge-proposal.v1"
    proposal_id: str=Field(min_length=1)
    task_id: str=Field(min_length=1)
    candidate_sha256: str=Field(pattern=r"^[0-9a-f]{64}$")
    candidate_payload: dict
    source_bindings: tuple[dict,...]=Field(default_factory=tuple)
    proposer_actor: str=Field(min_length=1)
    proposed_at: datetime
    status: Literal["PENDING_REVIEW","APPROVED_FOR_OFFLINE_EVALUATION","REJECTED"]="PENDING_REVIEW"
    active_kg_write_allowed: Literal[False]=False

    @model_validator(mode="after")
    def valid_candidate(self):
        _utc(self.proposed_at)
        if _sha(self.candidate_payload)!=self.candidate_sha256:
            raise ValueError("knowledge candidate hash mismatch")
        if self.candidate_payload.get("active_kg_modified") is not False:
            raise ValueError("knowledge proposal must originate from a review-only candidate")
        return self

class ImprovementCandidate(Strict):
    schema_version: Literal["astrograph-improvement-candidate.v1"]="astrograph-improvement-candidate.v1"
    candidate_id: str
    feedback_id: str
    target_layer: TargetLayer
    proposed_change: str
    evidence_refs: tuple[str,...]
    status: Literal["PENDING_REVIEW","APPROVED_FOR_OFFLINE_EVALUATION","REJECTED"]="PENDING_REVIEW"
    production_change_allowed: Literal[False]=False
    active_kg_modified: Literal[False]=False
    active_model_modified: Literal[False]=False
    confirmed_memory_modified: Literal[False]=False


class KnowledgeFeedbackStore:
    def __init__(self,path: str|Path,policy: LifecyclePolicy):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.policy=policy
        with self._db() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS document_proposals(document_id TEXT PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL, review TEXT);
              CREATE TABLE IF NOT EXISTS feedback(feedback_id TEXT PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS knowledge_proposals(proposal_id TEXT PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL, review TEXT);
              CREATE TABLE IF NOT EXISTS improvement_candidates(candidate_id TEXT PRIMARY KEY, feedback_id TEXT NOT NULL, target_layer TEXT NOT NULL, body TEXT NOT NULL, digest TEXT NOT NULL, status TEXT NOT NULL, review TEXT);
              CREATE TABLE IF NOT EXISTS receipts(event_id TEXT PRIMARY KEY, digest TEXT NOT NULL, response TEXT NOT NULL);
            ''')

    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.path,timeout=30,isolation_level=None);db.row_factory=sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL");db.execute("BEGIN IMMEDIATE");yield db;db.commit()
        except Exception: db.rollback();raise
        finally: db.close()

    def _reviewer(self,actor: str):
        if actor not in self.policy.approved_reviewers: raise PermissionError("reviewer is not approved by lifecycle policy")

    def _permitted_path(self,path: Path) -> Path:
        p=path.expanduser().resolve()
        roots=[Path(x).expanduser().resolve() for x in self.policy.permitted_document_roots]
        if not any(p.is_relative_to(root) for root in roots): raise PermissionError("document path is outside permitted library roots")
        return p

    @staticmethod
    def _old(db,event_id,payload):
        row=db.execute("SELECT digest,response FROM receipts WHERE event_id=?",(event_id,)).fetchone()
        if row:
            if row["digest"]!=_sha(payload): raise ValueError("lifecycle idempotency key reused with different payload")
            return json.loads(row["response"])
        return None

    @staticmethod
    def _receipt(db,event_id,payload,response):
        db.execute("INSERT INTO receipts VALUES(?,?,?)",(event_id,_sha(payload),_json(response)))

    def stage_document(self, proposal: DocumentProposal, *, event_id: str) -> dict:
        path=self._permitted_path(Path(proposal.source_path))
        if not path.is_file(): raise FileNotFoundError(path)
        if _file_sha(path)!=proposal.source_sha256: raise ValueError("document changed before staging")
        body=proposal.model_dump(mode="json");payload={"op":"stage_document","proposal":body}
        with self._db() as db:
            old=self._old(db,event_id,payload)
            if old is not None:return old
            row=db.execute("SELECT digest FROM document_proposals WHERE document_id=?",(proposal.document_id,)).fetchone()
            if row and row["digest"]!=_sha(body): raise ValueError("document_id reused with different content")
            db.execute("INSERT OR IGNORE INTO document_proposals VALUES(?,?,?,?,NULL)",(proposal.document_id,_json(body),_sha(body),"PENDING_REVIEW"))
            response={"document_id":proposal.document_id,"status":"PENDING_REVIEW","library_visible":False,"active_kg_modified":False}
            self._receipt(db,event_id,payload,response);return response

    def review_document(self, *, document_id: str, decision: Literal["APPROVE","REJECT"], reviewer: str,
                        reason: str, event_id: str) -> dict:
        self._reviewer(reviewer);payload={"op":"review_document","document_id":document_id,"decision":decision,"reviewer":reviewer,"reason":reason}
        with self._db() as db:
            old=self._old(db,event_id,payload)
            if old is not None:return old
            row=db.execute("SELECT body,status FROM document_proposals WHERE document_id=?",(document_id,)).fetchone()
            if row is None: raise KeyError("document proposal not found")
            proposal=DocumentProposal.model_validate_json(row["body"]);path=self._permitted_path(Path(proposal.source_path))
            if _file_sha(path)!=proposal.source_sha256: raise ValueError("document changed after proposal")
            status="APPROVED_LIBRARY" if decision=="APPROVE" else "REJECTED"
            review={"reviewer":reviewer,"decision":decision,"reason":reason,"reviewed_at":datetime.now(timezone.utc).isoformat()}
            db.execute("UPDATE document_proposals SET status=?,review=? WHERE document_id=?",(status,_json(review),document_id))
            response={"document_id":document_id,"status":status,"library_visible":status=="APPROVED_LIBRARY","active_kg_modified":False}
            self._receipt(db,event_id,payload,response);return response

    def approved_documents(self) -> list[dict]:
        with self._db() as db:
            rows=db.execute("SELECT body,review FROM document_proposals WHERE status='APPROVED_LIBRARY' ORDER BY document_id").fetchall()
            out=[]
            for row in rows:
                p=DocumentProposal.model_validate_json(row["body"]);path=self._permitted_path(Path(p.source_path))
                if _file_sha(path)!=p.source_sha256: raise ValueError("approved document changed on disk")
                out.append({**p.model_dump(mode="json"),"review":json.loads(row["review"]),"diagnostic_vote_allowed":False})
            return out


    def stage_knowledge_proposal(self, proposal: KnowledgeProposal, *, event_id: str) -> dict:
        body=proposal.model_dump(mode="json");payload={"op":"stage_knowledge_proposal","proposal":body}
        with self._db() as db:
            old=self._old(db,event_id,payload)
            if old is not None:return old
            row=db.execute("SELECT digest FROM knowledge_proposals WHERE proposal_id=?",(proposal.proposal_id,)).fetchone()
            if row and row["digest"]!=_sha(body): raise ValueError("knowledge proposal id reused with different content")
            db.execute("INSERT OR IGNORE INTO knowledge_proposals VALUES(?,?,?,?,NULL)",(proposal.proposal_id,_json(body),_sha(body),proposal.status))
            response={"proposal_id":proposal.proposal_id,"status":"PENDING_REVIEW","active_kg_modified":False,
                      "active_kg_write_allowed":False,"requires_expert_review":True}
            self._receipt(db,event_id,payload,response);return response

    def review_knowledge_proposal(self, *, proposal_id: str,
                                  decision: Literal["APPROVE_FOR_OFFLINE_EVALUATION","REJECT"],
                                  reviewer: str, reason: str, event_id: str) -> dict:
        self._reviewer(reviewer);payload={"op":"review_knowledge_proposal","proposal_id":proposal_id,"decision":decision,"reviewer":reviewer,"reason":reason}
        with self._db() as db:
            old=self._old(db,event_id,payload)
            if old is not None:return old
            row=db.execute("SELECT body FROM knowledge_proposals WHERE proposal_id=?",(proposal_id,)).fetchone()
            if row is None: raise KeyError("knowledge proposal not found")
            current=KnowledgeProposal.model_validate_json(row["body"])
            status="APPROVED_FOR_OFFLINE_EVALUATION" if decision=="APPROVE_FOR_OFFLINE_EVALUATION" else "REJECTED"
            updated=current.model_copy(update={"status":status})
            review={"reviewer":reviewer,"decision":decision,"reason":reason,"reviewed_at":datetime.now(timezone.utc).isoformat()}
            db.execute("UPDATE knowledge_proposals SET body=?,digest=?,status=?,review=? WHERE proposal_id=?",
                       (_json(updated.model_dump(mode="json")),_sha(updated.model_dump(mode="json")),status,_json(review),proposal_id))
            response={"proposal_id":proposal_id,"status":status,"active_kg_modified":False,"active_kg_write_allowed":False,
                      "next_step":"OFFLINE_GRAPH_REGRESSION_AND_EXISTING_KNOWLEDGE_GOVERNANCE" if status=="APPROVED_FOR_OFFLINE_EVALUATION" else "NONE"}
            self._receipt(db,event_id,payload,response);return response

    def export_knowledge_proposal(self, proposal_id: str) -> dict:
        with self._db() as db:
            row=db.execute("SELECT body,review,status FROM knowledge_proposals WHERE proposal_id=?",(proposal_id,)).fetchone()
            if row is None: raise KeyError("knowledge proposal not found")
            body=json.loads(row["body"]);body["review"]=json.loads(row["review"]) if row["review"] else None
            body["status"]=row["status"];body["active_kg_write_allowed"]=False
            body["next_step"]="OFFLINE_GRAPH_REGRESSION_AND_EXISTING_KNOWLEDGE_GOVERNANCE" if row["status"]=="APPROVED_FOR_OFFLINE_EVALUATION" else "EXPERT_REVIEW_REQUIRED"
            return body

    def submit_feedback(self, feedback: ExpertFeedback, *, event_id: str) -> dict:
        self._reviewer(feedback.reviewer);body=feedback.model_dump(mode="json");payload={"op":"submit_feedback","feedback":body}
        with self._db() as db:
            old=self._old(db,event_id,payload)
            if old is not None:return old
            row=db.execute("SELECT digest FROM feedback WHERE feedback_id=?",(feedback.feedback_id,)).fetchone()
            if row and row["digest"]!=_sha(body): raise ValueError("feedback_id reused with different content")
            db.execute("INSERT OR IGNORE INTO feedback VALUES(?,?,?)",(feedback.feedback_id,_json(body),_sha(body)))
            ids=[]
            for layer in feedback.target_layers:
                cid=f"IMPROVE-{hashlib.sha256((feedback.feedback_id+'|'+layer).encode()).hexdigest()[:20].upper()}"
                candidate=ImprovementCandidate(candidate_id=cid,feedback_id=feedback.feedback_id,target_layer=layer,
                    proposed_change=feedback.recommended_change,evidence_refs=feedback.evidence_refs)
                cbody=candidate.model_dump(mode="json")
                db.execute("INSERT OR IGNORE INTO improvement_candidates VALUES(?,?,?,?,?,?,NULL)",(cid,feedback.feedback_id,layer,_json(cbody),_sha(cbody),candidate.status))
                ids.append(cid)
            response={"feedback_id":feedback.feedback_id,"status":"RECORDED","candidate_ids":ids,
                      "production_modified":False,"requires_offline_evaluation":True}
            self._receipt(db,event_id,payload,response);return response

    def review_candidate(self, *, candidate_id: str, decision: Literal["APPROVE_FOR_OFFLINE_EVALUATION","REJECT"],
                         reviewer: str, reason: str, event_id: str) -> dict:
        self._reviewer(reviewer);payload={"op":"review_candidate","candidate_id":candidate_id,"decision":decision,"reviewer":reviewer,"reason":reason}
        with self._db() as db:
            old=self._old(db,event_id,payload)
            if old is not None:return old
            row=db.execute("SELECT body FROM improvement_candidates WHERE candidate_id=?",(candidate_id,)).fetchone()
            if row is None: raise KeyError("improvement candidate not found")
            current=ImprovementCandidate.model_validate_json(row["body"])
            status="APPROVED_FOR_OFFLINE_EVALUATION" if decision=="APPROVE_FOR_OFFLINE_EVALUATION" else "REJECTED"
            updated=current.model_copy(update={"status":status});review={"reviewer":reviewer,"decision":decision,"reason":reason,"reviewed_at":datetime.now(timezone.utc).isoformat()}
            db.execute("UPDATE improvement_candidates SET body=?,digest=?,status=?,review=? WHERE candidate_id=?",(_json(updated.model_dump(mode="json")),_sha(updated.model_dump(mode="json")),status,_json(review),candidate_id))
            response={"candidate_id":candidate_id,"status":status,"production_modified":False,"active_kg_modified":False,
                      "active_model_modified":False,"confirmed_memory_modified":False}
            self._receipt(db,event_id,payload,response);return response

    def export_candidate(self,candidate_id: str) -> dict:
        with self._db() as db:
            row=db.execute("SELECT body,review,status FROM improvement_candidates WHERE candidate_id=?",(candidate_id,)).fetchone()
            if row is None: raise KeyError("improvement candidate not found")
            body=json.loads(row["body"]);body["review"]=json.loads(row["review"]) if row["review"] else None
            body["status"]=row["status"];body["active_release_allowed"]=False
            body["next_step"]="OFFLINE_ABLATION_OR_EXISTING_KNOWLEDGE_GOVERNANCE" if body["target_layer"]=="knowledge" else "OFFLINE_AB_TEST_REQUIRED"
            return body
