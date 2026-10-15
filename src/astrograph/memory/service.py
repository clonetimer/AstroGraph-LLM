"""Transactional, host-owned case memory with independent confirmation and snapshots.

This small SQLite implementation is an initial case store, not a vector database.
PPL owns policy resolution; this service is the sole durable writer. Caller context
must be authenticated by the host -- it is not accepted from an Agent tool payload.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Literal, Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
from astrograph.orchestration.models import PermissionScope, WorkflowContext
from .ppl_bridge import PPLBridge


def _json(value) -> str:
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:raise ValueError('timezone required')
    return value.astimezone(timezone.utc).isoformat()


class MemoryPolicy(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True)
    schema_version: Literal['astrograph-memory-policy.v1']='astrograph-memory-policy.v1'
    profile_sha256: str=Field(pattern=r'^[0-9a-f]{64}$')
    approved_reviewers: tuple[str,...]=Field(min_length=1)
    max_cases: int=Field(default=5,ge=1,le=50)
    commit_end_reasons: tuple[Literal['completed'],...]=('completed',)


class CheckExperience(BaseModel):
    """Observed tool outcome, not a claim that the check proves a fault."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    tool_id: str = Field(min_length=1)
    feature_definition_ids: tuple[str, ...] = ()
    outcome: Literal['ADMITTED', 'UNAVAILABLE', 'NO_CHANGE', 'FAILED']
    evidence_ids: tuple[str, ...] = ()


class PendingCase(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True)
    case_id: str=Field(min_length=1)
    source_group_id: str=Field(min_length=1)
    producer_actor: str=Field(min_length=1)
    symptom_ids: tuple[str,...]=Field(min_length=1)
    operating_mode: str=Field(min_length=1)
    proposed_fault_id: str | None=None
    evidence_ids: tuple[str,...]=Field(min_length=1)
    observed_end: datetime
    end_reason: Literal['completed','max-tokens','error']
    output_valid: bool
    checks: tuple[CheckExperience, ...] = ()
    decision_status: str = ''
    decision_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    knowledge_version: str = ''

    @model_validator(mode='after')
    def validate_case(self) -> Self:
        _utc(self.observed_end)
        if any(not value for value in (*self.symptom_ids,*self.evidence_ids)):
            raise ValueError('empty symptom/evidence reference')
        return self


class MemoryConflictError(ValueError):
    pass


class CaseMemoryService:
    def __init__(self, path: str | Path, policy: MemoryPolicy, *, bridge: PPLBridge | None=None):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.policy=policy;self.bridge=bridge
        if bridge is not None and bridge.profile_sha256!=policy.profile_sha256:
            raise ValueError('memory and PPL profile versions differ')
        with self._db() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS pending(case_id TEXT PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS head(id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL);
              INSERT OR IGNORE INTO head VALUES(1,0);
              CREATE TABLE IF NOT EXISTS confirmed(case_id TEXT PRIMARY KEY, version INTEGER UNIQUE NOT NULL,
                  body TEXT NOT NULL, available_at TEXT NOT NULL, source_group_id TEXT NOT NULL);
              CREATE INDEX IF NOT EXISTS confirmed_version_available ON confirmed(version,available_at);
              CREATE INDEX IF NOT EXISTS confirmed_source ON confirmed(source_group_id);
              CREATE INDEX IF NOT EXISTS confirmed_mode_knowledge ON confirmed(json_extract(body,'$.operating_mode'),COALESCE(json_extract(body,'$.knowledge_version'),''),version);
              CREATE TABLE IF NOT EXISTS receipts(event_id TEXT PRIMARY KEY, digest TEXT NOT NULL, response TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS outbox(event_id TEXT PRIMARY KEY, body TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
              CREATE TABLE IF NOT EXISTS profile_state(profile_sha256 TEXT PRIMARY KEY, version INTEGER NOT NULL, state TEXT NOT NULL);
            ''')

    @contextmanager
    def _db(self, *, read_only=False):
        from astrograph.integrations.execution_budget import remaining_time
        db=sqlite3.connect(self.path,timeout=remaining_time(30),isolation_level=None)
        db.row_factory=sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('BEGIN' if read_only else 'BEGIN IMMEDIATE')
            yield db
            db.commit()
        except Exception:
            db.rollback();raise
        finally:db.close()

    @staticmethod
    def _require(context, permission):
        if permission not in context.permissions:raise PermissionError('memory permission denied')

    @staticmethod
    def _existing(db, event_id, payload):
        row=db.execute('SELECT digest,response FROM receipts WHERE event_id=?',(event_id,)).fetchone()
        if row:
            if row['digest']!=_digest(payload):raise MemoryConflictError('idempotency key reused with different payload')
            return json.loads(row['response'])
        return None

    @staticmethod
    def _receipt(db, event_id, payload, response):
        db.execute('INSERT INTO receipts VALUES(?,?,?)',(event_id,_digest(payload),_json(response)))

    def stage_case(self, case: PendingCase, *, event_id: str, context: WorkflowContext) -> dict:
        self._require(context,PermissionScope.WRITE_MEMORY)
        if case.producer_actor!=context.actor:raise PermissionError('case producer identity mismatch')
        body=case.model_dump(mode='json');payload={'operation':'stage','case':body}
        with self._db() as db:
            old=self._existing(db,event_id,payload)
            if old is not None:return old
            row=db.execute('SELECT digest FROM pending WHERE case_id=?',(case.case_id,)).fetchone()
            if row and row['digest']!=_digest(body):raise MemoryConflictError('case ID reused with changed evidence')
            db.execute('INSERT OR IGNORE INTO pending VALUES(?,?,?)',(case.case_id,_json(body),_digest(body)))
            response={'case_id':case.case_id,'status':'PENDING','confirmed':False}
            self._receipt(db,event_id,payload,response)
            return response

    def snapshot(self) -> int:
        with self._db(read_only=True) as db:return db.execute('SELECT version FROM head WHERE id=1').fetchone()[0]

    def confirm_case(self, *, case_id: str, event_id: str, verified_fault_id: str,
                     truth_source_ref: str, verified_at: datetime, available_at: datetime,
                     expected_snapshot: int, context: WorkflowContext) -> dict:
        self._require(context,PermissionScope.REVIEW_MEMORY)
        if context.actor not in self.policy.approved_reviewers:raise PermissionError('reviewer is not approved')
        if not verified_fault_id or not truth_source_ref:raise ValueError('independent truth and source reference required')
        verified,available=_utc(verified_at),_utc(available_at)
        if available_at<verified_at:raise ValueError('confirmation cannot be available before verification')
        payload={'operation':'confirm','case_id':case_id,'truth':verified_fault_id,'source':truth_source_ref,
                 'verified_at':verified,'available_at':available,'reviewer':context.actor}
        with self._db() as db:
            old=self._existing(db,event_id,payload)
            if old is not None:return old  # replay must not advance a snapshot
            row=db.execute('SELECT body FROM pending WHERE case_id=?',(case_id,)).fetchone()
            if row is None:raise KeyError('pending case not found')
            case=PendingCase.model_validate_json(row['body'])
            if case.producer_actor==context.actor:raise PermissionError('producer cannot self-confirm')
            if case.end_reason not in self.policy.commit_end_reasons or not case.output_valid:
                raise ValueError('incomplete/invalid output cannot enter confirmed memory')
            if verified_at<case.observed_end:raise ValueError('verification precedes observation')
            head=db.execute('SELECT version FROM head WHERE id=1').fetchone()[0]
            if head!=expected_snapshot:raise MemoryConflictError('memory compare-and-swap failed')
            if db.execute('SELECT 1 FROM confirmed WHERE case_id=?',(case_id,)).fetchone():
                raise MemoryConflictError('case already confirmed; corrections need a new reviewed revision')
            version=head+1
            body={**case.model_dump(mode='json'),'verified_fault_id':verified_fault_id,'truth_source_ref':truth_source_ref,
                  'reviewer':context.actor,'verified_at':verified,'available_at':available,'snapshot':version}
            db.execute('INSERT INTO confirmed VALUES(?,?,?,?,?)',(case_id,version,_json(body),available,case.source_group_id))
            db.execute('UPDATE head SET version=? WHERE id=1',(version,))
            response={'case_id':case_id,'snapshot':version,'status':'CONFIRMED'}
            db.execute('INSERT INTO outbox(event_id,body) VALUES(?,?)',(event_id,_json(response)))
            self._receipt(db,event_id,payload,response)
            return response

    def retrieve(self, symptom_ids: tuple[str,...], *, operating_mode: str, snapshot: int,
                 cutoff_at: datetime, exclude_case_id: str, exclude_source_group_id: str,
                 context: WorkflowContext, knowledge_version: str | None = None) -> list[dict]:
        self._require(context,PermissionScope.READ_MEMORY)
        cutoff=_utc(cutoff_at)
        if snapshot<0:raise ValueError('negative memory snapshot')
        query=set(symptom_ids)
        with self._db(read_only=True) as db:
            if snapshot>db.execute('SELECT version FROM head WHERE id=1').fetchone()[0]:
                raise ValueError('unknown future snapshot')
            # sqlite JSON1 filters before deserializing unrelated cases; no new truth enters retrieval.
            sql="""SELECT body FROM confirmed WHERE version<=? AND available_at<=?
                 AND case_id<>? AND source_group_id<>?
                 AND json_extract(body,'$.operating_mode')=?"""
            params=[snapshot,cutoff,exclude_case_id,exclude_source_group_id,operating_mode]
            if knowledge_version is not None:
                sql+=" AND COALESCE(json_extract(body,'$.knowledge_version'),'')=?"
                params.append(knowledge_version)
            # Exact Jaccard ranking in SQL: fetch only the final top-k case bodies.
            ranked_sql="""WITH eligible AS ("""+sql+"""), scored AS (
                SELECT body,json_extract(body,'$.case_id') AS case_id,
                (SELECT COUNT(DISTINCT value) FROM json_each(body,'$.symptom_ids')
                    WHERE value IN (SELECT value FROM json_each(?))) AS overlap,
                (SELECT COUNT(DISTINCT value) FROM json_each(body,'$.symptom_ids')) AS symptom_count
                FROM eligible)
                SELECT body FROM scored WHERE overlap>0
                ORDER BY CAST(overlap AS REAL) / (? + symptom_count - overlap) DESC,case_id
                LIMIT ?"""
            rows=db.execute(ranked_sql,[*params,_json(sorted(query)),len(query),self.policy.max_cases]).fetchall()
        ranked=[]
        for row in rows:
            case=json.loads(row['body'])
            if case['operating_mode']!=operating_mode:continue
            if knowledge_version is not None and case.get('knowledge_version', '') != knowledge_version:continue
            # Unknown modes must not make unrelated equipment look applicable.
            if not operating_mode or operating_mode == 'UNKNOWN':continue
            symptoms=set(case['symptom_ids']);score=len(query&symptoms)/max(1,len(query|symptoms))
            if score>0:ranked.append({'case':case,'retrieval_score':score,'diagnostic_vote_allowed':False})
        return sorted(ranked,key=lambda x:(-x['retrieval_score'],x['case']['case_id']))[:self.policy.max_cases]

    def apply_profile_event(self, event: dict, *, event_id: str, expected_version: int,
                            end_reason: str, context: WorkflowContext) -> dict:
        self._require(context,PermissionScope.WRITE_MEMORY)
        if self.bridge is None:raise RuntimeError('PPL binding not configured')
        payload={'operation':'profile','profile_sha256':self.policy.profile_sha256,'event':event,
                 'end_reason':end_reason,'actor':context.actor}
        with self._db() as db:
            old=self._existing(db,event_id,payload)
            if old is not None:return old
            row=db.execute('SELECT version,state FROM profile_state WHERE profile_sha256=?',
                           (self.policy.profile_sha256,)).fetchone()
            version,state=(row['version'],json.loads(row['state'])) if row else (0,self.bridge.profile['initialState'])
            if version!=expected_version:raise MemoryConflictError('profile compare-and-swap failed')
        # Do not hold SQLite's write lock across the external deterministic resolver.
        resolved=self.bridge.resolve(event=event,context=context,state=state,end_reason=end_reason)
        with self._db() as db:
            old=self._existing(db,event_id,payload)
            if old is not None:return old
            row=db.execute('SELECT version FROM profile_state WHERE profile_sha256=?',(self.policy.profile_sha256,)).fetchone()
            actual=row['version'] if row else 0
            if actual!=version:raise MemoryConflictError('profile changed during resolution')
            if resolved['committed']:
                version+=1
                db.execute('INSERT OR REPLACE INTO profile_state VALUES(?,?,?)',
                           (self.policy.profile_sha256,version,_json(resolved['state'])))
            response={**resolved,'version':version}
            self._receipt(db,event_id,payload,response)
            return response

    def pending_outbox(self) -> list[dict]:
        with self._db() as db:
            return [dict(row) for row in db.execute('SELECT event_id,body FROM outbox WHERE delivered=0 ORDER BY event_id')]

    def mark_delivered(self, event_id: str) -> None:
        with self._db() as db:db.execute('UPDATE outbox SET delivered=1 WHERE event_id=?',(event_id,))
