"""Host-owned, review-gated memory for *material* fault reviews.

Separate namespace from telemetry CaseMemoryService: document cues must not be
cast into telemetry EvidenceDefinition IDs or silently become confirmed truth.
Both stores follow the same pending -> independent confirmation -> snapshot
principle. No writes to the Active KG and no self-reinforcing generated claims.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serial(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_serial(value).encode('utf-8')).hexdigest()


class MaterialMemoryConflict(ValueError):
    pass


class MaterialCaseMemory:
    """Append-only approvals; pending input never appears in retrieve_confirmed.

    Callers must authenticate reviewers *outside* this class. The caller passes
    an administrator-owned allowlist; a network user-supplied reviewer header is
    NOT a substitute for authentication. A non-empty independent truth source
    and an exact equipment scope are mandatory for publication.
    """

    def __init__(self, path: str | Path, *, reviewers: tuple[str, ...] = ()) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reviewers = frozenset(reviewers)
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS pending (
                    case_id TEXT PRIMARY KEY, source_group_id TEXT NOT NULL,
                    producer TEXT NOT NULL, body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL, disposition TEXT NOT NULL DEFAULT 'PENDING'
                );
                CREATE TABLE IF NOT EXISTS confirmed (
                    case_id TEXT PRIMARY KEY, version INTEGER UNIQUE NOT NULL,
                    body TEXT NOT NULL, available_at TEXT NOT NULL,
                    source_group_id TEXT NOT NULL, equipment_scope TEXT NOT NULL,
                    verified_fault_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS head (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL);
                INSERT OR IGNORE INTO head(id,version) VALUES (1,0);
                CREATE TABLE IF NOT EXISTS decisions (
                    case_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reviewer TEXT NOT NULL,
                    truth_source_ref TEXT NOT NULL, decided_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS confirmed_scope_fault ON confirmed(equipment_scope,verified_fault_id,version);
                CREATE TABLE IF NOT EXISTS reference_pending (
                    document_id TEXT PRIMARY KEY, source_group_id TEXT NOT NULL,
                    producer TEXT NOT NULL, body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL, disposition TEXT NOT NULL DEFAULT 'PENDING'
                );
                CREATE TABLE IF NOT EXISTS reference_confirmed (
                    document_id TEXT PRIMARY KEY, version INTEGER UNIQUE NOT NULL,
                    body TEXT NOT NULL, available_at TEXT NOT NULL,
                    source_group_id TEXT NOT NULL, equipment_scope TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS confirmed_ref_scope ON reference_confirmed(equipment_scope,version);
                CREATE TABLE IF NOT EXISTS reference_decisions (
                    document_id TEXT PRIMARY KEY, decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL, truth_source_ref TEXT NOT NULL,
                    decided_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph_pending (
                    proposal_id TEXT PRIMARY KEY, source_group_id TEXT NOT NULL,
                    producer TEXT NOT NULL, body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL, disposition TEXT NOT NULL DEFAULT 'PENDING'
                );
                CREATE INDEX IF NOT EXISTS graph_pending_status ON graph_pending(disposition,created_at);
                CREATE TABLE IF NOT EXISTS graph_confirmed (
                    proposal_id TEXT PRIMARY KEY, version INTEGER UNIQUE NOT NULL,
                    body TEXT NOT NULL, available_at TEXT NOT NULL,
                    source_group_id TEXT NOT NULL, equipment_scope TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS graph_confirmed_scope ON graph_confirmed(equipment_scope,version);
                CREATE TABLE IF NOT EXISTS graph_decisions (
                    proposal_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reviewer TEXT NOT NULL,
                    truth_source_ref TEXT NOT NULL, decided_at TEXT NOT NULL
                );
                -- Append-only tombstones keep historical snapshots reproducible.
                -- Retraction does not rewrite the approval or source material.
                CREATE TABLE IF NOT EXISTS graph_retractions (
                    proposal_id TEXT PRIMARY KEY, version INTEGER UNIQUE NOT NULL,
                    reviewer TEXT NOT NULL, reason TEXT NOT NULL,
                    truth_source_ref TEXT NOT NULL, revoked_at TEXT NOT NULL
                );
                -- Knowledge gaps are review tickets, never confirmed graph facts.
                CREATE TABLE IF NOT EXISTS reasoning_gaps (
                    gap_id TEXT PRIMARY KEY, source_group_id TEXT NOT NULL,
                    equipment_scope TEXT NOT NULL, fault_id TEXT NOT NULL,
                    body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN',
                    resolver TEXT, resolution_ref TEXT, resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS reasoning_gaps_queue ON reasoning_gaps(status,created_at);
            ''')

    @contextmanager
    def _db(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA busy_timeout=15000')
            db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _scope(value: str) -> str:
        value = value.strip().upper()
        if not re.fullmatch(r'[A-Z0-9][A-Z0-9_.:-]{1,79}', value) or value in {'UNKNOWN', 'UNSPECIFIED'}:
            raise ValueError('equipment_scope must be an explicit approved equipment/model identifier')
        return value

    def snapshot(self) -> int:
        with self._db(write=False) as db:
            return int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])

    def stage_review(self, *, source_group_id: str, review: dict[str, Any],
                     producer: str = 'ASTROGRAPH_MATERIAL_ANALYZER') -> list[dict[str, Any]]:
        """Idempotently propose one case per candidate, never publish knowledge."""
        if not re.fullmatch(r'JOB-[A-F0-9]{24}', source_group_id):
            raise ValueError('source_group_id is not an AstroGraph job identifier')
        proposals: list[dict[str, Any]] = []
        with self._db() as db:
            for candidate in review.get('candidates', [])[:5]:
                fault_id = str(candidate['fault_id'])
                case_id = 'MRC-' + _digest({'job_id': source_group_id, 'fault_id': fault_id})[:24].upper()
                body = {
                    'schema_version': 'astrograph-material-case-proposal.v1',
                    'case_id': case_id, 'source_group_id': source_group_id,
                    'producer': producer, 'proposed_fault_id': fault_id,
                    'fault_name': str(candidate['fault_name'])[:240],
                    'equipment_scope': str(review.get('equipment_scope') or 'UNKNOWN').upper(),
                    'source_manifest': list(review.get('source_manifest') or []),
                    'diagnosis_source_ids': list(candidate.get('diagnosis_source_ids') or []),
                    'log_evidence_ids': list(candidate.get('log_evidence_ids') or []),
                    'reference_citation_ids': list(candidate.get('reference_citation_ids') or []),
                    'hypotheses': list(candidate.get('cause_hypotheses') or []),
                    'candidate_graph_relations': list(candidate.get('knowledge_candidate_relations') or []),
                    'review_actions': list(candidate.get('handling_measures') or []),
                    'verification_checks': list(candidate.get('verification_checks') or []),
                    'review_status': 'UNVERIFIED_PENDING_INDEPENDENT_CONFIRMATION',
                    'active_kg_modified': False,
                }
                digest = _digest(body)
                row = db.execute('SELECT body_sha256, disposition FROM pending WHERE case_id=?', (case_id,)).fetchone()
                if row and row['body_sha256'] != digest:
                    raise MaterialMemoryConflict('case ID reused with different input/provenance')
                if not row:
                    db.execute('INSERT INTO pending(case_id,source_group_id,producer,body,body_sha256,created_at) VALUES(?,?,?,?,?,?)',
                               (case_id, source_group_id, producer, _serial(body), digest, _now()))
                proposals.append({'case_id': case_id, 'status': row['disposition'] if row else 'PENDING',
                                  'content_sha256': digest, 'confirmed': False})
        return proposals

    def pending(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._db(write=False) as db:
            rows = db.execute('SELECT case_id,body,created_at,source_group_id FROM pending WHERE disposition=? ORDER BY created_at DESC,case_id LIMIT ?',
                              ('PENDING', min(max(1,limit),200))).fetchall()
        return [{**json.loads(row['body']), 'created_at': row['created_at']} for row in rows]

    def decide(self, *, case_id: str, reviewer: str, decision: str, truth_source_ref: str,
               verified_fault_id: str | None = None, equipment_scope: str | None = None,
               approved_actions: list[str] | None = None, expected_snapshot: int | None = None) -> dict[str, Any]:
        if reviewer not in self.reviewers:
            raise PermissionError('reviewer is not in the administrator-controlled allowlist')
        if decision not in {'CONFIRM', 'REJECT'}:
            raise ValueError('decision must be CONFIRM or REJECT')
        if not isinstance(truth_source_ref,str) or not truth_source_ref.strip() or len(truth_source_ref) > 500:
            raise ValueError('independent truth source reference is required')
        with self._db() as db:
            row = db.execute('SELECT body,producer,disposition FROM pending WHERE case_id=?', (case_id,)).fetchone()
            if not row:
                raise KeyError('pending case not found')
            if reviewer == row['producer']:
                raise PermissionError('case producer cannot confirm their own output')
            if row['disposition'] != 'PENDING':
                raise MaterialMemoryConflict('case already decided; corrections require a new revision')
            current = int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            if expected_snapshot is not None and expected_snapshot != current:
                raise MaterialMemoryConflict('memory snapshot compare-and-swap failed')
            decided_at = _now()
            proposed = json.loads(row['body'])
            db.execute('INSERT INTO decisions(case_id,decision,reviewer,truth_source_ref,decided_at) VALUES(?,?,?,?,?)',
                       (case_id, decision, reviewer, truth_source_ref.strip(), decided_at))
            db.execute('UPDATE pending SET disposition=? WHERE case_id=?',
                       ('CONFIRMED' if decision == 'CONFIRM' else 'REJECTED', case_id))
            if decision == 'REJECT':
                return {'case_id': case_id, 'status': 'REJECTED', 'snapshot': current}
            fault_id = str(verified_fault_id or '').strip()
            if not fault_id or len(fault_id) > 160:
                raise ValueError('independently verified fault ID is required')
            scope = self._scope(equipment_scope or '')
            if proposed['equipment_scope'] not in {'UNKNOWN', scope}:
                raise ValueError('approved scope conflicts with documented input scope')
            actions = approved_actions or []
            if (len(actions) > 12 or any(not isinstance(x, str) or not x.strip() or len(x) > 1500 for x in actions)):
                raise ValueError('approved_actions must be bounded nonempty strings')
            version = current + 1
            body = {
                **proposed, 'verified_fault_id': fault_id,
                'equipment_scope': scope, 'reviewer': reviewer,
                'truth_source_ref': truth_source_ref.strip(), 'verified_at': decided_at,
                'available_at': decided_at, 'approved_actions': actions,
                'review_status': 'INDEPENDENTLY_CONFIRMED', 'snapshot': version,
                'active_kg_modified': False,
            }
            db.execute('INSERT INTO confirmed(case_id,version,body,available_at,source_group_id,equipment_scope,verified_fault_id) VALUES(?,?,?,?,?,?,?)',
                       (case_id,version,_serial(body),decided_at,proposed['source_group_id'],scope,fault_id))
            db.execute('UPDATE head SET version=? WHERE id=1',(version,))
            return {'case_id': case_id,'status':'CONFIRMED','snapshot':version,'verified_fault_id':fault_id}

    def retrieve_confirmed(self, *, fault_id: str, equipment_scope: str | None,
                           exclude_source_group_id: str, snapshot: int | None = None,
                           limit: int = 5) -> dict[str, Any]:
        """Do not offer fuzzy cross-equipment matches or pending material as proof."""
        if not equipment_scope or str(equipment_scope).strip().upper() in {'UNKNOWN','UNSPECIFIED'}:
            return {'status':'EQUIPMENT_BINDING_REQUIRED', 'snapshot':self.snapshot(), 'hits':[]}
        scope = self._scope(equipment_scope)
        if not fault_id:
            return {'status':'NO_CANDIDATE','snapshot':self.snapshot(),'hits':[]}
        with self._db(write=False) as db:
            head = int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            version = head if snapshot is None else snapshot
            if not 0 <= version <= head:
                raise ValueError('unknown memory snapshot')
            rows = db.execute('''SELECT body FROM confirmed
                  WHERE version <= ? AND equipment_scope=?
                  AND (verified_fault_id=? OR json_extract(body,'$.proposed_fault_id')=?)
                  AND source_group_id <> ? AND available_at <= ?
                  ORDER BY version DESC, case_id LIMIT ?''',
                  (version, scope, fault_id, fault_id, exclude_source_group_id, _now(), min(max(1,limit),12))).fetchall()
        return {'status':'SUCCEEDED', 'snapshot':version,
                'hits':[{'case_id':case['case_id'],'verified_fault_id':case['verified_fault_id'],
                         'reviewer':case['reviewer'],'truth_source_ref':case['truth_source_ref'],
                         'equipment_scope':case['equipment_scope'],'snapshot':case['snapshot'],
                         'approved_actions':case['approved_actions'],
                         'source_group_id':case['source_group_id'],
                         'proposed_fault_id':case['proposed_fault_id'],
                         'relation_status':('PRIOR_UPSTREAM_MISCLASSIFICATION' if case['proposed_fault_id']!=case['verified_fault_id'] else 'CONSISTENT_HISTORICAL_CASE'),
                         'status':'CONFIRMED_HISTORICAL_CASE_NOT_CURRENT_TRUTH',
                         'diagnostic_vote_allowed':False}
                        for row in rows for case in [json.loads(row['body'])]]}
    def stage_reference_documents(self, *, source_group_id: str,
                                  chunks: list[Any], producer: str = 'ASTROGRAPH_MATERIAL_ANALYZER') -> list[dict[str, Any]]:
        """Freeze uploaded document excerpts as proposals, not a searchable approved corpus."""
        if not re.fullmatch(r'JOB-[A-F0-9]{24}', source_group_id):
            raise ValueError('invalid source_group_id')
        by_source: dict[tuple[str,str],list[Any]] = {}
        for chunk in chunks:
            by_source.setdefault((chunk.source_sha256,chunk.filename),[]).append(chunk)
        output=[]
        with self._db() as db:
            for (sha,filename),source_chunks in sorted(by_source.items()):
                document_id='MRD-'+_digest({'job':source_group_id,'sha':sha,'filename':filename})[:24].upper()
                sections=[{'knowledge_id':c.chunk_id,'source_locators':list(c.source_locators),
                           'section_path':list(c.section_path),'quote':c.text[:2400]}
                          for c in source_chunks[:60]]
                body={'schema_version':'astrograph-material-reference-proposal.v1',
                      'document_id':document_id,'source_group_id':source_group_id,
                      'filename':filename,'source_sha256':sha,'producer':producer,
                      'chunks':sections,'chunks_truncated':len(source_chunks)>60,
                      'review_status':'PENDING_EXPERT_VERIFICATION','active_kg_modified':False}
                digest=_digest(body)
                row=db.execute('SELECT body_sha256,disposition FROM reference_pending WHERE document_id=?',
                               (document_id,)).fetchone()
                if row and row['body_sha256']!=digest:
                    raise MaterialMemoryConflict('source document proposal mutated')
                if not row:
                    db.execute('INSERT INTO reference_pending(document_id,source_group_id,producer,body,body_sha256,created_at) VALUES(?,?,?,?,?,?)',
                               (document_id,source_group_id,producer,_serial(body),digest,_now()))
                output.append({'document_id':document_id,'status':row['disposition'] if row else 'PENDING',
                               'source_sha256':sha,'chunk_count':len(sections)})
        return output

    def pending_documents(self, *, limit: int = 40) -> list[dict[str, Any]]:
        with self._db(write=False) as db:
            rows=db.execute('SELECT body,created_at FROM reference_pending WHERE disposition=? ORDER BY created_at DESC,document_id LIMIT ?',
                            ('PENDING',min(max(1,limit),200))).fetchall()
        return [{**json.loads(row['body']), 'created_at':row['created_at']} for row in rows]

    def decide_document(self, *, document_id: str, reviewer: str, decision: str,
                        truth_source_ref: str, equipment_scope: str | None = None,
                        expected_snapshot: int | None = None) -> dict[str, Any]:
        if reviewer not in self.reviewers:
            raise PermissionError('reviewer is not in administrator allowlist')
        if decision not in {'CONFIRM','REJECT'}:
            raise ValueError('invalid document review decision')
        if not truth_source_ref or not truth_source_ref.strip():
            raise ValueError('reference provenance/independent check is required')
        with self._db() as db:
            row=db.execute('SELECT body,producer,disposition FROM reference_pending WHERE document_id=?',
                           (document_id,)).fetchone()
            if not row:
                raise KeyError('pending document not found')
            if row['disposition']!='PENDING':
                raise MaterialMemoryConflict('document already decided')
            if reviewer==row['producer']:
                raise PermissionError('reviewer cannot approve own source')
            head=int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            if expected_snapshot is not None and expected_snapshot!=head:
                raise MaterialMemoryConflict('snapshot compare-and-swap failed')
            scope=self._scope(equipment_scope or '') if decision=='CONFIRM' else None
            db.execute('INSERT INTO reference_decisions(document_id,decision,reviewer,truth_source_ref,decided_at) VALUES(?,?,?,?,?)',
                       (document_id,decision,reviewer,truth_source_ref.strip()[:500],_now()))
            db.execute('UPDATE reference_pending SET disposition=? WHERE document_id=?',
                       ('CONFIRMED' if decision=='CONFIRM' else 'REJECTED',document_id))
            if decision=='REJECT':
                return {'document_id':document_id,'status':'REJECTED','snapshot':head}
            body={**json.loads(row['body']),'equipment_scope':scope,'reviewer':reviewer,
                  'truth_source_ref':truth_source_ref.strip()[:500],'available_at':_now(),
                  'review_status':'APPROVED_REFERENCE_NOT_OBSERVATION','snapshot':head+1}
            db.execute('INSERT INTO reference_confirmed(document_id,version,body,available_at,source_group_id,equipment_scope) VALUES(?,?,?,?,?,?)',
                       (document_id,head+1,_serial(body),body['available_at'],body['source_group_id'],scope))
            db.execute('UPDATE head SET version=? WHERE id=1',(head+1,))
            return {'document_id':document_id,'status':'CONFIRMED','snapshot':head+1,
                    'equipment_scope':scope}

    def retrieve_reference_chunks(self, *, equipment_scope: str | None,
                                  exclude_source_group_id: str, snapshot: int,
                                  max_chunks: int = 120) -> list[dict[str, Any]]:
        if not equipment_scope or str(equipment_scope).upper() in {'UNKNOWN','UNSPECIFIED'}:
            return []
        scope=self._scope(equipment_scope)
        with self._db(write=False) as db:
            head=int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            if snapshot<0 or snapshot>head:
                raise ValueError('invalid reference snapshot')
            rows=db.execute("""SELECT body FROM reference_confirmed WHERE version<=?
                 AND equipment_scope=? AND source_group_id<>? AND available_at<=?
                 ORDER BY version DESC,document_id LIMIT 80""",
                 (snapshot,scope,exclude_source_group_id,_now())).fetchall()
        found=[]
        for row in rows:
            doc=json.loads(row['body'])
            for chunk in doc['chunks']:
                found.append({**chunk,'filename':doc['filename'],
                              'source_sha256':doc['source_sha256'],
                              'source_group_id':doc['source_group_id'],
                              'document_id':doc['document_id'],
                              'reviewer':doc['reviewer'],
                              'truth_source_ref':doc['truth_source_ref'],
                              'snapshot':doc['snapshot']})
                if len(found)>=max_chunks:
                    return found
        return found

    def stage_graph_proposals(self, *, source_group_id: str,
                              proposals: list[dict[str, Any]],
                              producer: str = 'ASTROGRAPH_MATERIAL_GRAPH_EXTRACTOR') -> list[dict[str, Any]]:
        """Idempotent staging, without granting a graph write or query vote."""
        from astrograph.integrations.material_graph_builder import GRAPH_PROPOSAL_SCHEMA, ALLOWED_TRIPLES
        if not re.fullmatch(r'JOB-[0-9A-F]{24}', source_group_id):
            raise ValueError('graph source_group_id must be a valid job ID')
        output=[]
        with self._db() as db:
            for proposal in proposals[:500]:
                if (proposal.get('schema_version') != GRAPH_PROPOSAL_SCHEMA or
                    proposal.get('source_group_id') != source_group_id or
                    not re.fullmatch(r'MGR-[0-9A-F]{24}', str(proposal.get('proposal_id'))) or
                    (proposal['source']['entity_type'], proposal['relation_type'],
                     proposal['target']['entity_type']) not in ALLOWED_TRIPLES or
                    not re.fullmatch(r'[0-9a-f]{64}', str(proposal['provenance']['source_sha256'])) or
                    not proposal['provenance'].get('source_locators') or
                    not proposal.get('evidence_quote') or
                    proposal.get('polarity') not in {'AFFIRMATIVE','NEGATED'}):
                    raise ValueError('invalid sourced, ontology-constrained graph proposal')
                identity=proposal['proposal_id']
                data=_serial(proposal)
                digest=hashlib.sha256(data.encode('utf-8')).hexdigest()
                row=db.execute('SELECT body_sha256,disposition FROM graph_pending WHERE proposal_id=?',(identity,)).fetchone()
                if row and row['body_sha256']!=digest:
                    raise MaterialMemoryConflict('graph proposal ID reused with different content')
                if not row:
                    db.execute('INSERT INTO graph_pending(proposal_id,source_group_id,producer,body,body_sha256,created_at) VALUES(?,?,?,?,?,?)',
                               (identity,source_group_id,producer,data,digest,_now()))
                output.append({'proposal_id':identity,'status':row['disposition'] if row else 'PENDING',
                               'content_sha256':digest,'source_sha256':proposal['provenance']['source_sha256']})
        return output

    def pending_graph(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._db(write=False) as db:
            rows=db.execute('SELECT body,created_at FROM graph_pending WHERE disposition=? ORDER BY created_at DESC,proposal_id LIMIT ?',
                            ('PENDING',min(max(1,limit),200))).fetchall()
        return [{**json.loads(r['body']),'created_at':r['created_at']} for r in rows]

    def decide_graph(self, *, proposal_id: str, reviewer: str, decision: str,
                     truth_source_ref: str, equipment_scope: str | None = None,
                     expected_snapshot: int | None = None) -> dict[str, Any]:
        """Independently approve one sourced proposition at a time (not a whole LLM batch)."""
        if reviewer not in self.reviewers:
            raise PermissionError('graph reviewer not in administrator allowlist')
        if decision not in {'CONFIRM','REJECT'}:
            raise ValueError('graph decision must be CONFIRM or REJECT')
        if not isinstance(truth_source_ref,str) or not 3 <= len(truth_source_ref.strip()) <= 500:
            raise ValueError('independent validation reference required')
        with self._db() as db:
            row=db.execute('SELECT body,producer,disposition FROM graph_pending WHERE proposal_id=?',
                           (proposal_id,)).fetchone()
            if row is None:
                raise KeyError('graph proposal not found')
            if reviewer==row['producer']:
                raise PermissionError('extractor must not approve its own graph proposition')
            if row['disposition']!='PENDING':
                raise MaterialMemoryConflict('graph proposition already reviewed; create a new revision')
            head=int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            if expected_snapshot is not None and expected_snapshot!=head:
                raise MaterialMemoryConflict('graph snapshot compare-and-swap failed')
            scope=self._scope(equipment_scope or '') if decision=='CONFIRM' else None
            timestamp=_now()
            db.execute('INSERT INTO graph_decisions(proposal_id,decision,reviewer,truth_source_ref,decided_at) VALUES(?,?,?,?,?)',
                       (proposal_id,decision,reviewer,truth_source_ref.strip(),timestamp))
            db.execute('UPDATE graph_pending SET disposition=? WHERE proposal_id=?',
                       ('CONFIRMED' if decision=='CONFIRM' else 'REJECTED',proposal_id))
            if decision=='REJECT':
                return {'proposal_id':proposal_id,'status':'REJECTED','snapshot':head}
            item=json.loads(row['body'])
            version=head+1
            body={**item,'equipment_scope':scope, 'reviewer':reviewer,
                  'truth_source_ref':truth_source_ref.strip(),'available_at':timestamp,
                  'snapshot':version, 'review_status':'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
                  'current_root_cause_proven':False, 'active_kg_modified':False}
            db.execute('INSERT INTO graph_confirmed(proposal_id,version,body,available_at,source_group_id,equipment_scope) VALUES(?,?,?,?,?,?)',
                       (proposal_id,version,_serial(body),timestamp,item['source_group_id'],scope))
            db.execute('UPDATE head SET version=? WHERE id=1',(version,))
            return {'proposal_id':proposal_id,'status':'CONFIRMED','snapshot':version,
                    'equipment_scope':scope}

    def retract_graph(self, *, proposal_id: str, reviewer: str, reason: str,
                      truth_source_ref: str, expected_snapshot: int | None = None) -> dict[str, Any]:
        """Withdraw an approved reference claim from *future* snapshots.

        A withdrawn claim remains readable in an earlier pinned snapshot and
        its approval/retraction audit trails are never silently rewritten.
        Publication of a newer Neo4j release must synchronize inactive claims.
        """
        if reviewer not in self.reviewers:
            raise PermissionError('graph retractor not in administrator allowlist')
        if not isinstance(reason,str) or not 8 <= len(reason.strip()) <= 1000:
            raise ValueError('an explicit retraction reason (8-1000 chars) is required')
        if not isinstance(truth_source_ref,str) or not 3 <= len(truth_source_ref.strip()) <= 500:
            raise ValueError('independent retraction evidence reference required')
        with self._db() as db:
            approved=db.execute('SELECT equipment_scope FROM graph_confirmed WHERE proposal_id=?',
                                (proposal_id,)).fetchone()
            if approved is None:
                raise KeyError('approved graph claim not found')
            if db.execute('SELECT 1 FROM graph_retractions WHERE proposal_id=?',
                          (proposal_id,)).fetchone() is not None:
                raise MaterialMemoryConflict('graph relation has already been retracted')
            head=int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            if expected_snapshot is not None and expected_snapshot!=head:
                raise MaterialMemoryConflict('graph snapshot compare-and-swap failed')
            version=head+1
            timestamp=_now()
            db.execute('''INSERT INTO graph_retractions
                          (proposal_id,version,reviewer,reason,truth_source_ref,revoked_at)
                          VALUES(?,?,?,?,?,?)''',
                       (proposal_id,version,reviewer,reason.strip(),truth_source_ref.strip(),timestamp))
            db.execute('UPDATE head SET version=? WHERE id=1',(version,))
            return {'proposal_id':proposal_id,'status':'RETRACTED','snapshot':version,
                    'equipment_scope':approved['equipment_scope'],
                    'historical_approvals_preserved':True}

    def graph_retraction_audit(self, *, equipment_scope: str, snapshot: int) -> list[dict[str, Any]]:
        """Host-side audit of tombstones visible in a particular snapshot."""
        scope=self._scope(equipment_scope)
        with self._db(write=False) as db:
            head=int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            if not 0 <= snapshot <= head:
                raise ValueError('invalid graph retraction snapshot')
            rows=db.execute('''SELECT r.proposal_id,r.version,r.reviewer,r.reason,
                                      r.truth_source_ref,r.revoked_at
                               FROM graph_retractions AS r
                               JOIN graph_confirmed AS c ON c.proposal_id=r.proposal_id
                               WHERE c.equipment_scope=? AND r.version<=?
                               ORDER BY r.version,r.proposal_id''',(scope,snapshot)).fetchall()
        return [dict(row) for row in rows]

    def graph_status(self, *, equipment_scope: str | None = None) -> dict[str, Any]:
        """Operational counters only, not a model accuracy or diagnosis metric."""
        scope=self._scope(equipment_scope) if equipment_scope else None
        with self._db(write=False) as db:
            head=int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            pending=int(db.execute("SELECT COUNT(*) FROM graph_pending WHERE disposition='PENDING'").fetchone()[0])
            if scope:
                confirmed=int(db.execute('SELECT COUNT(*) FROM graph_confirmed WHERE equipment_scope=?',
                                         (scope,)).fetchone()[0])
                revoked=int(db.execute('''SELECT COUNT(*) FROM graph_retractions AS r
                                          JOIN graph_confirmed AS c ON c.proposal_id=r.proposal_id
                                          WHERE c.equipment_scope=?''',(scope,)).fetchone()[0])
            else:
                confirmed=int(db.execute('SELECT COUNT(*) FROM graph_confirmed').fetchone()[0])
                revoked=int(db.execute('SELECT COUNT(*) FROM graph_retractions').fetchone()[0])
        status={'schema_version':'astrograph-material-graph-status.v1',
                'snapshot':head,'equipment_scope':scope or 'ALL_SCOPES',
                'pending_unreviewed_relations':pending,
                'approved_relations_ever':confirmed,'retracted_relations':revoked,
                'active_kg_modified':False,
                'meaning':'Counts reference claims and approvals, not verified fault diagnosis precision.'}
        if scope:
            active=self._approved_graph_at(equipment_scope=scope,snapshot=head)
            triples={}
            for entry in active:
                key=(entry['source']['entity_id'],entry['relation_type'],entry['target']['entity_id'])
                triples.setdefault(key,set()).add(entry['polarity'])
            status['active_approved_relations']=len(active)
            status['disputed_relation_groups']=sum(1 for values in triples.values() if len(values)>1)
        return status

    def _approved_graph_at(self, *, equipment_scope: str, snapshot: int,
                           exclude_source_group_id: str = '') -> list[dict[str, Any]]:
        scope=self._scope(equipment_scope)
        with self._db(write=False) as db:
            head=int(db.execute('SELECT version FROM head WHERE id=1').fetchone()[0])
            if not 0 <= snapshot <= head:
                raise ValueError('invalid approved graph snapshot')
            rows=db.execute('''SELECT c.body FROM graph_confirmed AS c
                               LEFT JOIN graph_retractions AS r
                                 ON c.proposal_id=r.proposal_id AND r.version<=?
                               WHERE c.equipment_scope=? AND c.version<=?
                                 AND c.source_group_id<>? AND c.available_at<=?
                                 AND r.proposal_id IS NULL
                               ORDER BY c.version,c.proposal_id LIMIT 10001''',
                            (snapshot,scope,snapshot,exclude_source_group_id,_now())).fetchall()
        if len(rows)>10000:
            raise MaterialMemoryConflict('approved reference graph exceeds safe 10000-edge export/retrieval bound; explicit pagination required')
        return [json.loads(r['body']) for r in rows]

    def approved_graph_for_reasoning(self, *, equipment_scope: str | None,
                                     snapshot: int, exclude_source_group_id: str) -> list[dict[str, Any]]:
        """Only the snapshot's approved, unretracted, exact-model graph claims.

        This is separate from the legacy 2-hop retrieval so that a 3-hop path
        cannot accidentally extend from an already truncated 2-hop edge list.
        """
        if not equipment_scope or equipment_scope.upper() in {'UNKNOWN','UNSPECIFIED'}:
            return []
        return self._approved_graph_at(equipment_scope=equipment_scope, snapshot=snapshot,
                                       exclude_source_group_id=exclude_source_group_id)

    def retrieve_graph(self, *, fault_id: str, fault_name: str,
                       equipment_scope: str | None, exclude_source_group_id: str,
                       snapshot: int, limit: int = 16) -> dict[str, Any]:
        """Two-hop reference graph retrieval, exact equipment and entity identity.

        No fuzzy join across device models and no unreviewed graph proposals.
        Multi-hop paths share stable normalized entity IDs, not text similarity.
        """
        if not equipment_scope or equipment_scope.upper() in {'UNKNOWN','UNSPECIFIED'}:
            return {'status':'EQUIPMENT_BINDING_REQUIRED','paths':[],'snapshot':snapshot}
        from astrograph.integrations.material_graph_builder import _normal
        entries=self._approved_graph_at(equipment_scope=equipment_scope,snapshot=snapshot,
                                        exclude_source_group_id=exclude_source_group_id)
        # Contradictory, independently reviewed reference claims remain visible
        # for investigation; they never silently cancel or endorse one another.
        triples={}
        for edge in entries:
            key=(edge['source']['entity_id'],edge['relation_type'],edge['target']['entity_id'])
            triples.setdefault(key,[]).append(edge)
        conflicts={key for key,claims in triples.items()
                   if {claim['polarity'] for claim in claims}=={'AFFIRMATIVE','NEGATED'}}
        for edge in entries:
            key=(edge['source']['entity_id'],edge['relation_type'],edge['target']['entity_id'])
            edge['conflict_status']=('CONTESTED_BY_APPROVED_REFERENCE' if key in conflicts
                                     else 'NO_APPROVED_POLARITY_CONFLICT')
            edge['opposing_proposal_ids']=[claim['proposal_id'] for claim in triples[key]
                if claim['polarity']!=edge['polarity']][:8]
        seeds={_normal(fault_id),_normal(fault_name)}
        graph={}
        for entry in entries:
            for side in ('source','target'):
                node=entry[side]
                if node['entity_type']=='FAULT_MODE' and any(_normal(v) in seeds for v in [node['name'],*node.get('aliases',[])]) :
                    graph[node['entity_id']]=True
        matched=[]
        frontier=set(graph)
        seen=set()
        for _depth in range(2):
            next_frontier=set()
            for edge in entries:
                a=edge['source']['entity_id']; b=edge['target']['entity_id']
                if a not in frontier or edge['proposal_id'] in seen:
                    continue
                matched.append(edge)
                seen.add(edge['proposal_id'])
                if edge['polarity']=='AFFIRMATIVE':
                    next_frontier.add(b)
                if len(matched)>=min(max(1,limit),60):
                    break
            if len(matched)>=min(max(1,limit),60):
                break
            frontier=next_frontier
        return {'status':'SUCCEEDED','snapshot':snapshot,'paths':matched,
                'matched_edges':len(matched),'knowledge_is_not_current_observation':True,
                'contradicted_relation_groups':len(conflicts),
                'diagnostic_vote_allowed':False}

    def export_graph_release(self, *, equipment_scope: str, snapshot: int,
                             output_root: str | Path) -> dict[str, Any]:
        """Immutable reviewed-claims export; separated from legacy Active KG.

        Nodes/edges conform to AstroGraph's CSV reference envelope, but no
        Neo4j credentials or automatic write are used during graph construction.
        """
        import csv
        import os
        scope=self._scope(equipment_scope)
        edges=self._approved_graph_at(equipment_scope=scope,snapshot=snapshot)
        retractions=self.graph_retraction_audit(equipment_scope=scope,snapshot=snapshot)
        directory=Path(output_root).expanduser().resolve()
        release_name=f'{scope}-SNAPSHOT-{snapshot:08d}'
        final=directory/release_name
        if final.exists():
            raise FileExistsError('immutable graph release already exists')
        staging=directory/(release_name+'.tmp')
        if staging.exists():
            raise FileExistsError('graph release staging directory exists')
        directory.mkdir(parents=True,exist_ok=True)
        staging.mkdir()
        try:
            nodes={node['entity_id']:node for edge in edges for node in (edge['source'],edge['target'])}
            def csv_write(file, headers, rows):
                with (staging/file).open('w',encoding='utf-8',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
            csv_write('nodes.csv',['node_id','node_type','name','properties_json'],[
                {'node_id':key,'node_type':n['entity_type'],'name':n['name'],
                 'properties_json':_serial({'aliases':n.get('aliases',[]),'review_status':'EXPERT_REVIEWED'})}
                for key,n in sorted(nodes.items())])
            csv_write('edges.csv',['source_id','target_id','relation','properties_json'],[
                {'source_id':item['source']['entity_id'],'target_id':item['target']['entity_id'],
                 'relation':item['relation_type'],
                 'properties_json':_serial({'proposal_id':item['proposal_id'],
                   'polarity':item['polarity'],'evidence_quote':item['evidence_quote'],
                   'provenance':item['provenance'],'reviewer':item['reviewer'],
                   'truth_source_ref':item['truth_source_ref'],'version':item['snapshot']})}
                for item in edges])
            # All approved claims (including negated claims) are retained in
            # export with explicit polarity. Downstream MUST filter appropriately.
            (staging/'graph.json').write_text(_serial({'schema_version':'astrograph-material-reviewed-graph.v1',
                'equipment_scope':scope,'snapshot':snapshot,'nodes':list(nodes.values()),
                'relations':edges,'retractions':retractions})+'\n',encoding='utf-8')
            digest={name:hashlib.sha256((staging/name).read_bytes()).hexdigest()
                    for name in ('nodes.csv','edges.csv','graph.json')}
            manifest={'schema_version':'astrograph-material-graph-release.v1',
                      'release_id':release_name,'snapshot':snapshot,'equipment_scope':scope,
                      'node_count':len(nodes),'edge_count':len(edges),
                      'retracted_claim_count':len(retractions),'files_sha256':digest,
                      'status':'EXPERT_REVIEWED_REFERENCE_CLAIMS',
                      'active_kg_modified':False}
            # A checksum manifest catches accidents; it is not an authentic
            # publication authorization. Sign with an administrator-owned
            # secret when available. Only signed releases may be sent to Neo4j.
            signing_key=os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','')
            if signing_key:
                import hmac
                if len(signing_key)<32:
                    raise ValueError('material graph release signing key must have at least 32 characters')
                manifest['signature_algorithm']='HMAC-SHA256'
                signature=hmac.new(signing_key.encode('utf-8'),_serial(manifest).encode('utf-8'),
                                   hashlib.sha256).hexdigest()
                manifest['signature_hmac_sha256']=signature
            (staging/'release.json').write_text(_serial(manifest)+'\n',encoding='utf-8')
            os.rename(staging,final)
            return {**manifest,'directory':str(final)}
        except BaseException:
            import shutil
            shutil.rmtree(staging,ignore_errors=True)
            raise



    def stage_reasoning_gaps(self, *, source_group_id: str, equipment_scope: str,
                             candidates: list[dict[str, Any]], source_digests: list[str],
                             snapshot: int | None) -> list[dict[str, Any]]:
        """Durable per-task knowledge-gap queue; not a graph publication channel."""
        if not re.fullmatch(r'JOB-[A-F0-9]{24}', source_group_id):
            raise ValueError('invalid source group for reasoning gaps')
        if len(candidates) > 5:
            raise ValueError('reasoning gap task exceeds candidate capacity')
        results=[]
        with self._db() as db:
            for candidate in candidates:
                codes=list(dict.fromkeys(candidate.get('knowledge_gap_codes') or []))[:20]
                if not codes:
                    continue
                fault_id=str(candidate.get('fault_id') or 'NO_UPSTREAM_CANDIDATE')[:160]
                scope=str(equipment_scope or 'UNKNOWN').upper()
                gap_id='MRGAP-'+_digest({'job':source_group_id,'fault':fault_id,
                                       'codes':codes})[:24].upper()
                body={'schema_version':'astrograph-material-reasoning-gap.v1',
                      'gap_id':gap_id,'source_group_id':source_group_id,
                      'equipment_scope':scope,'fault_id':fault_id,
                      'knowledge_gap_codes':codes,'graph_snapshot':snapshot,
                      'source_sha256_list':[x for x in source_digests[:40]
                          if isinstance(x,str) and re.fullmatch(r'[a-f0-9]{64}',x)],
                      'claim_boundary':'QUESTION_FOR_EXPERT_NOT_APPROVED_GRAPH_FACT'}
                digest=_digest(body)
                row=db.execute('SELECT body_sha256,status FROM reasoning_gaps WHERE gap_id=?',
                               (gap_id,)).fetchone()
                if row and row['body_sha256']!=digest:
                    raise MaterialMemoryConflict('reasoning gap identity reused with changed content')
                if not row:
                    db.execute("""INSERT INTO reasoning_gaps
                        (gap_id,source_group_id,equipment_scope,fault_id,body,body_sha256,created_at)
                        VALUES(?,?,?,?,?,?,?)""",
                        (gap_id,source_group_id,scope,fault_id,_serial(body),digest,_now()))
                results.append({'gap_id':gap_id,'status':row['status'] if row else 'OPEN',
                                'fault_id':fault_id,'knowledge_gap_codes':codes})
        return results

    def pending_reasoning_gaps(self, *, equipment_scope: str | None = None,
                               limit: int = 40) -> list[dict[str, Any]]:
        """Protected expert worklist, sorted deterministically, no log excerpts."""
        with self._db(write=False) as db:
            if equipment_scope:
                scope=self._scope(equipment_scope)
                rows=db.execute("""SELECT body,created_at FROM reasoning_gaps
                    WHERE status='OPEN' AND equipment_scope=? ORDER BY created_at DESC,gap_id LIMIT ?""",
                    (scope,min(max(1,limit),200))).fetchall()
            else:
                rows=db.execute("""SELECT body,created_at FROM reasoning_gaps
                    WHERE status='OPEN' ORDER BY created_at DESC,gap_id LIMIT ?""",
                    (min(max(1,limit),200),)).fetchall()
        return [{**json.loads(r['body']),'created_at':r['created_at'],
                 'status':'OPEN'} for r in rows]

    def resolve_reasoning_gap(self, *, gap_id: str, reviewer: str,
                              resolution_ref: str) -> dict[str, Any]:
        """Mark review ticket addressed; no automatic graph promotion."""
        if reviewer not in self.reviewers:
            raise PermissionError('reviewer is not in administrator allowlist')
        if not isinstance(resolution_ref,str) or not 8 <= len(resolution_ref.strip()) <= 500:
            raise ValueError('independent resolution/evidence reference required')
        with self._db() as db:
            row=db.execute('SELECT status FROM reasoning_gaps WHERE gap_id=?',(gap_id,)).fetchone()
            if not row:
                raise KeyError('knowledge gap not found')
            if row['status']!='OPEN':
                raise MaterialMemoryConflict('knowledge gap is already resolved')
            db.execute("""UPDATE reasoning_gaps SET status='RESOLVED', resolver=?,
                       resolution_ref=?, resolved_at=? WHERE gap_id=?""",
                       (reviewer,resolution_ref.strip(),_now(),gap_id))
        return {'gap_id':gap_id,'status':'RESOLVED',
                'graph_promoted':False,'current_root_cause_proven':False}


__all__ = ['MaterialCaseMemory','MaterialMemoryConflict']
