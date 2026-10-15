"""Bounded, provenance-first hierarchical hybrid retrieval for material diagnosis.

Document -> section -> original fragment retrieval.  This is *not* Microsoft
GraphRAG global community summarization; approved graph terms supply auditable
query hints, while a separate graph engine checks causal paths.  Retrieval never
confirms a root cause or authorizes an action.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping, Sequence

from astrograph.integrations.platform_knowledge import _bm25, _tokens
from astrograph.retrieval.spacecraft_terminology import (
    expand_spacecraft_terms, load_pinned_domain_lexicon,
)

SCHEMA = "astrograph-material-hierarchical-hybrid-rag.v1"
MAX_CHUNKS = 6000
MAX_INDEX_CHARS = 12000
MAX_DOCS = 8
MAX_SECTIONS = 16
MAX_QUERY_VARIANTS = 14


def _ngrams(text: str) -> set[str]:
    import re
    latin = set(re.findall(r"[a-z][a-z0-9_-]{2,}", text.casefold()))
    grams = {s[i:i + length] for s in re.findall(r"[\u4e00-\u9fff]+", text)
             for length in (2, 3) for i in range(max(0, len(s) - length + 1))}
    return latin | grams


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().strip().split())


def _source_key(chunk: Any) -> str:
    return str(chunk.source_id)


def _section_key(chunk: Any) -> tuple[str, tuple[str, ...]]:
    return (_source_key(chunk), tuple(chunk.section_path))


def _safe_chunk(chunk: Any) -> bool:
    # Unreviewed OCR/caption/formula/table reconstruction cannot be a factual
    # quotation in the answer. Native source text or independently approved
    # historical references remain eligible for topical retrieval only.
    return (chunk.content_safety_status in {"PASS", "REVIEWED"}
            and chunk.review_status in {"HUMAN_REVIEWED", "EXPERT_APPROVED_REFERENCE"})


@dataclass(frozen=True)
class RetrievalBudgets:
    documents: int = MAX_DOCS
    sections: int = MAX_SECTIONS
    hits: int = 5
    query_variants: int = MAX_QUERY_VARIANTS

    def __post_init__(self) -> None:
        if not 1 <= self.documents <= 12 or not 1 <= self.sections <= 32:
            raise ValueError("invalid coarse retrieval budgets")
        if not 1 <= self.hits <= 12 or not 1 <= self.query_variants <= 20:
            raise ValueError("invalid fine retrieval budgets")


def _anchored_excerpt(text: str, plan: Sequence[Mapping[str, Any]],
                      max_chars: int = 1200) -> tuple[str, int]:
    """Return a literal contiguous excerpt containing the strongest exact probe.

    Prevent citing the head of a long section/table when the actual matching
    identifier appears only near the end.  Offsets refer to the indexed chunk,
    not a rewritten or stitched quotation.
    """
    if len(text) <= max_chars:
        return text, 0
    source = text.casefold()
    matches = []
    for probe in plan:
        query = str(probe['query']).casefold().strip()
        if len(query) < 3:
            continue
        where = source.find(query)
        if where >= 0:
            specificity = len(query) if len(query) < 140 else 140
            matches.append((float(probe['weight']) * (1 + specificity / 100),
                            where, len(query)))
    if not matches:
        # Probe could match only individual terms. Prefer the first longest
        # alphanumeric token instead of a fabricated paraphrase.
        tokens = sorted({word for probe in plan for word in
                         str(probe['query']).casefold().split() if len(word) >= 4},
                        key=lambda word: -len(word))
        for word in tokens:
            position = source.find(word)
            if position >= 0:
                matches.append((1.0, position, len(word)))
                break
    if not matches:
        return text[:max_chars], 0
    _, position, term_length = sorted(matches, key=lambda row:(-row[0], row[1]))[0]
    start = min(max(0, position - max_chars // 4), len(text) - max_chars)
    if position + term_length > start + max_chars:
        start = min(position, len(text) - max_chars)
    return text[start:start + max_chars], start


def _table_row_windows(chunk: Any, max_chars: int = 1400) -> list[Any]:
    """Return contiguous original table line windows plus separate header context.

    The header is only used for matching/interpretation, NOT spliced into the
    evidence quote.  Each locator retains the original source reference and a
    one-based line range relative to the parsed table.
    """
    if getattr(chunk, 'modalities', ()) != ('TABLE',) or len(chunk.text) <= max_chars:
        return [chunk]
    lines = chunk.text.splitlines(keepends=True)
    if len(lines) < 3 or not lines[0].lstrip().startswith('|'):
        return [chunk]
    if not all(line.lstrip().startswith('|') for line in lines if line.strip()):
        return [chunk]
    header = lines[0].rstrip('\r\n')
    # Markdown separator is a header-affiliated cell-format indicator, not a
    # data row; keep it out of excerpts, without changing source coordinates.
    base = 2 if len(lines) > 1 and set(lines[1].strip()) <= set('|- :') else 1
    out: list[Any] = []
    start = base
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines):
            next_size = len(lines[end])
            if end > start and size + next_size > max_chars:
                break
            size += next_size
            end += 1
        quote = ''.join(lines[start:end]).strip('\r\n')
        if quote.strip():
            locators = tuple(f'{loc}#table-lines:{start + 1}-{end}'
                             for loc in chunk.source_locators)
            out.append(replace(
                chunk, chunk_id=f'{chunk.chunk_id}:table-window:{len(out)+1:04d}',
                text=quote, source_locators=locators, table_header_context=header,
                derived_table_window=True,
            ))
        start = end
    return out or [chunk]


class HierarchicalMaterialIndex:
    """Pre-index chunks once per task; serve each upstream candidate separately."""

    def __init__(self, chunks: Sequence[Any]) -> None:
        self.all_count = len(chunks)
        # The cap is explicit in every result; no hidden overflow.
        self.truncated = len(chunks) > MAX_CHUNKS
        self.chunks = list(chunks[:MAX_CHUNKS])
        # A large native table is searched as precise row windows; each
        # evidence quote remains a contiguous excerpt of the original table.
        self.eligible = [part for c in self.chunks if _safe_chunk(c)
                         for part in _table_row_windows(c)]
        self.suppressed = [c for c in self.chunks if not _safe_chunk(c)]
        self.chunk_by_id = {c.chunk_id: c for c in self.eligible}
        self.docs = defaultdict(list)
        self.sections = defaultdict(list)
        self.doc_modalities = defaultdict(Counter)
        for chunk in self.eligible:
            self.docs[_source_key(chunk)].append(chunk)
            self.sections[_section_key(chunk)].append(chunk)
        self.tokens = [_tokens(" ".join(c.section_path) + " "
                               + getattr(c, 'table_header_context', '') + " "
                               + c.text[:MAX_INDEX_CHARS])
                       for c in self.eligible]
        self.ngrams = [_ngrams(" ".join(c.section_path) + " "
                                 + getattr(c, 'table_header_context', '') + " "
                                 + c.text[:MAX_INDEX_CHARS])
                       for c in self.eligible]
        self.document_count = len({_source_key(c) for c in self.chunks})

    @staticmethod
    def _graph_hints(fault_id: str, fault_name: str,
                     approved_edges: Sequence[Mapping[str, Any]], *,
                     equipment_scope: str | None,
                     snapshot: int | None = None) -> list[str]:
        """Expand only from independently reviewed, scope-bound positive edges.

        Untrusted graph suggestions and negative/contested claims must never
        expand a live retrieval query.  A reverse neighbor is a *lookup hint*,
        not a reversed causal statement.  Contradictions veto that exact triple.
        """
        from astrograph.retrieval.approved_community_search import _edge_ok

        if not equipment_scope or equipment_scope.upper() in {"UNKNOWN", "UNSPECIFIED"}:
            return []
        eligible = [edge for edge in approved_edges if _edge_ok(
            edge, equipment_scope, snapshot=snapshot, allow_negated=True)
            and edge.get('claim_modality') in (None, 'REFERENCE_ASSERTION_REQUIRES_REVIEW')]
        conflicts = {
            (edge['source']['entity_id'], edge['relation_type'], edge['target']['entity_id'])
            for edge in eligible if edge['polarity'] == 'NEGATED'
        }
        names = {_norm(fault_id), _norm(fault_name)}
        hints: list[str] = []
        for edge in sorted(eligible, key=lambda e: str(e.get('proposal_id', ''))):
            if edge['polarity'] != 'AFFIRMATIVE' or edge.get('claim_modality') == 'HYPOTHETICAL_OR_CONDITIONAL':
                continue
            triple = (edge['source']['entity_id'], edge['relation_type'], edge['target']['entity_id'])
            if triple in conflicts:
                continue
            for near, far in ((edge['source'], edge['target']), (edge['target'], edge['source'])):
                # A graph node with a coincidental alias is not a valid fault anchor.
                if near.get('entity_type') != 'FAULT_MODE':
                    continue
                labels = [near.get('name', ''), near.get('entity_id', ''),
                          *(near.get('aliases') or [])]
                if any(_norm(str(label)) in names for label in labels if label):
                    hint = str(far.get('name', '')).strip()
                    if 3 <= len(hint) <= 100 and hint not in hints:
                        hints.append(hint)
                        if len(hints) >= 6:
                            return hints
        return hints

    @staticmethod
    def _plan(base_query: str, *, fault_id: str, fault_name: str,
              reference_terms: Sequence[str], graph_hints: Sequence[str],
              budgets: RetrievalBudgets,
              extra_concepts: tuple[tuple[str, tuple[str, ...]], ...] = (),
              glossary_receipt: Mapping[str, Any] | None = None,
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        # Avoid one enormous, token-diluted query.  Keep separate plans for
        # entity, mechanism/measurement and approved graph-neighbor hints.
        candidates: list[tuple[str, str, float]] = [
            ("ORIGINAL_CANDIDATE", base_query[:900], 1.2),
            ("FAULT_ID", fault_id[:120], 1.5),
            ("FAULT_NAME", fault_name[:200], 1.35),
        ]
        for hint in list(reference_terms)[:6]:
            candidates.append(("CURATED_REFERENCE_TERM", str(hint)[:120], 0.85))
        expansion = expand_spacecraft_terms(
            " ".join([base_query, fault_name, *list(reference_terms)[:6]]),
            extra_concepts=extra_concepts)
        expansion['mission_glossary'] = dict(glossary_receipt or {})
        for hint in expansion["alternatives"]:
            candidates.append(("CURATED_SYNONYM", str(hint), 0.7))
        for hint in graph_hints:
            candidates.append(("APPROVED_GRAPH_QUERY_HINT", hint, 0.5))
        seen: set[str] = set()
        plan: list[dict[str, Any]] = []
        for role, text, weight in candidates:
            key = _norm(text)
            if key in seen or len(key) < 2:
                continue
            seen.add(key)
            plan.append({"role": role, "query": text, "weight": weight})
            if len(plan) == budgets.query_variants:
                break
        return plan, expansion

    def retrieve(self, *, base_query: str, fault_id: str, fault_name: str,
                 reference_terms: Sequence[str] = (),
                 semantic_ids: Sequence[str] = (),
                 approved_edges: Sequence[Mapping[str, Any]] = (),
                 equipment_scope: str | None = None,
                 graph_snapshot: int | None = None,
                 budgets: RetrievalBudgets | None = None) -> dict[str, Any]:
        limits = budgets or RetrievalBudgets()
        hints = self._graph_hints(fault_id, fault_name, approved_edges,
                                  equipment_scope=equipment_scope, snapshot=graph_snapshot)
        extra_concepts, glossary_receipt = load_pinned_domain_lexicon(equipment_scope)
        plan, expansion = self._plan(base_query, fault_id=fault_id,
                                      fault_name=fault_name,
                                      reference_terms=reference_terms,
                                      graph_hints=hints, budgets=limits,
                                      extra_concepts=extra_concepts,
                                      glossary_receipt=glossary_receipt)
        modality_counts = Counter(kind for chunk in self.chunks
                                  for kind in (getattr(chunk, "modalities", ()) or ("UNKNOWN",)))
        unsafe_modality_counts = Counter(kind for chunk in self.suppressed
                                         for kind in (getattr(chunk, "modalities", ()) or ("UNKNOWN",)))
        trace: dict[str, Any] = {
            "document_index_count": self.document_count,
            "chunk_index_count": len(self.chunks),
            "eligible_chunk_count": len(self.eligible),
            "unsafe_representation_excluded_count": len(self.suppressed),
            "index_truncated": self.truncated,
            "corpus_completeness": "PARTIAL_UNINDEXED_CONTENT_REQUIRES_REVIEW" if self.truncated else "INDEXED_INPUT_COMPLETE",
            "unindexed_chunk_count": max(0, self.all_count - len(self.chunks)),
            "budgets": {"documents": limits.documents, "sections": limits.sections,
                        "fine_hits": limits.hits, "query_variants": limits.query_variants},
            "modalities": dict(sorted(modality_counts.items())),
            "excluded_modalities": dict(sorted(unsafe_modality_counts.items())),
            "stages": [],
        }
        methods = ["BM25", "CHAR_NGRAM"]
        if semantic_ids:
            methods.append("LOCAL_EMBEDDING")
        if hints:
            methods.append("APPROVED_GRAPH_HINTS")
        skeleton = {"schema_version": SCHEMA, "status": "NO_RELEVANT_PASSAGES",
                    "retrieval_mode": "HIERARCHICAL_HYBRID_GRAPH_HINTS" if hints else "HIERARCHICAL_HYBRID",
                    "query": base_query, "methods": methods,
                    "query_plan": plan, "terminology": expansion, "retrieval_trace": trace,
                    "hits": [], "causal_proof": False, "automatic_control_authorized": False,
                    "claim_boundary": "Topical snippets; no generated summaries, causal proof or automatic action."}
        if not self.eligible or not plan:
            skeleton["status"] = ("PARTIAL_CORPUS_LIMIT_REVIEW_REQUIRED" if self.truncated
                                  else "NO_ELIGIBLE_REPRESENTATIONS" if self.suppressed
                                  else "NO_RELEVANT_PASSAGES")
            trace["stages"].append({"stage": "ADMISSION", "status": skeleton["status"]})
            return skeleton

        # Stage 1: candidate-specific bounded parallel probes: BM25, character
        # overlap and optional local embedding ranking.  No LLM rewrite.
        ranked_score: dict[str, float] = defaultdict(float)
        channels: dict[str, set[str]] = defaultdict(set)
        for probe in plan:
            terms = _tokens(probe["query"])
            bm = _bm25(terms, self.tokens)
            good = sorted(((-value, i) for i, value in enumerate(bm) if value > 0),
                          key=lambda pair: (pair[0], self.eligible[pair[1]].chunk_id))[:100]
            for rank, (_, i) in enumerate(good, 1):
                cid = self.eligible[i].chunk_id
                ranked_score[cid] += float(probe["weight"]) / (60 + rank)
                channels[cid].add("BM25")
            grams = _ngrams(probe["query"])
            if grams:
                positive = []
                for i, tokens in enumerate(self.ngrams):
                    common = len(grams & tokens)
                    if common:
                        normalized = common / math.sqrt(max(1, len(grams)) * max(1, len(tokens)))
                        positive.append((-normalized, i))
                for rank, (_, i) in enumerate(sorted(positive,
                    key=lambda pair: (pair[0], self.eligible[pair[1]].chunk_id))[:100], 1):
                    cid = self.eligible[i].chunk_id
                    ranked_score[cid] += float(probe["weight"]) * 0.6 / (60 + rank)
                    channels[cid].add("CHAR_NGRAM")
        for rank, cid in enumerate(semantic_ids[:100], 1):
            if cid in self.chunk_by_id:
                ranked_score[cid] += 1.2 / (60 + rank)
                channels[cid].add("LOCAL_EMBEDDING")
        trace["stages"].append({"stage": "PARALLEL_LEXICAL_AND_SEMANTIC_PROBES",
                                "scored_chunks": len(ranked_score),
                                "query_variants_executed": len(plan), "channels": methods})
        if not ranked_score:
            skeleton["status"] = ("PARTIAL_CORPUS_LIMIT_REVIEW_REQUIRED" if self.truncated
                                  else "NO_RELEVANT_PASSAGES")
            trace["stages"].append({"stage": "ABSTENTION_GATE",
                                     "reason": "NO_QUERY_SUPPORTED_CHUNK"})
            return skeleton

        # Stage 2: coarse doc-level + section-level selection, populated by the
        # BEST scored leaf anywhere within the document (including late pages).
        doc_score: dict[str, float] = defaultdict(float)
        section_score: dict[tuple[str, tuple[str, ...]], float] = defaultdict(float)
        for cid, score in ranked_score.items():
            chunk = self.chunk_by_id[cid]
            doc = _source_key(chunk)
            sec = _section_key(chunk)
            doc_score[doc] = max(doc_score[doc], score)
            section_score[sec] = max(section_score[sec], score)
        sorted_docs = sorted(doc_score, key=lambda key: (-doc_score[key], key))[:limits.documents]
        doc_set = set(sorted_docs)
        sorted_sections = sorted(
            (sec for sec in section_score if sec[0] in doc_set),
            key=lambda key: (-section_score[key], key),
        )[:limits.sections]
        sec_set = set(sorted_sections)
        trace["stages"].append({"stage": "DOCUMENT_SELECTION",
                                "candidate_count": len(doc_score),
                                "selected": [{"document_id": key, "score": round(doc_score[key], 8)}
                                             for key in sorted_docs]})
        trace["stages"].append({"stage": "SECTION_SELECTION",
                                "candidate_count": len(section_score),
                                "selected": [{"document_id": doc, "section_path": list(section),
                                              "score": round(section_score[(doc, section)], 8)}
                                             for doc, section in sorted_sections]})

        # Stage 3: choose *original* fine-grained citations.  Interleave top
        # sections before filling remaining slots to reduce near-duplicate hits.
        fine_ids = [cid for cid in ranked_score
                    if _section_key(self.chunk_by_id[cid]) in sec_set]
        fine_ids.sort(key=lambda cid: (-ranked_score[cid], cid))
        selected: list[str] = []
        used_sections: set[tuple[str, tuple[str, ...]]] = set()
        for cid in fine_ids:
            sec = _section_key(self.chunk_by_id[cid])
            if sec not in used_sections:
                selected.append(cid)
                used_sections.add(sec)
                if len(selected) == limits.hits:
                    break
        if len(selected) < limits.hits:
            for cid in fine_ids:
                if cid not in selected:
                    selected.append(cid)
                if len(selected) == limits.hits:
                    break
        hits = []
        for rank, cid in enumerate(selected, 1):
            c = self.chunk_by_id[cid]
            text, excerpt_offset = _anchored_excerpt(c.text, plan)
            hits.append({"rank": rank, "knowledge_id": cid,
                         "filename": c.filename, "source_id": c.source_id,
                         "source_sha256": c.source_sha256,
                         "section_path": list(c.section_path),
                         "source_locators": list(c.source_locators),
                         "modalities": list(getattr(c, "modalities", ()) or ("UNKNOWN",)),
                         "quote": text, "quote_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "table_header_context": getattr(c, 'table_header_context', ''),
                         "table_row_window": getattr(c, 'derived_table_window', False),
                         "quote_truncated": len(c.text) > len(text),
                         "quote_chunk_char_offset": excerpt_offset,
                         "quote_is_contiguous_original_representation": True,
                         "fusion_score": round(ranked_score[cid], 8),
                         "retrieval_channels": sorted(channels[cid]),
                         "knowledge_validation_status": (
                             "EXPERT_APPROVED_HISTORICAL_REFERENCE"
                             if c.review_status == "EXPERT_APPROVED_REFERENCE"
                             else "UNREVIEWED_UPLOADED_REFERENCE"),
                         "representation_safety_status": c.content_safety_status,
                         "support_scope": "TOPICAL_KNOWLEDGE_ONLY_NOT_OBSERVED_CAUSE",
                         "diagnostic_vote_allowed": False})
        trace["stages"].append({"stage": "FINE_CITATION_RERANK_AND_DIVERSIFY",
                                "eligible_fine_count": len(fine_ids),
                                "returned": [hit["knowledge_id"] for hit in hits],
                                "score_semantics": "RRF_ORDINAL_NOT_PROBABILITY"})
        skeleton["status"] = ("PARTIAL_CORPUS_LIMIT_REVIEW_REQUIRED" if self.truncated
                              else "SUCCEEDED" if hits else "NO_RELEVANT_PASSAGES")
        skeleton["hits"] = hits
        return skeleton


    @staticmethod
    def retrieve_reviewed_counterevidence(*, fault_id: str, fault_name: str,
                                          approved_edges: Sequence[Mapping[str, Any]],
                                          equipment_scope: str | None,
                                          graph_snapshot: int | None = None,
                                          limit: int = 12) -> dict[str, Any]:
        """Expose signed, independently reviewed *source statements* for review.

        This is NOT an automatic fault refutation. Only directly fault-anchored
        negative or contradictory triples are returned; a missing graph relation
        is never equated with a negative claim. Provenance remains on each quote.
        """
        from astrograph.integrations.material_multihop import _scoped_approved

        if not 1 <= limit <= 32:
            raise ValueError("counterevidence limit must be 1..32")
        scope = str(equipment_scope or "UNKNOWN").upper()
        empty = {'schema_version': 'astrograph-reviewed-counterevidence-rag.v1',
                 'status': 'ABSTAIN_NO_REVIEWED_COUNTERCLAIM',
                 'fault_id': fault_id, 'items': [],
                 'negative_claim_is_not_current_observation': True,
                 'automatic_diagnostic_vote_allowed': False}
        if scope in {"UNKNOWN", "UNSPECIFIED"}:
            return {**empty, 'status': 'ABSTAIN_EQUIPMENT_SCOPE_REQUIRED'}
        eligible, _blocked = _scoped_approved(approved_edges, scope)
        eligible = [e for e in eligible if graph_snapshot is None
                    or e.get('snapshot') is None
                    or (type(e['snapshot']) is int and 0 <= e['snapshot'] <= graph_snapshot)]
        polarity_by_triple: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for e in eligible:
            if e['claim_modality'] != 'REFERENCE_ASSERTION_REQUIRES_REVIEW':
                continue
            triple = (e['source']['entity_id'], e['relation_type'],
                      e['target']['entity_id'])
            polarity_by_triple[triple].add(e['polarity'])
        names = {_norm(fault_id), _norm(fault_name)}
        rows = []
        for e in sorted(eligible, key=lambda record: record['edge_id']):
            triple = (e['source']['entity_id'],e['relation_type'],
                      e['target']['entity_id'])
            conflicted = (e['claim_modality'] == 'REFERENCE_ASSERTION_REQUIRES_REVIEW'
                          and len(polarity_by_triple[triple]) > 1)
            if not conflicted and e['polarity'] != 'NEGATED':
                continue
            if not any(node.get('entity_type') == 'FAULT_MODE' and
                    names.intersection({_norm(str(label)) for label in
                      [node.get('name'),*(node.get('aliases') or [])] if label})
                    for node in (e['source'],e['target'])):
                continue
            rows.append({
                'edge_id': e['edge_id'], 'polarity': e['polarity'],
                'status': ('CONDITIONAL_NEGATION_REQUIRES_REVIEW'
                           if e['claim_modality'] == 'HYPOTHETICAL_OR_CONDITIONAL'
                           else 'REVIEWED_CONFLICTING_TRIPLE' if conflicted
                           else 'REVIEWED_NEGATIVE_RELATION'),
                'relation_type': e['relation_type'],
                'subject': e['source']['name'], 'object': e['target']['name'],
                'quote': e['evidence_quote'],
                'source_sha256': e['provenance']['source_sha256'],
                'source_locators': list(e['provenance']['source_locators']),
                'truth_source_ref': e['truth_source_ref'],
                'claim_modality': e['claim_modality'],
                'observed_telemetry_fact': False,
            })
        return {**empty, 'status': 'REVIEWED_COUNTERCLAIMS_FOUND' if rows else
                empty['status'], 'items': rows[:limit],
                'truncated': len(rows) > limit}

    def retrieve_contrastive(self, *, base_query: str, fault_id: str,
                             fault_name: str, competitor_fault_id: str,
                             competitor_fault_name: str,
                             approved_edges: Sequence[Mapping[str, Any]] = (),
                             equipment_scope: str | None = None,
                             graph_snapshot: int | None = None,
                             budgets: RetrievalBudgets | None = None) -> dict[str, Any]:
        """A two-sided literature comparison, not a diagnosis or negative vote.

        Each side uses exactly the same governed corpus and budgets. References
        appearing only on one side are *discrimination review pointers*, not
        proof that the other hypothesis is false. Shared sources are explicitly
        flagged to avoid counting them as independent corroboration.
        """
        if not fault_id or not competitor_fault_id or fault_id == competitor_fault_id:
            raise ValueError("contrastive retrieval requires two distinct candidate IDs")
        common = dict(approved_edges=approved_edges,
                      equipment_scope=equipment_scope,
                      graph_snapshot=graph_snapshot, budgets=budgets)
        primary = self.retrieve(base_query=base_query, fault_id=fault_id,
                                fault_name=fault_name, **common)
        alternative = self.retrieve(base_query=base_query,
                                    fault_id=competitor_fault_id,
                                    fault_name=competitor_fault_name, **common)
        primary_sources = {h['source_sha256'] for h in primary['hits']}
        alternative_sources = {h['source_sha256'] for h in alternative['hits']}
        return {
            'schema_version': 'astrograph-contrastive-material-rag.v1',
            'status': 'REVIEW_POINTERS_AVAILABLE' if primary['hits'] or alternative['hits']
                      else 'ABSTAIN_NO_RELEVANT_PASSAGES',
            'primary_fault_id': fault_id,
            'competitor_fault_id': competitor_fault_id,
            'primary': primary, 'competitor': alternative,
            'shared_source_sha256': sorted(primary_sources & alternative_sources),
            'primary_only_chunk_ids': sorted(
                {h['knowledge_id'] for h in primary['hits']} -
                {h['knowledge_id'] for h in alternative['hits']}),
            'competitor_only_chunk_ids': sorted(
                {h['knowledge_id'] for h in alternative['hits']} -
                {h['knowledge_id'] for h in primary['hits']}),
            'status_boundary': 'NON_VOTING_DIFFERENTIAL_LITERATURE_POINTERS',
            'evidence_absence_is_not_refutation': True,
            'current_root_cause_proven': False,
            'automatic_control_authorized': False,
        }


__all__ = ["HierarchicalMaterialIndex", "RetrievalBudgets", "SCHEMA"]
