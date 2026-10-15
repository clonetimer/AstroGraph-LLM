from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
import re
from typing import Protocol

from .document_governance import evaluate_document_access

from .models import (
    GovernedTextChunk,
    LocalRetrievalHit,
    OntologyConcept,
    OntologyPathHop,
    OntologyRelation,
    OntologyRetrievalPath,
    PreReasoningRetrievalPackage,
    PreReasoningRetrievalRequest,
    RetrievalStatus,
)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.-]*|[\u4e00-\u9fff]", re.I)


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in _TOKEN_RE.findall(text)]


def _sha256_json(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _bm25(query_terms: list[str], documents: list[list[str]]) -> list[float]:
    if not query_terms or not documents:
        return [0.0 for _ in documents]
    average_length = sum(len(item) for item in documents) / len(documents)
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    result: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        # Stable summation order avoids process-dependent last-bit drift from
        # Python hash randomization, which otherwise changes package identities.
        for term in sorted(set(query_terms)):
            frequency = frequencies.get(term, 0)
            if frequency <= 0:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * len(document) / max(1.0, average_length)
            )
            score += inverse * frequency * 2.5 / max(denominator, 1e-12)
        result.append(score)
    return result


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    minimum = min(values)
    if maximum <= minimum:
        return [1.0 if maximum > 0.0 else 0.0 for _ in values]
    return [(item - minimum) / (maximum - minimum) for item in values]


class DenseScoreProvider(Protocol):
    """Read-only adapter; implementations return normalized scores for supplied chunks only."""

    def score(
        self,
        *,
        query: str,
        chunks: list[GovernedTextChunk],
    ) -> dict[str, float]: ...


class PreReasoningRetrievalService:
    service_id = "pre-reasoning-local-oag-retrieval"
    service_version = "0.1.0"

    def __init__(self, *, dense_score_provider: DenseScoreProvider | None = None) -> None:
        self.dense_score_provider = dense_score_provider

    @staticmethod
    def _eligible_concepts(request: PreReasoningRetrievalRequest) -> dict[str, OntologyConcept]:
        caller_tags = set(request.query.caller_access_tags)
        return {
            item.concept_id: item
            for item in request.concepts
            if item.lifecycle_status == "ACTIVE"
            and item.review_status == "HUMAN_REVIEWED"
            and set(item.required_access_tags).issubset(caller_tags)
        }

    @staticmethod
    def _seed_concepts(
        request: PreReasoningRetrievalRequest,
        concepts: dict[str, OntologyConcept],
    ) -> set[str]:
        normalized_query = request.query.text.casefold()
        seeds = set(request.query.seed_concept_ids)
        for concept_id, concept in concepts.items():
            if any(term.casefold() in normalized_query for term in [*concept.labels, *concept.aliases]):
                seeds.add(concept_id)
        return seeds & set(concepts)

    @staticmethod
    def _ontology_paths(
        request: PreReasoningRetrievalRequest,
        concepts: dict[str, OntologyConcept],
        seeds: set[str],
    ) -> tuple[list[OntologyRetrievalPath], set[str], bool]:
        allowed_relations = set(request.config.allowed_relation_types)
        caller_tags = set(request.query.caller_access_tags)
        adjacency: dict[str, list[OntologyRelation]] = defaultdict(list)
        for relation in request.relations:
            if (
                relation.lifecycle_status == "ACTIVE"
                and relation.review_status == "HUMAN_REVIEWED"
                and relation.relation_type in allowed_relations
                and relation.source_concept_id in concepts
                and relation.target_concept_id in concepts
                and set(relation.required_access_tags).issubset(caller_tags)
            ):
                adjacency[relation.source_concept_id].append(relation)
        paths_truncated = False
        for concept_id, values in adjacency.items():
            values.sort(key=lambda item: (item.relation_type, item.target_concept_id, item.relation_id))
            if len(values) > request.config.maximum_relations_per_concept:
                paths_truncated = True
                adjacency[concept_id] = values[: request.config.maximum_relations_per_concept]

        paths: list[OntologyRetrievalPath] = []
        reached = set(seeds)
        for seed in sorted(seeds):
            queue = deque([(seed, [])])
            visited_depth = {seed: 0}
            while queue and len(paths) < request.config.maximum_ontology_paths:
                current, history = queue.popleft()
                if len(history) >= request.query.maximum_ontology_hops:
                    continue
                for relation in adjacency.get(current, []):
                    visited_nodes = {seed}
                    visited_nodes.update(hop.target_concept_id for hop in history)
                    if relation.target_concept_id in visited_nodes:
                        continue
                    hop = OntologyPathHop(
                        relation_id=relation.relation_id,
                        source_concept_id=relation.source_concept_id,
                        relation_type=relation.relation_type,
                        target_concept_id=relation.target_concept_id,
                        source_id=relation.source_id,
                        source_sha256=relation.source_sha256,
                        knowledge_version=relation.knowledge_version,
                        source_fragment_ids=relation.source_fragment_ids,
                        epistemic_status=relation.epistemic_status,
                        association_semantics=relation.association_semantics,
                        retrieval_only=relation.retrieval_only,
                        diagnostic_evidence_allowed=relation.diagnostic_evidence_allowed,
                        runtime_vote_allowed=relation.runtime_vote_allowed,
                    )
                    next_history = [*history, hop]
                    paths.append(
                        OntologyRetrievalPath(
                            seed_concept_id=seed,
                            target_concept_id=relation.target_concept_id,
                            hops=next_history,
                        )
                    )
                    if len(paths) >= request.config.maximum_ontology_paths:
                        paths_truncated = True
                        break
                    reached.add(relation.target_concept_id)
                    depth = len(next_history)
                    if depth < visited_depth.get(relation.target_concept_id, 99):
                        visited_depth[relation.target_concept_id] = depth
                        queue.append((relation.target_concept_id, next_history))
        paths.sort(
            key=lambda item: (
                len(item.hops), item.seed_concept_id, item.target_concept_id,
                tuple(hop.relation_id for hop in item.hops),
            )
        )
        return paths, reached, paths_truncated

    def _local_hits(
        self,
        request: PreReasoningRetrievalRequest,
        concepts: dict[str, OntologyConcept],
        reached: set[str],
    ) -> tuple[list[LocalRetrievalHit], int, int, list[str], bool, dict[str, int]]:
        caller_tags = set(request.query.caller_access_tags)
        eligible: list[GovernedTextChunk] = []
        filtered_count = 0
        governance_filtered_count = 0
        deduplicated: list[str] = []
        document_access_reasons: Counter[str] = Counter()
        seen_hashes: set[str] = set()
        for chunk in sorted(request.chunks, key=lambda item: item.chunk_id):
            if chunk.document_descriptor is not None:
                context = request.query.document_access_context
                if context is None:
                    filtered_count += 1
                    document_access_reasons["DOCUMENT_ACCESS_CONTEXT_REQUIRED"] += 1
                    continue
                decision = evaluate_document_access(chunk.document_descriptor, context)
                if not decision.allowed:
                    filtered_count += 1
                    document_access_reasons[decision.reason_code] += 1
                    continue
            if not set(chunk.required_access_tags).issubset(caller_tags):
                filtered_count += 1
                continue
            if (
                chunk.lifecycle_status != "ACTIVE"
                or chunk.review_status != "HUMAN_REVIEWED"
                or chunk.content_safety_status != "PASS"
            ):
                governance_filtered_count += 1
                continue
            if chunk.chunk_sha256 in seen_hashes:
                deduplicated.append(chunk.chunk_id)
                continue
            seen_hashes.add(chunk.chunk_sha256)
            eligible.append(chunk)

        expansion_terms: list[str] = []
        for concept_id in sorted(reached):
            concept = concepts[concept_id]
            expansion_terms.extend([*concept.labels, *concept.aliases])
        query_terms = _tokens(" ".join([request.query.text, *expansion_terms]))
        documents = [
            _tokens(
                "\n".join([*item.section_path, item.text])
                if request.config.include_section_path_in_lexical_index
                else item.text
            )
            for item in eligible
        ]
        normalized_bm25 = _normalize(_bm25(query_terms, documents))
        query_term_set = set(query_terms)
        dense_scores: dict[str, float] = {}
        if self.dense_score_provider is not None and eligible:
            dense_scores = self.dense_score_provider.score(
                query=request.query.text,
                chunks=eligible,
            )
            unknown = set(dense_scores) - {item.chunk_id for item in eligible}
            if unknown:
                raise ValueError(f"dense score provider returned unknown chunk IDs: {sorted(unknown)}")
            if any(score < 0.0 or score > 1.0 for score in dense_scores.values()):
                raise ValueError("dense scores must be normalized to [0, 1]")
        elif request.config.dense_weight > 0.0:
            raise ValueError("dense_weight is positive but no dense score provider is configured")

        rows: list[tuple[float, GovernedTextChunk, float, float, float, list[str]]] = []
        for index, chunk in enumerate(eligible):
            document_terms = set(documents[index])
            term_coverage = (
                len(query_term_set & document_terms) / len(query_term_set) if query_term_set else 0.0
            )
            matched_concepts = sorted(set(chunk.concept_ids) & reached)
            dense_score = dense_scores.get(chunk.chunk_id, 0.0)
            score = (
                request.config.bm25_weight * normalized_bm25[index]
                + request.config.term_coverage_weight * term_coverage
                + request.config.dense_weight * dense_score
                + (request.config.ontology_concept_bonus if matched_concepts else 0.0)
            )
            if score > 0.0:
                rows.append(
                    (
                        score,
                        chunk,
                        normalized_bm25[index],
                        term_coverage,
                        dense_score,
                        matched_concepts,
                    )
                )
        rows.sort(key=lambda item: (-item[0], item[1].chunk_id))
        hits = [
            LocalRetrievalHit(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_uri=chunk.source_uri,
                source_sha256=chunk.source_sha256,
                chunk_sha256=chunk.chunk_sha256,
                knowledge_version=chunk.knowledge_version,
                rank=index,
                score=score,
                bm25_score=bm25_score,
                term_coverage=term_coverage,
                dense_score=dense_score,
                matched_concept_ids=matched_concepts,
                snippet=chunk.text[:500],
                lane=chunk.lane,
                section_id=chunk.section_id,
                section_path=chunk.section_path,
                modalities=chunk.modalities,
                source_locators=chunk.source_locators,
                structure_sha256=chunk.structure_sha256,
            )
            for index, (
                score,
                chunk,
                bm25_score,
                term_coverage,
                dense_score,
                matched_concepts,
            ) in enumerate(
                rows[: request.query.top_k], start=1
            )
        ]
        return (
            hits,
            filtered_count,
            governance_filtered_count,
            deduplicated,
            self.dense_score_provider is not None,
            dict(sorted(document_access_reasons.items())),
        )

    def retrieve(self, request: PreReasoningRetrievalRequest) -> PreReasoningRetrievalPackage:
        concepts = self._eligible_concepts(request)
        seeds = self._seed_concepts(request, concepts)
        paths, reached, paths_truncated = self._ontology_paths(request, concepts, seeds)
        (
            hits,
            filtered_count,
            governance_filtered_count,
            deduplicated,
            dense_retrieval_available,
            document_access_reason_counts,
        ) = self._local_hits(request, concepts, reached)
        caller_tags = set(request.query.caller_access_tags)
        ontology_filtered_count = sum(
            1
            for item in request.concepts
            if not set(item.required_access_tags).issubset(caller_tags)
        ) + sum(
            1
            for item in request.relations
            if not set(item.required_access_tags).issubset(caller_tags)
        )
        required = set(request.query.required_concept_ids)
        concept_coverage = len(required & reached) / len(required) if required else (1.0 if reached else 0.0)
        top_local_score = hits[0].score if hits else 0.0
        ontology_ok = concept_coverage >= request.config.minimum_concept_coverage
        local_ok = top_local_score >= request.config.minimum_local_score
        if request.config.require_local_and_ontology:
            sufficient = local_ok and ontology_ok
        else:
            sufficient = local_ok or ontology_ok
        reason_codes: list[str] = []
        if not local_ok:
            reason_codes.append("LOCAL_RETRIEVAL_BELOW_THRESHOLD")
        if not ontology_ok:
            reason_codes.append("ONTOLOGY_CONCEPT_COVERAGE_BELOW_THRESHOLD")
        if paths_truncated:
            reason_codes.append("ONTOLOGY_PATH_LIMIT_REACHED")
        if sufficient:
            status = RetrievalStatus.SUFFICIENT
        elif hits or reached:
            status = RetrievalStatus.LOW_MATCH
        else:
            status = RetrievalStatus.NO_MATCH
            reason_codes.append("NO_RETRIEVAL_EVIDENCE")
        query_payload = request.query.model_dump(mode="json")
        query_hash = _sha256_json(query_payload)
        experience_hits = [item for item in hits if item.lane.value == "EXPERIENCE"]
        hit_experience_ids = {
            item.source_id.removeprefix("experience:")
            for item in experience_hits
            if item.source_id.startswith("experience:")
        }
        selected_knowledge_candidates = [
            candidate for candidate in request.knowledge_update_candidates
            if candidate.get("experience_id") in hit_experience_ids
        ]
        hit_chunk_ids = {item.chunk_id for item in hits}
        selected_trace_links = [
            item for item in request.document_graph_trace_links
            if item.retrieval_chunk_id in hit_chunk_ids
        ]
        selected_link_keys = {
            (item.retrieval_chunk_id, item.knowledge_fragment_id)
            for item in selected_trace_links
        }
        selected_document_claims = [
            item for item in request.document_evidence_claims
            if (item.source_chunk_id, item.source_fragment_id) in selected_link_keys
        ]
        from .document_claims import evaluate_document_claim_admissions

        document_claim_admissions = evaluate_document_claim_admissions(
            selected_document_claims, request.document_claim_admission_policy
        )
        package_payload = {
            "query_sha256": query_hash,
            "hit_hashes": [item.chunk_sha256 for item in hits],
            "paths": [[hop.relation_id for hop in item.hops] for item in paths],
            "config": request.config.model_dump(mode="json"),
            "experience_reuse_receipt": request.experience_reuse_receipt,
            "knowledge_update_candidates": selected_knowledge_candidates,
            "ontology_seed_concept_ids": request.query.seed_concept_ids,
            "ontology_asset_audits": request.ontology_asset_audits,
            "document_graph_trace_links": [
                item.model_dump(mode="json") for item in selected_trace_links
            ],
            "document_evidence_claims": [
                item.model_dump(mode="json") for item in selected_document_claims
            ],
            "document_claim_admissions": [
                item.model_dump(mode="json") for item in document_claim_admissions
            ],
            "document_claim_admission_policy": (
                request.document_claim_admission_policy.model_dump(mode="json")
            ),
        }
        return PreReasoningRetrievalPackage(
            package_id="RETRIEVAL-" + _sha256_json(package_payload)[:20].upper(),
            query_id=request.query.query_id,
            episode_id=request.query.episode_id,
            query_sha256=query_hash,
            status=status,
            local_hits=hits,
            ontology_paths=paths,
            ontology_seed_concept_ids=request.query.seed_concept_ids,
            ontology_asset_audits=request.ontology_asset_audits,
            document_graph_trace_links=selected_trace_links,
            document_evidence_claims=selected_document_claims,
            document_claim_admissions=document_claim_admissions,
            matched_concept_ids=sorted(reached),
            required_concept_coverage=concept_coverage,
            top_local_score=top_local_score,
            dense_retrieval_available=dense_retrieval_available,
            access_filtered_chunk_count=filtered_count,
            document_access_reason_counts=document_access_reason_counts,
            governance_filtered_chunk_count=governance_filtered_count,
            access_filtered_ontology_item_count=ontology_filtered_count,
            ontology_paths_truncated=paths_truncated,
            deduplicated_chunk_ids=deduplicated,
            reason_codes=list(dict.fromkeys(reason_codes)),
            paperseek_fallback_recommended=status != RetrievalStatus.SUFFICIENT,
            query_construction_mode=request.query.construction_mode,
            grounding_fault_ids=request.query.grounding_fault_ids,
            grounding_evidence_fact_ids=request.query.grounding_evidence_fact_ids,
            experience_hit_count=len(experience_hits),
            experience_source_ids=sorted({item.source_id for item in experience_hits}),
            experience_reuse_receipt=request.experience_reuse_receipt,
            knowledge_update_candidates=selected_knowledge_candidates,
        )
