"""Shared graph/evidence/fusion diagnostic kernel, with no DOR or DAG dependency.

A session keeps its original model-input receipt unchanged. Evidence acquired
later is recorded as a separate, hash-linked observation revision by the Agent;
it is not relabelled as having been used to produce the original model output.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from astrograph.evidence.executable import evaluate_evidence
from astrograph.fusion import RuntimeFusionService
from astrograph.fusion.decision import FusionDecisionService, render_decision
from astrograph.knowledge import (
    CsvEvidenceGraphReferenceExecutor, associations_from_query_result,
    build_forward_evidence_query_plan, build_backward_fault_query_plan,
)
from astrograph.knowledge.evidence_graph_admin import load_evidence_graph_csv
from .contracts import OAGInput, content_hash
from .summary import build_diagnostic_summary


def validate_causal_evidence(effective):
    """Validate facts against admitted observations, shared by both entrypoints."""
    feature_defs = {d.feature_definition_id: d for d in effective.feature_definitions}
    features = {f.feature_result_id: f for f in effective.feature_results}
    if len(features) != len(effective.feature_results) or len({f.feature_definition_id for f in features.values()}) != len(features):
        raise ValueError("single-window diagnosis requires unique feature IDs and definitions")
    for feature in features.values():
        definition = feature_defs.get(feature.feature_definition_id)
        if definition is None or feature.unit != definition.unit:
            raise ValueError("feature semantic/unit contract mismatch")
        if (feature.episode_id != effective.episode.episode_id
                or feature.window_start_s > feature.window_end_s
                or feature.window_end_s > effective.episode.end_s + 1e-8
                or (definition.window_scope == "episode" and feature.window_start_s < effective.episode.start_s - 1e-8)):
            raise ValueError("feature is outside the admitted causal window")
    evidence_defs = {d.evidence_definition_id: d for d in effective.evidence_definitions}
    for fact in effective.evidence_facts:
        definition = evidence_defs[fact.evidence_definition_id]
        if (fact.unit != definition.threshold_unit or fact.comparison != definition.comparison
                or fact.threshold_value != definition.threshold):
            raise ValueError("fact rewrites its registered threshold or unit")
        if fact.status.value in {"SATISFIED", "CONTRADICTED"}:
            feature = features.get(fact.feature_result_id)
            if (feature is None or feature.feature_definition_id != definition.feature_definition_id
                    or feature.quality_flags or feature.sample_count <= 0):
                raise ValueError("positive/negative evidence lacks a usable primary observation")
            for conjunct in definition.and_feature_conditions:
                others = [f for f in features.values() if f.feature_definition_id == conjunct.feature_definition_id]
                if (not others or others[0].quality_flags or others[0].value is None
                        or others[0].unit != conjunct.threshold_unit):
                    raise ValueError("positive/negative evidence lacks a usable conjunct observation")
            expected = evaluate_evidence(definition, feature, all_feature_results=list(features.values()))
            if expected.status != fact.status:
                raise ValueError("claimed evidence status differs from executable condition")


def apply_fusion_decision(data, bundle, package, summary):
    validate_causal_evidence(bundle)
    heads = data.three_task_result
    projection = heads.evidence_projection if heads is not None else None
    decision = FusionDecisionService().decide(
        bundle=bundle, package=package, summary=summary,
        detected=heads.detected if heads is not None else None,
        identical_reference=bool(projection and projection.source_window_sha256 == projection.nominal_window_sha256),
        maximum_candidates=data.maximum_diagnostic_candidates,
    )
    summary = dict(summary)
    summary["diagnostic_decision"] = decision
    summary["suspected_fault_id"] = decision["selected_fault_id"]
    summary["status"] = ({"SUPPORTED_SINGLETON": "READY_FOR_EXPLANATION",
                          "MODEL_ONLY": "MODEL_HYPOTHESIS",
                          "CANDIDATE_SET": "AMBIGUOUS"}.get(decision["status"], decision["status"]))
    # Keep legacy gates visible without mislabelling support gaps as hard evidence.
    summary["legacy_blocking_conflicts"] = summary["blocking_conflicts"]
    summary["blocking_conflicts"] = decision["blocking_conflicts"]
    summary["quality_warnings"] = decision["quality_warnings"]
    if heads is not None:
        summary["three_task_results"] = heads.prompt_projection()
        summary["three_task_results"]["additional_fusion_votes"] = 0
    summary.pop("summary_sha256", None)
    summary["summary_sha256"] = content_hash(summary)
    return summary


class EvidenceDiagnosisKernel:
    """Same native graph queries, rankers and BiRRF as the existing OAG runtime."""
    def __init__(self, graph_dir: str | Path):
        root = Path(graph_dir).resolve()
        nodes, edges = root / "nodes.csv", root / "edges.csv"
        load_evidence_graph_csv(nodes, edges)
        self.identity = {"backend": "NATIVE_CSV_READ_ONLY",
                         "nodes_sha256": hashlib.sha256(nodes.read_bytes()).hexdigest(),
                         "edges_sha256": hashlib.sha256(edges.read_bytes()).hexdigest()}
        self._nodes, self._edges = nodes, edges
        self.executor = CsvEvidenceGraphReferenceExecutor(nodes, edges)
        self.knowledge_version = content_hash(self.identity)

    def evaluate(self, data: OAGInput, *, bundle=None) -> dict:
        # Root model/evidence binding is always validated. Only the Agent's
        # admitted revisions are passed as the effective bundle thereafter.
        data = OAGInput.model_validate_json(data.model_dump_json())
        if (hashlib.sha256(self._nodes.read_bytes()).hexdigest() != self.identity["nodes_sha256"]
                or hashlib.sha256(self._edges.read_bytes()).hexdigest() != self.identity["edges_sha256"]):
            raise ValueError("knowledge release changed during this diagnostic session")
        effective = bundle if bundle is not None else data.bundle
        if effective.episode != data.bundle.episode or effective.evaluation_metadata:
            raise ValueError("effective evidence changes the episode or exposes evaluation truth")
        for field in ("feature_definitions", "evidence_definitions", "source_protocol_hashes"):
            if getattr(effective, field) != getattr(data.bundle, field):
                raise ValueError("online evidence must not rewrite published knowledge or protocols")
        validate_causal_evidence(effective)
        definitions = sorted(d.evidence_definition_id for d in effective.evidence_definitions)
        faults = sorted({d.target_fault_id for d in effective.evidence_definitions})
        queries = [
            self.executor.execute(build_forward_evidence_query_plan(definitions), knowledge_version=self.knowledge_version),
            self.executor.execute(build_backward_fault_query_plan(faults), knowledge_version=self.knowledge_version),
        ] if definitions and faults else []
        if any(q.truncated for q in queries):
            raise ValueError("incomplete graph results cannot establish diagnostic support")
        package = RuntimeFusionService().build(
            bundle=effective, data_ranking=data.diagnostic_ranking,
            config=data.fusion_config, registered_fault_ids=data.registered_fault_ids,
            forward_associations=associations_from_query_result(queries[0]) if queries else None,
            backward_associations=associations_from_query_result(queries[1]) if queries else None,
            runtime_metadata={"fusion_controls_llm_candidate_order_only": False,
                              "decision_consumer": FusionDecisionService.policy_id},
        )
        summary = build_diagnostic_summary(
            effective, package, queries,
            top_k=max(1, len(package.birrf_result.candidates)), graph_identity=self.identity,
        )
        summary = apply_fusion_decision(data, effective, package, summary)
        decision = summary["diagnostic_decision"]
        return {"input_sha256": content_hash(data.model_dump(mode="json")),
                "evidence_bundle_sha256": content_hash(effective.model_dump(mode="json")),
                "fusion": package.model_dump(mode="json"), "summary": summary,
                "diagnosis": decision, "explanation": render_decision(decision),
                "formal_diagnosis_publishable": False}
