from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astrograph.knowledge import (
    CausalEdgeReviewPacket,
    StructuredEntityLinkingBenchmark,
    load_frozen_structured_entity_linking_benchmark,
    project_generic_fault_ontology,
    validate_causal_edge_review_packet,
)


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "configs/knowledge/generic_fault_ontology.json"
ARCHETYPES = ROOT / "configs/semantics/core_component_archetypes.json"
SEMANTICS = ROOT / "configs/semantics/core_semantic_schema.json"
PACKET = ROOT / "governance/knowledge_reviews/kge2_causal_edges_review_packet_20260818.json"
BENCHMARK = ROOT / "configs/knowledge/benchmarks/structured_candidate_entity_linking_v1.json"
BENCHMARK_SHA = ROOT / "configs/knowledge/benchmarks/structured_candidate_entity_linking_v1.sha256"


def _graph():
    return project_generic_fault_ontology(ONTOLOGY, ARCHETYPES, SEMANTICS)


def test_real_review_packet_is_source_bound_but_promotes_no_edge() -> None:
    graph = _graph()
    before = graph.model_dump(mode="json")
    receipt = validate_causal_edge_review_packet(PACKET, project_root=ROOT, graph=graph)
    packet = CausalEdgeReviewPacket.model_validate_json(PACKET.read_text(encoding="utf-8"))

    assert receipt.status == "READY_FOR_HUMAN_REVIEW"
    assert receipt.checked_item_count == 3
    assert receipt.ready_item_ids == []
    assert receipt.deferred_item_ids == [item.review_item_id for item in packet.review_items]
    assert all(item.human_decision is None for item in packet.review_items)
    assert packet.publication_allowed is False
    assert receipt.publication_allowed is False
    assert receipt.active_graph_modified is False
    assert graph.model_dump(mode="json") == before


def test_review_packet_detects_source_hash_drift(tmp_path: Path) -> None:
    payload = json.loads(PACKET.read_text(encoding="utf-8"))
    payload["review_items"][0]["source_anchors"][0]["source_sha256"] = "0" * 64
    changed = tmp_path / "changed-review-packet.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    receipt = validate_causal_edge_review_packet(changed, project_root=ROOT, graph=_graph())
    assert receipt.status == "BLOCKED"
    assert any(item.startswith("SOURCE_HASH_MISMATCH:") for item in receipt.findings)


def test_frozen_benchmark_targets_post_llm_structured_output() -> None:
    benchmark = load_frozen_structured_entity_linking_benchmark(
        BENCHMARK,
        BENCHMARK_SHA,
        ontology_path=ONTOLOGY,
        graph=_graph(),
    )
    assert benchmark.input_stage == "POST_LLM_STRUCTURED_OUTPUT"
    assert benchmark.usage == "DEVELOPMENT_GATE_ONLY"
    assert benchmark.label_review_status == "MACHINE_DRAFT"
    assert benchmark.human_label_review_required is True
    assert benchmark.frozen is True
    assert len(benchmark.cases) == 8
    assert sum(item.expected_action == "LINK" for item in benchmark.cases) == 6
    assert sum(item.expected_action == "ABSTAIN" for item in benchmark.cases) == 2


def test_frozen_benchmark_checksum_drift_fails_closed(tmp_path: Path) -> None:
    checksum = tmp_path / "wrong.sha256"
    checksum.write_text(f"{'0' * 64}  benchmark.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_frozen_structured_entity_linking_benchmark(
            BENCHMARK,
            checksum,
            ontology_path=ONTOLOGY,
            graph=_graph(),
        )


def test_frozen_benchmark_rejects_target_type_mismatch(tmp_path: Path) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_node_id"] = "effect:command-response-residual-growth"
    changed = tmp_path / "changed-benchmark.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(changed.read_bytes()).hexdigest()
    checksum = tmp_path / "changed-benchmark.sha256"
    checksum.write_text(f"{digest}  changed-benchmark.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target type mismatch"):
        load_frozen_structured_entity_linking_benchmark(
            changed,
            checksum,
            ontology_path=ONTOLOGY,
            graph=_graph(),
        )


def test_frozen_benchmark_rejects_ontology_identity_mismatch(tmp_path: Path) -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["ontology_version"] = "unexpected"
    changed = tmp_path / "changed-benchmark.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(changed.read_bytes()).hexdigest()
    checksum = tmp_path / "changed-benchmark.sha256"
    checksum.write_text(f"{digest}  changed-benchmark.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ontology identity mismatch"):
        load_frozen_structured_entity_linking_benchmark(
            changed,
            checksum,
            ontology_path=ONTOLOGY,
            graph=_graph(),
        )


def test_benchmark_schema_forbids_forced_link_without_target() -> None:
    payload = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_node_id"] = None
    with pytest.raises(ValueError, match="LINK requires"):
        StructuredEntityLinkingBenchmark.model_validate(payload)
