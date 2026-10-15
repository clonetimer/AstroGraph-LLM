from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KG = ROOT / "benchmarks" / "knowledge_graph" / "online_extended"


def _csv(name: str):
    with (KG / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_v058_kg_gold_manifest_and_graph_are_consistent():
    manifest = json.loads((KG / "GOLD_STANDARD_MANIFEST.json").read_text(encoding="utf-8"))
    nodes = _csv("nodes.csv")
    edges = _csv("edges.csv")
    negative = _csv("negative_constraints.csv")
    queries = json.loads((KG / "query_gold.json").read_text(encoding="utf-8"))["queries"]
    composite = json.loads((KG / "composite_research_gold.json").read_text(encoding="utf-8"))
    node_ids = {item["node_id"] for item in nodes}
    assert manifest["passed"] is True
    assert manifest["formal_fault_count"] == 16
    assert manifest["composite_template_count"] == 4
    assert len(nodes) == manifest["node_count"]
    assert len(edges) == manifest["edge_count"]
    assert len(negative) >= 12
    assert len(queries) == 32
    assert len(composite["templates"]) == 4
    assert all(edge["source_id"] in node_ids and edge["target_id"] in node_ids for edge in edges)
