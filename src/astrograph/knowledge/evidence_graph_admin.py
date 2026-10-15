from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .query import ReadOnlyGraphQueryPolicy


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"invalid graph identifier: {value}")
    return value


def load_evidence_graph_csv(
    nodes_csv: str | Path,
    edges_csv: str | Path,
    *,
    policy: ReadOnlyGraphQueryPolicy | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = policy or ReadOnlyGraphQueryPolicy()
    nodes_path = Path(nodes_csv)
    edges_path = Path(edges_csv)
    nodes: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    with nodes_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            node_type = _identifier(row["node_type"])
            if node_type not in policy.allowed_node_labels:
                raise ValueError(f"node label is not allowed: {node_type}")
            node_id = row["node_id"]
            if node_id in by_id:
                raise ValueError(f"duplicate graph node ID: {node_id}")
            properties = json.loads(row.get("properties_json") or "{}")
            item = {
                "id": node_id,
                "node_type": node_type,
                "name": row.get("name", ""),
                "properties": properties,
            }
            nodes.append(item)
            by_id[node_id] = item
    edges: list[dict[str, Any]] = []
    seen_edges: set[str] = set()
    with edges_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relation = _identifier(row["relation"])
            if relation not in policy.allowed_relation_types:
                raise ValueError(f"relation type is not allowed: {relation}")
            source = by_id.get(row["source_id"])
            target = by_id.get(row["target_id"])
            if source is None or target is None:
                raise ValueError(f"edge references missing node: {row['edge_id']}")
            edge_id = row["edge_id"]
            if edge_id in seen_edges:
                raise ValueError(f"duplicate graph edge ID: {edge_id}")
            seen_edges.add(edge_id)
            edges.append(
                {
                    "edge_id": edge_id,
                    "source_id": source["id"],
                    "source_type": source["node_type"],
                    "target_id": target["id"],
                    "target_type": target["node_type"],
                    "relation": relation,
                    "properties": json.loads(row.get("properties_json") or "{}"),
                }
            )
    return nodes, edges


class Neo4jEvidenceGraphAdmin:
    """Explicit administrative importer for a reviewed evidence-graph CSV release.

    This write-capable class must not be exposed as an online diagnostic tool. Dynamic labels and
    relationship types are interpolated only after strict policy allow-list validation.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        policy: ReadOnlyGraphQueryPolicy | None = None,
    ) -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.policy = policy or ReadOnlyGraphQueryPolicy()

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def ensure_schema(self, labels: list[str]) -> None:
        with self.driver.session() as session:
            for label in sorted(set(labels)):
                label = _identifier(label)
                if label not in self.policy.allowed_node_labels:
                    raise ValueError(f"node label is not allowed: {label}")
                name = f"astrograph_{label.lower()}_id_unique"
                query = f"CREATE CONSTRAINT `{name}` IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.id IS UNIQUE"
                session.run(query).consume()
            session.run(
                "CREATE CONSTRAINT astrograph_evidence_graph_release_id_unique IF NOT EXISTS "
                "FOR (r:EvidenceGraphRelease) REQUIRE r.release_id IS UNIQUE"
            ).consume()

    def import_release(
        self,
        *,
        release_id: str,
        knowledge_version: str,
        nodes_csv: str | Path,
        edges_csv: str | Path,
    ) -> dict[str, Any]:
        nodes_path = Path(nodes_csv)
        edges_path = Path(edges_csv)
        nodes, edges = load_evidence_graph_csv(nodes_path, edges_path, policy=self.policy)
        self.verify_connectivity()
        self.ensure_schema([item["node_type"] for item in nodes])
        nodes_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in nodes:
            nodes_by_type[item["node_type"]].append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "properties": {
                        **item["properties"],
                        "knowledge_version": knowledge_version,
                        "release_id": release_id,
                    },
                }
            )
        edges_by_signature: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in edges:
            edges_by_signature[(item["source_type"], item["relation"], item["target_type"])].append(
                {
                    "edge_id": item["edge_id"],
                    "source_id": item["source_id"],
                    "target_id": item["target_id"],
                    "properties": {
                        **item["properties"],
                        "knowledge_version": knowledge_version,
                        "release_id": release_id,
                    },
                }
            )
        with self.driver.session() as session:
            session.run(
                """
MERGE (release:EvidenceGraphRelease {release_id: $release_id})
SET release.knowledge_version = $knowledge_version,
    release.nodes_sha256 = $nodes_sha256,
    release.edges_sha256 = $edges_sha256,
    release.node_count = $node_count,
    release.edge_count = $edge_count
""",
                release_id=release_id,
                knowledge_version=knowledge_version,
                nodes_sha256=_sha256_file(nodes_path),
                edges_sha256=_sha256_file(edges_path),
                node_count=len(nodes),
                edge_count=len(edges),
            ).consume()
            for label, rows in sorted(nodes_by_type.items()):
                query = f"""
UNWIND $rows AS row
MERGE (node:`{label}` {{id: row.id}})
SET node.name = row.name
SET node += row.properties
"""
                session.run(query, rows=rows).consume()
            for (source_type, relation, target_type), rows in sorted(edges_by_signature.items()):
                source_type = _identifier(source_type)
                relation = _identifier(relation)
                target_type = _identifier(target_type)
                if relation not in self.policy.allowed_relation_types:
                    raise ValueError(f"relation type is not allowed: {relation}")
                query = f"""
UNWIND $rows AS row
MATCH (source:`{source_type}` {{id: row.source_id}})
MATCH (target:`{target_type}` {{id: row.target_id}})
MERGE (source)-[rel:`{relation}` {{edge_id: row.edge_id}}]->(target)
SET rel += row.properties
"""
                session.run(query, rows=rows).consume()
        parity = self.verify_release(
            release_id=release_id,
            knowledge_version=knowledge_version,
            expected_nodes=nodes,
            expected_edges=edges,
        )
        return {
            "schema_version": "astrograph-evidence-graph-import-report.v1",
            "status": parity["status"],
            "release_id": release_id,
            "knowledge_version": knowledge_version,
            "imported_node_count": len(nodes),
            "imported_edge_count": len(edges),
            "nodes_sha256": _sha256_file(nodes_path),
            "edges_sha256": _sha256_file(edges_path),
            "parity": parity,
            "claim_boundary": "Administrative graph import; online diagnostic identities must use a read-only Neo4j account.",
        }

    def verify_release(
        self,
        *,
        release_id: str,
        knowledge_version: str,
        expected_nodes: list[dict[str, Any]],
        expected_edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with self.driver.session() as session:
            node_record = session.run(
                "MATCH (n) WHERE NOT n:EvidenceGraphRelease AND n.release_id = $release_id AND n.knowledge_version = $knowledge_version "
                "RETURN count(n) AS count",
                release_id=release_id,
                knowledge_version=knowledge_version,
            ).single()
            edge_record = session.run(
                "MATCH ()-[r]->() WHERE r.release_id = $release_id AND r.knowledge_version = $knowledge_version "
                "RETURN count(r) AS count",
                release_id=release_id,
                knowledge_version=knowledge_version,
            ).single()
        actual_nodes = 0 if node_record is None else int(node_record["count"])
        actual_edges = 0 if edge_record is None else int(edge_record["count"])
        status = "PASS" if actual_nodes == len(expected_nodes) and actual_edges == len(expected_edges) else "FAIL"
        return {
            "schema_version": "astrograph-evidence-graph-parity-report.v1",
            "status": status,
            "release_id": release_id,
            "knowledge_version": knowledge_version,
            "expected_node_count": len(expected_nodes),
            "actual_node_count": actual_nodes,
            "expected_edge_count": len(expected_edges),
            "actual_edge_count": actual_edges,
        }
