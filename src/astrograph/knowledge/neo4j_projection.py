"""Neo4j writer for V3 five-layer graph projections.

Writes a validated KnowledgeGraphV3 into a local Neo4j instance with an
idempotent graph-scoped MERGE (deletes the prior projection for the same
graph_id first). Connection details come from environment or constructor, never
from the graph payload.
"""

from __future__ import annotations

import json
import os


class Neo4jGraphWriter:
    """Idempotent writer for V3 graph projections into Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        from neo4j import GraphDatabase

        resolved_uri = uri or os.environ.get("ASTROGRAPH_NEO4J_URI", "neo4j://127.0.0.1:7687")
        resolved_user = user or os.environ.get("ASTROGRAPH_NEO4J_USER", "neo4j")
        resolved_password = password or os.environ.get("ASTROGRAPH_NEO4J_PASSWORD")
        if not resolved_password:
            raise ValueError(
                "Neo4j password is required via constructor or ASTROGRAPH_NEO4J_PASSWORD"
            )
        self._driver = GraphDatabase.driver(
            resolved_uri,
            auth=(resolved_user, resolved_password),
        )

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    def write_graph(self, graph) -> tuple[int, int]:
        """Write a KnowledgeGraphV3, replacing any prior graph_id projection.

        Returns (node_count, edge_count).
        """
        self._driver.verify_connectivity()
        with self._driver.session() as session:
            session.run(
                "MATCH (n) WHERE n.graph_id = $gid DETACH DELETE n",
                gid=graph.graph_id,
            ).consume()
            for node in graph.nodes:
                session.run(
                    "MERGE (n:V3Node {node_id: $id}) "
                    "SET n.graph_id = $gid, n.node_type = $type, n.properties = $props",
                    id=node.node_id,
                    gid=graph.graph_id,
                    type=node.node_type.value,
                    props=json.dumps(node.properties),
                ).consume()
            for edge in graph.edges:
                session.run(
                    "MATCH (a {node_id: $s}), (b {node_id: $t}) "
                    "MERGE (a)-[r:V3REL {relation_type: $rel}]->(b) "
                    "SET r.graph_id = $gid",
                    s=edge.source_id,
                    t=edge.target_id,
                    rel=edge.relation_type.value,
                    gid=graph.graph_id,
                ).consume()
            node_count = session.run(
                "MATCH (n {graph_id: $gid}) RETURN count(n) AS c", gid=graph.graph_id
            ).single()["c"]
            edge_count = session.run(
                "MATCH ()-[r {graph_id: $gid}]->() RETURN count(r) AS c",
                gid=graph.graph_id,
            ).single()["c"]
            return int(node_count), int(edge_count)


__all__ = ["Neo4jGraphWriter"]
