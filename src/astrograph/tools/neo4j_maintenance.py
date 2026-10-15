from __future__ import annotations

import os
from typing import Any


def neo4j_configured() -> bool:
    return bool(os.environ.get("NEO4J_URI") and os.environ.get("NEO4J_USER") and os.environ.get("NEO4J_PASSWORD"))


def _driver():
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Neo4j driver is not installed; install astrograph-core[neo4j]") from exc
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USER", "").strip()
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not uri or not user or not password:
        raise RuntimeError("Neo4j service is not configured")
    return GraphDatabase.driver(uri, auth=(user, password))


def neo4j_health() -> dict[str, Any]:
    if not neo4j_configured():
        return {"status": "NOT_CONFIGURED"}
    driver = _driver()
    try:
        driver.verify_connectivity()
        database = os.environ.get("NEO4J_DATABASE") or None
        with driver.session(database=database) as session:
            record = session.run("MATCH (n) RETURN count(n) AS node_count").single()
        return {"status": "PASS", "node_count": int(record["node_count"] if record else 0), "database": database or "default"}
    finally:
        driver.close()


def get_fault_mechanism(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, read-only fault/anomaly neighborhood for maintenance review."""
    fault_id = str(payload.get("fault_id") or "").strip()
    if not fault_id:
        raise ValueError("fault_id is required")
    if not neo4j_configured():
        return {"status":"NOT_CONFIGURED","fault_id":fault_id,"nodes":[],"edges":[],"read_only":True,"graph_query_executed":False}
    max_nodes = max(1, min(100, int(payload.get("max_nodes", 40))))
    driver = _driver()
    database = os.environ.get("NEO4J_DATABASE") or None
    try:
        with driver.session(database=database) as session:
            records = session.run(
                """
                MATCH (f:FaultMode {id:$fault_id})-[r]->(n)
                RETURN f.id AS from_id, coalesce(f.name,f.id) AS from_name,
                       type(r) AS relation, coalesce(n.id, elementId(n)) AS to_id,
                       coalesce(n.name,n.id,elementId(n)) AS to_name,
                       labels(n) AS to_labels,
                       coalesce(r.source_ref,'') AS source_ref
                LIMIT $max_nodes
                """,
                fault_id=fault_id, max_nodes=max_nodes,
            )
            rows = [dict(row) for row in records]
        nodes: dict[str, dict[str, Any]] = {fault_id: {"id": fault_id, "label": fault_id, "kind": "FaultMode"}}
        edges: list[dict[str, Any]] = []
        for row in rows:
            nodes[fault_id]["label"] = row.get("from_name") or fault_id
            tid = str(row.get("to_id"))
            nodes[tid] = {"id": tid, "label": row.get("to_name") or tid, "kind": ",".join(row.get("to_labels") or []) or "KnowledgeNode"}
            edges.append({"from": fault_id, "to": tid, "relation": row.get("relation"), "source_ref": row.get("source_ref") or None})
        return {
            "status": "SUCCEEDED",
            "fault_id": fault_id,
            "nodes": list(nodes.values()),
            "edges": edges,
            "read_only": True,
            "graph_query_executed": True,
            "claim_boundary": "Read-only Neo4j neighborhood for maintenance review; graph paths do not independently promote a diagnosis.",
        }
    finally:
        driver.close()


def match_observations(payload: dict[str, Any]) -> dict[str, Any]:
    anomaly_ids = sorted({str(x).strip() for x in payload.get("anomaly_ids", []) if str(x).strip()})
    if not anomaly_ids:
        raise ValueError("anomaly_ids is required")
    if not neo4j_configured():
        return {"status":"NOT_CONFIGURED","anomaly_ids":anomaly_ids,"candidates":[],"read_only":True,"graph_query_executed":False}
    driver = _driver()
    database = os.environ.get("NEO4J_DATABASE") or None
    try:
        with driver.session(database=database) as session:
            records = session.run(
                """
                MATCH (f:FaultMode)-[r:CAUSES]->(a:Anomaly)
                WHERE a.id IN $anomaly_ids
                RETURN f.id AS fault_id, coalesce(f.name,f.id) AS fault_name,
                       collect({anomaly_id:a.id, anomaly_name:coalesce(a.name,a.id), weight:coalesce(r.weight,0.0), source_ref:coalesce(r.source_ref,'')}) AS matches,
                       sum(coalesce(r.weight,0.0)) AS support
                ORDER BY support DESC, fault_id
                LIMIT 20
                """,
                anomaly_ids=anomaly_ids,
            )
            candidates = [dict(row) for row in records]
        return {"status": "SUCCEEDED", "anomaly_ids": anomaly_ids, "candidates": candidates, "read_only": True, "graph_query_executed": True}
    finally:
        driver.close()
