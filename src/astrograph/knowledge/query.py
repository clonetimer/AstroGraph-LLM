from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Iterable, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.investigation import GraphQueryPlan


class GraphQueryKind(str, Enum):
    AUTO = "AUTO"
    FORWARD_EVIDENCE_TO_FAULT = "FORWARD_EVIDENCE_TO_FAULT"
    BACKWARD_FAULT_TO_EVIDENCE = "BACKWARD_FAULT_TO_EVIDENCE"
    GENERAL_READ_ONLY_PATH = "GENERAL_READ_ONLY_PATH"


class ReadOnlyGraphQueryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-read-only-graph-query-policy.v1"
    policy_id: str = "astrograph-evidence-graph-read-only-v1"
    allowed_node_labels: list[str] = Field(
        default_factory=lambda: [
            "FaultMode",
            "EvidenceDefinition",
            "FeatureDefinition",
            "EvidenceSource",
            "ComponentInstance",
            "TelemetryField",
            "TelemetryFieldTemplate",
            "TelemetryChannel",
            "CompositeResearchTemplate",
        ]
    )
    allowed_relation_types: list[str] = Field(
        default_factory=lambda: [
            "REQUIRES_EVIDENCE",
            "SUPPORTED_BY",
            "CONTRADICTED_BY",
            "EXCLUDES",
            "USES_FEATURE",
            "DOCUMENTED_IN",
            "AFFECTS",
            "USES_TELEMETRY_FIELD",
            "HAS_MEMBER_CANDIDATE",
        ]
    )
    allowed_property_filter_keys: list[str] = Field(
        default_factory=lambda: [
            "knowledge_version",
            "object_id",
            "component_instance_id",
            "operating_condition_id",
            "evidence_level",
            "review_status",
        ]
    )
    maximum_depth: int = Field(default=4, ge=1, le=6)
    maximum_results: int = Field(default=500, ge=1, le=1000)

    @field_validator("allowed_node_labels", "allowed_relation_types", "allowed_property_filter_keys")
    @classmethod
    def unique_allowlist(cls, value: list[str]) -> list[str]:
        if not value or len(value) != len(set(value)):
            raise ValueError("graph query allow-lists must be nonempty and unique")
        for item in value:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item):
                raise ValueError(f"invalid graph allow-list identifier: {item}")
        return value


class CompiledGraphQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-compiled-graph-query.v1"
    query_plan_id: str
    query_kind: GraphQueryKind
    cypher: str = Field(min_length=1)
    parameters: dict[str, Any]
    expected_columns: list[str]
    read_only: bool = True
    policy_id: str
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Cypher is generated from fixed read-only templates. Entity IDs and filters remain parameters; "
        "allow-listed labels, relations, depth, and result limits are validated before compilation."
    )

    @model_validator(mode="after")
    def enforce_read_only(self) -> "CompiledGraphQuery":
        if not self.read_only:
            raise ValueError("compiled graph query must be read-only")
        validate_read_only_cypher(self.cypher)
        return self


class GraphQueryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_entity_id: str
    target_entity_id: str
    start_entity_types: list[str]
    target_entity_types: list[str]
    relation_type: str | None = None
    path_entity_ids: list[str] = Field(default_factory=list)
    path_relation_types: list[str] = Field(default_factory=list)
    relationship_properties: dict[str, Any] = Field(default_factory=dict)
    target_properties: dict[str, Any] = Field(default_factory=dict)


class GraphQueryExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-graph-query-execution-result.v1"
    query_plan_id: str
    query_kind: GraphQueryKind
    row_count: int = Field(ge=0)
    rows: list[GraphQueryRow]
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    knowledge_version: str | None = None
    truncated: bool = False
    claim_boundary: str = "Read-only graph retrieval result; ranking and formal diagnosis remain downstream responsibilities."


class GraphEvidenceAssociation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    evidence_definition_id: str
    relation_type: str
    knowledge_version: str | None = None
    source_query_plan_id: str


_FORBIDDEN_CYPHER = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|GRANT|DENY|REVOKE|LOAD\s+CSV|FOREACH|CALL|APOC\.)\b",
    flags=re.IGNORECASE,
)


def validate_read_only_cypher(cypher: str) -> None:
    if ";" in cypher.strip().rstrip(";"):
        raise ValueError("multiple Cypher statements are forbidden")
    match = _FORBIDDEN_CYPHER.search(cypher)
    if match:
        raise ValueError(f"forbidden Cypher token in read-only query: {match.group(1)}")
    if not re.search(r"\b(MATCH|OPTIONAL\s+MATCH)\b", cypher, flags=re.IGNORECASE):
        raise ValueError("read-only query must contain MATCH")
    if not re.search(r"\bRETURN\b", cypher, flags=re.IGNORECASE):
        raise ValueError("read-only query must contain RETURN")


def _query_hash(cypher: str, parameters: dict[str, Any], policy_id: str) -> str:
    payload = {"cypher": cypher, "parameters": parameters, "policy_id": policy_id}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ReadOnlyCypherCompiler:
    def __init__(self, policy: ReadOnlyGraphQueryPolicy | None = None) -> None:
        self.policy = policy or ReadOnlyGraphQueryPolicy()

    def _validate_plan(self, plan: GraphQueryPlan) -> None:
        if not plan.start_entity_ids:
            raise ValueError("GraphQueryPlan requires start_entity_ids")
        if len(plan.start_entity_ids) != len(set(plan.start_entity_ids)):
            raise ValueError("GraphQueryPlan start_entity_ids must be unique")
        if not plan.allowed_relation_types:
            raise ValueError("GraphQueryPlan requires allowed_relation_types")
        unknown_relations = set(plan.allowed_relation_types) - set(self.policy.allowed_relation_types)
        if unknown_relations:
            raise ValueError(f"relation types are not allowed: {sorted(unknown_relations)}")
        if not plan.target_entity_types:
            raise ValueError("GraphQueryPlan requires target_entity_types")
        unknown_labels = set(plan.target_entity_types) - set(self.policy.allowed_node_labels)
        if unknown_labels:
            raise ValueError(f"target entity types are not allowed: {sorted(unknown_labels)}")
        unknown_filters = set(plan.operating_condition_filters) - set(self.policy.allowed_property_filter_keys)
        if unknown_filters:
            raise ValueError(f"property filters are not allowed: {sorted(unknown_filters)}")
        if plan.max_depth > self.policy.maximum_depth:
            raise ValueError("query max_depth exceeds policy")
        if plan.max_results > self.policy.maximum_results:
            raise ValueError("query max_results exceeds policy")
        for key, value in plan.operating_condition_filters.items():
            if isinstance(value, (dict, list, tuple, set)):
                raise ValueError(f"property filter {key} must be scalar")

    @staticmethod
    def _infer_kind(plan: GraphQueryPlan) -> GraphQueryKind:
        targets = set(plan.target_entity_types)
        if targets == {"FaultMode"}:
            return GraphQueryKind.FORWARD_EVIDENCE_TO_FAULT
        if targets == {"EvidenceDefinition"} and plan.max_depth == 1:
            return GraphQueryKind.BACKWARD_FAULT_TO_EVIDENCE
        return GraphQueryKind.GENERAL_READ_ONLY_PATH

    def compile(
        self,
        plan: GraphQueryPlan,
        *,
        query_kind: GraphQueryKind = GraphQueryKind.AUTO,
        knowledge_version: str | None = None,
    ) -> CompiledGraphQuery:
        self._validate_plan(plan)
        kind = self._infer_kind(plan) if query_kind == GraphQueryKind.AUTO else query_kind
        parameters: dict[str, Any] = {
            "start_entity_ids": list(plan.start_entity_ids),
            "allowed_relation_types": list(plan.allowed_relation_types),
            "target_entity_types": list(plan.target_entity_types),
            "property_filters": dict(plan.operating_condition_filters),
            "max_results": plan.max_results,
            "knowledge_version": knowledge_version,
        }
        if kind == GraphQueryKind.FORWARD_EVIDENCE_TO_FAULT:
            cypher = """
MATCH (start:EvidenceDefinition)<-[rel]-(target:FaultMode)
WHERE start.id IN $start_entity_ids
  AND type(rel) IN $allowed_relation_types
  AND ($knowledge_version IS NULL OR coalesce(rel.knowledge_version, $knowledge_version) = $knowledge_version)
  AND all(key IN keys($property_filters) WHERE target[key] = $property_filters[key])
RETURN start.id AS start_entity_id,
       target.id AS target_entity_id,
       labels(start) AS start_entity_types,
       labels(target) AS target_entity_types,
       type(rel) AS relation_type,
       [start.id, target.id] AS path_entity_ids,
       [type(rel)] AS path_relation_types,
       properties(rel) AS relationship_properties,
       properties(target) AS target_properties
ORDER BY target_entity_id, start_entity_id
LIMIT $max_results
""".strip()
        elif kind == GraphQueryKind.BACKWARD_FAULT_TO_EVIDENCE:
            cypher = """
MATCH (start:FaultMode)-[rel]->(target:EvidenceDefinition)
WHERE start.id IN $start_entity_ids
  AND type(rel) IN $allowed_relation_types
  AND ($knowledge_version IS NULL OR coalesce(rel.knowledge_version, $knowledge_version) = $knowledge_version)
  AND all(key IN keys($property_filters) WHERE target[key] = $property_filters[key])
RETURN start.id AS start_entity_id,
       target.id AS target_entity_id,
       labels(start) AS start_entity_types,
       labels(target) AS target_entity_types,
       type(rel) AS relation_type,
       [start.id, target.id] AS path_entity_ids,
       [type(rel)] AS path_relation_types,
       properties(rel) AS relationship_properties,
       properties(target) AS target_properties
ORDER BY start_entity_id, target_entity_id
LIMIT $max_results
""".strip()
        elif kind == GraphQueryKind.GENERAL_READ_ONLY_PATH:
            depth = int(plan.max_depth)
            cypher = f"""
MATCH path=(start)-[*1..{depth}]-(target)
WHERE start.id IN $start_entity_ids
  AND all(rel IN relationships(path) WHERE type(rel) IN $allowed_relation_types)
  AND any(label IN labels(target) WHERE label IN $target_entity_types)
  AND all(key IN keys($property_filters) WHERE target[key] = $property_filters[key])
RETURN start.id AS start_entity_id,
       target.id AS target_entity_id,
       labels(start) AS start_entity_types,
       labels(target) AS target_entity_types,
       CASE WHEN length(path) = 1 THEN type(relationships(path)[0]) ELSE NULL END AS relation_type,
       [node IN nodes(path) | node.id] AS path_entity_ids,
       [rel IN relationships(path) | type(rel)] AS path_relation_types,
       CASE WHEN length(path) = 1 THEN properties(relationships(path)[0]) ELSE {{}} END AS relationship_properties,
       properties(target) AS target_properties
ORDER BY length(path), target_entity_id
LIMIT $max_results
""".strip()
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported graph query kind: {kind}")
        validate_read_only_cypher(cypher)
        expected = [
            "start_entity_id",
            "target_entity_id",
            "start_entity_types",
            "target_entity_types",
            "relation_type",
            "path_entity_ids",
            "path_relation_types",
            "relationship_properties",
            "target_properties",
        ]
        return CompiledGraphQuery(
            query_plan_id=plan.query_plan_id,
            query_kind=kind,
            cypher=cypher,
            parameters=parameters,
            expected_columns=expected,
            policy_id=self.policy.policy_id,
            query_sha256=_query_hash(cypher, parameters, self.policy.policy_id),
        )


class _SessionProtocol(Protocol):
    def run(self, query: str, **parameters: Any) -> Iterable[Any]: ...
    def __enter__(self) -> "_SessionProtocol": ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...


class _DriverProtocol(Protocol):
    def session(self, **kwargs: Any) -> _SessionProtocol: ...


class ReadOnlyGraphQueryExecutor:
    def __init__(self, driver: _DriverProtocol) -> None:
        self.driver = driver

    def execute(
        self,
        compiled: CompiledGraphQuery,
        *,
        knowledge_version: str | None = None,
    ) -> GraphQueryExecutionResult:
        validate_read_only_cypher(compiled.cypher)
        try:
            session_context = self.driver.session(default_access_mode="READ")
        except TypeError:  # small fake drivers and old compatible wrappers
            session_context = self.driver.session()
        with session_context as session:
            raw_records = session.run(compiled.cypher, **compiled.parameters)
            rows: list[GraphQueryRow] = []
            for record in raw_records:
                payload = dict(record)
                missing = set(compiled.expected_columns) - set(payload)
                if missing:
                    raise ValueError(f"graph query row is missing expected columns: {sorted(missing)}")
                payload = {key: payload[key] for key in compiled.expected_columns}
                rows.append(GraphQueryRow.model_validate(payload))
        max_results = int(compiled.parameters["max_results"])
        return GraphQueryExecutionResult(
            query_plan_id=compiled.query_plan_id,
            query_kind=compiled.query_kind,
            row_count=len(rows),
            rows=rows,
            query_sha256=compiled.query_sha256,
            knowledge_version=knowledge_version,
            truncated=len(rows) >= max_results,
        )


def build_forward_evidence_query_plan(
    evidence_definition_ids: list[str],
    *,
    max_results: int = 500,
) -> GraphQueryPlan:
    return GraphQueryPlan(
        query_plan_id=_stable_plan_id("GQF", evidence_definition_ids),
        purpose="Resolve activated evidence definitions to registered fault modes.",
        start_entity_ids=sorted(set(evidence_definition_ids)),
        allowed_relation_types=["REQUIRES_EVIDENCE", "SUPPORTED_BY", "CONTRADICTED_BY", "EXCLUDES"],
        target_entity_types=["FaultMode"],
        max_depth=1,
        max_results=max_results,
        read_only=True,
    )


def build_backward_fault_query_plan(
    fault_ids: list[str],
    *,
    max_results: int = 500,
) -> GraphQueryPlan:
    return GraphQueryPlan(
        query_plan_id=_stable_plan_id("GQB", fault_ids),
        purpose="Resolve candidate faults to expected evidence definitions.",
        start_entity_ids=sorted(set(fault_ids)),
        allowed_relation_types=["REQUIRES_EVIDENCE", "SUPPORTED_BY", "CONTRADICTED_BY", "EXCLUDES"],
        target_entity_types=["EvidenceDefinition"],
        max_depth=1,
        max_results=max_results,
        read_only=True,
    )


def _stable_plan_id(prefix: str, ids: list[str]) -> str:
    raw = json.dumps(sorted(set(ids)), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:20].upper()}"


def associations_from_query_result(result: GraphQueryExecutionResult) -> list[GraphEvidenceAssociation]:
    associations: list[GraphEvidenceAssociation] = []
    for row in result.rows:
        labels_start = set(row.start_entity_types)
        labels_target = set(row.target_entity_types)
        if "EvidenceDefinition" in labels_start and "FaultMode" in labels_target:
            evidence_id, fault_id = row.start_entity_id, row.target_entity_id
        elif "FaultMode" in labels_start and "EvidenceDefinition" in labels_target:
            fault_id, evidence_id = row.start_entity_id, row.target_entity_id
        else:
            continue
        if row.relation_type is None:
            continue
        associations.append(
            GraphEvidenceAssociation(
                fault_id=fault_id,
                evidence_definition_id=evidence_id,
                relation_type=row.relation_type,
                knowledge_version=result.knowledge_version,
                source_query_plan_id=result.query_plan_id,
            )
        )
    unique: dict[tuple[str, str, str], GraphEvidenceAssociation] = {}
    for item in associations:
        unique[(item.fault_id, item.evidence_definition_id, item.relation_type)] = item
    return [unique[key] for key in sorted(unique)]


class CsvEvidenceGraphReferenceExecutor:
    """Offline semantic reference for the direct forward/backward query templates.

    This class does not emulate Neo4j or claim database acceptance. It is used to verify that
    the versioned graph CSV release and the parameterized query result schema produce the same
    associations expected by BiRRF before a real Neo4j service is available.
    """

    def __init__(self, nodes_csv: str | "Path", edges_csv: str | "Path") -> None:
        import csv
        from pathlib import Path

        self.nodes_csv = Path(nodes_csv)
        self.edges_csv = Path(edges_csv)
        self.nodes: dict[str, dict[str, Any]] = {}
        with self.nodes_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                properties = json.loads(row.get("properties_json") or "{}")
                self.nodes[row["node_id"]] = {
                    "id": row["node_id"],
                    "node_type": row["node_type"],
                    "name": row.get("name", ""),
                    **properties,
                }
        self.edges: list[dict[str, Any]] = []
        with self.edges_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                properties = json.loads(row.get("properties_json") or "{}")
                self.edges.append(
                    {
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                        "relation": row["relation"],
                        "properties": properties,
                    }
                )

    def execute(
        self,
        plan: GraphQueryPlan,
        *,
        query_kind: GraphQueryKind = GraphQueryKind.AUTO,
        knowledge_version: str | None = None,
        compiler: ReadOnlyCypherCompiler | None = None,
    ) -> GraphQueryExecutionResult:
        compiler = compiler or ReadOnlyCypherCompiler()
        compiled = compiler.compile(plan, query_kind=query_kind, knowledge_version=knowledge_version)
        if compiled.query_kind not in {
            GraphQueryKind.FORWARD_EVIDENCE_TO_FAULT,
            GraphQueryKind.BACKWARD_FAULT_TO_EVIDENCE,
        }:
            raise ValueError("CSV reference executor supports only direct forward/backward evidence queries")
        starts = set(plan.start_entity_ids)
        relations = set(plan.allowed_relation_types)
        rows: list[GraphQueryRow] = []
        for edge in self.edges:
            if edge["relation"] not in relations:
                continue
            source = self.nodes.get(edge["source_id"])
            target = self.nodes.get(edge["target_id"])
            if source is None or target is None:
                raise ValueError("graph edge references a node absent from the node catalog")
            if compiled.query_kind == GraphQueryKind.FORWARD_EVIDENCE_TO_FAULT:
                if target["node_type"] != "EvidenceDefinition" or source["node_type"] != "FaultMode":
                    continue
                if target["id"] not in starts:
                    continue
                start_node, target_node = target, source
            else:
                if source["node_type"] != "FaultMode" or target["node_type"] != "EvidenceDefinition":
                    continue
                if source["id"] not in starts:
                    continue
                start_node, target_node = source, target
            if any(target_node.get(key) != value for key, value in plan.operating_condition_filters.items()):
                continue
            rows.append(
                GraphQueryRow(
                    start_entity_id=start_node["id"],
                    target_entity_id=target_node["id"],
                    start_entity_types=[start_node["node_type"]],
                    target_entity_types=[target_node["node_type"]],
                    relation_type=edge["relation"],
                    path_entity_ids=[start_node["id"], target_node["id"]],
                    path_relation_types=[edge["relation"]],
                    relationship_properties=edge["properties"],
                    target_properties={key: value for key, value in target_node.items() if key != "node_type"},
                )
            )
        rows.sort(key=lambda item: (item.start_entity_id, item.target_entity_id, item.relation_type or ""))
        limited = rows[: plan.max_results]
        return GraphQueryExecutionResult(
            query_plan_id=plan.query_plan_id,
            query_kind=compiled.query_kind,
            row_count=len(limited),
            rows=limited,
            query_sha256=compiled.query_sha256,
            knowledge_version=knowledge_version,
            truncated=len(rows) > len(limited),
            claim_boundary=(
                "Offline CSV semantic reference result; this is not a Neo4j connectivity, transaction, index, "
                "authorization, or performance acceptance result."
            ),
        )
