from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.provider_lifecycle import BUILTIN_PROVIDER_QUALIFICATIONS, ProviderLifecycleProfile

from .query import (
    CsvEvidenceGraphReferenceExecutor,
    GraphEvidenceAssociation,
    ReadOnlyCypherCompiler,
    ReadOnlyGraphQueryExecutor,
    associations_from_query_result,
    build_backward_fault_query_plan,
    build_forward_evidence_query_plan,
)


class Neo4jKGEvidenceBindingConfig(BaseModel):
    """Versioned binding between online diagnosis and one read-only KG release."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(default="bolt://127.0.0.1:7687", min_length=1)
    read_user: str = Field(min_length=1)
    read_password_env: str = Field(default="ASTROGRAPH_NEO4J_READ_PASSWORD", min_length=1)
    release_id: str = Field(min_length=1)
    knowledge_version: str = Field(min_length=1)
    nodes_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    edges_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    graph_backend: str = "neo4j"
    fallback_mode: Literal["disabled", "csv_on_unavailable"] = "disabled"
    fallback_nodes_csv: str | None = None
    fallback_edges_csv: str | None = None

    @model_validator(mode="after")
    def validate_fallback(self) -> "Neo4jKGEvidenceBindingConfig":
        configured = bool(self.fallback_nodes_csv) or bool(self.fallback_edges_csv)
        if self.fallback_mode == "csv_on_unavailable":
            if not self.fallback_nodes_csv or not self.fallback_edges_csv:
                raise ValueError("csv_on_unavailable requires fallback_nodes_csv and fallback_edges_csv")
        elif configured:
            raise ValueError("fallback CSV paths require fallback_mode=csv_on_unavailable")
        return self


class KGEvidenceRuntimeReference(BaseModel):
    """KG-native receipt; diagnosis layers may adapt it to their release contract."""

    model_config = ConfigDict(extra="forbid")

    release_id: str
    knowledge_version: str
    nodes_sha256: str
    edges_sha256: str
    query_ids: list[str]
    query_sha256_values: list[str]
    read_user: str
    row_count: int
    graph_backend: str = "neo4j"
    claim_boundary: str = (
        "KG runtime evidence is scoped to a bound release and read-only query templates; "
        "it is not by itself a diagnosis or release authorization."
    )


class KGEvidenceRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference: KGEvidenceRuntimeReference
    forward_associations: list[GraphEvidenceAssociation]
    backward_associations: list[GraphEvidenceAssociation]
    selected_backend: str = "neo4j"
    degraded_to_reference: bool = False
    fallback_reason: str | None = None


class KGEvidenceRuntimeService:
    """Own read-only KG backend selection, integrity checks, and fallback semantics."""

    @staticmethod
    def lifecycle_profile(*, selected_backend: str = "neo4j") -> ProviderLifecycleProfile:
        if selected_backend == "csv_reference_fallback":
            return BUILTIN_PROVIDER_QUALIFICATIONS.get("kg-csv-reference.v1")
        return BUILTIN_PROVIDER_QUALIFICATIONS.get("neo4j-readonly.v1")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _reference_from_results(
        *,
        config: Neo4jKGEvidenceBindingConfig,
        forward: Any,
        backward: Any,
        read_user: str,
        graph_backend: str,
        claim_boundary: str | None = None,
    ) -> KGEvidenceRuntimeReference:
        payload: dict[str, Any] = {
            "release_id": config.release_id,
            "knowledge_version": config.knowledge_version,
            "nodes_sha256": config.nodes_sha256,
            "edges_sha256": config.edges_sha256,
            "query_ids": [forward.query_plan_id, backward.query_plan_id],
            "query_sha256_values": [forward.query_sha256, backward.query_sha256],
            "read_user": read_user,
            "row_count": forward.row_count + backward.row_count,
            "graph_backend": graph_backend,
        }
        if claim_boundary is not None:
            payload["claim_boundary"] = claim_boundary
        return KGEvidenceRuntimeReference(**payload)

    def query_live(
        self,
        *,
        config: Neo4jKGEvidenceBindingConfig,
        bundle: EpisodeEvidenceBundle,
        compiler: Any | None = None,
    ) -> KGEvidenceRuntimeResult:
        password = os.environ.get(config.read_password_env)
        if not password:
            raise ValueError(f"missing Neo4j password environment variable: {config.read_password_env}")
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(config.uri, auth=(config.read_user, password))
        try:
            driver.verify_connectivity()
            active_compiler = compiler or ReadOnlyCypherCompiler()
            executor = ReadOnlyGraphQueryExecutor(driver)
            evidence_ids = [item.evidence_definition_id for item in bundle.evidence_definitions]
            fault_ids = sorted({item.target_fault_id for item in bundle.evidence_definitions})
            forward = executor.execute(
                active_compiler.compile(
                    build_forward_evidence_query_plan(evidence_ids),
                    knowledge_version=config.knowledge_version,
                ),
                knowledge_version=config.knowledge_version,
            )
            backward = executor.execute(
                active_compiler.compile(
                    build_backward_fault_query_plan(fault_ids),
                    knowledge_version=config.knowledge_version,
                ),
                knowledge_version=config.knowledge_version,
            )
        finally:
            driver.close()
        return KGEvidenceRuntimeResult(
            reference=self._reference_from_results(
                config=config,
                forward=forward,
                backward=backward,
                read_user=config.read_user,
                graph_backend=config.graph_backend,
            ),
            forward_associations=associations_from_query_result(forward),
            backward_associations=associations_from_query_result(backward),
            selected_backend=config.graph_backend,
            degraded_to_reference=False,
        )

    def query_csv_reference(
        self,
        *,
        config: Neo4jKGEvidenceBindingConfig,
        bundle: EpisodeEvidenceBundle,
        fallback_reason: str,
        compiler: Any | None = None,
    ) -> KGEvidenceRuntimeResult:
        nodes_path = Path(str(config.fallback_nodes_csv))
        edges_path = Path(str(config.fallback_edges_csv))
        if not nodes_path.is_file() or not edges_path.is_file():
            raise FileNotFoundError("configured CSV KG fallback files are missing")
        if self._sha256_file(nodes_path).lower() != config.nodes_sha256.lower():
            raise ValueError("fallback nodes CSV hash does not match bound KG release")
        if self._sha256_file(edges_path).lower() != config.edges_sha256.lower():
            raise ValueError("fallback edges CSV hash does not match bound KG release")
        active_compiler = compiler or ReadOnlyCypherCompiler()
        executor = CsvEvidenceGraphReferenceExecutor(nodes_path, edges_path)
        evidence_ids = [item.evidence_definition_id for item in bundle.evidence_definitions]
        fault_ids = sorted({item.target_fault_id for item in bundle.evidence_definitions})
        forward = executor.execute(
            build_forward_evidence_query_plan(evidence_ids),
            knowledge_version=config.knowledge_version,
            compiler=active_compiler,
        )
        backward = executor.execute(
            build_backward_fault_query_plan(fault_ids),
            knowledge_version=config.knowledge_version,
            compiler=active_compiler,
        )
        return KGEvidenceRuntimeResult(
            reference=self._reference_from_results(
                config=config,
                forward=forward,
                backward=backward,
                read_user=f"csv-reference:{config.read_user}",
                graph_backend="csv_reference_fallback",
                claim_boundary=(
                    "Versioned CSV semantic fallback selected after Neo4j unavailability. "
                    "This is deterministic KG evidence, not Neo4j availability or performance evidence."
                ),
            ),
            forward_associations=associations_from_query_result(forward),
            backward_associations=associations_from_query_result(backward),
            selected_backend="csv_reference_fallback",
            degraded_to_reference=True,
            fallback_reason=fallback_reason,
        )

    def query_bound(
        self,
        *,
        config: Neo4jKGEvidenceBindingConfig,
        bundle: EpisodeEvidenceBundle,
        compiler: Any | None = None,
        live_query: Callable[[], Any] | None = None,
        fallback_query: Callable[[str], Any] | None = None,
    ) -> Any:
        """Execute the bound backend and apply the configured deterministic fallback.

        Optional callables exist only for backward-compatible test seams and adapters;
        the fallback decision remains owned by this service.
        """
        try:
            if live_query is not None:
                return live_query()
            return self.query_live(config=config, bundle=bundle, compiler=compiler)
        except Exception as exc:
            if config.fallback_mode != "csv_on_unavailable":
                raise
            reason = f"{type(exc).__name__}: {exc}"
            if fallback_query is not None:
                return fallback_query(reason)
            return self.query_csv_reference(
                config=config,
                bundle=bundle,
                fallback_reason=reason,
                compiler=compiler,
            )
