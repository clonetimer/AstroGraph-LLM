from __future__ import annotations

import os
from pathlib import Path
import shutil

from astrograph.anomaly import EpisodeTemporalConstraintCatalog, JsonEpisodeLifecycleStore
from astrograph.audit import JsonlAuditStore
from astrograph.cases import DiagnosticCaseRegistry
from astrograph.config import DiagnosisPolicy
from astrograph.evidence import EvidenceConstraintCatalog, EvidenceRegistry
from astrograph.experiments import JsonExperimentRegistry
from astrograph.idempotency import JsonIdempotencyStore, SqliteIdempotencyStore
from astrograph.knowledge import JsonKnowledgeGovernanceStore, KnowledgeRelease
from astrograph.knowledge.repository import CsvKnowledgeRepository, Neo4jKnowledgeRepository
from astrograph.lifecycle import JsonRequestLifecycleStore
from astrograph.persistence import JsonStatePersistenceCoordinator
from astrograph.service.core import AstroGraphService
from astrograph.service.paths import ServicePathLayout


def _seed_active_release(project_root: Path, state_root: Path) -> tuple[Path, Path]:
    packaged = project_root / "data" / "knowledge_releases" / "active_release.json"
    state_path = state_root / "knowledge_releases" / "active_release.json"
    if not state_path.exists() and packaged.exists():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packaged, state_path)
    selected = state_path if state_path.exists() else packaged
    return state_path, selected


def build_service_from_environment(
    project_root: str | Path | None = None,
    *,
    state_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> AstroGraphService:
    layout = ServicePathLayout.resolve(
        project_root or os.environ.get("ASTROGRAPH_PROJECT_ROOT", "."),
        state_dir=state_dir,
        artifact_dir=artifact_dir,
    )
    root = layout.project_root
    state_root = layout.state_root
    repository_mode = os.environ.get("ASTROGRAPH_REPOSITORY", "csv").strip().lower()
    active_release, readable_active_release = _seed_active_release(root, state_root)
    configured_version = os.environ.get("ASTROGRAPH_KNOWLEDGE_VERSION", "").strip()
    if configured_version:
        knowledge_version = configured_version
    elif readable_active_release.exists():
        knowledge_version = KnowledgeRelease.from_json(readable_active_release).knowledge_version
    else:
        knowledge_version = "astrograph-ontology-v1-e0"
    if repository_mode == "neo4j":
        required = {
            "NEO4J_URI": os.environ.get("NEO4J_URI", ""),
            "NEO4J_USER": os.environ.get("NEO4J_USER", ""),
            "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", ""),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Neo4j mode requires environment variables: {', '.join(missing)}")
        repository = Neo4jKnowledgeRepository(
            required["NEO4J_URI"],
            required["NEO4J_USER"],
            required["NEO4J_PASSWORD"],
            knowledge_version=knowledge_version,
        )
    elif repository_mode == "csv":
        configured_edges = os.environ.get("ASTROGRAPH_CSV_KNOWLEDGE_EDGES", "").strip()
        if configured_edges:
            candidate = Path(configured_edges)
            edges_path = candidate if candidate.is_absolute() else root / candidate
        elif readable_active_release.exists():
            release = KnowledgeRelease.from_json(readable_active_release)
            edges_path = root / release.edges_file
        else:
            edges_path = root / "data" / "knowledge_seed" / "causes.csv"
        repository = CsvKnowledgeRepository(
            edges_path,
            knowledge_version=knowledge_version,
            release_path=readable_active_release if readable_active_release.exists() else None,
            project_root=root,
        )
    else:
        raise RuntimeError(f"unsupported ASTROGRAPH_REPOSITORY: {repository_mode}")
    audit_path = state_root / "audit" / "api_diagnostic_audit_v043.jsonl"
    case_path = state_root / "case_registry" / "diagnostic_cases_v043_api.json"
    idempotency_backend = os.environ.get("ASTROGRAPH_IDEMPOTENCY_BACKEND", "sqlite").strip().lower()
    if idempotency_backend == "sqlite":
        idempotency_path = state_root / "idempotency" / "requests_v0530.sqlite3"
        idempotency_store = SqliteIdempotencyStore(
            idempotency_path,
            wait_timeout_s=float(os.environ.get("ASTROGRAPH_IDEMPOTENCY_WAIT_TIMEOUT_S", "120")),
            stale_after_s=float(os.environ.get("ASTROGRAPH_IDEMPOTENCY_STALE_AFTER_S", "600")),
            heartbeat_interval_s=float(
                os.environ.get("ASTROGRAPH_IDEMPOTENCY_HEARTBEAT_INTERVAL_S", "0")
            ) or None,
        )
    elif idempotency_backend == "json":
        idempotency_path = state_root / "idempotency" / "requests_v043.json"
        idempotency_store = JsonIdempotencyStore(idempotency_path)
    else:
        raise RuntimeError(f"unsupported ASTROGRAPH_IDEMPOTENCY_BACKEND: {idempotency_backend}")
    episode_path = state_root / "episode_state" / "online_episodes.json"
    request_path = state_root / "request_lifecycle" / "requests.json"
    governance_path = state_root / "knowledge_governance" / "changes.json"
    experiment_path = state_root / "experiment_registry" / "experiments.json"

    episode_store = JsonEpisodeLifecycleStore(
        episode_path,
        max_gap_s=float(os.environ.get("ASTROGRAPH_EPISODE_MAX_GAP_S", "30")),
    )
    request_store = JsonRequestLifecycleStore(request_path)
    persistence = JsonStatePersistenceCoordinator(
        {
            "audit": audit_path,
            "cases": case_path,
            "idempotency": idempotency_path,
            "episodes": episode_path,
            "requests": request_path,
            "knowledge_governance": governance_path,
            "experiments": experiment_path,
            "active_release": active_release,
        },
        episode_store=episode_store,
        request_store=request_store,
    )
    service = AstroGraphService(
        repository=repository,
        policy=DiagnosisPolicy.from_json(root / "configs" / "diagnosis.json"),
        audit_store=JsonlAuditStore(audit_path),
        project_root=root,
        active_release_path=active_release,
        evidence_registry=EvidenceRegistry(state_root / "evidence_registry"),
        case_registry=DiagnosticCaseRegistry(case_path),
        evidence_constraints=EvidenceConstraintCatalog.from_json(root / "configs" / "evidence_constraints.json"),
        temporal_constraints=EpisodeTemporalConstraintCatalog.from_json(
            root / "configs" / "episode_temporal_constraints.json"
        ),
        idempotency_store=idempotency_store,
        episode_store=episode_store,
        request_lifecycle_store=request_store,
        knowledge_governance_store=JsonKnowledgeGovernanceStore(
            governance_path,
            project_root=root,
            active_release_path=active_release,
            history_dir=state_root / "knowledge_releases" / "history",
        ),
        experiment_registry=JsonExperimentRegistry(experiment_path),
        persistence_coordinator=persistence,
    )
    # Expose resolved roots for downstream adapters without making them infer
    # mutable locations from the immutable project tree.
    service.state_root = state_root
    service.artifact_root = layout.artifact_root
    return service
