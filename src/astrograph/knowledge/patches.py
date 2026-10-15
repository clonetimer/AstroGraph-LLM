"""Versioned, provenance-bound knowledge patches and sandbox impact replay."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.structured_output import StructuredDiagnosisOutput
from astrograph.fusion.models import CandidateFusionPackage

from .release import KnowledgeRelease, sha256_file, validate_knowledge_release


EDGE_FIELDS = (
    "fault_id", "fault_name", "anomaly_id", "anomaly_name", "weight", "edge_role",
    "evidence_level", "evidence_source", "source_ref", "review_status", "provisional",
    "knowledge_version", "applicable_condition_ids", "excluded_condition_ids",
)
ALLOWED_EDGE_ROLES = {"direct", "indirect", "cascade", "counter", "exclusion"}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def knowledge_release_sha256(release: KnowledgeRelease) -> str:
    return _canonical_sha256(release.model_dump(mode="json"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _resolve_inside(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("knowledge patch paths must remain inside project_root")
    return resolved


class KnowledgePatchOperationKind(str, Enum):
    ADD = "ADD"
    SUPERSEDE = "SUPERSEDE"
    DEPRECATE = "DEPRECATE"


class KnowledgePatchSourceSplit(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    DEVELOPMENT = "DEVELOPMENT"
    EXTERNAL = "EXTERNAL"
    HIL = "HIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    TEST = "TEST"


class KnowledgeEdgeValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fault_id: str = Field(min_length=1)
    fault_name: str = Field(min_length=1)
    anomaly_id: str = Field(min_length=1)
    anomaly_name: str = Field(min_length=1)
    weight: float = Field(ge=0.0)
    edge_role: str = Field(min_length=1)
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"]
    evidence_source: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    review_status: str = Field(min_length=1)
    provisional: bool = True
    applicable_condition_ids: list[str] = Field(default_factory=list)
    excluded_condition_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_role(self) -> "KnowledgeEdgeValue":
        if self.edge_role not in ALLOWED_EDGE_ROLES:
            raise ValueError(f"unsupported knowledge edge_role: {self.edge_role}")
        if set(self.applicable_condition_ids) & set(self.excluded_condition_ids):
            raise ValueError("applicable and excluded conditions must be disjoint")
        return self

    def as_csv_row(self, *, knowledge_version: str) -> dict[str, str]:
        return {
            "fault_id": self.fault_id,
            "fault_name": self.fault_name,
            "anomaly_id": self.anomaly_id,
            "anomaly_name": self.anomaly_name,
            "weight": format(self.weight, ".12g"),
            "edge_role": self.edge_role,
            "evidence_level": self.evidence_level,
            "evidence_source": self.evidence_source,
            "source_ref": self.source_ref,
            "review_status": self.review_status,
            "provisional": str(self.provisional).lower(),
            "knowledge_version": knowledge_version,
            "applicable_condition_ids": ";".join(self.applicable_condition_ids),
            "excluded_condition_ids": ";".join(self.excluded_condition_ids),
        }


class KnowledgePatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    kind: KnowledgePatchOperationKind
    fault_id: str = Field(min_length=1)
    anomaly_id: str = Field(min_length=1)
    edge: KnowledgeEdgeValue | None = None
    reason: str = Field(min_length=1)
    valid_from: str | None = None
    valid_to: str | None = None

    @model_validator(mode="after")
    def validate_edge(self) -> "KnowledgePatchOperation":
        if self.kind in {KnowledgePatchOperationKind.ADD, KnowledgePatchOperationKind.SUPERSEDE}:
            if self.edge is None:
                raise ValueError("ADD/SUPERSEDE operation requires an edge payload")
            if (self.edge.fault_id, self.edge.anomaly_id) != (self.fault_id, self.anomaly_id):
                raise ValueError("knowledge patch operation key and edge payload differ")
        elif self.edge is not None:
            raise ValueError("DEPRECATE operation must not replace its edge payload")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("knowledge patch valid_to precedes valid_from")
        return self


class KnowledgePatchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-patch-candidate.v1"] = (
        "astrograph-knowledge-patch-candidate.v1"
    )
    candidate_id: str = Field(min_length=1)
    base_release_id: str = Field(min_length=1)
    base_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposed_release_id: str = Field(min_length=1)
    proposed_knowledge_version: str = Field(min_length=1)
    operations: list[KnowledgePatchOperation] = Field(min_length=1)
    source_candidate_ids: list[str] = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    provenance_independence_groups: list[str] = Field(min_length=1)
    source_split: KnowledgePatchSourceSplit
    created_by: str = Field(min_length=1)
    transaction_time: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    human_review_required: Literal[True] = True
    active_graph_modified: Literal[False] = False
    patch_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Candidate delta only. It cannot alter the active graph until deterministic validation, "
        "sandbox replay, human approval, and atomic KnowledgeRelease publication complete."
    )

    @model_validator(mode="after")
    def validate_candidate(self) -> "KnowledgePatchCandidate":
        ids = [item.operation_id for item in self.operations]
        keys = [(item.fault_id, item.anomaly_id) for item in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("knowledge patch operation IDs must be unique")
        if len(keys) != len(set(keys)):
            raise ValueError("knowledge patch may mutate an edge key only once")
        expected = self.compute_sha256()
        if self.patch_sha256 is not None and self.patch_sha256 != expected:
            raise ValueError("knowledge patch_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json", exclude={"patch_sha256"}))


class KnowledgePatchCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: str
    check_id: str
    status: Literal["PASS", "FAIL", "PENDING"]
    details: dict[str, Any] = Field(default_factory=dict)


class KnowledgePatchValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-patch-validation.v1"] = (
        "astrograph-knowledge-patch-validation.v1"
    )
    candidate_id: str
    status: Literal["BLOCKED", "READY_FOR_SANDBOX"]
    checks: list[KnowledgePatchCheck]
    affected_fault_ids: list[str]
    affected_edge_keys: list[str]
    candidate_edge_count: int = Field(ge=0)
    candidate_edges_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    active_graph_modified: Literal[False] = False


class CompiledKnowledgePatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-compiled-knowledge-patch.v1"] = (
        "astrograph-compiled-knowledge-patch.v1"
    )
    candidate_id: str
    release_path: str
    release: KnowledgeRelease
    validation: KnowledgePatchValidationReport
    active_graph_modified: Literal[False] = False


class KnowledgePatchImpactReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-patch-impact.v1"] = (
        "astrograph-knowledge-patch-impact.v1"
    )
    candidate_id: str
    status: Literal["PASS", "BLOCKED"]
    base_release_id: str
    candidate_release_id: str
    base_output_sha256: str
    candidate_output_sha256: str
    changed_paths: list[str]
    declared_expected_change_paths: list[str]
    unexpected_change_paths: list[str]
    active_graph_modified: Literal[False] = False
    human_review_required: Literal[True] = True


KnowledgePatchReplayDimension = Literal[
    "kg_forward_fault_ids",
    "kg_backward_fault_ids",
    "birrf_top_fault_id",
    "governor_disposition",
    "policy_status",
    "structured_output_sha256",
]


class AstroGraphKnowledgePatchReplayEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(min_length=1)
    kg_forward_fault_ids: list[str] = Field(default_factory=list)
    kg_backward_fault_ids: list[str] = Field(default_factory=list)
    birrf_top_fault_id: str | None = None
    governor_disposition: str = Field(min_length=1)
    policy_status: str = Field(min_length=1)
    structured_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AstroGraphKnowledgePatchReplayOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-patch-replay-output.v1"] = (
        "astrograph-knowledge-patch-replay-output.v1"
    )
    scenario_set_id: str = Field(min_length=1)
    scenario_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episodes: list[AstroGraphKnowledgePatchReplayEpisode] = Field(min_length=1)
    active_graph_modified: Literal[False] = False

    @model_validator(mode="after")
    def unique_episodes(self) -> "AstroGraphKnowledgePatchReplayOutput":
        ids = [item.episode_id for item in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("knowledge patch replay episode IDs must be unique")
        return self


class AstroGraphKnowledgePatchReplayContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-knowledge-patch-replay-contract.v1"] = (
        "astrograph-knowledge-patch-replay-contract.v1"
    )
    contract_id: str = Field(min_length=1)
    scenario_set_id: str = Field(min_length=1)
    scenario_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_episode_ids: list[str] = Field(min_length=1)
    allowed_change_dimensions: list[KnowledgePatchReplayDimension] = Field(default_factory=list)
    contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_contract(self) -> "AstroGraphKnowledgePatchReplayContract":
        if len(self.required_episode_ids) != len(set(self.required_episode_ids)):
            raise ValueError("knowledge patch replay required episode IDs must be unique")
        if len(self.allowed_change_dimensions) != len(set(self.allowed_change_dimensions)):
            raise ValueError("knowledge patch replay allowed dimensions must be unique")
        expected = self.compute_sha256()
        if self.contract_sha256 is not None and self.contract_sha256 != expected:
            raise ValueError("knowledge patch replay contract_sha256 mismatch")
        return self

    def compute_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json", exclude={"contract_sha256"}))


def build_astrograph_knowledge_patch_replay_output(
    *,
    scenario_set_id: str,
    scenario_set_sha256: str,
    episode_payloads: list[dict[str, Any]],
) -> AstroGraphKnowledgePatchReplayOutput:
    """Project real whole-spacecraft outputs onto governed replay dimensions.

    The adapter deliberately validates the typed fusion and final structured
    artifacts instead of accepting a caller-authored summary. A replay therefore
    fails closed when an episode is incomplete, internally inconsistent, or has a
    tampered final output.
    """
    episodes: list[AstroGraphKnowledgePatchReplayEpisode] = []
    for payload in episode_payloads:
        fusion_payload = payload.get("candidate_fusion")
        if not fusion_payload:
            raise ValueError("knowledge patch replay requires candidate_fusion for every episode")
        fusion = CandidateFusionPackage.model_validate(fusion_payload)

        structured_payload = payload.get("structured_diagnosis_output")
        if not structured_payload:
            raise ValueError(
                "knowledge patch replay requires structured_diagnosis_output for every episode"
            )
        structured = StructuredDiagnosisOutput.model_validate(structured_payload)
        if structured.episode_id != fusion.episode_id:
            raise ValueError("knowledge patch replay fusion and structured episode IDs differ")

        decision = payload.get("decision_record") or {}
        policy = payload.get("diagnosis_policy_decision") or {}
        if decision.get("episode_id") != fusion.episode_id:
            raise ValueError("knowledge patch replay governor episode ID differs")
        disposition = decision.get("disposition")
        policy_status = policy.get("status")
        if not disposition or not policy_status:
            raise ValueError("knowledge patch replay requires governor disposition and policy status")

        episodes.append(AstroGraphKnowledgePatchReplayEpisode(
            episode_id=fusion.episode_id,
            kg_forward_fault_ids=[
                item.fault_id
                for item in sorted(
                    fusion.forward_ranking.candidates,
                    key=lambda item: (item.rank, item.fault_id),
                )
            ],
            kg_backward_fault_ids=[
                item.fault_id
                for item in sorted(
                    fusion.backward_ranking.candidates,
                    key=lambda item: (item.rank, item.fault_id),
                )
            ],
            birrf_top_fault_id=fusion.birrf_result.top_fault_id,
            governor_disposition=str(disposition),
            policy_status=str(policy_status),
            structured_output_sha256=structured.output_sha256,
        ))
    return AstroGraphKnowledgePatchReplayOutput(
        scenario_set_id=scenario_set_id,
        scenario_set_sha256=scenario_set_sha256,
        episodes=sorted(episodes, key=lambda item: item.episode_id),
    )


def build_knowledge_patch_candidate(
    *,
    base_release: KnowledgeRelease,
    proposed_release_id: str,
    proposed_knowledge_version: str,
    operations: list[KnowledgePatchOperation],
    source_candidate_ids: list[str],
    source_refs: list[str],
    provenance_independence_groups: list[str],
    source_split: KnowledgePatchSourceSplit,
    created_by: str,
) -> KnowledgePatchCandidate:
    payload = {
        "base_release_id": base_release.release_id,
        "base_release_sha256": knowledge_release_sha256(base_release),
        "proposed_release_id": proposed_release_id,
        "proposed_knowledge_version": proposed_knowledge_version,
        "operations": [item.model_dump(mode="json") for item in operations],
        "source_candidate_ids": sorted(source_candidate_ids),
    }
    provisional = KnowledgePatchCandidate(
        candidate_id="KPATCH-" + _canonical_sha256(payload)[:20].upper(),
        base_release_id=base_release.release_id,
        base_release_sha256=knowledge_release_sha256(base_release),
        proposed_release_id=proposed_release_id,
        proposed_knowledge_version=proposed_knowledge_version,
        operations=operations,
        source_candidate_ids=sorted(set(source_candidate_ids)),
        source_refs=sorted(set(source_refs)),
        provenance_independence_groups=sorted(set(provenance_independence_groups)),
        source_split=source_split,
        created_by=created_by,
    )
    return provisional.model_copy(update={"patch_sha256": provisional.compute_sha256()})


def _apply_operations(
    rows: list[dict[str, str]], candidate: KnowledgePatchCandidate
) -> tuple[list[dict[str, str]], list[str]]:
    by_key = {(row.get("fault_id", ""), row.get("anomaly_id", "")): dict(row) for row in rows}
    errors: list[str] = []
    for operation in candidate.operations:
        key = (operation.fault_id, operation.anomaly_id)
        exists = key in by_key
        if operation.kind is KnowledgePatchOperationKind.ADD:
            if exists:
                errors.append(f"ADD_EXISTING_EDGE:{key[0]}:{key[1]}")
            else:
                by_key[key] = operation.edge.as_csv_row(
                    knowledge_version=candidate.proposed_knowledge_version
                )
        elif operation.kind is KnowledgePatchOperationKind.SUPERSEDE:
            if not exists:
                errors.append(f"SUPERSEDE_MISSING_EDGE:{key[0]}:{key[1]}")
            else:
                by_key[key] = operation.edge.as_csv_row(
                    knowledge_version=candidate.proposed_knowledge_version
                )
        else:
            if not exists:
                errors.append(f"DEPRECATE_MISSING_EDGE:{key[0]}:{key[1]}")
            else:
                by_key[key]["review_status"] = "retired"
                by_key[key]["provisional"] = "true"
                by_key[key]["source_ref"] = (
                    by_key[key].get("source_ref", "") + f"; deprecated: {operation.reason}"
                ).strip("; ")
    for row in by_key.values():
        row["knowledge_version"] = candidate.proposed_knowledge_version
    return sorted(by_key.values(), key=lambda row: (row["fault_id"], row["anomaly_id"])), errors


def validate_knowledge_patch(
    candidate: KnowledgePatchCandidate,
    *,
    base_release: KnowledgeRelease,
    project_root: str | Path,
) -> KnowledgePatchValidationReport:
    root = Path(project_root).resolve()
    checks: list[KnowledgePatchCheck] = []

    def check(layer: str, check_id: str, passed: bool, **details: Any) -> None:
        checks.append(KnowledgePatchCheck(
            layer=layer, check_id=check_id, status="PASS" if passed else "FAIL", details=details
        ))

    check(
        "K1_STRUCTURE", "candidate_hash",
        candidate.patch_sha256 == candidate.compute_sha256(),
    )
    check(
        "K2_PROVENANCE", "source_refs_present",
        bool(candidate.source_refs and candidate.provenance_independence_groups),
    )
    check(
        "K2_PROVENANCE", "test_split_not_promotion_basis",
        candidate.source_split is not KnowledgePatchSourceSplit.TEST,
        source_split=candidate.source_split.value,
    )
    check(
        "K3_VERSION", "base_release_identity",
        candidate.base_release_id == base_release.release_id
        and candidate.base_release_sha256 == knowledge_release_sha256(base_release),
        actual_release_id=base_release.release_id,
        actual_release_sha256=knowledge_release_sha256(base_release),
    )

    edges_path = _resolve_inside(root, base_release.edges_file)
    faults_path = _resolve_inside(root, base_release.fault_modes_file)
    anomalies_path = _resolve_inside(root, base_release.anomalies_file)
    paths_exist = all(path.is_file() for path in (edges_path, faults_path, anomalies_path))
    check("K1_STRUCTURE", "base_assets_exist", paths_exist)
    candidate_rows: list[dict[str, str]] = []
    operation_errors: list[str] = []
    known_faults: set[str] = set()
    known_anomalies: set[str] = set()
    if paths_exist:
        known_faults = {row["fault_id"] for row in _read_csv(faults_path)}
        known_anomalies = {row["anomaly_id"] for row in _read_csv(anomalies_path)}
        candidate_rows, operation_errors = _apply_operations(_read_csv(edges_path), candidate)
    entity_errors = [
        f"UNKNOWN_FAULT:{item.fault_id}"
        for item in candidate.operations if item.fault_id not in known_faults
    ] + [
        f"UNKNOWN_ANOMALY:{item.anomaly_id}"
        for item in candidate.operations if item.anomaly_id not in known_anomalies
    ]
    check("K1_STRUCTURE", "known_entities", not entity_errors, errors=entity_errors)
    check("K3_VERSION", "operation_conflicts", not operation_errors, errors=operation_errors)
    edge_digest = _canonical_sha256(candidate_rows) if candidate_rows else None
    check(
        "K4_STATIC_IMPACT", "affected_scope_declared", bool(candidate.operations),
        affected_edge_keys=[f"{item.fault_id}->{item.anomaly_id}" for item in candidate.operations],
    )
    blocked = any(item.status == "FAIL" for item in checks)
    return KnowledgePatchValidationReport(
        candidate_id=candidate.candidate_id,
        status="BLOCKED" if blocked else "READY_FOR_SANDBOX",
        checks=checks,
        affected_fault_ids=sorted({item.fault_id for item in candidate.operations}),
        affected_edge_keys=sorted({
            f"{item.fault_id}->{item.anomaly_id}" for item in candidate.operations
        }),
        candidate_edge_count=len(candidate_rows),
        candidate_edges_sha256=edge_digest,
    )


def compile_knowledge_patch_release(
    candidate: KnowledgePatchCandidate,
    *,
    base_release: KnowledgeRelease,
    project_root: str | Path,
    output_dir: str | Path,
) -> CompiledKnowledgePatch:
    root = Path(project_root).resolve()
    target = _resolve_inside(root, output_dir)
    validation = validate_knowledge_patch(candidate, base_release=base_release, project_root=root)
    if validation.status != "READY_FOR_SANDBOX":
        raise ValueError("knowledge patch candidate failed deterministic preflight")
    rows, errors = _apply_operations(_read_csv(_resolve_inside(root, base_release.edges_file)), candidate)
    if errors:
        raise ValueError(f"knowledge patch conflicts after validation: {errors}")
    target.mkdir(parents=True, exist_ok=True)
    edges_path = target / "causes.csv"
    with edges_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EDGE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    release = KnowledgeRelease(
        release_id=candidate.proposed_release_id,
        knowledge_version=candidate.proposed_knowledge_version,
        ontology_version=base_release.ontology_version,
        release_status="engineering_released",
        edges_file=str(edges_path.relative_to(root)).replace("\\", "/"),
        edges_sha256=sha256_file(edges_path),
        fault_modes_file=base_release.fault_modes_file,
        anomalies_file=base_release.anomalies_file,
        evidence_levels=sorted({row["evidence_level"] for row in rows}),
        review_statuses=sorted({row["review_status"] for row in rows}),
        edge_count=len(rows),
        claim_boundary=(
            f"Compiled from {candidate.candidate_id}; publication still requires sandbox impact "
            "PASS and the existing human-governed KnowledgeRelease workflow."
        ),
    )
    release_path = target / "release.json"
    release_path.write_text(
        json.dumps(release.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    release_validation = validate_knowledge_release(release, root)
    if release_validation["status"] != "PASS":
        raise ValueError("compiled KnowledgeRelease failed deterministic validation")
    return CompiledKnowledgePatch(
        candidate_id=candidate.candidate_id,
        release_path=str(release_path.relative_to(root)).replace("\\", "/"),
        release=release,
        validation=validation,
    )


def _changed_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix]
        paths = []
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            paths.extend(_changed_paths(l_item, r_item, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def run_knowledge_patch_impact_replay(
    compiled: CompiledKnowledgePatch,
    *,
    base_release: KnowledgeRelease,
    replay_runner: Callable[[KnowledgeRelease], dict[str, Any]],
    declared_expected_change_paths: list[str],
) -> KnowledgePatchImpactReport:
    """Execute the same frozen replay against base and candidate releases."""
    base_output = replay_runner(base_release)
    candidate_output = replay_runner(compiled.release)
    changed = sorted(set(_changed_paths(base_output, candidate_output)))
    declared = sorted(set(declared_expected_change_paths))
    unexpected = [
        path for path in changed
        if not any(path == allowed or path.startswith(allowed + ".") for allowed in declared)
    ]
    return KnowledgePatchImpactReport(
        candidate_id=compiled.candidate_id,
        status="BLOCKED" if unexpected else "PASS",
        base_release_id=base_release.release_id,
        candidate_release_id=compiled.release.release_id,
        base_output_sha256=_canonical_sha256(base_output),
        candidate_output_sha256=_canonical_sha256(candidate_output),
        changed_paths=changed,
        declared_expected_change_paths=declared,
        unexpected_change_paths=unexpected,
    )


def run_astrograph_knowledge_patch_impact_replay(
    compiled: CompiledKnowledgePatch,
    *,
    base_release: KnowledgeRelease,
    replay_runner: Callable[[KnowledgeRelease], AstroGraphKnowledgePatchReplayOutput | dict[str, Any]],
    contract: AstroGraphKnowledgePatchReplayContract,
) -> KnowledgePatchImpactReport:
    """Replay a patch against the fixed AstroGraph diagnostic impact dimensions."""
    if contract.contract_sha256 is None:
        raise ValueError("knowledge patch replay contract requires embedded contract_sha256")
    base_output = AstroGraphKnowledgePatchReplayOutput.model_validate(replay_runner(base_release))
    candidate_output = AstroGraphKnowledgePatchReplayOutput.model_validate(
        replay_runner(compiled.release)
    )
    for output in (base_output, candidate_output):
        if (
            output.scenario_set_id != contract.scenario_set_id
            or output.scenario_set_sha256 != contract.scenario_set_sha256
        ):
            raise ValueError("knowledge patch replay scenario identity differs from contract")
        if sorted(item.episode_id for item in output.episodes) != sorted(
            contract.required_episode_ids
        ):
            raise ValueError("knowledge patch replay episode coverage differs from contract")
    base_by_id = {item.episode_id: item for item in base_output.episodes}
    candidate_by_id = {item.episode_id: item for item in candidate_output.episodes}
    dimensions = [
        "kg_forward_fault_ids", "kg_backward_fault_ids", "birrf_top_fault_id",
        "governor_disposition", "policy_status", "structured_output_sha256",
    ]
    changed = sorted(
        f"episode:{episode_id}.{dimension}"
        for episode_id in contract.required_episode_ids
        for dimension in dimensions
        if getattr(base_by_id[episode_id], dimension) != getattr(
            candidate_by_id[episode_id], dimension
        )
    )
    allowed = set(contract.allowed_change_dimensions)
    unexpected = sorted(
        path for path in changed if path.rsplit(".", 1)[-1] not in allowed
    )
    return KnowledgePatchImpactReport(
        candidate_id=compiled.candidate_id,
        status="BLOCKED" if unexpected else "PASS",
        base_release_id=base_release.release_id,
        candidate_release_id=compiled.release.release_id,
        base_output_sha256=_canonical_sha256(base_output.model_dump(mode="json")),
        candidate_output_sha256=_canonical_sha256(candidate_output.model_dump(mode="json")),
        changed_paths=changed,
        declared_expected_change_paths=sorted(
            f"*.{dimension}" for dimension in contract.allowed_change_dimensions
        ),
        unexpected_change_paths=unexpected,
    )


def experience_candidate_is_compilable(payload: Any) -> tuple[bool, list[str]]:
    """Fail closed until an experience candidate has an explicit graph-edge mapping."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        return False, ["EXPERIENCE_CANDIDATE_PAYLOAD_INVALID"]
    reasons: list[str] = []
    if payload.get("review_status") != "HUMAN_REVIEW_REQUIRED":
        reasons.append("EXPERIENCE_CANDIDATE_REVIEW_STATUS_INVALID")
    if payload.get("active_graph_modified") is not False:
        reasons.append("EXPERIENCE_CANDIDATE_ALREADY_MUTATED_ACTIVE_GRAPH")
    if not payload.get("fault_ids"):
        reasons.append("EXPERIENCE_CANDIDATE_MISSING_FAULT_SCOPE")
    if not payload.get("proposed_edge_operations"):
        reasons.append("HUMAN_GRAPH_EDGE_MAPPING_REQUIRED")
    return (not reasons, reasons)


__all__ = [
    "CompiledKnowledgePatch",
    "AstroGraphKnowledgePatchReplayContract",
    "AstroGraphKnowledgePatchReplayEpisode",
    "AstroGraphKnowledgePatchReplayOutput",
    "KnowledgeEdgeValue",
    "KnowledgePatchCandidate",
    "KnowledgePatchImpactReport",
    "KnowledgePatchOperation",
    "KnowledgePatchOperationKind",
    "KnowledgePatchSourceSplit",
    "KnowledgePatchValidationReport",
    "build_astrograph_knowledge_patch_replay_output",
    "build_knowledge_patch_candidate",
    "compile_knowledge_patch_release",
    "experience_candidate_is_compilable",
    "knowledge_release_sha256",
    "run_knowledge_patch_impact_replay",
    "run_astrograph_knowledge_patch_impact_replay",
    "validate_knowledge_patch",
]
