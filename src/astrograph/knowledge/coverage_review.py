"""Human-review package for closing dataset-to-executable-KG coverage gaps."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class KGReviewSourceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    logical_uri: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["GENERATOR_SOURCE", "DATASET_INDEX", "ASTROGRAPH_ASSET"]
    independence_group: str = Field(min_length=1)


class GeneratorConditionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str
    effect_id: str
    component_source: str
    description: str
    parameter_names: list[str] = Field(default_factory=list)
    calibration_source_ids: list[str] = Field(default_factory=list)
    calibration_confidence: str | None = None
    severity_band: str | None = None


class FaultCoverageReviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    dataset_fault_class_id: str
    physical_condition_ids: list[str] = Field(min_length=1)
    generator_condition_summaries: list[GeneratorConditionSummary] = Field(min_length=1)
    runtime_fault_id: str | None = None
    proposed_runtime_fault_id: str
    component_id: str
    mechanism_id: str
    expected_effect_pattern_ids: list[str] = Field(min_length=1)
    evidence_definition_ids: list[str] = Field(default_factory=list)
    semantic_tbox_bound: bool
    formal_runtime_id_registered: bool
    executable_kg_evidence: bool
    coverage_status: Literal["EXECUTABLE", "FORMAL_ID_ONLY", "SEMANTIC_ONLY"]
    review_status: Literal["HUMAN_REVIEW_REQUIRED"] = "HUMAN_REVIEW_REQUIRED"


class KnowledgeMappingProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    dataset_id: str
    dataset_fault_class_id: str
    proposed_runtime_fault_id: str
    requires_new_runtime_fault_id: bool
    component_id: str
    mechanism_id: str
    expected_effect_pattern_ids: list[str] = Field(min_length=1)
    candidate_feature_family_ids: list[str] = Field(min_length=1)
    physical_condition_ids: list[str] = Field(min_length=1)
    generator_condition_summaries: list[GeneratorConditionSummary] = Field(min_length=1)
    proposed_operations: list[str] = Field(min_length=1)
    blocking_reasons: list[str] = Field(min_length=1)
    source_anchor_ids: list[str] = Field(min_length=1)
    review_status: Literal["HUMAN_REVIEW_REQUIRED"] = "HUMAN_REVIEW_REQUIRED"
    human_decision: None = None
    active_graph_modified: Literal[False] = False


class KnowledgeMappingCandidateSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dataset-kg-mapping-candidates.v1"] = (
        "astrograph-dataset-kg-mapping-candidates.v1"
    )
    candidate_set_id: str
    candidates: list[KnowledgeMappingProposal] = Field(min_length=1)
    candidate_count: int = Field(ge=1)
    candidate_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    claim_boundary: str = (
        "Machine-generated mapping and evidence-definition proposals only. Generator truth is audit-only; "
        "human semantic review, independent provenance, thresholds, deterministic validation and sandbox "
        "replay are required before a KnowledgePatch may be compiled or published."
    )

    @model_validator(mode="after")
    def validate_content(self) -> "KnowledgeMappingCandidateSet":
        if self.candidate_count != len(self.candidates):
            raise ValueError("knowledge mapping candidate_count mismatch")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"candidate_set_sha256"}))
        if self.candidate_set_sha256 != expected:
            raise ValueError("knowledge mapping candidate_set_sha256 mismatch")
        return self


class ReplayEpisodeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    dataset_id: str
    dataset_fault_class_id: str
    expected_runtime_fault_id: str
    development_group_id: Literal["3", "8"]
    operating_context_id: str
    physical_condition_id: str
    positive_sample_id: str
    positive_telemetry_path: str
    positive_telemetry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nominal_sample_id: str
    nominal_telemetry_path: str
    nominal_telemetry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    truth_usage: Literal["OFFLINE_REPLAY_ASSERTION_ONLY"] = "OFFLINE_REPLAY_ASSERTION_ONLY"


class ReplayScenarioSetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-kg-replay-scenario-set-candidate.v1"] = (
        "astrograph-kg-replay-scenario-set-candidate.v1"
    )
    scenario_set_id: str
    allowed_development_group_ids: list[Literal["3", "8"]] = ["3", "8"]
    permanently_post_hoc_group_ids: list[Literal["4", "9"]] = ["4", "9"]
    allowed_change_dimensions: list[str] = [
        "kg_forward_fault_ids", "kg_backward_fault_ids", "structured_output_sha256"
    ]
    episodes: list[ReplayEpisodeCandidate] = Field(min_length=1)
    required_episode_ids: list[str] = Field(min_length=1)
    scenario_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["CANDIDATE_NOT_FROZEN"] = "CANDIDATE_NOT_FROZEN"
    human_review_required: Literal[True] = True
    active_graph_modified: Literal[False] = False
    claim_boundary: str = (
        "Development replay candidate using groups 3/8 only. Expected labels are offline assertions and "
        "must never enter Runtime telemetry, retrieval context or evidence features. Human approval is "
        "required before this becomes an AstroGraphKnowledgePatchReplayContract."
    )

    @model_validator(mode="after")
    def validate_content(self) -> "ReplayScenarioSetCandidate":
        ids = [item.episode_id for item in self.episodes]
        if ids != self.required_episode_ids or len(ids) != len(set(ids)):
            raise ValueError("replay required_episode_ids must exactly match unique ordered episodes")
        if {item.development_group_id for item in self.episodes} - {"3", "8"}:
            raise ValueError("replay candidate may use development groups 3/8 only")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"scenario_set_sha256"}))
        if self.scenario_set_sha256 != expected:
            raise ValueError("replay scenario_set_sha256 mismatch")
        return self


class DatasetKGReviewPackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-dataset-kg-review-package.v1"] = (
        "astrograph-dataset-kg-review-package.v1"
    )
    package_id: str
    source_anchors: list[KGReviewSourceAnchor] = Field(min_length=1)
    dataset_fault_class_count: int
    semantic_tbox_coverage_count: int
    formal_runtime_id_count: int
    executable_kg_evidence_count: int
    uncovered_candidate_count: int
    replay_episode_count: int
    artifacts: dict[str, str]
    artifact_sha256: dict[str, str]
    status: Literal["READY_FOR_HUMAN_REVIEW"] = "READY_FOR_HUMAN_REVIEW"
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    claim_boundary: str = (
        "Review package only. Complete semantic binding is not executable KG coverage, and generator/data "
        "artifacts share one provenance family. No Active KG, Runtime vote or qualification state changes."
    )

    @model_validator(mode="after")
    def validate_counts_and_sources(self) -> "DatasetKGReviewPackageManifest":
        if not (
            0 <= self.executable_kg_evidence_count <= self.formal_runtime_id_count
            <= self.semantic_tbox_coverage_count <= self.dataset_fault_class_count
        ):
            raise ValueError("dataset KG review coverage counts are inconsistent")
        if self.uncovered_candidate_count != (
            self.dataset_fault_class_count - self.executable_kg_evidence_count
        ):
            raise ValueError("dataset KG review uncovered count mismatch")
        source_ids = [item.source_id for item in self.source_anchors]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("dataset KG review source IDs must be unique")
        if set(self.artifacts) != set(self.artifact_sha256):
            raise ValueError("dataset KG review artifact names and hashes differ")
        return self


class SupplementalReviewFreezeReceipt(BaseModel):
    """Content-addressed freeze of advisory review comments, not KG approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-supplemental-kg-review-freeze.v1"] = (
        "astrograph-supplemental-kg-review-freeze.v1"
    )
    freeze_id: str
    candidate_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blank_decision_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supplemental_review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_candidate_count: int = Field(ge=1)
    advisory_status_counts: dict[str, int]
    formal_decision_count: Literal[0] = 0
    reviewer_identity_present: Literal[False] = False
    temporary_frozen: Literal[True] = True
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    status: Literal["SUPPLEMENTAL_REVIEW_FROZEN_PENDING_FORMAL_DECISION"] = (
        "SUPPLEMENTAL_REVIEW_FROZEN_PENDING_FORMAL_DECISION"
    )
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "The returned Chinese comments are frozen as advisory review evidence only. Empty decision and "
        "reviewer fields cannot be interpreted as approval, and no mapping, Runtime ID, threshold, replay "
        "contract or Active KG state is promoted."
    )

    @model_validator(mode="after")
    def validate_content(self) -> "SupplementalReviewFreezeReceipt":
        if sum(self.advisory_status_counts.values()) != self.reviewed_candidate_count:
            raise ValueError("supplemental review advisory status count mismatch")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"freeze_sha256"}))
        if self.freeze_sha256 != expected:
            raise ValueError("supplemental review freeze_sha256 mismatch")
        return self


class SourceAdjudicationRecord(BaseModel):
    """Machine source finding that remains subordinate to formal human review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    source_status: Literal[
        "SOURCE_CONFIRMED",
        "SOURCE_CONFIRMED_WITH_REQUIRED_CANDIDATE_MODIFICATION",
        "SOURCE_SCOPE_LIMIT_REQUIRED",
    ]
    proposed_action: Literal[
        "RETAIN_CANDIDATE_FOR_HUMAN_APPROVAL",
        "REVISE_CANDIDATE_BEFORE_HUMAN_APPROVAL",
        "SPLIT_OR_SCOPE_BEFORE_HUMAN_APPROVAL",
    ]
    source_finding: str
    source_anchor_ids: list[str]
    proposed_candidate_patch: dict[str, Any] = Field(default_factory=dict)
    remaining_human_gate: str
    formal_decision: Literal[None] = None


class SourceAdjudicationPackage(BaseModel):
    """Content-addressed source resolution of advisory comments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-kg-source-adjudication.v1"] = (
        "astrograph-kg-source-adjudication.v1"
    )
    adjudication_id: str
    candidate_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supplemental_review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_source_sha256: dict[str, str]
    records: list[SourceAdjudicationRecord] = Field(min_length=1)
    status_counts: dict[str, int]
    proposed_action_counts: dict[str, int]
    formal_decision_count: Literal[0] = 0
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    status: Literal["SOURCE_ADJUDICATED_PENDING_FORMAL_HUMAN_DECISION"] = (
        "SOURCE_ADJUDICATED_PENDING_FORMAL_HUMAN_DECISION"
    )
    adjudication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Static generator-source findings resolve what the synthetic fault injection means and may "
        "propose candidate edits. They do not establish independent provenance, evidence thresholds, "
        "formal human approval, executable KG publication or Runtime voting eligibility."
    )

    @model_validator(mode="after")
    def validate_content(self) -> "SourceAdjudicationPackage":
        ids = [item.candidate_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("source adjudication candidate IDs must be unique")
        if sum(self.status_counts.values()) != len(self.records):
            raise ValueError("source adjudication status count mismatch")
        if sum(self.proposed_action_counts.values()) != len(self.records):
            raise ValueError("source adjudication action count mismatch")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"adjudication_sha256"}))
        if self.adjudication_sha256 != expected:
            raise ValueError("source adjudication_sha256 mismatch")
        return self


class SupplementalReviewUnfreezeReceipt(BaseModel):
    """Auditable transition from temporary freeze into source adjudication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-supplemental-kg-review-unfreeze.v1"] = (
        "astrograph-supplemental-kg-review-unfreeze.v1"
    )
    unfreeze_id: str
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sota_comparison_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_adjudication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    temporary_frozen: Literal[False] = False
    formal_decision_count: Literal[0] = 0
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    status: Literal["UNFROZEN_FOR_SOURCE_ADJUDICATION_PENDING_FORMAL_DECISION"] = (
        "UNFROZEN_FOR_SOURCE_ADJUDICATION_PENDING_FORMAL_DECISION"
    )
    unfreeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Temporary workflow freeze is released only for source adjudication. The review remains "
        "formally undecided and cannot publish or modify Active KG."
    )

    @model_validator(mode="after")
    def validate_content(self) -> "SupplementalReviewUnfreezeReceipt":
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"unfreeze_sha256"}))
        if self.unfreeze_sha256 != expected:
            raise ValueError("supplemental review unfreeze_sha256 mismatch")
        return self


class SupplementalReviewRefreezeReceipt(BaseModel):
    """Content-addressed pause after source adjudication and before formal decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-supplemental-kg-review-refreeze.v1"] = (
        "astrograph-supplemental-kg-review-refreeze.v1"
    )
    refreeze_id: str
    unfreeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_adjudication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_handoff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=1)
    freeze_reason: str = Field(min_length=1)
    temporary_frozen: Literal[True] = True
    formal_decision_count: Literal[0] = 0
    publication_allowed: Literal[False] = False
    active_graph_modified: Literal[False] = False
    status: Literal["REFROZEN_PENDING_FORMAL_DECISION_AFTER_SOURCE_ADJUDICATION"] = (
        "REFROZEN_PENDING_FORMAL_DECISION_AFTER_SOURCE_ADJUDICATION"
    )
    refreeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "The source-adjudicated handoff is paused before formal human decisions. Concurrent graph "
        "construction refactoring cannot promote these candidates, publish a KnowledgePatch or modify "
        "Active KG."
    )

    @model_validator(mode="after")
    def validate_content(self) -> "SupplementalReviewRefreezeReceipt":
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"refreeze_sha256"}))
        if self.refreeze_sha256 != expected:
            raise ValueError("supplemental review refreeze_sha256 mismatch")
        return self


def _assignment(tree: ast.AST, name: str) -> ast.AST:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node.value
    raise ValueError(f"generator source does not define {name}")


def _constant(node: ast.AST) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


def extract_adcs_generator_conditions(path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    conditions: dict[str, dict[str, Any]] = {}
    value = _assignment(tree, "CONDITIONS")
    for item in getattr(value, "elts", []):
        if not isinstance(item, ast.Call) or len(item.args) < 10:
            continue
        condition_id = _constant(item.args[0])
        if not isinstance(condition_id, str) or condition_id == "nominal":
            continue
        sources = [
            str(_constant(source))
            for source in getattr(item.args[7], "elts", [])
            if isinstance(_constant(source), str)
        ]
        conditions[condition_id] = {
            "category": _constant(item.args[1]),
            "effect": _constant(item.args[2]),
            "component_source": _constant(item.args[4]),
            "description": _constant(item.args[5]),
            "source_ids": sources,
            "calibration_confidence": _constant(item.args[8]),
            "severity_band": _constant(item.args[9]),
            "parameter_names": [
                str(_constant(key)) for key in getattr(item.args[3], "keys", [])
                if isinstance(_constant(key), str)
            ],
        }
    overrides = ast.literal_eval(_assignment(tree, "DIAGNOSTIC_CLASS_OVERRIDES"))
    return conditions, {str(key): str(value) for key, value in overrides.items()}


def extract_phm_generator_conditions(path: str | Path) -> dict[str, dict[str, Any]]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    conditions: dict[str, dict[str, Any]] = {}
    value = _assignment(tree, "CONDITIONS")
    for item in getattr(value, "elts", []):
        if not isinstance(item, ast.Call) or len(item.args) < 6:
            continue
        condition_id = _constant(item.args[0])
        if not isinstance(condition_id, str):
            continue
        parameters = [
            str(_constant(key)) for key in getattr(item.args[3], "keys", [])
            if isinstance(_constant(key), str)
        ]
        conditions[condition_id] = {
            "subsystem": _constant(item.args[1]),
            "effect": _constant(item.args[2]),
            "parameter_names": parameters,
            "component_source": _constant(item.args[4]),
            "description": _constant(item.args[5]),
            "source_ids": [],
            "calibration_confidence": None,
            "severity_band": None,
        }
    return conditions


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _content_hashed_model(model_type: type[BaseModel], body: dict[str, Any], hash_field: str) -> BaseModel:
    provisional = model_type.model_construct(**body, **{hash_field: "0" * 64})
    digest = canonical_sha256(provisional.model_dump(mode="json", exclude={hash_field}))
    return model_type(**body, **{hash_field: digest})


def _dataset_assets(project_root: Path, dataset_key: str) -> dict[str, Any]:
    root = project_root / "configs" / "datasets" / dataset_key
    candidate = _read_json(root / "candidate_universe.json")
    class_map = _read_json(root / "fault_class_map.json")["mappings"]
    hierarchy = _read_json(root / "diagnosis_hierarchy.json")["fault_to_component"]
    bindings = {
        item["fault_class_id"]: item
        for item in _read_json(root / "fault_ontology_bindings.json")["bindings"]
    }
    evidence = _read_json(root / "runtime_evidence.json")
    evidence_by_fault: dict[str, list[str]] = defaultdict(list)
    for definition in evidence["evidence_definitions"]:
        evidence_by_fault[definition["target_fault_id"]].append(
            definition["evidence_definition_id"]
        )
    return {
        "candidate": candidate, "class_map": class_map, "hierarchy": hierarchy,
        "bindings": bindings, "evidence_by_fault": evidence_by_fault,
        "asset_paths": [
            root / "candidate_universe.json", root / "fault_class_map.json",
            root / "diagnosis_hierarchy.json", root / "fault_ontology_bindings.json",
            root / "runtime_evidence.json",
        ],
    }


def _scenario_candidates(
    *, dataset_id: str, dataset_root: Path, rows: list[dict[str, str]],
    coverage_rows: list[FaultCoverageReviewRow],
) -> list[ReplayEpisodeCandidate]:
    nominal_by_pair = {
        (row["pair_id"], row["sample_index"]): row
        for row in rows if row["condition_id"] == "nominal"
    }
    episodes: list[ReplayEpisodeCandidate] = []
    for coverage in coverage_rows:
        candidates = [
            row for row in rows
            if row["diagnostic_class"] == coverage.dataset_fault_class_id
        ]
        for group in ("3", "8"):
            group_rows = [row for row in candidates if str(int(row["sample_index"])) == group]
            if not group_rows:
                raise ValueError(
                    f"missing group {group} replay sample for {dataset_id}:"
                    f"{coverage.dataset_fault_class_id}"
                )
            selected = sorted(group_rows, key=lambda row: (
                row.get("scenario_id") or row.get("operating_phase_id") or "",
                row["condition_id"], row["sample_id"],
            ))[0]
            nominal = nominal_by_pair.get((selected["pair_id"], selected["sample_index"]))
            if nominal is None:
                raise ValueError(f"missing paired nominal for replay sample {selected['sample_id']}")
            positive_path = dataset_root / selected["telemetry_file"]
            nominal_path = dataset_root / nominal["telemetry_file"]
            context = selected.get("scenario_id") or selected.get("operating_phase_id") or "unknown"
            episode_id = (
                "KGREPLAY-" + dataset_id.upper().replace(".", "-") + "-"
                + coverage.dataset_fault_class_id.upper().replace("_", "-") + f"-G{group}"
            )
            episodes.append(ReplayEpisodeCandidate(
                episode_id=episode_id,
                dataset_id=dataset_id,
                dataset_fault_class_id=coverage.dataset_fault_class_id,
                expected_runtime_fault_id=coverage.proposed_runtime_fault_id,
                development_group_id=group,
                operating_context_id=context,
                physical_condition_id=selected["condition_id"],
                positive_sample_id=selected["sample_id"],
                positive_telemetry_path=selected["telemetry_file"],
                positive_telemetry_sha256=sha256_file(positive_path),
                nominal_sample_id=nominal["sample_id"],
                nominal_telemetry_path=nominal["telemetry_file"],
                nominal_telemetry_sha256=sha256_file(nominal_path),
            ))
    return sorted(episodes, key=lambda item: item.episode_id)


def build_dataset_kg_review_package(
    *, project_root: str | Path, adcs_generator_path: str | Path,
    phm_generator_path: str | Path, adcs_dataset_root: str | Path,
    phm_dataset_root: str | Path, output_dir: str | Path,
) -> DatasetKGReviewPackageManifest:
    project_root = Path(project_root).resolve()
    adcs_generator_path = Path(adcs_generator_path).resolve()
    adcs_base_generator = adcs_generator_path.with_name(
        "generate_whole_spacecraft_adcs_eps_dataset.py"
    )
    phm_generator_path = Path(phm_generator_path).resolve()
    dataset_roots = {
        "adcs.v5_2_2": Path(adcs_dataset_root).resolve(),
        "phm.v6_0": Path(phm_dataset_root).resolve(),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adcs_conditions, overrides = extract_adcs_generator_conditions(adcs_base_generator)
    phm_conditions = extract_phm_generator_conditions(phm_generator_path)
    generator_conditions = {
        "adcs.v5_2_2": adcs_conditions,
        "phm.v6_0": phm_conditions,
    }
    source_anchors = [
        KGReviewSourceAnchor(
            source_id="SOURCE-ADCS-V522-GENERATOR", logical_uri="sat://scripts/"
            "generate_whole_spacecraft_adcs_eps_dataset_v522.py",
            source_sha256=sha256_file(adcs_generator_path), role="GENERATOR_SOURCE",
            independence_group="SAT-SIM-WHOLE-SPACECRAFT-GENERATOR",
        ),
        KGReviewSourceAnchor(
            source_id="SOURCE-ADCS-BASE-GENERATOR", logical_uri="sat://scripts/"
            "generate_whole_spacecraft_adcs_eps_dataset.py",
            source_sha256=sha256_file(adcs_base_generator), role="GENERATOR_SOURCE",
            independence_group="SAT-SIM-WHOLE-SPACECRAFT-GENERATOR",
        ),
        KGReviewSourceAnchor(
            source_id="SOURCE-PHM-V6-GENERATOR", logical_uri="sat://scripts/"
            "generate_whole_spacecraft_phm_dataset_v6.py",
            source_sha256=sha256_file(phm_generator_path), role="GENERATOR_SOURCE",
            independence_group="SAT-SIM-WHOLE-SPACECRAFT-GENERATOR",
        ),
    ]

    ontology_path = project_root / "configs/knowledge/generic_fault_ontology.json"
    ontology = _read_json(ontology_path)
    effect_features = {
        item["effect_pattern_id"]: item["feature_family_ids"]
        for item in ontology["effect_patterns"]
    }
    source_anchors.append(KGReviewSourceAnchor(
        source_id="SOURCE-ASTROGRAPH-GENERIC-TBOX",
        logical_uri="astrograph://configs/knowledge/generic_fault_ontology.json",
        source_sha256=sha256_file(ontology_path), role="ASTROGRAPH_ASSET",
        independence_group="ASTROGRAPH-CURATED-KNOWLEDGE",
    ))

    all_coverage: list[FaultCoverageReviewRow] = []
    all_episodes: list[ReplayEpisodeCandidate] = []
    proposals: list[KnowledgeMappingProposal] = []
    for dataset_id, dataset_root in dataset_roots.items():
        assets = _dataset_assets(project_root, dataset_id.replace(".", "_"))
        index_path = dataset_root / "sample_index.csv"
        source_id = "SOURCE-" + dataset_id.upper().replace(".", "-") + "-INDEX"
        source_anchors.append(KGReviewSourceAnchor(
            source_id=source_id,
            logical_uri=f"dataset://{dataset_id}/sample_index.csv",
            source_sha256=sha256_file(index_path), role="DATASET_INDEX",
            independence_group="SAT-SIM-WHOLE-SPACECRAFT-GENERATOR",
        ))
        for asset_path in assets["asset_paths"]:
            source_anchors.append(KGReviewSourceAnchor(
                source_id="SOURCE-ASSET-" + asset_path.stem.upper().replace("_", "-")
                + "-" + dataset_id.upper().replace(".", "-"),
                logical_uri="astrograph://" + asset_path.relative_to(project_root).as_posix(),
                source_sha256=sha256_file(asset_path), role="ASTROGRAPH_ASSET",
                independence_group="ASTROGRAPH-CURATED-KNOWLEDGE",
            ))
        index_rows = _read_csv(index_path)
        conditions_by_class: dict[str, set[str]] = defaultdict(set)
        for row in index_rows:
            if row["condition_id"] != "nominal":
                conditions_by_class[row["diagnostic_class"]].add(row["condition_id"])
        dataset_coverage: list[FaultCoverageReviewRow] = []
        for class_id in sorted(conditions_by_class):
            binding = assets["bindings"].get(class_id)
            if binding is None:
                raise ValueError(f"missing semantic TBox binding for {dataset_id}:{class_id}")
            runtime_fault_id = assets["class_map"].get(class_id)
            proposed_fault_id = runtime_fault_id or class_id.upper()
            evidence_ids = sorted(assets["evidence_by_fault"].get(proposed_fault_id, []))
            formal = runtime_fault_id is not None
            executable = bool(evidence_ids)
            status = "EXECUTABLE" if executable else ("FORMAL_ID_ONLY" if formal else "SEMANTIC_ONLY")
            condition_summaries = [
                GeneratorConditionSummary(
                    condition_id=condition_id,
                    effect_id=str(generator_conditions[dataset_id][condition_id]["effect"]),
                    component_source=str(
                        generator_conditions[dataset_id][condition_id]["component_source"]
                    ),
                    description=str(
                        generator_conditions[dataset_id][condition_id]["description"]
                    ),
                    parameter_names=generator_conditions[dataset_id][condition_id].get(
                        "parameter_names", []
                    ),
                    calibration_source_ids=generator_conditions[dataset_id][condition_id].get(
                        "source_ids", []
                    ),
                    calibration_confidence=generator_conditions[dataset_id][condition_id].get(
                        "calibration_confidence"
                    ),
                    severity_band=generator_conditions[dataset_id][condition_id].get(
                        "severity_band"
                    ),
                )
                for condition_id in sorted(conditions_by_class[class_id])
            ]
            coverage = FaultCoverageReviewRow(
                dataset_id=dataset_id,
                dataset_fault_class_id=class_id,
                physical_condition_ids=sorted(conditions_by_class[class_id]),
                generator_condition_summaries=condition_summaries,
                runtime_fault_id=runtime_fault_id,
                proposed_runtime_fault_id=proposed_fault_id,
                component_id=assets["hierarchy"][class_id],
                mechanism_id=binding["mechanism_id"],
                expected_effect_pattern_ids=sorted(binding["expected_effect_pattern_ids"]),
                evidence_definition_ids=evidence_ids,
                semantic_tbox_bound=True,
                formal_runtime_id_registered=formal,
                executable_kg_evidence=executable,
                coverage_status=status,
            )
            dataset_coverage.append(coverage)
            if not executable:
                feature_families = sorted({
                    feature
                    for effect in coverage.expected_effect_pattern_ids
                    for feature in effect_features[effect]
                })
                blockers = ["HUMAN_EVIDENCE_DEFINITION_AND_THRESHOLD_REVIEW_REQUIRED",
                            "INDEPENDENT_PROVENANCE_REQUIRED"]
                operations = [
                    "BIND_RUNTIME_FAULT_TO_COMPONENT",
                    "BIND_RUNTIME_FAULT_TO_MECHANISM",
                    "DECLARE_EXPECTED_EFFECT_PATTERN",
                    "DRAFT_EXECUTABLE_EVIDENCE_DEFINITION",
                ]
                if not formal:
                    blockers.insert(0, "HUMAN_RUNTIME_FAULT_ID_REVIEW_REQUIRED")
                    operations.insert(0, "REGISTER_RUNTIME_FAULT_ID")
                dataset_suffix = dataset_id.upper().replace(".", "-")
                generator_anchor_ids = (
                    ["SOURCE-ADCS-V522-GENERATOR", "SOURCE-ADCS-BASE-GENERATOR"]
                    if dataset_id == "adcs.v5_2_2" else ["SOURCE-PHM-V6-GENERATOR"]
                )
                proposals.append(KnowledgeMappingProposal(
                    candidate_id="KGMAP-" + canonical_sha256({
                        "dataset": dataset_id, "class": class_id, "fault": proposed_fault_id,
                    })[:20].upper(),
                    dataset_id=dataset_id,
                    dataset_fault_class_id=class_id,
                    proposed_runtime_fault_id=proposed_fault_id,
                    requires_new_runtime_fault_id=not formal,
                    component_id=coverage.component_id,
                    mechanism_id=coverage.mechanism_id,
                    expected_effect_pattern_ids=coverage.expected_effect_pattern_ids,
                    candidate_feature_family_ids=feature_families,
                    physical_condition_ids=coverage.physical_condition_ids,
                    generator_condition_summaries=condition_summaries,
                    proposed_operations=operations,
                    blocking_reasons=blockers,
                    source_anchor_ids=[
                        *generator_anchor_ids,
                        source_id,
                        "SOURCE-ASTROGRAPH-GENERIC-TBOX",
                        "SOURCE-ASSET-FAULT-ONTOLOGY-BINDINGS-" + dataset_suffix,
                        "SOURCE-ASSET-DIAGNOSIS-HIERARCHY-" + dataset_suffix,
                    ],
                ))
        # Ensure the static generator extractor observed every physical condition used by the data.
        extracted = generator_conditions[dataset_id]
        missing_conditions = sorted({
            condition for row in dataset_coverage for condition in row.physical_condition_ids
            if condition not in extracted
        })
        if missing_conditions:
            raise ValueError(f"dataset conditions missing from generator AST: {missing_conditions}")
        if dataset_id == "adcs.v5_2_2":
            inconsistent = sorted(
                condition for row in dataset_coverage for condition in row.physical_condition_ids
                if overrides.get(condition, condition) != row.dataset_fault_class_id
            )
            if inconsistent:
                raise ValueError(f"ADCS diagnostic class merge drift: {inconsistent}")
        all_coverage.extend(dataset_coverage)
        all_episodes.extend(_scenario_candidates(
            dataset_id=dataset_id, dataset_root=dataset_root, rows=index_rows,
            coverage_rows=dataset_coverage,
        ))

    all_coverage.sort(key=lambda item: (item.dataset_id, item.dataset_fault_class_id))
    proposals.sort(key=lambda item: (item.dataset_id, item.dataset_fault_class_id))
    candidate_body = {
        "candidate_set_id": "KGMAPSET-" + canonical_sha256([
            item.candidate_id for item in proposals
        ])[:20].upper(),
        "candidates": proposals,
        "candidate_count": len(proposals),
    }
    candidate_set = _content_hashed_model(
        KnowledgeMappingCandidateSet, candidate_body, "candidate_set_sha256"
    )
    required_ids = [item.episode_id for item in all_episodes]
    scenario_body = {
        "scenario_set_id": "KGREPLAYSET-" + canonical_sha256(required_ids)[:20].upper(),
        "episodes": all_episodes,
        "required_episode_ids": required_ids,
    }
    scenario_set = _content_hashed_model(
        ReplayScenarioSetCandidate, scenario_body, "scenario_set_sha256"
    )

    matrix_path = output_dir / "fault_coverage_matrix.csv"
    with matrix_path.open("w", encoding="utf-8-sig", newline="") as stream:
        rows = [item.model_dump(mode="json") for item in all_coverage]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
    candidates_path = output_dir / "knowledge_mapping_candidates.json"
    candidates_path.write_text(candidate_set.model_dump_json(indent=2) + "\n", encoding="utf-8")
    replay_path = output_dir / "replay_scenario_set_candidate.json"
    replay_path.write_text(scenario_set.model_dump_json(indent=2) + "\n", encoding="utf-8")
    review_path = output_dir / "human_review_decisions.csv"
    review_fields = [
        "candidate_id", "dataset_id", "dataset_fault_class_id",
        "proposed_runtime_fault_id", "requires_new_runtime_fault_id", "component_id",
        "mechanism_id", "expected_effect_pattern_ids", "candidate_feature_family_ids",
        "physical_condition_ids", "blocking_reasons", "decision",
        "approved_runtime_fault_id", "edge_role_decision", "reviewer", "review_comment",
    ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields, lineterminator="\n")
        writer.writeheader()
        for item in proposals:
            writer.writerow({
                "candidate_id": item.candidate_id,
                "dataset_id": item.dataset_id,
                "dataset_fault_class_id": item.dataset_fault_class_id,
                "proposed_runtime_fault_id": item.proposed_runtime_fault_id,
                "requires_new_runtime_fault_id": str(item.requires_new_runtime_fault_id).lower(),
                "component_id": item.component_id,
                "mechanism_id": item.mechanism_id,
                "expected_effect_pattern_ids": ";".join(item.expected_effect_pattern_ids),
                "candidate_feature_family_ids": ";".join(item.candidate_feature_family_ids),
                "physical_condition_ids": ";".join(item.physical_condition_ids),
                "blocking_reasons": ";".join(item.blocking_reasons),
                "decision": "",
                "approved_runtime_fault_id": "",
                "edge_role_decision": "",
                "reviewer": "",
                "review_comment": "",
            })

    artifacts = {
        "fault_coverage_matrix": matrix_path.name,
        "knowledge_mapping_candidates": candidates_path.name,
        "replay_scenario_set_candidate": replay_path.name,
        "human_review_decisions": review_path.name,
    }
    artifact_hashes = {
        key: sha256_file(output_dir / filename) for key, filename in artifacts.items()
    }
    manifest = DatasetKGReviewPackageManifest(
        package_id="KGREVIEW-" + canonical_sha256(artifact_hashes)[:20].upper(),
        source_anchors=source_anchors,
        dataset_fault_class_count=len(all_coverage),
        semantic_tbox_coverage_count=sum(item.semantic_tbox_bound for item in all_coverage),
        formal_runtime_id_count=sum(item.formal_runtime_id_registered for item in all_coverage),
        executable_kg_evidence_count=sum(item.executable_kg_evidence for item in all_coverage),
        uncovered_candidate_count=len(proposals),
        replay_episode_count=len(all_episodes),
        artifacts=artifacts,
        artifact_sha256=artifact_hashes,
    )
    (output_dir / "review_package_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def validate_dataset_kg_review_package(
    output_dir: str | Path,
) -> DatasetKGReviewPackageManifest:
    """Validate the generated review package without resolving external logical URIs."""
    output_dir = Path(output_dir).resolve()
    manifest = DatasetKGReviewPackageManifest.model_validate_json(
        (output_dir / "review_package_manifest.json").read_text(encoding="utf-8")
    )
    resolved_artifacts: dict[str, Path] = {}
    for role, relative in manifest.artifacts.items():
        path = (output_dir / relative).resolve()
        if not path.is_relative_to(output_dir) or not path.is_file():
            raise ValueError(f"dataset KG review artifact path invalid: {role}")
        if sha256_file(path) != manifest.artifact_sha256[role]:
            raise ValueError(f"dataset KG review artifact hash mismatch: {role}")
        resolved_artifacts[role] = path

    candidates = KnowledgeMappingCandidateSet.model_validate_json(
        resolved_artifacts["knowledge_mapping_candidates"].read_text(encoding="utf-8")
    )
    scenarios = ReplayScenarioSetCandidate.model_validate_json(
        resolved_artifacts["replay_scenario_set_candidate"].read_text(encoding="utf-8")
    )
    matrix_rows = _read_csv(resolved_artifacts["fault_coverage_matrix"])
    if len(matrix_rows) != manifest.dataset_fault_class_count:
        raise ValueError("dataset KG review matrix count mismatch")
    matrix_keys = {
        (row["dataset_id"], row["dataset_fault_class_id"]) for row in matrix_rows
    }
    uncovered_keys = {
        (row["dataset_id"], row["dataset_fault_class_id"])
        for row in matrix_rows if row["executable_kg_evidence"].lower() != "true"
    }
    candidate_keys = {
        (item.dataset_id, item.dataset_fault_class_id) for item in candidates.candidates
    }
    if uncovered_keys != candidate_keys or len(candidate_keys) != manifest.uncovered_candidate_count:
        raise ValueError("dataset KG review candidates do not exactly cover non-executable rows")
    review_rows = _read_csv(resolved_artifacts["human_review_decisions"])
    if {row["candidate_id"] for row in review_rows} != {
        item.candidate_id for item in candidates.candidates
    }:
        raise ValueError("dataset KG human review rows do not match candidates")
    if any(
        row.get(field) for row in review_rows
        for field in ("decision", "approved_runtime_fault_id", "edge_role_decision", "reviewer")
    ):
        raise ValueError("machine-generated human review decisions must remain blank")
    scenario_keys = {
        (item.dataset_id, item.dataset_fault_class_id) for item in scenarios.episodes
    }
    if scenario_keys != matrix_keys or len(scenarios.episodes) != manifest.replay_episode_count:
        raise ValueError("dataset KG review scenarios do not cover the matrix")
    group_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in scenarios.episodes:
        group_counts[(item.dataset_id, item.dataset_fault_class_id)].add(
            item.development_group_id
        )
    if any(groups != {"3", "8"} for groups in group_counts.values()):
        raise ValueError("every dataset fault requires replay groups 3 and 8")
    return manifest


def freeze_supplemental_human_review(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    output_path: str | Path,
) -> SupplementalReviewFreezeReceipt:
    """Validate and freeze a returned advisory review without manufacturing approval."""
    package_dir = Path(package_dir).resolve()
    supplemental_review_path = Path(supplemental_review_path).resolve()
    output_path = Path(output_path).resolve()
    validate_dataset_kg_review_package(package_dir)
    candidates = KnowledgeMappingCandidateSet.model_validate_json(
        (package_dir / "knowledge_mapping_candidates.json").read_text(encoding="utf-8")
    )
    template_path = package_dir / "human_review_decisions.csv"
    template_rows = _read_csv(template_path)
    supplemental_rows = _read_csv(supplemental_review_path)
    template_by_id = {row["candidate_id"]: row for row in template_rows}
    supplemental_by_id = {row["candidate_id"]: row for row in supplemental_rows}
    expected_ids = {item.candidate_id for item in candidates.candidates}
    if set(template_by_id) != expected_ids or set(supplemental_by_id) != expected_ids:
        raise ValueError("supplemental review rows must exactly match the candidate set")
    if len(supplemental_rows) != len(supplemental_by_id):
        raise ValueError("supplemental review candidate IDs must be unique")
    protected_fields = list(template_rows[0])
    decision_fields = (
        "decision", "approved_runtime_fault_id", "edge_role_decision", "reviewer",
    )
    advisory_fields = (
        "review_status_zh", "supplemental_review_opinion_zh", "recommended_next_check_zh",
    )
    status_counts: dict[str, int] = defaultdict(int)
    for candidate_id in sorted(expected_ids):
        original = template_by_id[candidate_id]
        returned = supplemental_by_id[candidate_id]
        if any(returned.get(field, "") != original.get(field, "") for field in protected_fields):
            raise ValueError(f"supplemental review changed protected candidate fields: {candidate_id}")
        if any(returned.get(field, "").strip() for field in decision_fields):
            raise ValueError("supplemental review has formal decisions but no named reviewer")
        if any(not returned.get(field, "").strip() for field in advisory_fields):
            raise ValueError(f"supplemental review advisory fields incomplete: {candidate_id}")
        status_counts[returned["review_status_zh"].strip()] += 1
    body = {
        "freeze_id": "KGREVIEW-FREEZE-" + sha256_file(supplemental_review_path)[:20].upper(),
        "candidate_set_sha256": candidates.candidate_set_sha256,
        "blank_decision_template_sha256": sha256_file(template_path),
        "supplemental_review_sha256": sha256_file(supplemental_review_path),
        "reviewed_candidate_count": len(supplemental_rows),
        "advisory_status_counts": dict(sorted(status_counts.items())),
    }
    receipt = _content_hashed_model(
        SupplementalReviewFreezeReceipt, body, "freeze_sha256"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt


def validate_supplemental_review_freeze(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    receipt_path: str | Path,
) -> SupplementalReviewFreezeReceipt:
    receipt = SupplementalReviewFreezeReceipt.model_validate_json(
        Path(receipt_path).read_text(encoding="utf-8")
    )
    candidates = KnowledgeMappingCandidateSet.model_validate_json(
        (Path(package_dir) / "knowledge_mapping_candidates.json").read_text(encoding="utf-8")
    )
    if receipt.candidate_set_sha256 != candidates.candidate_set_sha256:
        raise ValueError("supplemental review freeze candidate set mismatch")
    if receipt.blank_decision_template_sha256 != sha256_file(
        Path(package_dir) / "human_review_decisions.csv"
    ):
        raise ValueError("supplemental review freeze template hash mismatch")
    if receipt.supplemental_review_sha256 != sha256_file(supplemental_review_path):
        raise ValueError("supplemental review freeze input hash mismatch")
    return receipt


def build_source_adjudication_package(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    adjudication_draft_path: str | Path, generator_source_paths: dict[str, str | Path],
    output_path: str | Path,
) -> SourceAdjudicationPackage:
    """Bind advisory comments to static generator-source findings without approval."""
    package_dir = Path(package_dir).resolve()
    supplemental_review_path = Path(supplemental_review_path).resolve()
    candidates = KnowledgeMappingCandidateSet.model_validate_json(
        (package_dir / "knowledge_mapping_candidates.json").read_text(encoding="utf-8")
    )
    draft = json.loads(Path(adjudication_draft_path).read_text(encoding="utf-8"))
    records = [SourceAdjudicationRecord.model_validate(item) for item in draft["records"]]
    expected_ids = {item.candidate_id for item in candidates.candidates}
    actual_ids = {item.candidate_id for item in records}
    if actual_ids != expected_ids or len(records) != len(expected_ids):
        raise ValueError("source adjudication must exactly cover the candidate set")
    candidate_by_id = {item.candidate_id: item for item in candidates.candidates}
    for item in records:
        allowed_anchors = set(candidate_by_id[item.candidate_id].source_anchor_ids)
        if not item.source_anchor_ids or not set(item.source_anchor_ids).issubset(allowed_anchors):
            raise ValueError(f"source adjudication contains an unbound source anchor: {item.candidate_id}")
        if item.proposed_action == "RETAIN_CANDIDATE_FOR_HUMAN_APPROVAL" and item.proposed_candidate_patch:
            raise ValueError("retained candidate cannot carry a patch")
        if item.proposed_action != "RETAIN_CANDIDATE_FOR_HUMAN_APPROVAL" and not item.proposed_candidate_patch:
            raise ValueError("candidate revision or scope action requires an explicit patch")
    source_hashes = {
        name: sha256_file(Path(path).resolve())
        for name, path in sorted(generator_source_paths.items())
    }
    status_counts: dict[str, int] = defaultdict(int)
    action_counts: dict[str, int] = defaultdict(int)
    for item in records:
        status_counts[item.source_status] += 1
        action_counts[item.proposed_action] += 1
    body = {
        "adjudication_id": "KGSOURCE-" + canonical_sha256({
            "candidate_set_sha256": candidates.candidate_set_sha256,
            "supplemental_review_sha256": sha256_file(supplemental_review_path),
            "generator_source_sha256": source_hashes,
            "records": [item.model_dump(mode="json") for item in records],
        })[:20].upper(),
        "candidate_set_sha256": candidates.candidate_set_sha256,
        "supplemental_review_sha256": sha256_file(supplemental_review_path),
        "generator_source_sha256": source_hashes,
        "records": records,
        "status_counts": dict(sorted(status_counts.items())),
        "proposed_action_counts": dict(sorted(action_counts.items())),
    }
    package = _content_hashed_model(
        SourceAdjudicationPackage, body, "adjudication_sha256"
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(package.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return package


def validate_source_adjudication_package(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    generator_source_paths: dict[str, str | Path], adjudication_path: str | Path,
) -> SourceAdjudicationPackage:
    """Revalidate source hashes and the complete frozen candidate inventory."""
    package = SourceAdjudicationPackage.model_validate_json(
        Path(adjudication_path).read_text(encoding="utf-8")
    )
    candidates = KnowledgeMappingCandidateSet.model_validate_json(
        (Path(package_dir) / "knowledge_mapping_candidates.json").read_text(encoding="utf-8")
    )
    if package.candidate_set_sha256 != candidates.candidate_set_sha256:
        raise ValueError("source adjudication candidate set mismatch")
    if {item.candidate_id for item in package.records} != {
        item.candidate_id for item in candidates.candidates
    }:
        raise ValueError("source adjudication does not cover the current candidate set")
    if package.supplemental_review_sha256 != sha256_file(supplemental_review_path):
        raise ValueError("source adjudication supplemental review hash mismatch")
    expected_source_hashes = {
        name: sha256_file(Path(path).resolve())
        for name, path in sorted(generator_source_paths.items())
    }
    if package.generator_source_sha256 != expected_source_hashes:
        raise ValueError("source adjudication generator source hash mismatch")
    return package


def write_source_adjudication_review_sheet(
    *, supplemental_review_path: str | Path, source_adjudication_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write a human handoff sheet without filling any formal decision field."""
    rows = _read_csv(Path(supplemental_review_path))
    package = SourceAdjudicationPackage.model_validate_json(
        Path(source_adjudication_path).read_text(encoding="utf-8")
    )
    by_id = {item.candidate_id: item for item in package.records}
    if {row["candidate_id"] for row in rows} != set(by_id):
        raise ValueError("source-adjudication review sheet candidate set mismatch")
    for row in rows:
        if any(row.get(field, "").strip() for field in (
            "decision", "approved_runtime_fault_id", "edge_role_decision", "reviewer",
        )):
            raise ValueError("source-adjudication handoff cannot overwrite formal decisions")
        finding = by_id[row["candidate_id"]]
        row.update({
            "source_adjudication_id": package.adjudication_id,
            "source_status": finding.source_status,
            "source_proposed_action": finding.proposed_action,
            "source_finding": finding.source_finding,
            "source_proposed_candidate_patch_json": json.dumps(
                finding.proposed_candidate_patch, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ),
            "source_remaining_human_gate": finding.remaining_human_gate,
        })
    source_fields = [
        "source_adjudication_id", "source_status", "source_proposed_action",
        "source_finding", "source_proposed_candidate_patch_json",
        "source_remaining_human_gate",
    ]
    original_fields = list(rows[0])
    for field in source_fields:
        original_fields.remove(field)
    decision_index = original_fields.index("decision")
    fields = original_fields[:decision_index] + source_fields + original_fields[decision_index:]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _validate_domain_sota_gate(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied_hash = payload.pop("audit_sha256", "")
    if not (
        supplied_hash == canonical_sha256(payload)
        and payload.get("status") == "PASS_DOMAIN_SOTA_FAIR_COMPARISON"
        and all(payload.get("anti_suppression_checks", {}).values())
    ):
        raise ValueError("domain SOTA comparison must pass before review unfreeze")
    payload["audit_sha256"] = supplied_hash
    return payload


def unfreeze_supplemental_human_review(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    freeze_receipt_path: str | Path, sota_comparison_audit_path: str | Path,
    source_adjudication_path: str | Path, output_path: str | Path,
) -> SupplementalReviewUnfreezeReceipt:
    """Release the temporary workflow freeze after comparison and source audit."""
    freeze = validate_supplemental_review_freeze(
        package_dir=package_dir,
        supplemental_review_path=supplemental_review_path,
        receipt_path=freeze_receipt_path,
    )
    sota = _validate_domain_sota_gate(sota_comparison_audit_path)
    source = SourceAdjudicationPackage.model_validate_json(
        Path(source_adjudication_path).read_text(encoding="utf-8")
    )
    if source.candidate_set_sha256 != freeze.candidate_set_sha256:
        raise ValueError("source adjudication candidate set does not match the freeze")
    body = {
        "unfreeze_id": "KGREVIEW-UNFREEZE-" + canonical_sha256({
            "freeze_sha256": freeze.freeze_sha256,
            "sota_comparison_audit_sha256": sota.get("audit_sha256"),
            "source_adjudication_sha256": source.adjudication_sha256,
        })[:20].upper(),
        "freeze_sha256": freeze.freeze_sha256,
        "sota_comparison_audit_sha256": str(sota.get("audit_sha256", "")),
        "source_adjudication_sha256": source.adjudication_sha256,
    }
    receipt = _content_hashed_model(
        SupplementalReviewUnfreezeReceipt, body, "unfreeze_sha256"
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt


def validate_supplemental_review_unfreeze(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    freeze_receipt_path: str | Path, sota_comparison_audit_path: str | Path,
    source_adjudication_path: str | Path, unfreeze_receipt_path: str | Path,
) -> SupplementalReviewUnfreezeReceipt:
    """Validate every content link in a completed source-adjudication transition."""
    freeze = validate_supplemental_review_freeze(
        package_dir=package_dir,
        supplemental_review_path=supplemental_review_path,
        receipt_path=freeze_receipt_path,
    )
    sota = _validate_domain_sota_gate(sota_comparison_audit_path)
    source = SourceAdjudicationPackage.model_validate_json(
        Path(source_adjudication_path).read_text(encoding="utf-8")
    )
    receipt = SupplementalReviewUnfreezeReceipt.model_validate_json(
        Path(unfreeze_receipt_path).read_text(encoding="utf-8")
    )
    expected = (
        freeze.freeze_sha256,
        sota["audit_sha256"],
        source.adjudication_sha256,
    )
    actual = (
        receipt.freeze_sha256,
        receipt.sota_comparison_audit_sha256,
        receipt.source_adjudication_sha256,
    )
    if actual != expected:
        raise ValueError("supplemental review unfreeze content links mismatch")
    if source.candidate_set_sha256 != freeze.candidate_set_sha256:
        raise ValueError("supplemental review unfreeze candidate set mismatch")
    return receipt


def refreeze_supplemental_human_review(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    freeze_receipt_path: str | Path, sota_comparison_audit_path: str | Path,
    source_adjudication_path: str | Path, unfreeze_receipt_path: str | Path,
    human_handoff_path: str | Path, freeze_reason: str, output_path: str | Path,
) -> SupplementalReviewRefreezeReceipt:
    """Pause the source-adjudicated branch without manufacturing formal approval."""
    unfreeze = validate_supplemental_review_unfreeze(
        package_dir=package_dir,
        supplemental_review_path=supplemental_review_path,
        freeze_receipt_path=freeze_receipt_path,
        sota_comparison_audit_path=sota_comparison_audit_path,
        source_adjudication_path=source_adjudication_path,
        unfreeze_receipt_path=unfreeze_receipt_path,
    )
    source = SourceAdjudicationPackage.model_validate_json(
        Path(source_adjudication_path).read_text(encoding="utf-8")
    )
    handoff_rows = _read_csv(Path(human_handoff_path))
    expected_ids = {item.candidate_id for item in source.records}
    if {row.get("candidate_id", "") for row in handoff_rows} != expected_ids:
        raise ValueError("supplemental review refreeze handoff candidate set mismatch")
    formal_fields = (
        "decision", "approved_runtime_fault_id", "edge_role_decision", "reviewer",
    )
    if any(row.get(field, "").strip() for row in handoff_rows for field in formal_fields):
        raise ValueError("cannot refreeze a handoff containing ungoverned formal decisions")
    body = {
        "refreeze_id": "KGREVIEW-REFREEZE-" + canonical_sha256({
            "unfreeze_sha256": unfreeze.unfreeze_sha256,
            "source_adjudication_sha256": source.adjudication_sha256,
            "human_handoff_sha256": sha256_file(human_handoff_path),
            "freeze_reason": freeze_reason,
        })[:20].upper(),
        "unfreeze_sha256": unfreeze.unfreeze_sha256,
        "source_adjudication_sha256": source.adjudication_sha256,
        "human_handoff_sha256": sha256_file(human_handoff_path),
        "candidate_count": len(handoff_rows),
        "freeze_reason": freeze_reason,
    }
    receipt = _content_hashed_model(
        SupplementalReviewRefreezeReceipt, body, "refreeze_sha256"
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt


def validate_supplemental_review_refreeze(
    *, package_dir: str | Path, supplemental_review_path: str | Path,
    freeze_receipt_path: str | Path, sota_comparison_audit_path: str | Path,
    source_adjudication_path: str | Path, unfreeze_receipt_path: str | Path,
    human_handoff_path: str | Path, refreeze_receipt_path: str | Path,
) -> SupplementalReviewRefreezeReceipt:
    """Validate every content link and blank formal field in a refrozen review branch."""
    unfreeze = validate_supplemental_review_unfreeze(
        package_dir=package_dir,
        supplemental_review_path=supplemental_review_path,
        freeze_receipt_path=freeze_receipt_path,
        sota_comparison_audit_path=sota_comparison_audit_path,
        source_adjudication_path=source_adjudication_path,
        unfreeze_receipt_path=unfreeze_receipt_path,
    )
    source = SourceAdjudicationPackage.model_validate_json(
        Path(source_adjudication_path).read_text(encoding="utf-8")
    )
    receipt = SupplementalReviewRefreezeReceipt.model_validate_json(
        Path(refreeze_receipt_path).read_text(encoding="utf-8")
    )
    handoff_rows = _read_csv(Path(human_handoff_path))
    formal_fields = (
        "decision", "approved_runtime_fault_id", "edge_role_decision", "reviewer",
    )
    if any(row.get(field, "").strip() for row in handoff_rows for field in formal_fields):
        raise ValueError("refrozen supplemental review handoff contains formal decisions")
    expected = (
        unfreeze.unfreeze_sha256,
        source.adjudication_sha256,
        sha256_file(human_handoff_path),
        len(handoff_rows),
    )
    actual = (
        receipt.unfreeze_sha256,
        receipt.source_adjudication_sha256,
        receipt.human_handoff_sha256,
        receipt.candidate_count,
    )
    if actual != expected:
        raise ValueError("supplemental review refreeze content links mismatch")
    return receipt


__all__ = [
    "DatasetKGReviewPackageManifest", "FaultCoverageReviewRow", "KGReviewSourceAnchor",
    "KnowledgeMappingCandidateSet", "KnowledgeMappingProposal", "ReplayEpisodeCandidate",
    "ReplayScenarioSetCandidate", "SourceAdjudicationPackage", "SourceAdjudicationRecord",
    "SupplementalReviewFreezeReceipt", "SupplementalReviewRefreezeReceipt",
    "SupplementalReviewUnfreezeReceipt",
    "build_dataset_kg_review_package", "canonical_sha256",
    "extract_adcs_generator_conditions", "extract_phm_generator_conditions", "sha256_file",
    "build_source_adjudication_package", "freeze_supplemental_human_review",
    "refreeze_supplemental_human_review",
    "unfreeze_supplemental_human_review", "validate_dataset_kg_review_package",
    "validate_source_adjudication_package", "validate_supplemental_review_freeze",
    "validate_supplemental_review_refreeze",
    "validate_supplemental_review_unfreeze", "write_source_adjudication_review_sheet",
]
