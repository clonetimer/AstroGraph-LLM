"""Contracts for leakage-resistant numeric--document paired corpora.

The corpus models intentionally keep runtime and evaluator indexes separate.
They store opaque episode references rather than dataset paths or condition
names so that an index cannot become an accidental label side channel.
"""
from __future__ import annotations

from collections import Counter
from enum import Enum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .document_governance import (
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
)


class CorpusSplitRole(str, Enum):
    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"
    POSTHOC = "POSTHOC"
    CONFIRMATION = "CONFIRMATION"


class CorpusArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)
    descriptor: GovernedDocumentDescriptor

    @model_validator(mode="after")
    def path_is_relative_and_opaque(self) -> "CorpusArtifact":
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("corpus artifact path must be a safe relative path")
        return self


class PairedDocumentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-paired-document-bundle.v1"
    bundle_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    opaque_episode_id: str = Field(min_length=1)
    source_episode_ref_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    telemetry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_group_id: str = Field(min_length=1)
    split_role: CorpusSplitRole
    artifacts: list[CorpusArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_episode_and_group_binding(self) -> "PairedDocumentBundle":
        ids: set[str] = set()
        paths: set[str] = set()
        for artifact in self.artifacts:
            descriptor = artifact.descriptor
            if descriptor.document_id in ids:
                raise ValueError("document_id must be unique inside a bundle")
            if artifact.relative_path in paths:
                raise ValueError("artifact relative_path must be unique inside a bundle")
            ids.add(descriptor.document_id)
            paths.add(artifact.relative_path)
            if (
                descriptor.visibility_class == DocumentVisibilityClass.OBSERVATION_CAUSAL
                and descriptor.episode_id != self.opaque_episode_id
            ):
                raise ValueError("causal document must bind to the opaque bundle episode")
            if (
                descriptor.visibility_class
                == DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY
                and descriptor.provenance_group_id != self.provenance_group_id
            ):
                raise ValueError("historical document must bind to the bundle provenance group")
        return self


class DocumentCorpusIndex(BaseModel):
    """Physical index membership; runtime and evaluator truth cannot overlap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_document_ids: list[str] = Field(default_factory=list)
    evaluator_truth_document_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def indexes_are_disjoint(self) -> "DocumentCorpusIndex":
        runtime = set(self.runtime_document_ids)
        truth = set(self.evaluator_truth_document_ids)
        if len(runtime) != len(self.runtime_document_ids):
            raise ValueError("runtime index contains duplicate document IDs")
        if len(truth) != len(self.evaluator_truth_document_ids):
            raise ValueError("evaluator index contains duplicate document IDs")
        if runtime & truth:
            raise ValueError("runtime and evaluator truth indexes must be disjoint")
        return self


class RuntimeIndexAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checked_document_count: int = Field(ge=0)
    violations: list[str] = Field(default_factory=list)
    visibility_counts: dict[str, int] = Field(default_factory=dict)


def audit_runtime_index(
    artifacts: list[CorpusArtifact],
    *,
    forbidden_tokens: tuple[str, ...] = (),
) -> RuntimeIndexAudit:
    """Reject truth classes and obvious path/identifier proxy leakage."""

    violations: list[str] = []
    counts: Counter[str] = Counter()
    lowered_tokens = tuple(token.casefold() for token in forbidden_tokens if token)
    for artifact in artifacts:
        descriptor = artifact.descriptor
        counts[descriptor.visibility_class.value] += 1
        locator = " ".join(
            [artifact.relative_path, descriptor.document_id, descriptor.source_uri]
        ).casefold()
        if descriptor.visibility_class == DocumentVisibilityClass.TRUTH_QUARANTINED:
            violations.append(f"{descriptor.document_id}:TRUTH_IN_RUNTIME_INDEX")
        if (
            descriptor.contains_episode_truth
            and descriptor.visibility_class
            != DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY
        ):
            violations.append(f"{descriptor.document_id}:EPISODE_TRUTH_FLAGGED")
        for token in lowered_tokens:
            if token in locator:
                violations.append(f"{descriptor.document_id}:FORBIDDEN_TOKEN:{token}")
    return RuntimeIndexAudit(
        passed=not violations,
        checked_document_count=len(artifacts),
        violations=violations,
        visibility_counts=dict(sorted(counts.items())),
    )


class DocumentCorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-document-corpus-manifest.v1"
    corpus_id: str = Field(min_length=1)
    release_state: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dataset_sha256: dict[str, str] = Field(default_factory=dict)
    generator_sha256: dict[str, str] = Field(default_factory=dict)
    bundles: list[PairedDocumentBundle] = Field(default_factory=list)
    index: DocumentCorpusIndex

    @model_validator(mode="after")
    def manifest_references_are_complete(self) -> "DocumentCorpusManifest":
        bundle_ids = [bundle.bundle_id for bundle in self.bundles]
        episode_ids = [bundle.opaque_episode_id for bundle in self.bundles]
        if len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError("bundle IDs must be unique")
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError("opaque episode IDs must be unique")
        descriptors = {
            artifact.descriptor.document_id: artifact
            for bundle in self.bundles
            for artifact in bundle.artifacts
        }
        indexed = set(self.index.runtime_document_ids) | set(
            self.index.evaluator_truth_document_ids
        )
        unknown = indexed - set(descriptors)
        if unknown:
            raise ValueError(f"index references unknown documents: {sorted(unknown)}")
        runtime_artifacts = [
            descriptors[document_id] for document_id in self.index.runtime_document_ids
        ]
        audit = audit_runtime_index(runtime_artifacts)
        if not audit.passed:
            raise ValueError("runtime index contains quarantined truth")
        return self


__all__ = [
    "CorpusArtifact",
    "CorpusSplitRole",
    "DocumentCorpusIndex",
    "DocumentCorpusManifest",
    "PairedDocumentBundle",
    "RuntimeIndexAudit",
    "audit_runtime_index",
]
