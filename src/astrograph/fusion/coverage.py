from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re

from astrograph.evidence import EpisodeEvidenceBundle, EvidenceRole

from .models import KGFaultCoverageDeclaration

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_kg_fault_coverage_declaration(
    bundle: EpisodeEvidenceBundle,
) -> KGFaultCoverageDeclaration:
    """Derive executable KG/rule capability from frozen evidence definitions.

    Positive-vote capability coverage is schema-level and therefore invariant to which facts are
    satisfied, missing, or ranked in one episode.  This function deliberately does
    not inspect labels, predictions, per-episode truth, or ranking cutoffs.
    """

    feature_ids = {item.feature_definition_id for item in bundle.feature_definitions}
    definitions_by_fault: dict[str, list[str]] = defaultdict(list)
    for definition in bundle.evidence_definitions:
        referenced_features = {
            definition.feature_definition_id,
            *(item.feature_definition_id for item in definition.and_feature_conditions),
        }
        missing_features = sorted(referenced_features - feature_ids)
        if missing_features:
            raise ValueError(
                "cannot declare executable KG coverage: evidence definition "
                f"{definition.evidence_definition_id} references absent features "
                f"{missing_features}"
            )
        if definition.role in {EvidenceRole.REQUIRED, EvidenceRole.SUPPORTING}:
            definitions_by_fault[definition.target_fault_id].append(
                definition.evidence_definition_id
            )

    package_sha256 = bundle.source_protocol_hashes.get(
        "configured_runtime_evidence_package"
    )
    package_id = bundle.analysis_metadata.get("configured_runtime_package_id")
    package_version = bundle.analysis_metadata.get("configured_runtime_package_version")
    if package_sha256 is not None:
        package_sha256 = str(package_sha256).lower()
        if not _SHA256.fullmatch(package_sha256):
            raise ValueError(
                "configured runtime evidence package hash is not a lowercase SHA-256"
            )
        source_artifact_id = str(package_id or "CONFIGURED_RUNTIME_EVIDENCE_PACKAGE")
        if package_version:
            source_artifact_id = f"{source_artifact_id}@{package_version}"
    else:
        canonical_definitions = sorted(
            (
                definition.model_dump(mode="json")
                for definition in bundle.evidence_definitions
            ),
            key=lambda item: item["evidence_definition_id"],
        )
        package_sha256 = _canonical_sha256(canonical_definitions)
        source_artifact_id = f"EPISODE_EVIDENCE_SCHEMA:{bundle.domain_id}"

    evidence_map = {
        fault_id: tuple(sorted(definition_ids))
        for fault_id, definition_ids in sorted(definitions_by_fault.items())
    }
    return KGFaultCoverageDeclaration(
        source_artifact_id=source_artifact_id,
        source_artifact_sha256=package_sha256,
        supported_fault_ids=tuple(evidence_map),
        evidence_definition_ids_by_fault=evidence_map,
    )


__all__ = ["build_kg_fault_coverage_declaration"]
