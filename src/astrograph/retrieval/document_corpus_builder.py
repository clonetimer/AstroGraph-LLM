"""Deterministic builder for paired telemetry and governed document corpora."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Iterable

from .document_corpus import (
    CorpusArtifact,
    CorpusSplitRole,
    DocumentCorpusIndex,
    DocumentCorpusManifest,
    PairedDocumentBundle,
    audit_runtime_index,
)
from .document_governance import (
    DocumentAccessContext,
    DocumentAuthority,
    DocumentReviewState,
    DocumentType,
    DocumentVisibilityClass,
    GovernedDocumentDescriptor,
    evaluate_document_access,
)


TRAIN_GROUPS = frozenset({0, 1, 2, 5, 6, 7})
DEVELOPMENT_GROUPS = frozenset({3, 8})
POSTHOC_GROUPS = frozenset({4, 9})
FORBIDDEN_RUNTIME_TOKENS = (
    "condition_id",
    "diagnostic_class",
    "fault_injection",
    "ground_truth",
    "event_parameters",
    "active_effects",
)
OBSERVATION_CANONICAL_BYTES = 32 * 1024
ANOMALY_CANONICAL_BYTES = 4 * 1024
SYNTHETIC_LICENSE_ID = "LicenseRef-AstroGraph-Synthetic-Research-Only"
SYNTHETIC_USAGE_BOUNDARY = "RESEARCH_DEVELOPMENT_ONLY_NO_FLIGHT_OR_RUNTIME_VOTE"
CORPUS_TEMPLATE_VERSION = "paired-document-canonical-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opaque(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:24]}"


def _split_role(group: int) -> CorpusSplitRole:
    if group in TRAIN_GROUPS:
        return CorpusSplitRole.TRAIN
    if group in DEVELOPMENT_GROUPS:
        return CorpusSplitRole.DEVELOPMENT
    if group in POSTHOC_GROUPS:
        return CorpusSplitRole.POSTHOC
    raise ValueError(f"unregistered Monte Carlo group: {group}")


def _format_number(value: float) -> str:
    return f"{value:.9g}"


def _pad_canonical_markdown(text: str, *, target_bytes: int) -> str:
    """Append semantically empty whitespace to remove byte-length label channels."""

    normalized = text.rstrip("\n") + "\n"
    encoded = normalized.encode("utf-8")
    if len(encoded) > target_bytes:
        raise ValueError(
            f"canonical runtime document exceeds fixed byte budget: {len(encoded)} > {target_bytes}"
        )
    return normalized + (" " * (target_bytes - len(encoded)))


def _load_telemetry(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "time_s" not in reader.fieldnames:
            raise ValueError(f"telemetry has no explicit time_s column: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"telemetry is empty: {path}")
    headers = list(reader.fieldnames)
    bad_headers = [
        field
        for field in headers
        if any(token in field.casefold() for token in FORBIDDEN_RUNTIME_TOKENS)
    ]
    if bad_headers:
        raise ValueError(f"truth-like telemetry headers are forbidden: {bad_headers}")
    return headers, rows


def _rows_to_cutoff(rows: list[dict[str, str]], cutoff_s: float | None) -> list[dict[str, str]]:
    if cutoff_s is None:
        return rows
    selected = [
        row for row in rows
        if row.get("time_s") and float(row["time_s"]) <= cutoff_s + 1e-12
    ]
    if not selected:
        raise ValueError(f"no telemetry observations at or before cutoff {cutoff_s}")
    return selected


def _summarize_loaded_telemetry(
    headers: list[str],
    rows: list[dict[str, str]],
    *,
    cutoff_s: float | None = None,
) -> tuple[str, float, float, int, int]:
    selected = _rows_to_cutoff(rows, cutoff_s)
    times = [float(row["time_s"]) for row in selected if row.get("time_s")]
    if not times:
        raise ValueError("telemetry has no valid time values")

    lines = [
        "# Telemetry observation record",
        "",
        "This record summarizes only fields present in the observable telemetry file.",
        "It contains no sample-index metadata, injected-event configuration, label, or closure conclusion.",
        "",
        "## Observation interval",
        "",
        f"- Start time (s): {_format_number(min(times))}",
        f"- Cutoff time (s): {_format_number(max(times))}",
        f"- Telemetry rows: {len(selected)}",
        f"- Observable fields: {len(headers)}",
        "",
        "## Channel summaries",
        "",
        "| Channel | Present | First | Last | Minimum | Maximum | Mean / distinct |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for field in headers:
        raw_values = [row.get(field, "").strip() for row in selected]
        present = [value for value in raw_values if value != ""]
        numeric: list[float] = []
        numeric_ok = bool(present)
        for value in present:
            try:
                parsed = float(value)
            except ValueError:
                numeric_ok = False
                break
            if math.isfinite(parsed):
                numeric.append(parsed)
        first = present[0] if present else ""
        last = present[-1] if present else ""
        if numeric_ok and numeric:
            minimum = _format_number(min(numeric))
            maximum = _format_number(max(numeric))
            aggregate = _format_number(sum(numeric) / len(numeric))
        else:
            minimum = maximum = ""
            distinct = sorted(set(present))
            aggregate = f"{len(distinct)} values: " + ", ".join(distinct[:8])
        escaped = [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in (field, first, last, minimum, maximum, aggregate)
        ]
        lines.append(
            f"| {escaped[0]} | {len(present)} | {escaped[1]} | {escaped[2]} | "
            f"{escaped[3]} | {escaped[4]} | {escaped[5]} |"
        )
    return "\n".join(lines) + "\n", min(times), max(times), len(selected), len(headers)


def summarize_telemetry(
    path: Path,
    *,
    cutoff_s: float | None = None,
) -> tuple[str, float, float, int, int]:
    """Create a label-blind observation record from telemetry columns only."""

    headers, rows = _load_telemetry(path)
    return _summarize_loaded_telemetry(headers, rows, cutoff_s=cutoff_s)


def _numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field, "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            return []
        if math.isfinite(value):
            values.append(value)
    return values


def _anomaly_screening_report(
    headers: list[str],
    rows: list[dict[str, str]],
    *,
    cutoff_s: float,
    baseline_end_s: float = 30.0,
    top_k: int = 12,
) -> str:
    selected = _rows_to_cutoff(rows, cutoff_s)
    baseline = [row for row in selected if float(row["time_s"]) <= baseline_end_s]
    comparison = [row for row in selected if float(row["time_s"]) > baseline_end_s]
    ranked: list[tuple[float, str, float, float, float]] = []
    if len(baseline) >= 3 and len(comparison) >= 3:
        for field in headers:
            if field == "time_s":
                continue
            left = _numeric_values(baseline, field)
            right = _numeric_values(comparison, field)
            if len(left) < 3 or len(right) < 3:
                continue
            mean_left = sum(left) / len(left)
            mean_right = sum(right) / len(right)
            variance = sum((value - mean_left) ** 2 for value in left) / len(left)
            scale = max(math.sqrt(variance), abs(mean_left) * 0.01, 1e-9)
            score = abs(mean_right - mean_left) / scale
            ranked.append((score, field, mean_left, mean_right, scale))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    lines = [
        "# Telemetry anomaly screening candidate",
        "",
        "This report is computed from observable telemetry only and contains no fault label.",
        "The normalized shift is an uncalibrated ranking heuristic, not a diagnosis or evidence threshold.",
        "",
        "## Causal window",
        "",
        f"- Baseline end (s): {_format_number(baseline_end_s)}",
        f"- Observation cutoff (s): {_format_number(cutoff_s)}",
        f"- Baseline rows: {len(baseline)}",
        f"- Comparison rows: {len(comparison)}",
        "",
        "## Largest normalized mean shifts",
        "",
        "| Rank | Channel | Baseline mean | Later mean | Baseline scale | Heuristic shift |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, (score, field, before, after, scale) in enumerate(ranked[:top_k], 1):
        safe_field = field.replace("|", "\\|")
        lines.append(
            f"| {rank} | {safe_field} | {_format_number(before)} | "
            f"{_format_number(after)} | {_format_number(scale)} | {_format_number(score)} |"
        )
    if not ranked:
        lines.append("| 1 | INSUFFICIENT_NUMERIC_WINDOW |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Admission boundary",
            "",
            "Any downstream evidence claim requires a separately registered and calibrated evidence definition.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_causal_reports(
    path: Path,
    *,
    requested_cutoffs_s: tuple[float, ...] = (60.0, 120.0),
) -> list[tuple[float, str, str]]:
    """Return cutoff, observation text, and label-blind anomaly-screening text."""

    headers, rows = _load_telemetry(path)
    end_s = max(float(row["time_s"]) for row in rows if row.get("time_s"))
    cutoffs = sorted({min(value, end_s) for value in requested_cutoffs_s} | {end_s})
    reports: list[tuple[float, str, str]] = []
    for cutoff in cutoffs:
        observation, _, actual_cutoff, _, _ = _summarize_loaded_telemetry(
            headers, rows, cutoff_s=cutoff
        )
        anomaly = _anomaly_screening_report(
            headers, rows, cutoff_s=actual_cutoff
        )
        reports.append((
            actual_cutoff,
            _pad_canonical_markdown(
                observation, target_bytes=OBSERVATION_CANONICAL_BYTES
            ),
            _pad_canonical_markdown(
                anomaly, target_bytes=ANOMALY_CANONICAL_BYTES
            ),
        ))
    return reports


def _truth_payload(row: dict[str, str]) -> dict[str, str]:
    allowed = (
        "condition_id",
        "diagnostic_class",
        "event_category",
        "effect",
        "fault_injection_phase",
        "event_start_s",
        "activation_time_s",
        "event_parameters",
        "fault_manifestation_phase",
        "fault_manifestation_time_s",
    )
    return {key: row[key] for key in allowed if row.get(key, "") != ""}


def _historical_resolution_text(row: dict[str, str], provenance_group_id: str) -> str:
    truth = _truth_payload(row)
    lines = [
        "# Historical resolution candidate",
        "",
        f"- Provenance group: {provenance_group_id}",
        "- Source nature: synthetic simulator resolution",
        "- Review state: human review required",
        "",
        "## Resolution fields",
        "",
    ]
    for key, value in sorted(truth.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Use boundary",
            "",
            "This report may support cross-group historical analogy after review. ",
            "It cannot diagnose its own provenance group and cannot vote directly.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_artifact(root: Path, relative_path: str, text: str) -> tuple[Path, str]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def build_paired_document_corpus(
    *,
    datasets: Iterable[tuple[str, Path]],
    output_root: Path,
    policy_path: Path,
    generator_paths: dict[str, Path],
    include_development: bool = True,
    limit_per_dataset: int | None = None,
    corpus_revision: str = "r002",
) -> tuple[DocumentCorpusManifest, dict[str, object]]:
    """Build train/development documents while never opening post-hoc telemetry or labels."""

    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty corpus directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    policy_sha = sha256_file(policy_path)
    generator_hashes = {
        key: sha256_file(path) for key, path in generator_paths.items()
    }
    bundles: list[PairedDocumentBundle] = []
    runtime_ids: list[str] = []
    evaluator_ids: list[str] = []
    skipped_posthoc = 0
    split_counts: dict[str, int] = {}
    document_counts: dict[str, int] = {}
    document_type_counts: dict[str, int] = {}
    source_dataset_hashes: dict[str, str] = {}

    for dataset_id, dataset_root in datasets:
        index_path = dataset_root / "sample_index.csv"
        source_dataset_hashes[dataset_id] = sha256_file(index_path)
        with index_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        included = 0
        for row in rows:
            group = int(row["sample_index"])
            split_role = _split_role(group)
            if split_role == CorpusSplitRole.POSTHOC:
                skipped_posthoc += 1
                continue
            if split_role == CorpusSplitRole.DEVELOPMENT and not include_development:
                continue
            if limit_per_dataset is not None and included >= limit_per_dataset:
                break
            included += 1
            split_counts[split_role.value] = split_counts.get(split_role.value, 0) + 1

            sample_id = row["sample_id"]
            opaque_episode_id = _opaque("episode", dataset_id, sample_id)
            bundle_id = _opaque("bundle", dataset_id, sample_id)
            group_id = f"{dataset_id}:mc-group:{group}"
            telemetry_path = (dataset_root / row["telemetry_file"]).resolve()
            if not telemetry_path.is_relative_to(dataset_root.resolve()):
                raise ValueError("telemetry path escapes dataset root")
            telemetry_sha = sha256_file(telemetry_path)
            artifacts: list[CorpusArtifact] = []
            for cutoff_s, observation, anomaly in build_causal_reports(telemetry_path):
                cutoff_token = _format_number(cutoff_s).replace(".", "p")
                for kind, document_type, text in (
                    ("observation", DocumentType.TEST_LOG, observation),
                    ("anomaly", DocumentType.ANOMALY_REPORT, anomaly),
                ):
                    document_id = _opaque(
                        f"doc:{kind}", dataset_id, sample_id, _format_number(cutoff_s)
                    )
                    relative_path = (
                        f"runtime/{opaque_episode_id.split(':', 1)[1]}/"
                        f"{kind}_{cutoff_token}s.md"
                    )
                    _, document_sha = _write_artifact(output_root, relative_path, text)
                    descriptor = GovernedDocumentDescriptor(
                        document_id=document_id,
                        document_type=document_type,
                        visibility_class=DocumentVisibilityClass.OBSERVATION_CAUSAL,
                        source_uri=f"corpus://{relative_path}",
                        source_sha256=document_sha,
                        review_state=DocumentReviewState.MACHINE_VALIDATED,
                        authority=DocumentAuthority.NON_NORMATIVE,
                        provenance_group_id=group_id,
                        episode_id=opaque_episode_id,
                        valid_time_end_s=cutoff_s,
                        observable_cutoff_time_s=cutoff_s,
                        template_version=CORPUS_TEMPLATE_VERSION,
                        license_id=SYNTHETIC_LICENSE_ID,
                        usage_boundary=SYNTHETIC_USAGE_BOUNDARY,
                        generator_sha256=generator_hashes.get(dataset_id),
                        synthetic_from_simulation=True,
                        runtime_vote_allowed=False,
                    )
                    artifacts.append(
                        CorpusArtifact(
                            relative_path=relative_path,
                            sha256=document_sha,
                            media_type="text/markdown",
                            descriptor=descriptor,
                        )
                    )
                    runtime_ids.append(document_id)
                    document_counts["OBSERVATION_CAUSAL"] = (
                        document_counts.get("OBSERVATION_CAUSAL", 0) + 1
                    )
                    document_type_counts[document_type.value] = (
                        document_type_counts.get(document_type.value, 0) + 1
                    )

            truth_id = _opaque("doc:truth", dataset_id, sample_id)
            truth_rel = f"quarantine/{opaque_episode_id.split(':', 1)[1]}/truth.json"
            truth_text = json.dumps(_truth_payload(row), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            _, truth_sha = _write_artifact(output_root, truth_rel, truth_text)
            truth_descriptor = GovernedDocumentDescriptor(
                document_id=truth_id,
                document_type=DocumentType.CAMPAIGN_REPORT,
                visibility_class=DocumentVisibilityClass.TRUTH_QUARANTINED,
                source_uri=f"quarantine://{truth_rel}",
                source_sha256=truth_sha,
                review_state=DocumentReviewState.MACHINE_VALIDATED,
                authority=DocumentAuthority.NON_NORMATIVE,
                provenance_group_id=group_id,
                episode_id=opaque_episode_id,
                template_version=CORPUS_TEMPLATE_VERSION,
                license_id=SYNTHETIC_LICENSE_ID,
                usage_boundary=SYNTHETIC_USAGE_BOUNDARY,
                generator_sha256=generator_hashes.get(dataset_id),
                contains_episode_truth=True,
                synthetic_from_simulation=True,
                runtime_vote_allowed=False,
            )
            artifacts.append(
                CorpusArtifact(
                    relative_path=truth_rel,
                    sha256=truth_sha,
                    media_type="application/json",
                    descriptor=truth_descriptor,
                )
            )
            evaluator_ids.append(truth_id)
            document_counts["TRUTH_QUARANTINED"] = document_counts.get("TRUTH_QUARANTINED", 0) + 1
            document_type_counts[DocumentType.CAMPAIGN_REPORT.value] = (
                document_type_counts.get(DocumentType.CAMPAIGN_REPORT.value, 0) + 1
            )

            if split_role == CorpusSplitRole.TRAIN and row.get("condition_id") != "nominal":
                closure_id = _opaque("doc:closure", dataset_id, sample_id)
                closure_rel = f"historical/{opaque_episode_id.split(':', 1)[1]}/closure.md"
                closure_text = _historical_resolution_text(row, group_id)
                _, closure_sha = _write_artifact(output_root, closure_rel, closure_text)
                closure_descriptor = GovernedDocumentDescriptor(
                    document_id=closure_id,
                    document_type=DocumentType.CLOSURE_REPORT,
                    visibility_class=DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY,
                    source_uri=f"corpus://{closure_rel}",
                    source_sha256=closure_sha,
                    review_state=DocumentReviewState.REVIEW_REQUIRED,
                    authority=DocumentAuthority.NON_NORMATIVE,
                    provenance_group_id=group_id,
                    episode_id=opaque_episode_id,
                    template_version=CORPUS_TEMPLATE_VERSION,
                    license_id=SYNTHETIC_LICENSE_ID,
                    usage_boundary=SYNTHETIC_USAGE_BOUNDARY,
                    generator_sha256=generator_hashes.get(dataset_id),
                    contains_root_cause=True,
                    contains_episode_truth=True,
                    synthetic_from_simulation=True,
                    runtime_vote_allowed=False,
                )
                artifacts.append(
                    CorpusArtifact(
                        relative_path=closure_rel,
                        sha256=closure_sha,
                        media_type="text/markdown",
                        descriptor=closure_descriptor,
                    )
                )
                document_counts["RESOLVED_CASE_TRAIN_ONLY"] = document_counts.get("RESOLVED_CASE_TRAIN_ONLY", 0) + 1
                document_type_counts[DocumentType.CLOSURE_REPORT.value] = (
                    document_type_counts.get(DocumentType.CLOSURE_REPORT.value, 0) + 1
                )

            bundles.append(
                PairedDocumentBundle(
                    bundle_id=bundle_id,
                    dataset_id=dataset_id,
                    opaque_episode_id=opaque_episode_id,
                    source_episode_ref_sha256=hashlib.sha256(sample_id.encode("utf-8")).hexdigest(),
                    telemetry_sha256=telemetry_sha,
                    provenance_group_id=group_id,
                    split_role=split_role,
                    artifacts=artifacts,
                )
            )

    manifest = DocumentCorpusManifest(
        corpus_id=f"astrograph-paired-document-development-{corpus_revision}",
        release_state="DRAFT_MACHINE_GENERATED_REVIEW_REQUIRED",
        policy_sha256=policy_sha,
        source_dataset_sha256=source_dataset_hashes,
        generator_sha256=generator_hashes,
        bundles=bundles,
        index=DocumentCorpusIndex(
            runtime_document_ids=runtime_ids,
            evaluator_truth_document_ids=evaluator_ids,
        ),
    )
    descriptor_by_id = {
        artifact.descriptor.document_id: artifact
        for bundle in bundles
        for artifact in bundle.artifacts
    }
    runtime_audit = audit_runtime_index(
        [descriptor_by_id[item] for item in runtime_ids],
        forbidden_tokens=FORBIDDEN_RUNTIME_TOKENS,
    )
    if not runtime_audit.passed:
        raise ValueError(f"runtime index audit failed: {runtime_audit.violations}")
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "astrograph-paired-document-corpus-build-receipt.v1",
        "corpus_id": manifest.corpus_id,
        "release_state": manifest.release_state,
        "manifest_sha256": sha256_file(manifest_path),
        "policy_sha256": policy_sha,
        "source_dataset_sha256": source_dataset_hashes,
        "generator_sha256": manifest.generator_sha256,
        "bundle_count": len(bundles),
        "split_counts": dict(sorted(split_counts.items())),
        "document_counts": dict(sorted(document_counts.items())),
        "document_type_counts": dict(sorted(document_type_counts.items())),
        "runtime_index_count": len(runtime_ids),
        "evaluator_truth_index_count": len(evaluator_ids),
        "skipped_posthoc_row_count": skipped_posthoc,
        "posthoc_telemetry_or_truth_opened": False,
        "runtime_index_audit": runtime_audit.model_dump(mode="json"),
        "historical_closures_indexed_for_runtime": 0,
        "claim_boundary": (
            "Deterministic synthetic development corpus only. Post-hoc groups 4/9 were skipped "
            "before telemetry or truth files were opened. Historical closures remain unreviewed "
            "and unindexed; no retrieval-quality or diagnosis-gain claim is made."
        ),
    }
    return manifest, receipt


def audit_paired_document_corpus(
    *,
    manifest_path: Path,
    corpus_root: Path,
    datasets: dict[str, Path],
    generator_paths: dict[str, Path],
) -> dict[str, object]:
    """Recheck file hashes, index separation, source drift, and label-value proxies."""

    manifest = DocumentCorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    violations: list[str] = []
    rows_by_ref: dict[tuple[str, str], dict[str, str]] = {}
    for dataset_id, dataset_root in datasets.items():
        current = sha256_file(dataset_root / "sample_index.csv")
        if manifest.source_dataset_sha256.get(dataset_id) != current:
            violations.append(f"SOURCE_DATASET_DRIFT:{dataset_id}")
        with (dataset_root / "sample_index.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                ref = hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest()
                rows_by_ref[(dataset_id, ref)] = row
    for dataset_id, generator_path in generator_paths.items():
        if manifest.generator_sha256.get(dataset_id) != sha256_file(generator_path):
            violations.append(f"GENERATOR_DRIFT:{dataset_id}")

    runtime_ids = set(manifest.index.runtime_document_ids)
    evaluator_ids = set(manifest.index.evaluator_truth_document_ids)
    artifacts_by_id = {
        artifact.descriptor.document_id: artifact
        for bundle in manifest.bundles
        for artifact in bundle.artifacts
    }
    runtime_artifacts = [artifacts_by_id[item] for item in runtime_ids]
    runtime_audit = audit_runtime_index(
        runtime_artifacts, forbidden_tokens=FORBIDDEN_RUNTIME_TOKENS
    )
    violations.extend(runtime_audit.violations)
    checked_files = 0
    proxy_checked = 0
    future_probe_checked = 0
    cutoff_counts: dict[str, int] = {}
    for bundle in manifest.bundles:
        row = rows_by_ref.get((bundle.dataset_id, bundle.source_episode_ref_sha256))
        if row is None:
            violations.append(f"SOURCE_EPISODE_REF_MISSING:{bundle.bundle_id}")
            continue
        forbidden_values = {
            value.casefold()
            for value in (row.get("condition_id", ""), row.get("diagnostic_class", ""))
            if value and value.casefold() != "nominal"
        }
        causal_types_by_cutoff: dict[float, set[DocumentType]] = {}
        for artifact in bundle.artifacts:
            path = corpus_root / artifact.relative_path
            if not path.is_file():
                violations.append(f"ARTIFACT_MISSING:{artifact.descriptor.document_id}")
                continue
            checked_files += 1
            if sha256_file(path) != artifact.sha256:
                violations.append(f"ARTIFACT_HASH_DRIFT:{artifact.descriptor.document_id}")
            document_id = artifact.descriptor.document_id
            visibility = artifact.descriptor.visibility_class
            if document_id in runtime_ids:
                proxy_checked += 1
                cutoff = artifact.descriptor.observable_cutoff_time_s
                if cutoff is None:
                    violations.append(f"RUNTIME_CAUSAL_CUTOFF_MISSING:{document_id}")
                else:
                    cutoff_key = _format_number(cutoff)
                    cutoff_counts[cutoff_key] = cutoff_counts.get(cutoff_key, 0) + 1
                    causal_types_by_cutoff.setdefault(cutoff, set()).add(
                        artifact.descriptor.document_type
                    )
                    before = evaluate_document_access(
                        artifact.descriptor,
                        DocumentAccessContext(
                            episode_id=bundle.opaque_episode_id,
                            current_provenance_group_id=bundle.provenance_group_id,
                            observation_time_s=cutoff - 1e-6,
                        ),
                    )
                    at_cutoff = evaluate_document_access(
                        artifact.descriptor,
                        DocumentAccessContext(
                            episode_id=bundle.opaque_episode_id,
                            current_provenance_group_id=bundle.provenance_group_id,
                            observation_time_s=cutoff,
                        ),
                    )
                    future_probe_checked += 1
                    if before.allowed or before.reason_code != "FUTURE_OBSERVATION_BLOCKED":
                        violations.append(f"FUTURE_CUTOFF_GATE_FAILED:{document_id}")
                    if not at_cutoff.allowed:
                        violations.append(f"AT_CUTOFF_GATE_FAILED:{document_id}")
                text = path.read_text(encoding="utf-8").casefold()
                for value in forbidden_values:
                    if value in text:
                        violations.append(
                            f"RUNTIME_TRUTH_VALUE_PROXY:{document_id}:{value}"
                        )
            if (
                visibility == DocumentVisibilityClass.TRUTH_QUARANTINED
                and document_id not in evaluator_ids
            ):
                violations.append(f"TRUTH_NOT_IN_EVALUATOR_INDEX:{document_id}")
            if (
                visibility == DocumentVisibilityClass.RESOLVED_CASE_TRAIN_ONLY
                and document_id in runtime_ids
                and artifact.descriptor.review_state
                != DocumentReviewState.HUMAN_REVIEWED
            ):
                violations.append(f"UNREVIEWED_CLOSURE_INDEXED:{document_id}")
        expected_pair = {DocumentType.TEST_LOG, DocumentType.ANOMALY_REPORT}
        for cutoff, document_types in causal_types_by_cutoff.items():
            if document_types != expected_pair:
                violations.append(
                    f"CAUSAL_DOCUMENT_PAIR_INCOMPLETE:{bundle.bundle_id}:{_format_number(cutoff)}"
                )
    return {
        "schema_version": "astrograph-paired-document-corpus-audit.v1",
        "status": "PASS" if not violations else "FAIL",
        "manifest_sha256": sha256_file(manifest_path),
        "bundle_count": len(manifest.bundles),
        "artifact_file_count_checked": checked_files,
        "runtime_document_count": len(runtime_ids),
        "runtime_truth_value_proxy_documents_checked": proxy_checked,
        "future_cutoff_gate_documents_checked": future_probe_checked,
        "runtime_document_cutoff_counts": dict(sorted(cutoff_counts.items())),
        "runtime_index_audit": runtime_audit.model_dump(mode="json"),
        "violations": violations,
        "claim_boundary": (
            "Integrity and synthetic-corpus leakage audit only. PASS does not establish "
            "retrieval quality, document realism, diagnosis gain, or external validity."
        ),
    }


def verify_paired_document_corpus_rebuild(
    *,
    existing_manifest_path: Path,
    datasets: Iterable[tuple[str, Path]],
    policy_path: Path,
    generator_paths: dict[str, Path],
    temporary_parent: Path,
    corpus_revision: str = "r002",
) -> dict[str, object]:
    """Rebuild into a fresh directory and compare every manifested artifact hash."""

    existing = DocumentCorpusManifest.model_validate_json(
        existing_manifest_path.read_text(encoding="utf-8")
    )
    dataset_list = list(datasets)
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="paired-document-rebuild-", dir=temporary_parent
    ) as directory:
        rebuilt_root = Path(directory)
        rebuilt, rebuilt_receipt = build_paired_document_corpus(
            datasets=dataset_list,
            output_root=rebuilt_root,
            policy_path=policy_path,
            generator_paths=generator_paths,
            include_development=True,
            corpus_revision=corpus_revision,
        )
        existing_artifacts = {
            artifact.relative_path: artifact.sha256
            for bundle in existing.bundles
            for artifact in bundle.artifacts
        }
        rebuilt_artifacts = {
            artifact.relative_path: artifact.sha256
            for bundle in rebuilt.bundles
            for artifact in bundle.artifacts
        }
        path_set_equal = set(existing_artifacts) == set(rebuilt_artifacts)
        mismatched = sorted(
            path for path in set(existing_artifacts) & set(rebuilt_artifacts)
            if existing_artifacts[path] != rebuilt_artifacts[path]
            or sha256_file(rebuilt_root / path) != rebuilt_artifacts[path]
        )
        manifest_equal = existing.model_dump(mode="json") == rebuilt.model_dump(mode="json")
        status = "PASS" if manifest_equal and path_set_equal and not mismatched else "FAIL"
        body = {
            "schema_version": "astrograph-paired-document-rebuild-verification.v1",
            "status": status,
            "existing_manifest_sha256": sha256_file(existing_manifest_path),
            "rebuilt_manifest_sha256": rebuilt_receipt["manifest_sha256"],
            "manifest_semantically_identical": manifest_equal,
            "artifact_path_set_identical": path_set_equal,
            "artifact_count": len(rebuilt_artifacts),
            "artifact_hash_mismatch_count": len(mismatched),
            "artifact_hash_mismatch_paths": mismatched[:100],
            "fresh_temporary_directory_used": True,
            "temporary_rebuild_removed": True,
            "posthoc_telemetry_or_truth_opened": rebuilt_receipt[
                "posthoc_telemetry_or_truth_opened"
            ],
            "claim_boundary": (
                "Byte-deterministic rebuild in a fresh directory using the current local Python "
                "environment. This is not yet a container or independent-machine reproduction."
            ),
        }
    body["verification_sha256"] = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


__all__ = [
    "ANOMALY_CANONICAL_BYTES",
    "DEVELOPMENT_GROUPS",
    "FORBIDDEN_RUNTIME_TOKENS",
    "OBSERVATION_CANONICAL_BYTES",
    "POSTHOC_GROUPS",
    "TRAIN_GROUPS",
    "build_paired_document_corpus",
    "audit_paired_document_corpus",
    "sha256_file",
    "summarize_telemetry",
    "build_causal_reports",
    "verify_paired_document_corpus_rebuild",
]
