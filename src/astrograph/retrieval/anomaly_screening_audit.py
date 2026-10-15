"""Independent recomputation audit for label-blind anomaly screening documents."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .document_corpus import DocumentCorpusManifest
from .document_governance import DocumentType


REGISTERED_ALGORITHM = {
    "baseline_end_s": 30.0,
    "minimum_rows_per_window": 3,
    "scale": "max(population_std,abs(baseline_mean)*0.01,1e-9)",
    "score": "abs(later_mean-baseline_mean)/scale",
    "top_k": 12,
    "tie_break": "channel_name_ascending",
    "number_format": ".9g",
    "canonical_bytes": 4096,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _format(value: float) -> str:
    return f"{value:.9g}"


def _numeric(rows: list[dict[str, str]], field: str) -> list[float]:
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


def independently_render_anomaly_screening(
    telemetry_path: str | Path, *, cutoff_s: float, baseline_end_s: float = 30.0,
    minimum_rows_per_window: int = 3, top_k: int = 12, canonical_bytes: int = 4096,
) -> str:
    """Reimplement the registered algorithm without importing the production builder."""

    telemetry_path = Path(telemetry_path)
    with telemetry_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if "time_s" not in headers or not rows:
        raise ValueError("telemetry must contain time_s and at least one row")
    selected = [row for row in rows if row.get("time_s") and float(row["time_s"]) <= cutoff_s + 1e-12]
    baseline = [row for row in selected if float(row["time_s"]) <= baseline_end_s]
    comparison = [row for row in selected if float(row["time_s"]) > baseline_end_s]
    ranked: list[tuple[float, str, float, float, float]] = []
    if len(baseline) >= minimum_rows_per_window and len(comparison) >= minimum_rows_per_window:
        for field in headers:
            if field == "time_s":
                continue
            left, right = _numeric(baseline, field), _numeric(comparison, field)
            if len(left) < minimum_rows_per_window or len(right) < minimum_rows_per_window:
                continue
            before, after = sum(left) / len(left), sum(right) / len(right)
            variance = sum((value - before) ** 2 for value in left) / len(left)
            scale = max(math.sqrt(variance), abs(before) * 0.01, 1e-9)
            ranked.append((abs(after - before) / scale, field, before, after, scale))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    lines = [
        "# Telemetry anomaly screening candidate", "",
        "This report is computed from observable telemetry only and contains no fault label.",
        "The normalized shift is an uncalibrated ranking heuristic, not a diagnosis or evidence threshold.",
        "", "## Causal window", "", f"- Baseline end (s): {_format(baseline_end_s)}",
        f"- Observation cutoff (s): {_format(cutoff_s)}", f"- Baseline rows: {len(baseline)}",
        f"- Comparison rows: {len(comparison)}", "", "## Largest normalized mean shifts", "",
        "| Rank | Channel | Baseline mean | Later mean | Baseline scale | Heuristic shift |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, (score, field, before, after, scale) in enumerate(ranked[:top_k], 1):
        lines.append(f"| {rank} | {field.replace('|', chr(92) + '|')} | {_format(before)} | "
                     f"{_format(after)} | {_format(scale)} | {_format(score)} |")
    if not ranked:
        lines.append("| 1 | INSUFFICIENT_NUMERIC_WINDOW |  |  |  |  |")
    lines.extend(["", "## Admission boundary", "",
                  "Any downstream evidence claim requires a separately registered and calibrated evidence definition."])
    text = "\n".join(lines).rstrip("\n") + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > canonical_bytes:
        raise ValueError("independent anomaly report exceeds registered byte budget")
    return text + " " * (canonical_bytes - len(encoded))


def audit_anomaly_screening_documents(
    *, config_path: str | Path, manifest_path: str | Path, corpus_root: str | Path,
    dataset_roots: dict[str, str | Path], production_builder_path: str | Path,
) -> dict[str, Any]:
    config_path, manifest_path = Path(config_path).resolve(), Path(manifest_path).resolve()
    corpus_root, production_builder_path = Path(corpus_root).resolve(), Path(production_builder_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "astrograph-anomaly-screening-recomputation-config.v1":
        raise ValueError("unsupported anomaly recomputation config")
    if config.get("algorithm") != REGISTERED_ALGORITHM:
        raise ValueError("anomaly recomputation algorithm differs from frozen implementation")
    if set(dataset_roots) != set(config["dataset_index_sha256"]):
        raise ValueError("dataset roots differ from the frozen audit set")
    violations: list[str] = []
    if _sha(manifest_path) != config["source_manifest_sha256"]:
        violations.append("SOURCE_MANIFEST_HASH_MISMATCH")
    if _sha(production_builder_path) != config["production_builder_sha256"]:
        violations.append("PRODUCTION_BUILDER_HASH_MISMATCH")
    manifest = DocumentCorpusManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    row_maps: dict[str, dict[str, str]] = {}
    for dataset_id, root_value in dataset_roots.items():
        root = Path(root_value).resolve(); index_path = root / "sample_index.csv"
        if _sha(index_path) != config["dataset_index_sha256"][dataset_id]:
            violations.append(f"DATASET_INDEX_HASH_MISMATCH:{dataset_id}")
        with index_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        row_maps[dataset_id] = {
            hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(): row["telemetry_file"]
            for row in rows
        }
    checked = matched = 0
    dataset_counts: dict[str, dict[str, int]] = {}
    mismatch_examples: list[dict[str, str]] = []
    algorithm = config["algorithm"]
    for bundle in manifest.bundles:
        relative_telemetry = row_maps[bundle.dataset_id].get(bundle.source_episode_ref_sha256)
        if relative_telemetry is None:
            violations.append(f"SOURCE_EPISODE_NOT_FOUND:{bundle.bundle_id}")
            continue
        dataset_root = Path(dataset_roots[bundle.dataset_id]).resolve()
        telemetry = (dataset_root / relative_telemetry).resolve()
        if not telemetry.is_relative_to(dataset_root) or not telemetry.is_file():
            violations.append(f"TELEMETRY_PATH_INVALID:{bundle.bundle_id}")
            continue
        if _sha(telemetry) != bundle.telemetry_sha256:
            violations.append(f"TELEMETRY_HASH_MISMATCH:{bundle.bundle_id}")
            continue
        for artifact in bundle.artifacts:
            if artifact.descriptor.document_type is not DocumentType.ANOMALY_REPORT:
                continue
            cutoff = artifact.descriptor.observable_cutoff_time_s
            if cutoff is None:
                violations.append(f"ANOMALY_CUTOFF_MISSING:{artifact.descriptor.document_id}")
                continue
            expected = independently_render_anomaly_screening(
                telemetry, cutoff_s=cutoff, baseline_end_s=float(algorithm["baseline_end_s"]),
                minimum_rows_per_window=int(algorithm["minimum_rows_per_window"]),
                top_k=int(algorithm["top_k"]), canonical_bytes=int(algorithm["canonical_bytes"]),
            ).encode("utf-8")
            actual_path = (corpus_root / artifact.relative_path).resolve()
            if not actual_path.is_relative_to(corpus_root):
                violations.append(f"ANOMALY_PATH_ESCAPES_CORPUS:{artifact.descriptor.document_id}")
                actual = b""
            else:
                actual = actual_path.read_bytes() if actual_path.is_file() else b""
            checked += 1
            counts = dataset_counts.setdefault(bundle.dataset_id, {"checked": 0, "matched": 0})
            counts["checked"] += 1
            if actual == expected and hashlib.sha256(actual).hexdigest() == artifact.sha256:
                matched += 1; counts["matched"] += 1
            else:
                violations.append(f"ANOMALY_RECOMPUTATION_MISMATCH:{artifact.descriptor.document_id}")
                if len(mismatch_examples) < 20:
                    mismatch_examples.append({"document_id": artifact.descriptor.document_id,
                        "expected_sha256": hashlib.sha256(expected).hexdigest(),
                        "actual_sha256": hashlib.sha256(actual).hexdigest()})
    body = {
        "schema_version": "astrograph-anomaly-screening-recomputation-audit.v1",
        "status": "PASS" if not violations and checked > 0 and matched == checked else "FAIL",
        "configuration_sha256": _sha(config_path),
        "source_manifest_sha256": _sha(manifest_path),
        "independent_auditor_source_sha256": _sha(Path(__file__)),
        "anomaly_document_count_checked": checked,
        "exact_byte_match_count": matched,
        "exact_byte_match_fraction": matched / checked if checked else 0.0,
        "dataset_counts": dataset_counts,
        "accessed_sample_index_fields": config["accessed_sample_index_fields"],
        "forbidden_inputs_used": [],
        "mismatch_examples": mismatch_examples,
        "violations": violations[:200],
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "claim_boundary": config["claim_boundary"],
    }
    return {**body, "audit_sha256": _canonical_sha(body)}


__all__ = ["audit_anomaly_screening_documents", "independently_render_anomaly_screening"]
