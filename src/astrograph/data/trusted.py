"""Trusted-data provenance, classification and leakage auditing.

The module intentionally separates *fault effect validity* from *complete
simulation acceptance*.  A run is ``ACCEPTANCE_VALID`` only when every frozen
execution/physics/conservation gate is exactly ``PASS`` and no public trace
leaks labels or equivalent truth proxies.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from astrograph.semantics.leakage import ProhibitedFieldRole, prohibited_field_role
from astrograph.assurance.artifact_paths import ArtifactRoot


class AcceptanceLevel(str, Enum):
    ACCEPTANCE_VALID = "ACCEPTANCE_VALID"
    EFFECT_VALID_RESEARCH_ONLY = "EFFECT_VALID_RESEARCH_ONLY"
    INVALID = "INVALID"


FORMAL_USES = {"training", "calibration", "validation", "acceptance"}
ACCEPTANCE_USES = ["research", "debugging", "training", "calibration", "validation", "acceptance"]
RESEARCH_USES = ["research", "debugging"]
INVALID_USES = ["forensic_review"]

# Public traces are scanned through the same centralized information-role
# classifier used by semantic mapping.  This closes naming gaps such as
# ``fault_type``, ``condition_id``, ``diagnostic_class`` and ``event_start_s``.
# A clean name is still not an allowlist approval; downstream model-input
# construction remains responsible for explicit semantic binding.

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, *, ignore_names: Sequence[str] = ()) -> str:
    """Return a deterministic digest over relative paths and file bytes."""
    root = root.resolve()
    ignored = set(ignore_names)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if any(part in ignored for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _normalise_header(value: str) -> str:
    return value.lstrip("\ufeff").strip().lower()


def classify_leak_columns(columns: Iterable[str]) -> tuple[list[str], list[str]]:
    direct: list[str] = []
    proxies: list[str] = []
    for raw in columns:
        display = str(raw).lstrip("\ufeff")
        role = prohibited_field_role(_normalise_header(str(raw)))
        if role is None:
            continue
        if role == ProhibitedFieldRole.DIRECT_LABEL:
            direct.append(display)
        else:
            proxies.append(display)
    return sorted(set(direct)), sorted(set(proxies))


def scan_public_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": str(path),
            "exists": False,
            "sha256": "",
            "column_count": 0,
            "direct_label_columns": [],
            "truth_proxy_columns": [],
            "leakage": True,
            "error": "PUBLIC_TRACE_MISSING",
        }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        direct, proxies = classify_leak_columns(header)
        return {
            "path": str(path),
            "exists": True,
            "sha256": sha256_file(path),
            "column_count": len(header),
            "direct_label_columns": direct,
            "truth_proxy_columns": proxies,
            "leakage": bool(direct or proxies),
            "error": None,
        }
    except Exception as exc:  # defensive evidence collection
        return {
            "path": str(path),
            "exists": True,
            "sha256": sha256_file(path),
            "column_count": 0,
            "direct_label_columns": [],
            "truth_proxy_columns": [],
            "leakage": True,
            "error": f"CSV_HEADER_UNREADABLE:{type(exc).__name__}",
        }


def _is_pass(value: Any) -> bool:
    return str(value or "").strip().upper() == "PASS"


def _summary_gate_vector(summary: Mapping[str, Any]) -> dict[str, str]:
    return {
        "status": str(summary.get("status", "MISSING")),
        "execution_status": str(summary.get("execution_status", "MISSING")),
        "physics_status": str(summary.get("physics_status", "MISSING")),
        "data_conservation_status": str(summary.get("data_conservation_status", "MISSING")),
        "energy_conservation_status": str(summary.get("energy_conservation_status", "MISSING")),
    }


def _effect_evidence_valid(summary: Mapping[str, Any], variant: str, campaign_result: Mapping[str, Any]) -> bool:
    common = (
        summary.get("basilisk_simbase_used") is True
        and summary.get("execute_simulation_used") is True
        and _is_pass(summary.get("numerical_status"))
        and _is_pass(summary.get("mission_status"))
        and Path(str(summary.get("backend", "basilisk"))).name.lower() != "synthetic"
    )
    if not common:
        return False
    if variant == "nominal":
        return int(summary.get("runtime_fault_triggered_count", 0) or 0) == 0
    return (
        _is_pass(summary.get("effect_execution_status"))
        and int(summary.get("runtime_fault_triggered_count", 0) or 0)
        + int(summary.get("degradation_effect_count", 0) or 0)
        > 0
        and str(campaign_result.get("status", "")).upper() == "PASS"
    )


@dataclass(frozen=True)
class TrustedRunManifest:
    schema_version: str
    run_id: str
    case_id: str
    split: str
    variant: str
    scenario_id: str
    difficulty: str
    category: str
    source_version: str
    basilisk_version: str
    python_version: str
    source_summary_path: str
    source_summary_sha256: str
    source_manifest_path: str
    source_manifest_sha256: str
    internal_trace_path: str
    internal_trace_sha256: str
    public_trace_path: str
    public_trace_sha256: str
    gate_vector: dict[str, str]
    full_gate_status: str
    effect_evidence_status: str
    campaign_claim_status: str
    public_trace_leakage: bool
    direct_label_columns: list[str]
    truth_proxy_columns: list[str]
    source_evidence_level: str
    acceptance_level: str
    allowed_uses: list[str]
    classification_reasons: list[str]
    source_archive_sha256: str
    dataset_archive_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrustedDatasetAudit:
    schema_version: str
    status: str
    dataset_root: str
    sat_agent_root: str
    dataset_archive_sha256: str
    sat_agent_archive_sha256: str
    dataset_tree_sha256: str
    sat_agent_tree_sha256: str
    total_run_count: int
    acceptance_valid_count: int
    effect_valid_research_only_count: int
    invalid_count: int
    source_effect_valid_research_only_count: int
    source_invalid_count: int
    acceptance_valid_pair_count: int
    public_trace_count: int
    public_trace_leakage_count: int
    leakage_columns: list[str]
    source_status_counts: dict[str, dict[str, int]]
    campaign_claim_pass_count: int
    campaign_claim_fail_count: int
    false_acceptance_count: int
    formal_use_violation_count: int
    reproduction_ready: bool
    reproduction_checks: dict[str, Any]
    blockers: list[str]
    warnings: list[str]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _campaign_results(campaign_root: Path) -> dict[str, dict[str, Any]]:
    summary_path = campaign_root / "campaign_summary.json"
    if not summary_path.is_file():
        return {}
    payload = _read_json(summary_path)
    results = payload.get("results", [])
    return {
        str(row.get("case_id")): dict(row)
        for row in results
        if isinstance(row, Mapping) and row.get("case_id")
    }


def _status_counts(rows: Sequence[TrustedRunManifest]) -> dict[str, dict[str, int]]:
    keys = ["status", "execution_status", "physics_status", "data_conservation_status", "energy_conservation_status"]
    output: dict[str, dict[str, int]] = {}
    for key in keys:
        counts: dict[str, int] = {}
        for row in rows:
            value = row.gate_vector.get(key, "MISSING")
            counts[value] = counts.get(value, 0) + 1
        output[key] = dict(sorted(counts.items()))
    return output


def audit_round6_dataset(
    dataset_root: Path,
    sat_agent_root: Path,
    *,
    dataset_archive_sha256: str = "",
    sat_agent_archive_sha256: str = "",
) -> tuple[TrustedDatasetAudit, list[TrustedRunManifest], list[dict[str, Any]]]:
    """Audit the supplied Round6 package without trusting its campaign verdicts."""
    dataset_root = dataset_root.resolve()
    sat_agent_root = sat_agent_root.resolve()
    campaigns_root = dataset_root / "basilisk_campaigns"
    if not campaigns_root.is_dir():
        raise FileNotFoundError(f"Missing basilisk_campaigns: {campaigns_root}")

    rows: list[TrustedRunManifest] = []
    leakage_records: list[dict[str, Any]] = []
    for split in ("development", "holdout", "final"):
        split_root = campaigns_root / split
        results = _campaign_results(split_root)
        raw_root = split_root / "raw"
        for case_root in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            case_id = case_root.name
            campaign_result = results.get(case_id, {})
            for variant in ("nominal", "effect"):
                run_root = case_root / variant
                summary_path = run_root / "summary.json"
                manifest_path = run_root / "manifest.json"
                internal_trace = run_root / "trace.csv"
                public_trace = case_root / f"{variant}_trace.csv"
                summary = _read_json(summary_path)
                source_manifest = _read_json(manifest_path)
                platform = source_manifest.get("platform", {}) if isinstance(source_manifest.get("platform"), Mapping) else {}
                leak = scan_public_csv(public_trace)
                leakage_records.append({
                    "run_id": f"{case_id}-{variant}",
                    "case_id": case_id,
                    "split": split,
                    "variant": variant,
                    **leak,
                })

                gate_vector = _summary_gate_vector(summary)
                full_gate_pass = all(_is_pass(value) for value in gate_vector.values())
                effect_valid = _effect_evidence_valid(summary, variant, campaign_result)
                reasons: list[str] = []
                for gate_name, value in gate_vector.items():
                    if not _is_pass(value):
                        reasons.append(f"{gate_name}={value}")
                if leak["leakage"]:
                    reasons.append("PUBLIC_TRACE_LABEL_OR_TRUTH_PROXY_LEAKAGE")
                if not effect_valid:
                    reasons.append("EFFECT_OR_NOMINAL_EVIDENCE_NOT_VALID")

                if full_gate_pass and effect_valid:
                    source_level = AcceptanceLevel.ACCEPTANCE_VALID
                elif effect_valid:
                    source_level = AcceptanceLevel.EFFECT_VALID_RESEARCH_ONLY
                else:
                    source_level = AcceptanceLevel.INVALID

                if source_level is AcceptanceLevel.ACCEPTANCE_VALID and not leak["leakage"]:
                    level = AcceptanceLevel.ACCEPTANCE_VALID
                    uses = ACCEPTANCE_USES
                elif source_level is AcceptanceLevel.EFFECT_VALID_RESEARCH_ONLY and not leak["leakage"]:
                    level = AcceptanceLevel.EFFECT_VALID_RESEARCH_ONLY
                    uses = RESEARCH_USES
                else:
                    level = AcceptanceLevel.INVALID
                    uses = INVALID_USES

                rows.append(TrustedRunManifest(
                    schema_version="astrograph-trusted-case-manifest.v1",
                    run_id=f"{case_id}-{variant}",
                    case_id=case_id,
                    split=split,
                    variant=variant,
                    scenario_id=str(campaign_result.get("scenario_id", "UNKNOWN")),
                    difficulty=str(campaign_result.get("difficulty", "UNKNOWN")),
                    category=str(campaign_result.get("category", "UNKNOWN")),
                    source_version=str(platform.get("platform_version", "UNKNOWN")),
                    basilisk_version=str(platform.get("bsk_version", "UNKNOWN")),
                    python_version=str(platform.get("python_version", "UNKNOWN")),
                    source_summary_path=str(summary_path.relative_to(dataset_root)),
                    source_summary_sha256=sha256_file(summary_path),
                    source_manifest_path=str(manifest_path.relative_to(dataset_root)),
                    source_manifest_sha256=sha256_file(manifest_path),
                    internal_trace_path=str(internal_trace.relative_to(dataset_root)),
                    internal_trace_sha256=sha256_file(internal_trace) if internal_trace.is_file() else "",
                    public_trace_path=str(public_trace.relative_to(dataset_root)),
                    public_trace_sha256=sha256_file(public_trace) if public_trace.is_file() else "",
                    gate_vector=gate_vector,
                    full_gate_status="PASS" if full_gate_pass else "FAIL",
                    effect_evidence_status="PASS" if effect_valid else "FAIL",
                    campaign_claim_status=str(campaign_result.get("status", "MISSING")),
                    public_trace_leakage=bool(leak["leakage"]),
                    direct_label_columns=list(leak["direct_label_columns"]),
                    truth_proxy_columns=list(leak["truth_proxy_columns"]),
                    source_evidence_level=source_level.value,
                    acceptance_level=level.value,
                    allowed_uses=list(uses),
                    classification_reasons=reasons,
                    source_archive_sha256=sat_agent_archive_sha256,
                    dataset_archive_sha256=dataset_archive_sha256,
                ))

    row_by_pair: dict[str, dict[str, TrustedRunManifest]] = {}
    for row in rows:
        row_by_pair.setdefault(row.case_id, {})[row.variant] = row
    valid_pairs = sum(
        1
        for pair in row_by_pair.values()
        if set(pair) == {"nominal", "effect"}
        and all(item.acceptance_level == AcceptanceLevel.ACCEPTANCE_VALID.value for item in pair.values())
    )

    leakage_columns = sorted({
        column
        for record in leakage_records
        for column in record["direct_label_columns"] + record["truth_proxy_columns"]
    })
    acceptance_count = sum(row.acceptance_level == AcceptanceLevel.ACCEPTANCE_VALID.value for row in rows)
    research_count = sum(row.acceptance_level == AcceptanceLevel.EFFECT_VALID_RESEARCH_ONLY.value for row in rows)
    invalid_count = sum(row.acceptance_level == AcceptanceLevel.INVALID.value for row in rows)
    source_research_count = sum(row.source_evidence_level == AcceptanceLevel.EFFECT_VALID_RESEARCH_ONLY.value for row in rows)
    source_invalid_count = sum(row.source_evidence_level == AcceptanceLevel.INVALID.value for row in rows)
    false_acceptance_count = sum(
        row.acceptance_level == AcceptanceLevel.ACCEPTANCE_VALID.value and row.full_gate_status != "PASS"
        for row in rows
    )
    formal_use_violations = sum(
        row.acceptance_level != AcceptanceLevel.ACCEPTANCE_VALID.value
        and bool(FORMAL_USES.intersection(row.allowed_uses))
        for row in rows
    )

    generation_script = dataset_root / "reproduction" / "generate_coupled_fault_campaign.py"
    run_matrix_script = sat_agent_root / "scripts" / "run_basilisk_fault_matrix.py"
    basilisk_available = importlib.util.find_spec("Basilisk") is not None
    exact_generation_script = False
    freeze_manifest = dataset_root / "evidence" / "round6_real_coupling_freeze_manifest.json"
    frozen_generation_sha = ""
    if freeze_manifest.is_file():
        freeze = _read_json(freeze_manifest)
        for record in freeze.get("files", []):
            if isinstance(record, Mapping) and str(record.get("path", "")).endswith("generate_coupled_fault_campaign.py"):
                frozen_generation_sha = str(record.get("sha256", ""))
                break
        if generation_script.is_file() and frozen_generation_sha:
            exact_generation_script = sha256_file(generation_script) == frozen_generation_sha

    reproduction_checks = {
        "dataset_generation_script_present": generation_script.is_file(),
        "dataset_generation_script_sha256": sha256_file(generation_script) if generation_script.is_file() else "",
        "frozen_generation_script_sha256": frozen_generation_sha,
        "dataset_generation_script_exact_match": exact_generation_script,
        "sat_agent_run_matrix_script_present": run_matrix_script.is_file(),
        "basilisk_runtime_available": basilisk_available,
        "sat_agent_archive_sha256_present": bool(sat_agent_archive_sha256),
        "dataset_archive_sha256_present": bool(dataset_archive_sha256),
    }
    reproduction_ready = all((
        reproduction_checks["dataset_generation_script_present"],
        reproduction_checks["dataset_generation_script_exact_match"],
        reproduction_checks["sat_agent_run_matrix_script_present"],
        reproduction_checks["basilisk_runtime_available"],
        reproduction_checks["sat_agent_archive_sha256_present"],
        reproduction_checks["dataset_archive_sha256_present"],
    ))

    blockers: list[str] = []
    warnings: list[str] = []
    leakage_count = sum(record["leakage"] for record in leakage_records)
    if leakage_count:
        blockers.append(f"PUBLIC_TRACE_LABEL_LEAKAGE:{leakage_count}/{len(leakage_records)}")
    if false_acceptance_count:
        blockers.append(f"FALSE_ACCEPTANCE:{false_acceptance_count}")
    if not reproduction_ready:
        blockers.append("ROUND6_REPRODUCTION_NOT_READY")
    if acceptance_count == 0:
        warnings.append("NO_ACCEPTANCE_VALID_RUNS")
    if research_count:
        warnings.append(f"RESEARCH_ONLY_RUNS:{research_count}")
    if invalid_count:
        warnings.append(f"INVALID_RUNS:{invalid_count}")
    warnings.append("CAMPAIGN_PASS_IS_RECORDED_SEPARATELY_FROM_COMPLETE_SIMULATION_GATES")

    audit = TrustedDatasetAudit(
        schema_version="astrograph-trusted-dataset-audit.v1",
        status="BLOCKED" if blockers else "PASS",
        dataset_root=str(dataset_root),
        sat_agent_root=str(sat_agent_root),
        dataset_archive_sha256=dataset_archive_sha256,
        sat_agent_archive_sha256=sat_agent_archive_sha256,
        dataset_tree_sha256=sha256_tree(dataset_root),
        sat_agent_tree_sha256=sha256_tree(sat_agent_root),
        total_run_count=len(rows),
        acceptance_valid_count=acceptance_count,
        effect_valid_research_only_count=research_count,
        invalid_count=invalid_count,
        source_effect_valid_research_only_count=source_research_count,
        source_invalid_count=source_invalid_count,
        acceptance_valid_pair_count=valid_pairs,
        public_trace_count=len(leakage_records),
        public_trace_leakage_count=leakage_count,
        leakage_columns=leakage_columns,
        source_status_counts=_status_counts(rows),
        campaign_claim_pass_count=sum(row.campaign_claim_status.upper() == "PASS" for row in rows),
        campaign_claim_fail_count=sum(row.campaign_claim_status.upper() != "PASS" for row in rows),
        false_acceptance_count=false_acceptance_count,
        formal_use_violation_count=formal_use_violations,
        reproduction_ready=reproduction_ready,
        reproduction_checks=reproduction_checks,
        blockers=blockers,
        warnings=warnings,
        claim_boundary=(
            "Classifies the supplied laboratory data for provenance, complete-simulation gates, "
            "effect evidence, public-trace leakage and permitted use. It does not establish "
            "diagnosis accuracy, causal validity, flight qualification or safety qualification."
        ),
    )
    return audit, rows, leakage_records


def write_trusted_audit_outputs(
    audit: TrustedDatasetAudit,
    rows: Sequence[TrustedRunManifest],
    leakage_records: Sequence[Mapping[str, Any]],
    *,
    audit_path: Path,
    manifest_jsonl_path: Path,
    matrix_csv_path: Path,
    leakage_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[3]

    def artifacts_for(path: Path) -> ArtifactRoot:
        return ArtifactRoot(path.parent, source_root=source_root)

    artifacts_for(audit_path).write_text(
        audit_path.name,
        json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n",
    )
    artifacts_for(manifest_jsonl_path).write_text(
        manifest_jsonl_path.name,
        "".join(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    matrix_fields = [
        "run_id", "case_id", "split", "variant", "scenario_id", "full_gate_status",
        "effect_evidence_status", "campaign_claim_status", "public_trace_leakage",
        "source_evidence_level", "acceptance_level", "allowed_uses", "classification_reasons",
    ]

    def write_matrix(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=matrix_fields)
            writer.writeheader()
            for row in rows:
                payload = row.to_dict()
                writer.writerow({
                    key: json.dumps(payload[key], ensure_ascii=False)
                    if isinstance(payload[key], (list, dict)) else payload[key]
                    for key in matrix_fields
                })

    artifacts_for(matrix_csv_path).write_generated(matrix_csv_path.name, write_matrix)
    artifacts_for(leakage_path).write_text(
        leakage_path.name,
        json.dumps({
            "schema_version": "astrograph-public-trace-leakage-audit.v1",
            "trace_count": len(leakage_records),
            "leakage_count": sum(bool(record.get("leakage")) for record in leakage_records),
            "records": list(leakage_records),
        }, ensure_ascii=False, indent=2) + "\n",
    )


def audit_trusted_campaign(
    campaign_root: Path,
    sat_agent_root: Path,
    *,
    dataset_archive_sha256: str = "",
    sat_agent_archive_sha256: str = "",
) -> tuple[TrustedDatasetAudit, list[TrustedRunManifest], list[dict[str, Any]]]:
    """Audit a regenerated ``sat-agent-trusted-campaign.v1`` package.

    Unlike :func:`audit_round6_dataset`, this function consumes the new paired
    campaign layout.  It trusts neither the campaign-level PASS nor a derived
    run summary unless the source evidence, pair validation, public trace and
    frozen Basilisk identity are all present and internally consistent.
    """
    campaign_root = campaign_root.resolve()
    sat_agent_root = sat_agent_root.resolve()
    manifest_path = campaign_root / "campaign_manifest.json"
    runtime_path = campaign_root / "runtime_identity.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing campaign_manifest.json: {manifest_path}")
    campaign_manifest = _read_json(manifest_path)
    runtime_identity = _read_json(runtime_path) if runtime_path.is_file() else {}
    rows: list[TrustedRunManifest] = []
    leakage_records: list[dict[str, Any]] = []

    cases_root = campaign_root / "cases"
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Missing cases directory: {cases_root}")
    case_claims = {
        str(row.get("case_id")): dict(row)
        for row in campaign_manifest.get("case_results", [])
        if isinstance(row, Mapping) and row.get("case_id")
    }
    for case_root in sorted(p for p in cases_root.iterdir() if p.is_dir()):
        pair_validation_path = case_root / "pair_validation.json"
        pair_validation = _read_json(pair_validation_path) if pair_validation_path.is_file() else {}
        claim = case_claims.get(case_root.name, {})
        for variant in ("nominal", "effect"):
            run_root = case_root / variant
            summary_path = run_root / "summary.json"
            source_summary_path = run_root / "source_summary.json"
            source_manifest_path = run_root / "manifest.json"
            trace_path = run_root / "trace.csv"
            summary = _read_json(summary_path)
            source_summary = _read_json(source_summary_path)
            source_manifest = _read_json(source_manifest_path)
            leak = scan_public_csv(trace_path)
            leakage_records.append({
                "run_id": f"{case_root.name}-{variant}",
                "case_id": case_root.name,
                "split": str(summary.get("split", "UNKNOWN")),
                "variant": variant,
                **leak,
            })
            gate_vector = _summary_gate_vector(summary)
            full_gate_pass = all(_is_pass(value) for value in gate_vector.values())
            source_single_run_pass = (
                _is_pass(source_summary.get("status"))
                and _is_pass(source_summary.get("execution_status"))
                and str(source_summary.get("physics_status", "")).upper() == "PASS_SINGLE_RUN_LIMITED"
                and _is_pass(source_summary.get("data_conservation_status"))
                and _is_pass(source_summary.get("energy_conservation_status"))
                and source_summary.get("basilisk_simbase_used") is True
                and source_summary.get("execute_simulation_used") is True
                and str(source_summary.get("backend", "")).strip().lower()
                in {"basilisk", "whole_spacecraft_basilisk_assembly"}
            )
            effect_valid = (
                variant == "nominal"
                or (
                    _is_pass(summary.get("effect_execution_status"))
                    and _is_pass(source_summary.get("runtime_injection_status"))
                    and int(source_summary.get("runtime_fault_triggered_count", 0) or 0) > 0
                    and int(source_summary.get("runtime_unsupported_count", 0) or 0) == 0
                )
            )
            pair_pass = str(pair_validation.get("status", "")).upper() == "PASS"
            source_hash_matches = str(summary.get("source_summary_sha256", "")) == sha256_file(source_summary_path)
            pair_hash_matches = str(summary.get("pair_validation_sha256", "")) == sha256_file(pair_validation_path)
            public_hash_matches = str(summary.get("public_trace_sha256", "")) == sha256_file(trace_path)
            reasons: list[str] = []
            if not full_gate_pass:
                reasons.append("TRUSTED_SUMMARY_GATE_NOT_PASS")
            if not source_single_run_pass:
                reasons.append("SOURCE_SINGLE_RUN_GATE_NOT_PASS")
            if not effect_valid:
                reasons.append("EFFECT_EVIDENCE_NOT_PASS")
            if not pair_pass:
                reasons.append("PAIR_VALIDATION_NOT_PASS")
            if not source_hash_matches:
                reasons.append("SOURCE_SUMMARY_HASH_MISMATCH")
            if not pair_hash_matches:
                reasons.append("PAIR_VALIDATION_HASH_MISMATCH")
            if not public_hash_matches:
                reasons.append("PUBLIC_TRACE_HASH_MISMATCH")
            if leak["leakage"]:
                reasons.append("PUBLIC_TRACE_LABEL_OR_TRUTH_PROXY_LEAKAGE")
            accepted = not reasons and str(summary.get("acceptance_level")) == AcceptanceLevel.ACCEPTANCE_VALID.value
            level = AcceptanceLevel.ACCEPTANCE_VALID if accepted else AcceptanceLevel.INVALID
            uses = ACCEPTANCE_USES if accepted else INVALID_USES
            platform_payload = source_manifest.get("platform", {}) if isinstance(source_manifest.get("platform"), Mapping) else {}
            rows.append(TrustedRunManifest(
                schema_version="astrograph-trusted-case-manifest.v1",
                run_id=f"{case_root.name}-{variant}",
                case_id=case_root.name,
                split=str(summary.get("split", "UNKNOWN")),
                variant=variant,
                scenario_id=str(summary.get("scenario_id", "UNKNOWN")),
                difficulty="trusted_minimum",
                category="paired_counterfactual",
                source_version=str(source_manifest.get("source_version", "UNKNOWN")),
                basilisk_version=str(summary.get("basilisk_version", platform_payload.get("version", "UNKNOWN"))),
                python_version=str(platform_payload.get("python_version", "UNKNOWN")),
                source_summary_path=str(source_summary_path.relative_to(campaign_root)),
                source_summary_sha256=sha256_file(source_summary_path),
                source_manifest_path=str(source_manifest_path.relative_to(campaign_root)),
                source_manifest_sha256=sha256_file(source_manifest_path),
                internal_trace_path=str(trace_path.relative_to(campaign_root)),
                internal_trace_sha256=sha256_file(trace_path),
                public_trace_path=str(trace_path.relative_to(campaign_root)),
                public_trace_sha256=sha256_file(trace_path),
                gate_vector=gate_vector,
                full_gate_status="PASS" if full_gate_pass else "FAIL",
                effect_evidence_status="PASS" if effect_valid and source_single_run_pass and pair_pass else "FAIL",
                campaign_claim_status=str(claim.get("status", "MISSING")),
                public_trace_leakage=bool(leak["leakage"]),
                direct_label_columns=list(leak["direct_label_columns"]),
                truth_proxy_columns=list(leak["truth_proxy_columns"]),
                source_evidence_level=level.value,
                acceptance_level=level.value,
                allowed_uses=list(uses),
                classification_reasons=reasons,
                source_archive_sha256=sat_agent_archive_sha256,
                dataset_archive_sha256=dataset_archive_sha256,
            ))

    by_pair: dict[str, dict[str, TrustedRunManifest]] = {}
    for row in rows:
        by_pair.setdefault(row.case_id, {})[row.variant] = row
    valid_pairs = sum(
        set(pair) == {"nominal", "effect"}
        and all(row.acceptance_level == AcceptanceLevel.ACCEPTANCE_VALID.value for row in pair.values())
        for pair in by_pair.values()
    )
    leakage_columns = sorted({
        column
        for record in leakage_records
        for column in record["direct_label_columns"] + record["truth_proxy_columns"]
    })
    acceptance_count = sum(row.acceptance_level == AcceptanceLevel.ACCEPTANCE_VALID.value for row in rows)
    invalid_count = len(rows) - acceptance_count
    false_acceptance_count = sum(
        row.acceptance_level == AcceptanceLevel.ACCEPTANCE_VALID.value and row.full_gate_status != "PASS"
        for row in rows
    )
    formal_use_violations = sum(
        row.acceptance_level != AcceptanceLevel.ACCEPTANCE_VALID.value
        and bool(FORMAL_USES.intersection(row.allowed_uses))
        for row in rows
    )
    generation_script = sat_agent_root / "scripts" / "generate_coupled_fault_campaign.py"
    run_matrix_script = sat_agent_root / "scripts" / "run_basilisk_fault_matrix.py"
    install_script = sat_agent_root / "scripts" / "install_basilisk_runtime.py"
    exporter_script = sat_agent_root / "scripts" / "export_trusted_run_evidence.py"
    basilisk_available = importlib.util.find_spec("Basilisk") is not None
    expected_version = "2.11.0"
    reproduction_checks = {
        "campaign_manifest_status": campaign_manifest.get("status"),
        "campaign_manifest_total_matches": int(campaign_manifest.get("total_run_count", -1)) == len(rows),
        "campaign_manifest_acceptance_matches": int(campaign_manifest.get("acceptance_valid_count", -1)) == acceptance_count,
        "runtime_identity_present": runtime_path.is_file(),
        "runtime_identity_status": runtime_identity.get("status"),
        "runtime_identity_version": runtime_identity.get("version"),
        "runtime_identity_version_exact": runtime_identity.get("version") == expected_version,
        "basilisk_runtime_available": basilisk_available,
        "dataset_generation_script_present": generation_script.is_file(),
        "sat_agent_run_matrix_script_present": run_matrix_script.is_file(),
        "runtime_install_script_present": install_script.is_file(),
        "trusted_evidence_exporter_present": exporter_script.is_file(),
        "sat_agent_archive_sha256_present": bool(sat_agent_archive_sha256),
        "dataset_archive_sha256_present": bool(dataset_archive_sha256),
    }
    reproduction_ready = all((
        campaign_manifest.get("status") == "PASS",
        reproduction_checks["campaign_manifest_total_matches"],
        reproduction_checks["campaign_manifest_acceptance_matches"],
        reproduction_checks["runtime_identity_present"],
        reproduction_checks["runtime_identity_status"] == "PASS",
        reproduction_checks["runtime_identity_version_exact"],
        reproduction_checks["basilisk_runtime_available"],
        reproduction_checks["dataset_generation_script_present"],
        reproduction_checks["sat_agent_run_matrix_script_present"],
        reproduction_checks["runtime_install_script_present"],
        reproduction_checks["trusted_evidence_exporter_present"],
        reproduction_checks["sat_agent_archive_sha256_present"],
        reproduction_checks["dataset_archive_sha256_present"],
    ))
    leakage_count = sum(bool(record["leakage"]) for record in leakage_records)
    blockers: list[str] = []
    warnings: list[str] = []
    if leakage_count:
        blockers.append(f"PUBLIC_TRACE_LABEL_LEAKAGE:{leakage_count}/{len(leakage_records)}")
    if false_acceptance_count:
        blockers.append(f"FALSE_ACCEPTANCE:{false_acceptance_count}")
    if acceptance_count < 12:
        blockers.append(f"ACCEPTANCE_VALID_RUNS_BELOW_MINIMUM:{acceptance_count}/12")
    if valid_pairs < 3:
        blockers.append(f"ACCEPTANCE_VALID_PAIRS_BELOW_MINIMUM:{valid_pairs}/3")
    if not reproduction_ready:
        blockers.append("TRUSTED_CAMPAIGN_REPRODUCTION_NOT_READY")
    if invalid_count:
        warnings.append(f"INVALID_RUNS:{invalid_count}")
    audit = TrustedDatasetAudit(
        schema_version="astrograph-trusted-dataset-audit.v2",
        status="BLOCKED" if blockers else "PASS",
        dataset_root=str(campaign_root),
        sat_agent_root=str(sat_agent_root),
        dataset_archive_sha256=dataset_archive_sha256,
        sat_agent_archive_sha256=sat_agent_archive_sha256,
        dataset_tree_sha256=sha256_tree(campaign_root),
        sat_agent_tree_sha256=sha256_tree(sat_agent_root),
        total_run_count=len(rows),
        acceptance_valid_count=acceptance_count,
        effect_valid_research_only_count=0,
        invalid_count=invalid_count,
        source_effect_valid_research_only_count=0,
        source_invalid_count=invalid_count,
        acceptance_valid_pair_count=valid_pairs,
        public_trace_count=len(leakage_records),
        public_trace_leakage_count=leakage_count,
        leakage_columns=leakage_columns,
        source_status_counts=_status_counts(rows),
        campaign_claim_pass_count=sum(row.campaign_claim_status.upper() == "PASS" for row in rows),
        campaign_claim_fail_count=sum(row.campaign_claim_status.upper() != "PASS" for row in rows),
        false_acceptance_count=false_acceptance_count,
        formal_use_violation_count=formal_use_violations,
        reproduction_ready=reproduction_ready,
        reproduction_checks=reproduction_checks,
        blockers=blockers,
        warnings=warnings,
        claim_boundary=(
            "Validates the regenerated laboratory Basilisk campaign, paired source evidence, "
            "public telemetry isolation and reproducibility. It does not establish diagnosis "
            "accuracy, flight qualification or hardware safety."
        ),
    )
    return audit, rows, leakage_records
