"""Pytest evidence profiles for current-source and archival-replay deliveries.

The profile layer never fabricates missing historical artifacts. It only makes
archival replay dependencies explicit so source-only regression runs do not fail
accidentally on evidence that the delivery profile intentionally excludes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

MANIFEST_RELATIVE_PATH = Path("configs/historical_archive_test_manifest.v1.json")
EXECUTION_PROFILES_RELATIVE_PATH = Path("configs/test_execution_profiles.v1.json")
VALID_PROFILES = {
    "auto", "source-only", "source-core", "archival-replay", "historical-replay",
    "external-integration", "platform-e2e",
}


@dataclass(frozen=True)
class ArchiveTestRequirement:
    nodeid: str
    required_paths: tuple[str, ...]
    rationale: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_archive_test_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / MANIFEST_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"historical archive test manifest missing: {path}")
    data = _load_json(path)
    if data.get("schema_version") != "astrograph-historical-archive-test-manifest.v1":
        raise ValueError("unsupported historical archive test manifest schema")
    rows = data.get("tests")
    if not isinstance(rows, list):
        raise ValueError("historical archive test manifest tests must be a list")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("archive test entry must be an object")
        nodeid = str(row.get("nodeid", "")).strip()
        if not nodeid or nodeid in seen or "::" not in nodeid:
            raise ValueError(f"invalid or duplicate archive test nodeid: {nodeid!r}")
        paths = row.get("required_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"archive test {nodeid} must declare required_paths")
        for raw in paths:
            p = Path(str(raw))
            if p.is_absolute() or ".." in p.parts or not str(raw).strip():
                raise ValueError(f"unsafe archive dependency path for {nodeid}: {raw!r}")
        if not str(row.get("rationale", "")).strip():
            raise ValueError(f"archive test {nodeid} requires rationale")
        seen.add(nodeid)
    return data


def resolve_test_profile(root: str | Path, requested: str = "auto") -> str:
    requested = str(requested or "auto").strip().lower()
    if requested not in VALID_PROFILES:
        raise ValueError(f"unsupported AstroGraph test profile: {requested}")
    if requested != "auto":
        return {
            "source-core": "source-only",
            "historical-replay": "archival-replay",
        }.get(requested, requested)
    root = Path(root).resolve()
    profile_path = root / "DELIVERY_PROFILE.json"
    if not profile_path.is_file():
        return "archival-replay"
    profile = _load_json(profile_path)
    if (
        profile.get("source_kind") == "CURRENT_SOURCE_ONLY"
        and profile.get("historical_records_included") is False
    ):
        return "source-only"
    return "archival-replay"


def load_test_execution_profiles(root: str | Path) -> dict[str, Any]:
    path = Path(root).resolve() / EXECUTION_PROFILES_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"test execution profile config missing: {path}")
    data = _load_json(path)
    if data.get("schema_version") != "astrograph-test-execution-profiles.v1":
        raise ValueError("unsupported test execution profile schema")
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("test execution profiles must be a non-empty object")
    for name, row in profiles.items():
        if not isinstance(row, dict):
            raise ValueError(f"test execution profile {name!r} must be an object")
        if int(row.get("minimum_collected_items", -1)) < 0:
            raise ValueError(f"test execution profile {name!r} has invalid collection floor")
        if int(row.get("maximum_total_skips", -1)) < 0:
            raise ValueError(f"test execution profile {name!r} has invalid skip ceiling")
        budgets = row.get("maximum_skips_by_category")
        if not isinstance(budgets, dict) or "unknown" not in budgets:
            raise ValueError(f"test execution profile {name!r} lacks skip categories")
        if any(int(value) < 0 for value in budgets.values()):
            raise ValueError(f"test execution profile {name!r} has a negative skip budget")
    return data


def classify_skip_reason(reason: str) -> str:
    """Classify an observed pytest skip without treating it as a pass."""
    value = " ".join(str(reason).strip().lower().split())
    if (
        "source-only profile" in value
        or "source-only package" in value
        or "outside the source-only" in value
        or "not included in the source-only" in value
        or "not present in this checkout" in value
    ):
        return "historical_archive_missing"
    if "runtime is not installed" in value or "runtime is not configured" in value:
        return "external_runtime_missing"
    if (
        "external adcs.zip regression material" in value
        or "external reduced telemetry regression material" in value
        or "full-telemetry ui bundle" in value
    ):
        return "ui_regression_data_missing"
    if "dataset is not configured" in value or "external fixture is not configured" in value:
        return "external_dataset_missing"
    if "playwright" in value:
        return "browser_dependency_missing"
    if "langgraph" in value:
        return "agent_dependency_missing"
    if "pdf pressure tools" in value:
        return "pdf_dependency_missing"
    if "symlink" in value or "posix" in value:
        return "platform_not_applicable"
    if "optional source fields" in value or "alternate unit" in value:
        return "scenario_not_applicable"
    return "unknown"


def evaluate_skip_budget(
    root: str | Path,
    profile: str,
    reasons: list[str] | tuple[str, ...],
    *,
    collected_items: int,
) -> dict[str, Any]:
    """Evaluate a complete pytest run against the selected profile budget."""
    from collections import Counter

    resolved = resolve_test_profile(root, profile)
    config = load_test_execution_profiles(root)
    if resolved not in config["profiles"]:
        raise ValueError(f"test execution profile has no budget: {resolved}")
    row = config["profiles"][resolved]
    counts = Counter(classify_skip_reason(reason) for reason in reasons)
    violations: list[str] = []
    minimum = int(row["minimum_collected_items"])
    if collected_items < minimum:
        violations.append(f"collected_items={collected_items} below minimum={minimum}")
    maximum_total = int(row["maximum_total_skips"])
    if len(reasons) > maximum_total:
        violations.append(f"total_skips={len(reasons)} exceeds maximum={maximum_total}")
    for category, maximum in row["maximum_skips_by_category"].items():
        observed = int(counts.get(category, 0))
        if observed > int(maximum):
            violations.append(
                f"skip_category={category} observed={observed} exceeds maximum={int(maximum)}"
            )
    return {
        "schema_version": "astrograph-test-profile-evaluation.v1",
        "profile": resolved,
        "status": "PASS" if not violations else "FAIL",
        "collected_items": collected_items,
        "skip_count": len(reasons),
        "skip_counts_by_category": dict(sorted(counts.items())),
        "violations": violations,
        "claim_boundary": config["claim_boundary"],
    }


def archive_requirements(root: str | Path) -> tuple[ArchiveTestRequirement, ...]:
    data = load_archive_test_manifest(root)
    return tuple(
        ArchiveTestRequirement(
            nodeid=str(row["nodeid"]),
            required_paths=tuple(str(value) for value in row["required_paths"]),
            rationale=str(row["rationale"]),
        )
        for row in data["tests"]
    )


def missing_archive_paths(root: str | Path, requirement: ArchiveTestRequirement) -> tuple[str, ...]:
    root = Path(root).resolve()
    return tuple(
        relative for relative in requirement.required_paths if not (root / relative).exists()
    )


def validate_archival_replay_ready(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    rows = []
    all_missing: set[str] = set()
    for requirement in archive_requirements(root):
        missing = missing_archive_paths(root, requirement)
        rows.append({
            "nodeid": requirement.nodeid,
            "required_paths": list(requirement.required_paths),
            "missing_paths": list(missing),
            "ready": not missing,
        })
        all_missing.update(missing)
    return {
        "schema_version": "astrograph-archival-replay-readiness.v1",
        "status": "PASS" if not all_missing else "BLOCKED_MISSING_ARCHIVE",
        "missing_path_count": len(all_missing),
        "missing_paths": sorted(all_missing),
        "test_count": len(rows),
        "tests": rows,
        "claim_boundary": (
            "PASS only means the archive files required by registered replay tests are present. "
            "It does not qualify runtime algorithms or recreate historical evidence."
        ),
    }


__all__ = [
    "ArchiveTestRequirement",
    "EXECUTION_PROFILES_RELATIVE_PATH",
    "MANIFEST_RELATIVE_PATH",
    "VALID_PROFILES",
    "archive_requirements",
    "classify_skip_reason",
    "evaluate_skip_budget",
    "load_archive_test_manifest",
    "load_test_execution_profiles",
    "missing_archive_paths",
    "resolve_test_profile",
    "validate_archival_replay_ready",
]
