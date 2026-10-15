from __future__ import annotations

from pathlib import Path

from astrograph.architecture import (
    ArchitectureBoundaryPolicy,
    CompatibilityPath,
    SpecializationPattern,
    load_boundary_policy,
    scan_architecture_boundaries,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/architecture/module_boundaries.json"


def test_current_generic_core_has_no_specialization_and_all_legacy_hits_are_declared() -> None:
    report = scan_architecture_boundaries(ROOT, load_boundary_policy(POLICY))
    assert report.status == "PASS", report.model_dump(mode="json")
    assert report.generic_match_count == 0
    assert report.compatibility_match_count > 0


def tiny_policy(*, generic: list[str], compatibility: list[str] | None = None) -> ArchitectureBoundaryPolicy:
    return ArchitectureBoundaryPolicy(
        schema_version="astrograph-architecture-boundary-policy.v1",
        scan_roots=["src"],
        generic_paths=generic,
        compatibility_allowlist=[
            CompatibilityPath(path=path, rationale="test compatibility boundary")
            for path in (compatibility or [])
        ],
        patterns=[
            SpecializationPattern(
                pattern_id="source_system",
                expression=r"(?i)\bsat[-_ ]?(?:sim|agent)(?:\b|_)",
                description="source identity",
            )
        ],
    )


def test_generic_specialization_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "src/core.py"
    path.parent.mkdir()
    path.write_text("sat_agent_mode = True\n", encoding="utf-8")
    report = scan_architecture_boundaries(tmp_path, tiny_policy(generic=["src/core.py"]))
    assert report.status == "FAIL"
    assert {item.code for item in report.issues} == {"GENERIC_CORE_SPECIALIZATION"}


def test_undeclared_specialization_fails_but_explicit_compatibility_passes(tmp_path: Path) -> None:
    path = tmp_path / "src/adapter.py"
    path.parent.mkdir()
    path.write_text('SOURCE = "sat_sim"\n', encoding="utf-8")
    report = scan_architecture_boundaries(tmp_path, tiny_policy(generic=[]))
    assert {item.code for item in report.issues} == {"UNDECLARED_SPECIALIZATION"}

    report = scan_architecture_boundaries(
        tmp_path,
        tiny_policy(generic=[], compatibility=["src/adapter.py"]),
    )
    assert report.status == "PASS"
    assert report.compatibility_match_count == 1


def test_missing_declared_path_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    report = scan_architecture_boundaries(tmp_path, tiny_policy(generic=["src/missing.py"]))
    assert {item.code for item in report.issues} == {"DECLARED_BOUNDARY_PATH_MISSING"}
