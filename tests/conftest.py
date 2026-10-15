from __future__ import annotations

from pathlib import Path
import pytest

from astrograph.governance.test_profiles import (
    archive_requirements,
    evaluate_skip_budget,
    missing_archive_paths,
    resolve_test_profile,
    validate_archival_replay_ready,
)

ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_CONFIG: pytest.Config | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--astrograph-test-profile",
        action="store",
        default="auto",
        choices=(
            "auto", "source-only", "source-core", "archival-replay",
            "historical-replay", "external-integration", "platform-e2e",
        ),
        help="Evidence profile: auto resolves from DELIVERY_PROFILE.json.",
    )
    parser.addoption(
        "--astrograph-enforce-profile",
        action="store_true",
        default=False,
        help="Require complete collection and enforce the selected profile skip budget.",
    )


def pytest_configure(config: pytest.Config) -> None:
    global _ACTIVE_CONFIG
    requested = config.getoption("--astrograph-test-profile")
    profile = resolve_test_profile(ROOT, requested)
    config._astrograph_test_profile = profile  # type: ignore[attr-defined]
    config._astrograph_skip_reasons = []  # type: ignore[attr-defined]
    config._astrograph_collected_items = 0  # type: ignore[attr-defined]
    _ACTIVE_CONFIG = config
    config.addinivalue_line(
        "markers",
        "archival_replay: test consumes frozen historical evidence excluded from current-source-only delivery",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    profile = getattr(session.config, "_astrograph_test_profile", "source-only")
    if profile != "archival-replay":
        return
    readiness = validate_archival_replay_ready(ROOT)
    if readiness["status"] != "PASS":
        preview = ", ".join(readiness["missing_paths"][:8])
        if readiness["missing_path_count"] > 8:
            preview += f", ... (+{readiness['missing_path_count'] - 8})"
        raise pytest.UsageError(
            "archival-replay profile requires bundled historical archive evidence; "
            f"missing {readiness['missing_path_count']} path(s): {preview}"
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    profile = getattr(config, "_astrograph_test_profile", "source-only")
    config._astrograph_collected_items = len(items)  # type: ignore[attr-defined]
    requirements = {item.nodeid: item for item in archive_requirements(ROOT)}
    collected = {item.nodeid for item in items}
    collected_files = {nodeid.split("::", 1)[0] for nodeid in collected}
    unknown = sorted(
        nodeid for nodeid in requirements
        if nodeid.split("::", 1)[0] in collected_files and nodeid not in collected
    )
    if unknown:
        raise pytest.UsageError(
            "historical archive test manifest contains stale nodeid(s): " + ", ".join(unknown[:5])
        )
    for item in items:
        requirement = requirements.get(item.nodeid)
        if requirement is None:
            continue
        item.add_marker(pytest.mark.archival_replay)
        if profile in {"source-only", "external-integration", "platform-e2e"}:
            missing = missing_archive_paths(ROOT, requirement)
            if missing:
                item.add_marker(pytest.mark.skip(reason=(
                    "source-only profile: frozen archive evidence not bundled: "
                    + ", ".join(missing)
                )))


def _record_skip(report: pytest.CollectReport | pytest.TestReport) -> None:
    if not report.skipped or _ACTIVE_CONFIG is None:
        return
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        reason = str(longrepr[2])
    else:
        reason = str(longrepr)
    _ACTIVE_CONFIG._astrograph_skip_reasons.append(reason)  # type: ignore[attr-defined]


def pytest_collectreport(report: pytest.CollectReport) -> None:
    _record_skip(report)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _record_skip(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not session.config.getoption("--astrograph-enforce-profile"):
        return
    receipt = evaluate_skip_budget(
        ROOT,
        getattr(session.config, "_astrograph_test_profile", "source-only"),
        getattr(session.config, "_astrograph_skip_reasons", []),
        collected_items=getattr(session.config, "_astrograph_collected_items", 0),
    )
    session.config._astrograph_profile_receipt = receipt  # type: ignore[attr-defined]
    if receipt["status"] != "PASS" and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    receipt = getattr(terminalreporter.config, "_astrograph_profile_receipt", None)
    if receipt is None:
        return
    terminalreporter.section("AstroGraph test profile")
    terminalreporter.write_line(
        f"profile={receipt['profile']} status={receipt['status']} "
        f"collected={receipt['collected_items']} skips={receipt['skip_count']}"
    )
    terminalreporter.write_line(
        "skip categories=" + ", ".join(
            f"{name}:{count}" for name, count in receipt["skip_counts_by_category"].items()
        )
    )
    for violation in receipt["violations"]:
        terminalreporter.write_line("VIOLATION: " + violation, red=True)
