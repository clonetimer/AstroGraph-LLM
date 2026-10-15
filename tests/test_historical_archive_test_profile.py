from __future__ import annotations

from pathlib import Path

from astrograph.governance.test_profiles import (
    archive_requirements,
    classify_skip_reason,
    evaluate_skip_budget,
    load_archive_test_manifest,
    load_test_execution_profiles,
    resolve_test_profile,
    validate_archival_replay_ready,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_only_delivery_auto_selects_source_only_profile() -> None:
    assert resolve_test_profile(ROOT, "auto") == "source-only"
    assert resolve_test_profile(ROOT, "source-core") == "source-only"
    assert resolve_test_profile(ROOT, "historical-replay") == "archival-replay"


def test_execution_profiles_are_machine_readable_and_unknown_skips_fail_closed() -> None:
    profiles = load_test_execution_profiles(ROOT)
    assert set(profiles["profiles"]) == {
        "source-only", "archival-replay", "external-integration", "platform-e2e",
    }
    receipt = evaluate_skip_budget(
        ROOT, "source-only", ["unregistered environmental skip"], collected_items=3730,
    )
    assert receipt["status"] == "FAIL"
    assert receipt["skip_counts_by_category"] == {"unknown": 1}


def test_skip_reason_classifier_separates_release_integration_and_platform_gaps() -> None:
    assert classify_skip_reason(
        "source-only profile: frozen archive evidence not bundled: x.json"
    ) == "historical_archive_missing"
    assert classify_skip_reason(
        "optional external DOR runtime is not installed"
    ) == "external_runtime_missing"
    assert classify_skip_reason(
        "optional ADCS v5.2.2 external dataset is not configured"
    ) == "external_dataset_missing"
    assert classify_skip_reason(
        "could not import 'playwright.sync_api': No module named 'playwright'"
    ) == "browser_dependency_missing"
    assert classify_skip_reason("POSIX interrupt unit test") == "platform_not_applicable"


def test_historical_archive_manifest_is_precise_and_nonempty() -> None:
    manifest = load_archive_test_manifest(ROOT)
    requirements = archive_requirements(ROOT)
    assert manifest["default_resolution"] == "AUTO_FROM_DELIVERY_PROFILE"
    assert len(requirements) >= 40
    assert len({item.nodeid for item in requirements}) == len(requirements)
    assert all(item.nodeid.startswith("tests/") for item in requirements)
    assert all(item.required_paths for item in requirements)


def test_source_only_package_is_explicitly_not_archival_replay_ready() -> None:
    readiness = validate_archival_replay_ready(ROOT)
    assert readiness["status"] == "BLOCKED_MISSING_ARCHIVE"
    assert readiness["missing_path_count"] > 0
    assert any(path.startswith("research/") for path in readiness["missing_paths"])
