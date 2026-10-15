from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_maintained_documentation_and_release_paths_exist():
    # Maintained documentation surface.  (DIAGNOSIS.md / DATA_FORMAT.md /
    # KNOWLEDGE_SCHEMA.md / LIMITATIONS.md were removed from the root during
    # the 2026-08-04 doc cleanup; authoritative status lives in docs/history/root/DEV_GOAL.md
    # + docs/history/root/PROGRESS_LEDGER.md, which are not asserted here because they are
    # untracked-in-git work-in-progress artifacts.)
    for name in ["README.md", "ARCHITECTURE.md", "TESTING.md", "CHANGELOG.md", "release.json"]:
        assert (ROOT / name).is_file()
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    for path in release["production_paths"].values():
        if "." in path and not path.endswith((".json", ".joblib")):
            continue
        assert (ROOT / path).exists(), path


def test_distribution_has_no_runtime_or_history_directories():
    for name in ["reports", "outputs", "local_acceptance_data", "external_validation", "memory", "simulation"]:
        assert not (ROOT / name).exists()


def test_operational_scripts_are_semantically_named():
    scripts = [p.name for p in (ROOT / "scripts").glob("*.py")]
    assert scripts
    assert all(not name[0].isdigit() for name in scripts)
    assert all("round" not in name.lower() and "phase" not in name.lower() for name in scripts)


def test_source_and_scripts_do_not_embed_absolute_runtime_paths():
    prohibited = ("/mnt/data", "C:\\")
    for base in [ROOT / "src", ROOT / "scripts"]:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert not any(token in text for token in prohibited), path
