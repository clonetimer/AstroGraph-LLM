from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPHEMERAL_PARTS = {".git", ".claude", ".pytest_cache", "__pycache__", ".tmp_tests", "worktrees", ".venv", "astrograph_core.egg-info"}
EXCLUDED_PREFIXES = {"scripts/integration/out", "runs"}  # Local jobs and acceptance reports are not distributable source.


def _distribution_files() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(
            [
                "git", "-C", str(ROOT), "ls-files", "--cached", "--others",
                "--exclude-standard", "-z",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            candidates = [
                ROOT / item.decode("utf-8", errors="strict")
                for item in result.stdout.split(b"\0")
                if item
            ]
        else:
            candidates = ROOT.rglob("*")
    else:
        candidates = ROOT.rglob("*")
    return sorted(
        path for path in candidates
        if path.is_file()
        and not EPHEMERAL_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".pyo"}
        and not any(
            path.relative_to(ROOT).as_posix() == prefix
            or path.relative_to(ROOT).as_posix().startswith(prefix + "/")
            for prefix in EXCLUDED_PREFIXES
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_metadata_and_claim_manifest_are_current() -> None:
    release = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    stable = str(release.get("status", "")).startswith("STABLE_SOFTWARE_RELEASE")
    claims_name = "stable_release.json" if stable else "engineering_candidate.json"
    claims = json.loads((ROOT / "configs/release_claims" / claims_name).read_text(encoding="utf-8"))
    assert release["version"] == pyproject["project"]["version"]
    suffix = "stable" if stable else "engineering-candidate"
    assert claims["release_id"] == f"AstroGraph-v{release['version']}-{suffix}"
    for claim in claims["claims"]:
        for source in claim["source_artifacts"]:
            assert (ROOT / source).is_file(), source


def test_reaction_wheel_sources_use_semantic_existing_paths() -> None:
    assert (ROOT / "configs/reaction_wheel_diagnosis_rules.json").is_file()
    assert not (ROOT / "configs/d3_rw_diagnosis_rules.json").exists()
    catalog = json.loads((ROOT / "configs/reaction_wheel_evidence.json").read_text(encoding="utf-8"))
    serialized = json.dumps(catalog)
    assert "src/astrograph/diagnosis/d3_rw.py" not in serialized
    assert "configs/d3_rw_diagnosis_rules.json" not in serialized
    registry = json.loads((ROOT / "benchmarks/knowledge_graph/core/source_registry.json").read_text(encoding="utf-8"))
    for source in registry["sources"]:
        assert (ROOT / source["source_ref"]).is_file(), source


def test_knowledge_source_anchors_match_distribution_files() -> None:
    anchors = json.loads((ROOT / "benchmarks/knowledge_graph/core/source_anchors.json").read_text(encoding="utf-8"))
    for source_ref, expected in anchors["documents"].items():
        path = ROOT / source_ref
        assert path.is_file(), source_ref
        assert _sha256(path) == expected, source_ref


def test_production_metadata_has_no_round_phase_or_dev_suffixes() -> None:
    paths = [
        ROOT / "release.json",
        ROOT / "configs/release_claims/engineering_candidate.json",
        ROOT / "configs/release_claims/stable_release.json",
        ROOT / "reference/models/sat_sim_catalog/model_manifest.json",
        ROOT / "reference/knowledge/sat_sim_fault_catalog.json",
        ROOT / "scripts/legacy/build_sat_sim_fault_dataset.py",
        ROOT / "scripts/legacy/train_sat_sim_catalog_model.py",
        ROOT / "scripts/legacy/train_sat_sim_open_set_model.py",
        ROOT / "scripts/legacy/evaluate_sat_sim_open_set.py",
        ROOT / "src/astrograph/integrations/sat_sim_pipeline.py",
        ROOT / "src/astrograph/agents/diagnosis_planner.py",
    ]
    pattern = re.compile(r"round\d+|phase\d+[a-z]*|\bdev\d+\b", re.IGNORECASE)
    for path in paths:
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_distribution_metadata_is_self_consistent() -> None:
    source_manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    excluded = {"FILELIST.txt", "SHA256SUMS.txt", "SOURCE_MANIFEST.json"}
    files = _distribution_files()
    content = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in files if str(p.relative_to(ROOT)).replace("\\", "/") not in excluded)
    listed = (ROOT / "FILELIST.txt").read_text(encoding="utf-8").splitlines()
    assert listed == content
    assert source_manifest["content_file_count"] == len(content)
    checksum_rows = [line.split("  ", 1) for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()]
    checksum_files = sorted(row[1] for row in checksum_rows)
    expected_files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in files if p.name != "SHA256SUMS.txt")
    assert checksum_files == expected_files
    for expected, relative in checksum_rows:
        assert _sha256(ROOT / relative) == expected
