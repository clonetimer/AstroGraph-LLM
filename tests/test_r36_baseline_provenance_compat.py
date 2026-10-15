"""Reconstructed baseline/provenance compatibility checks for the source-only handoff."""
import json
from pathlib import Path


def test_r36_benchmark_identities_are_release_configuration():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/evaluation/r36_diagnostic_benchmarks.json").read_text())
    assert config["benchmark_ids"] == ["NASA_SMAP_MSL", "ESA_OPS_SAT", "ESA_MISSION1", "ADCS", "SATSIM"]
    assert "MISSING" in config["claim_boundary"]


def test_source_version_remains_baseline_version():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    assert 'version = "0.7.1.post49"' in text
