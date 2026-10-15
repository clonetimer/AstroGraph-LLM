from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pytest

from astrograph.diagnosis.sat_sim_catalog import load_catalog_artifact
from astrograph.diagnosis.sat_sim_open_set import load_open_set_artifact


@pytest.mark.parametrize(
    ("loader", "manifest_schema", "artifact_schema"),
    [
        (
            load_catalog_artifact,
            "astrograph-sat-sim-catalog-release.v1",
            "astrograph-sat-sim-catalog-model.v1",
        ),
        (
            load_open_set_artifact,
            "astrograph-sat-sim-open-set-model-manifest.v1",
            "astrograph-sat-sim-open-set-model.v1",
        ),
    ],
)
def test_model_loader_accepts_verified_release_manifest(
    tmp_path: Path,
    loader,
    manifest_schema: str,
    artifact_schema: str,
) -> None:
    artifact = tmp_path / "model.joblib"
    joblib.dump({"schema_version": artifact_schema, "sentinel": "loaded"}, artifact)
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": manifest_schema,
                "model_file": artifact.name,
                "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert loader(manifest)["sentinel"] == "loaded"


def test_model_release_manifest_rejects_hash_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    joblib.dump({"schema_version": "astrograph-sat-sim-catalog-model.v1"}, artifact)
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-sat-sim-catalog-release.v1",
                "model_file": artifact.name,
                "model_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        load_catalog_artifact(manifest)


def test_model_release_manifest_rejects_path_escape(tmp_path: Path) -> None:
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-sat-sim-catalog-release.v1",
                "model_file": "../model.joblib",
                "model_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest directory"):
        load_catalog_artifact(manifest)


def test_model_loader_cache_invalidates_when_artifact_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    joblib.dump({"schema_version": "astrograph-sat-sim-catalog-model.v1", "revision": 1}, artifact)
    manifest = tmp_path / "model_manifest.json"

    def write_manifest() -> None:
        manifest.write_text(json.dumps({
            "schema_version": "astrograph-sat-sim-catalog-release.v1",
            "model_file": artifact.name,
            "model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }), encoding="utf-8")

    write_manifest()
    assert load_catalog_artifact(manifest)["revision"] == 1
    joblib.dump({"schema_version": "astrograph-sat-sim-catalog-model.v1", "revision": 200}, artifact)
    write_manifest()
    assert load_catalog_artifact(manifest)["revision"] == 200


def test_raw_joblib_is_rejected_without_explicit_unsafe_opt_in(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "model.joblib"
    joblib.dump({"schema_version": "astrograph-sat-sim-catalog-model.v1"}, artifact)
    monkeypatch.delenv("ASTROGRAPH_ALLOW_UNMANIFESTED_JOBLIB", raising=False)
    with pytest.raises(ValueError, match="manifest JSON is required"):
        load_catalog_artifact(artifact)
