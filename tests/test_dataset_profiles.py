from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from astrograph.dataset_profiles import (
    DatasetProfile,
    DatasetProfileAsset,
    DatasetProfileReference,
    DatasetReferenceScope,
    REQUIRED_PROFILE_ASSET_ROLES,
    DatasetProfileRegistry,
    DatasetProfileRegistryIndex,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/dataset_profiles/registry.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reviewed_registry_resolves_adcs_and_phm_with_verified_assets() -> None:
    registry = DatasetProfileRegistry(index_path=REGISTRY)
    assert registry.list_profile_ids() == ["adcs.v5_2_2", "phm.v6_0"]
    adcs = registry.resolve("adcs.v5_2_2")
    phm = registry.resolve("phm.v6_0")
    assert adcs.asset_paths["runtime_evidence"].name == "runtime_evidence.json"
    assert phm.profile.dataset_id == "whole_spacecraft_phm_dataset_v6_0"
    assert adcs.asset_paths["fault_catalog"].name == "lab_fault_catalog.json"
    assert phm.asset_paths["evidence_calibration_receipt"].name == (
        "evidence_calibration_receipt.json"
    )
    assert set(adcs.receipt.verified_asset_sha256) == set(adcs.profile.assets)
    assert adcs.profile.fusion_policy_profile_id == "birrf.engineering-default.v1"
    assert adcs.protocol_root == ROOT / "protocols/phm_project_a"
    assert adcs.profile.reference_scope.to_dor_manifest() == {
        "kind": "ASSET_LOCAL",
        "asset_id": "simulated-spacecraft.adcs-v5_2_2",
    }
    assert phm.profile.reference_scope.to_dor_manifest() == {
        "kind": "ASSET_LOCAL",
        "asset_id": "simulated-spacecraft.phm-v6_0",
    }
    assert adcs.profile.reference_scope.asset_id != phm.profile.reference_scope.asset_id
    assert adcs.receipt.reference_scope == adcs.profile.reference_scope


def _temporary_registry(tmp_path: Path, *, unsafe_asset_path: bool = False) -> tuple[Path, Path]:
    asset_root = tmp_path / "root"
    config_dir = asset_root / "profiles"
    protocol_root = asset_root / "protocol"
    config_dir.mkdir(parents=True)
    protocol_root.mkdir()
    shared = asset_root / "asset.json"
    shared.write_text("{}\n", encoding="utf-8")
    asset_path = "../escape.json" if unsafe_asset_path else "asset.json"
    asset = DatasetProfileAsset(path=asset_path, sha256=_sha(shared))
    profile = DatasetProfile(
        profile_id="test.profile",
        version="1.0.0",
        dataset_id="test-dataset",
        protocol_root_path="protocol",
        reference_scope=DatasetReferenceScope(
            kind="ASSET_LOCAL",
            identity_basis="SYNTHETIC_DATASET_DOMAIN",
            asset_id="test.synthetic-domain",
            claim_boundary="test-only local reference",
        ),
        assets={role: asset for role in (
            "runtime_evidence",
            "candidate_universe",
            "fault_class_map",
            "diagnosis_hierarchy",
            "fault_ontology_bindings",
            "generic_fault_ontology",
            "component_archetypes",
            "semantic_schema",
            "protocol_contract",
        )},
        claim_boundary="test-only profile",
    )
    profile_path = config_dir / "profile.json"
    profile_path.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    index = DatasetProfileRegistryIndex(
        registry_id="test-registry",
        version="1.0.0",
        asset_root="root",
        profiles=[
            DatasetProfileReference(
                profile_id="test.profile",
                path="profiles/profile.json",
                sha256=_sha(profile_path),
            )
        ],
    )
    index_path = tmp_path / "registry.json"
    index_path.write_text(
        json.dumps(index.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return index_path, shared


def test_registry_fails_closed_on_asset_hash_drift(tmp_path: Path) -> None:
    index_path, shared = _temporary_registry(tmp_path)
    shared.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        DatasetProfileRegistry(index_path=index_path).resolve("test.profile")


def test_registry_rejects_path_traversal(tmp_path: Path) -> None:
    index_path, _ = _temporary_registry(tmp_path, unsafe_asset_path=True)
    with pytest.raises(ValueError, match="safe relative path"):
        DatasetProfileRegistry(index_path=index_path).resolve("test.profile")


def test_registry_fails_closed_on_profile_hash_drift(tmp_path: Path) -> None:
    copied = tmp_path / "profiles"
    shutil.copytree(ROOT / "configs/dataset_profiles", copied)
    index = json.loads((copied / "registry.json").read_text(encoding="utf-8"))
    index["asset_root"] = "."
    index["profiles"][0]["path"] = "adcs_v5_2_2.json"
    (copied / "registry.json").write_text(json.dumps(index), encoding="utf-8")
    (copied / "adcs_v5_2_2.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profile hash mismatch"):
        DatasetProfileRegistry(index_path=copied / "registry.json").resolve("adcs.v5_2_2")


def test_unknown_profile_is_not_guessed() -> None:
    with pytest.raises(KeyError, match="unknown dataset profile"):
        DatasetProfileRegistry(index_path=REGISTRY).resolve("future.dataset")


def test_reference_scope_fails_closed_on_missing_or_conflicting_identity() -> None:
    with pytest.raises(ValueError, match="requires asset_id"):
        DatasetReferenceScope(
            kind="ASSET_LOCAL",
            identity_basis="SYNTHETIC_DATASET_DOMAIN",
            claim_boundary="invalid test scope",
        )
    with pytest.raises(ValueError, match="forbids fleet fields"):
        DatasetReferenceScope(
            kind="ASSET_LOCAL",
            identity_basis="SYNTHETIC_DATASET_DOMAIN",
            asset_id="asset-a",
            fleet_group_id="fleet-a",
            claim_boundary="invalid mixed scope",
        )


def test_fleet_scope_requires_commissioning_receipt_binding() -> None:
    scope = DatasetReferenceScope(
        kind="FLEET_GROUP",
        identity_basis="PHYSICAL_ASSET_REGISTRY",
        fleet_group_id="fleet-a",
        member_asset_ids=("asset-a", "asset-b"),
        commissioning_receipt_asset_role="fleet_commissioning_receipt",
        claim_boundary="test fleet requires independent commissioning",
    )
    placeholder = DatasetProfileAsset(path="placeholder.json", sha256="0" * 64)
    with pytest.raises(ValueError, match="commissioning receipt asset role"):
        DatasetProfile(
            profile_id="fleet.profile",
            version="1.0.0",
            dataset_id="fleet-dataset",
            protocol_root_path="protocol",
            reference_scope=scope,
            assets={role: placeholder for role in REQUIRED_PROFILE_ASSET_ROLES},
            claim_boundary="invalid unbound fleet profile",
        )
