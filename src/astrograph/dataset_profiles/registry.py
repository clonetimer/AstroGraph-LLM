from __future__ import annotations

from pathlib import Path

from .models import (
    DatasetProfile,
    DatasetProfileReceipt,
    DatasetProfileRegistryIndex,
    ResolvedDatasetProfile,
    sha256_file,
)


def _safe_resolve(root: Path, relative: str, *, expect_directory: bool = False) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"dataset profile path must be a safe relative path: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dataset profile path escapes asset root: {relative}") from exc
    exists = resolved.is_dir() if expect_directory else resolved.is_file()
    if not exists:
        kind = "directory" if expect_directory else "file"
        raise FileNotFoundError(f"dataset profile {kind} is missing: {relative}")
    return resolved


class DatasetProfileRegistry:
    """Fail-closed registry for reviewed, hash-bound dataset profiles."""

    def __init__(self, *, index_path: str | Path) -> None:
        self.index_path = Path(index_path).resolve()
        self.index = DatasetProfileRegistryIndex.model_validate_json(
            self.index_path.read_text(encoding="utf-8-sig")
        )
        declared_root = Path(self.index.asset_root)
        if declared_root.is_absolute():
            raise ValueError("dataset profile registry asset_root must be relative")
        self.asset_root = (self.index_path.parent / declared_root).resolve()
        self._references = {item.profile_id: item for item in self.index.profiles}

    def resolve(self, profile_id: str) -> ResolvedDatasetProfile:
        try:
            reference = self._references[profile_id]
        except KeyError as exc:
            raise KeyError(f"unknown dataset profile: {profile_id}") from exc
        profile_path = _safe_resolve(self.asset_root, reference.path)
        actual_profile_sha = sha256_file(profile_path)
        if actual_profile_sha != reference.sha256:
            raise ValueError(f"dataset profile hash mismatch: {profile_id}")
        profile = DatasetProfile.model_validate_json(profile_path.read_text(encoding="utf-8-sig"))
        if profile.profile_id != profile_id:
            raise ValueError("dataset profile ID does not match registry reference")

        protocol_root = _safe_resolve(
            self.asset_root,
            profile.protocol_root_path,
            expect_directory=True,
        )
        asset_paths: dict[str, Path] = {}
        verified: dict[str, str] = {}
        for role, asset in sorted(profile.assets.items()):
            path = _safe_resolve(self.asset_root, asset.path)
            actual = sha256_file(path)
            if actual != asset.sha256:
                raise ValueError(f"dataset profile asset hash mismatch: {profile_id}:{role}")
            asset_paths[role] = path
            verified[role] = actual

        receipt = DatasetProfileReceipt(
            registry_id=self.index.registry_id,
            registry_version=self.index.version,
            profile_id=profile.profile_id,
            profile_version=profile.version,
            dataset_id=profile.dataset_id,
            reference_scope=profile.reference_scope,
            fusion_policy_profile_id=profile.fusion_policy_profile_id,
            profile_sha256=actual_profile_sha,
            verified_asset_sha256=verified,
            claim_boundary=profile.claim_boundary,
        )
        resolved = ResolvedDatasetProfile(
            profile=profile,
            protocol_root=protocol_root,
            asset_paths=asset_paths,
            receipt=receipt,
        )
        return resolved

    def list_profile_ids(self) -> list[str]:
        return sorted(self._references)


__all__ = ["DatasetProfileRegistry"]
