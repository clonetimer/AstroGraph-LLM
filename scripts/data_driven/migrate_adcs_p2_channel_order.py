#!/usr/bin/env python3
"""Migrate the legacy ADCS P2 package to an explicit tensor-order contract.

The v1 training pipeline called ``pandas.read_csv(usecols=...)`` and converted
that frame directly to a tensor.  Pandas retained source-header order rather
than the requested ``usecols`` order, while the package manifest recorded the
requested order.  This migration reconstructs the historical tensor order from
every dataset header, requires it to be globally consistent, and writes a v2
package whose positional contract matches the trained weights.

This is a contract repair, not model retraining or a new performance claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.assurance import ArtifactRoot
from astrograph.data_driven.ordered_features import (
    legacy_usecols_tensor_order,
    require_consistent_legacy_tensor_order,
)
from astrograph.integrations.adcs_three_task_torch import (
    ADCS_PACKAGE_SCHEMA_V1,
    ADCS_PACKAGE_SCHEMA_V2,
    declared_channel_order_contract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_csv_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            row = next(csv.reader(stream))
        except StopIteration as exc:
            raise ValueError(f"telemetry CSV has no header: {path}") from exc
    if not row:
        raise ValueError(f"telemetry CSV has an empty header: {path}")
    if len(row) != len(set(row)):
        raise ValueError(f"telemetry CSV header contains duplicate columns: {path}")
    return tuple(row)


def _load_telemetry_paths(dataset_root: Path) -> list[Path]:
    index_path = dataset_root / "sample_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"ADCS sample index not found: {index_path}")
    paths: list[Path] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "telemetry_file" not in reader.fieldnames:
            raise ValueError("sample_index.csv must contain telemetry_file")
        for row_number, row in enumerate(reader, start=2):
            relative = str(row.get("telemetry_file", "")).strip()
            if not relative:
                raise ValueError(f"sample_index.csv row {row_number} has no telemetry_file")
            child = Path(relative)
            if child.is_absolute() or ".." in child.parts:
                raise ValueError(f"unsafe telemetry_file at row {row_number}: {relative}")
            path = (dataset_root / child).resolve(strict=True)
            root = dataset_root.resolve(strict=True)
            if path != root and root not in path.parents:
                raise ValueError(f"telemetry_file escapes dataset root: {relative}")
            paths.append(path)
    if not paths:
        raise ValueError("sample_index.csv contains no telemetry paths")
    return paths


def recover_dataset_tensor_order(
    dataset_root: Path, requested_columns: Iterable[str]
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Recover and attest the single legacy order across all telemetry files."""

    telemetry_paths = _load_telemetry_paths(dataset_root)
    requested = tuple(map(str, requested_columns))
    headers: list[tuple[str, ...]] = []
    inventory: list[dict[str, str]] = []
    root = dataset_root.resolve(strict=True)
    for path in telemetry_paths:
        header = _read_csv_header(path)
        headers.append(header)
        relative = path.relative_to(root).as_posix()
        inventory.append({"path": relative, "header_sha256": _canonical_sha256(list(header))})
    recovered = require_consistent_legacy_tensor_order(headers, requested)
    # Re-run on the first header as an explicit assertion that this remains the
    # historical pandas usecols behavior being repaired.
    if legacy_usecols_tensor_order(headers[0], requested) != recovered:
        raise AssertionError("legacy tensor-order recovery is internally inconsistent")
    evidence = {
        "telemetry_file_count": len(telemetry_paths),
        "unique_full_header_count": len({_canonical_sha256(list(header)) for header in headers}),
        "source_header_inventory_sha256": _canonical_sha256(inventory),
        "recovered_tensor_order_sha256": _canonical_sha256(list(recovered)),
        "sample_index_sha256": _sha256(dataset_root / "sample_index.csv"),
        "dataset_manifest_sha256": (
            _sha256(dataset_root / "dataset_manifest.json")
            if (dataset_root / "dataset_manifest.json").is_file()
            else None
        ),
    }
    return recovered, evidence


def migrate_package(
    *,
    legacy_package: Path,
    expected_legacy_sha256: str,
    dataset_root: Path,
    output_package: Path,
    output_receipt: Path,
    model_version: str,
    feature_schema_id: str,
) -> dict[str, Any]:
    import torch

    expected = expected_legacy_sha256.lower()
    actual = _sha256(legacy_package)
    if actual != expected:
        raise ValueError(
            f"legacy ADCS package SHA-256 mismatch: expected {expected}, observed {actual}"
        )
    bundle = torch.load(legacy_package, map_location="cpu", weights_only=True)
    if bundle.get("schema_version") != ADCS_PACKAGE_SCHEMA_V1:
        raise ValueError("migration accepts only the legacy ADCS package v1 schema")
    if bundle.get("model_family") != "shared_residual_tcn":
        raise ValueError("unsupported legacy ADCS model family")

    old_order = tuple(map(str, bundle.get("selected_channels", ())))
    manifest = dict(bundle.get("manifest", {}))
    old_manifest_order = tuple(map(str, manifest.get("input_contract", {}).get("feature_order", ())))
    if not old_order or old_order != old_manifest_order:
        raise ValueError("legacy selected_channels and manifest feature_order must match")

    recovered_order, evidence = recover_dataset_tensor_order(dataset_root, old_order)
    if set(recovered_order) != set(old_order) or len(recovered_order) != len(old_order):
        raise ValueError("recovered tensor order does not preserve the legacy feature universe")

    contract = declared_channel_order_contract(
        recovered_order,
        provenance="LEGACY_PANDAS_USECOLS_HEADER_ORDER_RECOVERED",
        legacy_package_sha256=actual,
        source_header_order_sha256=evidence["recovered_tensor_order_sha256"],
    )
    input_contract = dict(manifest.get("input_contract", {}))
    input_contract["feature_order"] = list(recovered_order)
    input_contract["tensor_channel_order"] = list(recovered_order)
    input_contract["channel_order_contract_schema"] = contract["schema_version"]
    input_contract["channel_order_contract_sha256"] = contract["contract_sha256"]
    manifest["input_contract"] = input_contract
    manifest["model_version"] = model_version
    manifest["feature_schema_ids"] = [feature_schema_id]
    metadata = dict(manifest.get("metadata", {}))
    metadata["channel_order_migration"] = {
        "status": "RETROSPECTIVE_CONTRACT_DEFECT_CORRECTION",
        "legacy_package_sha256": actual,
        "legacy_manifest_order_sha256": _canonical_sha256(list(old_order)),
        "recovered_tensor_order_sha256": evidence["recovered_tensor_order_sha256"],
        "claim_boundary": "No weights were retrained and no performance claim is created by migration.",
    }
    manifest["metadata"] = metadata

    migrated = dict(bundle)
    migrated["schema_version"] = ADCS_PACKAGE_SCHEMA_V2
    migrated["manifest"] = manifest
    migrated["selected_channels"] = list(recovered_order)
    migrated["channel_order_contract"] = contract

    source_root = ROOT.resolve(strict=True)
    package_root = ArtifactRoot(output_package.parent, source_root=source_root)
    package_target = package_root.write_generated(
        output_package.name, lambda temporary: torch.save(migrated, temporary)
    )
    migrated_sha256 = _sha256(package_target)

    old_position = {name: index for index, name in enumerate(old_order)}
    receipt: dict[str, Any] = {
        "schema_version": "astrograph-adcs-channel-order-migration-receipt.v1",
        "status": "PASS",
        "migration_type": "RETROSPECTIVE_CONTRACT_DEFECT_CORRECTION",
        "legacy_package": {
            "sha256": actual,
            "schema_version": ADCS_PACKAGE_SCHEMA_V1,
            "model_version": bundle["manifest"]["model_version"],
        },
        "migrated_package": {
            "sha256": migrated_sha256,
            "schema_version": ADCS_PACKAGE_SCHEMA_V2,
            "model_version": model_version,
            "feature_schema_id": feature_schema_id,
            "channel_order_contract_sha256": contract["contract_sha256"],
        },
        "dataset_evidence": evidence,
        "legacy_manifest_order": list(old_order),
        "recovered_training_tensor_order": list(recovered_order),
        "recovered_position_to_legacy_manifest_position": [
            old_position[name] for name in recovered_order
        ],
        "moved_channel_count": sum(
            left != right for left, right in zip(old_order, recovered_order, strict=True)
        ),
        "weights_retrained": False,
        "performance_claim_created": False,
        "claim_boundary": (
            "The receipt proves only the positional contract repair. Test-set performance is a "
            "retrospective defect-correction audit, not untouched confirmatory evidence."
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    receipt_root = ArtifactRoot(output_receipt.parent, source_root=source_root)
    receipt_root.write_json(output_receipt.name, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-package", type=Path, required=True)
    parser.add_argument("--legacy-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument(
        "--model-version", default="0.7.1.post24-candidate.2-order-recovered"
    )
    parser.add_argument(
        "--feature-schema-id",
        default="adcs-v5.2.2-p2-innovation-window-v2-order-recovered",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    receipt = migrate_package(
        legacy_package=args.legacy_package,
        expected_legacy_sha256=args.legacy_sha256,
        dataset_root=args.dataset_root,
        output_package=args.output_package,
        output_receipt=args.output_receipt,
        model_version=args.model_version,
        feature_schema_id=args.feature_schema_id,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
