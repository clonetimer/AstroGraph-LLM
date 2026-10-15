#!/usr/bin/env python3
"""Run the frozen Route A development-selection protocol from hashed JSON inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from astrograph.assurance.analysis_plan import PrimaryAnalysisPlanV2
from astrograph.experiments.route_a_protocol import (
    FrozenPolicyPredictionBundle,
    OfflineTruthTable,
    freeze_route_a_development_selection,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_hashed_json(root: Path, item: dict[str, Any], *, role: str) -> dict[str, Any]:
    relative = Path(str(item["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{role} path must be relative and cannot traverse parents")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"{role} path escapes input root")
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {relative}")
    expected = str(item["sha256"])
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{role} SHA-256 mismatch: expected={expected} observed={observed}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def run_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "astrograph-route-a-development-run-manifest.v1":
        raise ValueError("unsupported Route A run manifest schema")
    root = manifest_path.parent
    analysis_plan = PrimaryAnalysisPlanV2.model_validate(
        _load_hashed_json(root, manifest["analysis_plan"], role="analysis plan")
    )
    truth_table = OfflineTruthTable.model_validate(
        _load_hashed_json(root, manifest["truth_table"], role="truth table")
    )
    bundles = [
        FrozenPolicyPredictionBundle.model_validate(
            _load_hashed_json(root, item, role=f"prediction bundle {index}")
        )
        for index, item in enumerate(manifest["prediction_bundles"], start=1)
    ]
    receipt = freeze_route_a_development_selection(
        analysis_plan=analysis_plan,
        prediction_bundles=bundles,
        truth_table=truth_table,
    )
    return receipt.model_dump(mode="json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = run_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
