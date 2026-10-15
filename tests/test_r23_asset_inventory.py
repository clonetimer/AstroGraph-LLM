from __future__ import annotations

import json
import zipfile

from astrograph.evaluation.r23_assets import build_zip_asset_inventory


def test_r23_asset_inventory_reads_schema_metadata_not_embedded_evaluation(tmp_path):
    path = tmp_path / "asset.zip"
    manifest = {
        "schema_version": "example.dataset.v1",
        "status": "PASS",
        "fault_count": 3,
        "secret_evaluation_metric": 0.999,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("dataset/dataset_manifest.json", json.dumps(manifest))
        archive.writestr("dataset/audit/evaluation.json", json.dumps({"accuracy": 1.0, "truth": [1, 0]}))
    receipt = build_zip_asset_inventory(path, asset_id="A", role="SCHEMA_ONLY")
    assert receipt.schema_facts["dataset_manifest"]["fault_count"] == 3
    assert "secret_evaluation_metric" not in receipt.schema_facts["dataset_manifest"]
    assert "evaluation" not in json.dumps(receipt.schema_facts).lower()
    assert receipt.evaluation_content_consumed_for_selection is False
