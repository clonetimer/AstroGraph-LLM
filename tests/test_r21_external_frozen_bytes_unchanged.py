import json
import os
from pathlib import Path

import pytest

from astrograph.evaluation.data_exposure import registry
from astrograph.evaluation.real_data_failure_analysis import verify_frozen_bytes


def _receipt_dir():
    configured = os.environ.get("ASTROGRAPH_R20_RECEIPTS")
    if configured:
        return Path(configured)
    root = Path(__file__).resolve().parents[1]
    candidate = root.parents[1].parent / "R20/results/R20_EXTERNAL_EVALUATION_RECEIPTS"
    return candidate


def test_all_79_frozen_external_files_are_unchanged_when_archive_is_available():
    root = _receipt_dir()
    if not root.exists():
        pytest.skip("R20.1 external receipts are deliberately outside the source-only package")
    result = verify_frozen_bytes(root, registry())
    assert result == {"status": "PASS", "original_frozen_files": 79, "analysis_sidecars": 3, "inference_rerun": False}


def test_changed_frozen_bytes_fail_before_any_analysis(tmp_path):
    (tmp_path / "x.json").write_text("changed", encoding="utf-8")
    fake = {"frozen_evidence": [{"relative_path": "x.json", "size": 1, "sha256": "0" * 64}], "analysis_result_inputs": []}
    with pytest.raises(ValueError, match="FROZEN_EXTERNAL_BYTES_CHANGED"):
        verify_frozen_bytes(tmp_path, fake)
