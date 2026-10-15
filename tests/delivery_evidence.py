"""Test helper for evidence intentionally external to current-source-only deliveries."""
from __future__ import annotations

from pathlib import Path
import pytest

from astrograph.governance.delivery_evidence import resolve_delivery_evidence


def require_delivery_evidence(root: Path, relative_path: str) -> Path:
    resolution = resolve_delivery_evidence(root, relative_path)
    path = root / relative_path
    if resolution.available:
        return path
    if resolution.external_declared:
        pytest.skip(
            f"historical evidence is declared external for CURRENT_SOURCE_ONLY delivery: "
            f"{resolution.dependency_id} -> {resolution.path}"
        )
    pytest.fail(f"required delivery evidence missing: {resolution.path} ({resolution.status})")
    raise AssertionError("unreachable")
