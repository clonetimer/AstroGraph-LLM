from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from astrograph.data_driven.ordered_features import (
    legacy_usecols_tensor_order,
    read_csv_in_declared_order,
    require_consistent_legacy_tensor_order,
    select_columns_in_declared_order,
)


def test_read_csv_restores_declared_order_before_tensor_conversion(tmp_path: Path):
    path = tmp_path / "telemetry.csv"
    path.write_text("b,a,c\n20,10,30\n", encoding="utf-8")

    frame = read_csv_in_declared_order(path, ("a", "b"))

    assert list(frame.columns) == ["a", "b"]
    assert frame.to_numpy().tolist() == [[10, 20]]


def test_select_columns_rejects_missing_and_duplicate_contracts():
    frame = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="missing"):
        select_columns_in_declared_order(frame, ("a", "c"))
    with pytest.raises(ValueError, match="duplicate"):
        select_columns_in_declared_order(frame, ("a", "a"))


def test_legacy_recovery_follows_header_order_and_requires_consistency():
    requested = ("a", "b", "c")
    assert legacy_usecols_tensor_order(("time", "c", "a", "b"), requested) == (
        "c",
        "a",
        "b",
    )
    assert require_consistent_legacy_tensor_order(
        [("time", "c", "a", "b"), ("other", "c", "a", "b")], requested
    ) == ("c", "a", "b")
    with pytest.raises(ValueError, match="not consistent"):
        require_consistent_legacy_tensor_order(
            [("time", "c", "a", "b"), ("time", "a", "b", "c")], requested
        )
