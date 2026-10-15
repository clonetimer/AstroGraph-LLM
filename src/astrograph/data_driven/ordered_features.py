"""Fail-closed helpers for preserving declared tensor feature order.

``pandas.read_csv(usecols=...)`` does not promise to return columns in the
``usecols`` argument order.  Model code must therefore perform an explicit
selection after loading before converting a frame to a tensor.  These helpers
centralize that contract and make legacy file-header-order recovery auditable.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def normalize_feature_order(columns: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze a non-empty, duplicate-free feature order."""

    order = tuple(str(value) for value in columns)
    if not order:
        raise ValueError("feature order must contain at least one column")
    if any(not value.strip() for value in order):
        raise ValueError("feature order contains an empty column name")
    duplicates = sorted(name for name, count in Counter(order).items() if count > 1)
    if duplicates:
        raise ValueError(f"feature order contains duplicate columns: {duplicates}")
    return order


def select_columns_in_declared_order(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a view copied in exactly ``columns`` order, rejecting drift."""

    order = normalize_feature_order(columns)
    missing = [name for name in order if name not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing declared feature columns: {missing}")
    # ``loc`` is intentional.  The result order is the declared tensor order,
    # independent of the source-file header and pandas' usecols behavior.
    return frame.loc[:, list(order)].copy()


def read_csv_in_declared_order(
    path: str | Path,
    columns: Sequence[str],
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read only requested CSV fields and restore their declared order."""

    order = normalize_feature_order(columns)
    frame = pd.read_csv(path, usecols=list(order), **read_csv_kwargs)
    return select_columns_in_declared_order(frame, order)


def legacy_usecols_tensor_order(
    source_header: Sequence[str],
    requested_columns: Sequence[str],
) -> tuple[str, ...]:
    """Recover the tensor order produced by legacy ``read_csv(usecols=...)`` code.

    Historical AstroGraph ADCS training loaded ``usecols=requested_columns`` and
    immediately called ``to_numpy``.  Pandas retained source-header order, not
    the requested order.  This helper reconstructs that exact order without
    reading any row values.  It is for migration/audit only; new training must
    use :func:`read_csv_in_declared_order`.
    """

    requested = normalize_feature_order(requested_columns)
    requested_set = set(requested)
    header = tuple(str(value) for value in source_header)
    actual = tuple(name for name in header if name in requested_set)
    missing = [name for name in requested if name not in actual]
    if missing:
        raise ValueError(f"source header is missing requested feature columns: {missing}")
    if len(actual) != len(requested):
        raise ValueError("source header contains duplicate requested feature columns")
    return actual


def require_consistent_legacy_tensor_order(
    headers: Iterable[Sequence[str]],
    requested_columns: Sequence[str],
) -> tuple[str, ...]:
    """Require every supplied header to imply the same legacy tensor order."""

    iterator = iter(headers)
    try:
        first_header = next(iterator)
    except StopIteration as exc:
        raise ValueError("at least one source header is required") from exc
    expected = legacy_usecols_tensor_order(first_header, requested_columns)
    for index, header in enumerate(iterator, start=2):
        observed = legacy_usecols_tensor_order(header, requested_columns)
        if observed != expected:
            raise ValueError(
                "legacy tensor order is not consistent across source headers: "
                f"header {index} differs"
            )
    return expected


__all__ = [
    "legacy_usecols_tensor_order",
    "normalize_feature_order",
    "read_csv_in_declared_order",
    "require_consistent_legacy_tensor_order",
    "select_columns_in_declared_order",
]
