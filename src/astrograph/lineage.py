from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def observation_correlation_group_id(
    *,
    episode_id: str,
    window_start_s: float | None,
    window_end_s: float | None,
    source_field: str,
    semantic_signal_id: str,
) -> str:
    """Return a branch-neutral correlation group for one physical observation.

    The token intentionally excludes feature transforms and model/graph identities.
    Two branches operating on the same episode, time window, source field and
    semantic signal therefore declare statistical dependence even when their
    downstream transformations differ.
    """

    payload = {
        "schema_version": "astrograph-observation-correlation.v1",
        "episode_id": str(episode_id),
        "window_start_s": None if window_start_s is None else float(window_start_s),
        "window_end_s": None if window_end_s is None else float(window_end_s),
        "source_field": str(source_field),
        "semantic_signal_id": str(semantic_signal_id),
    }
    return "CORR-OBS-" + _canonical_sha256(payload)[:32].upper()


__all__ = ["observation_correlation_group_id"]
