"""Stable semantic digests for retrospective replay records.

Raw replay artifacts retain full provenance, including run-local evidence bundle
hashes. Semantic repeatability intentionally ignores only that bundle hash while
remaining sensitive to every other persisted inference field.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any


def semantic_inference_digest(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash all decision-relevant replay fields, excluding only run-local bundle identity."""
    h = hashlib.sha256()
    for record in records:
        stable = {k: v for k, v in record.items() if k != "evidence_bundle_sha256"}
        payload = json.dumps(
            stable,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        h.update(payload)
        h.update(b"\n")
    return h.hexdigest()
