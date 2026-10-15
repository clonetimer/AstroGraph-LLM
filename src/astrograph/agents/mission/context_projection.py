"""Lossless Mission-only wire projection; never mutate ranking or fusion inputs.

Large correlation lineage lists stay in the hash-bound full context artifact.
The LLM view preserves exact set identity, cardinality and pairwise overlap. It
is not licensed to interpret a shared lineage as independent supporting votes.
Repeated record keys are represented as a reversible column table, not a summary.
"""
from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import PurePath
from typing import Any

from .storage import canonical, sha

_ENCODING = "_context_encoding"
_LINEAGES = "_correlation_lineage_sets"
_SHARED = "_shared_context_values"
_TABLE_KEYS = frozenset({"candidates", "candidate_explanations", "graph_paths"})
_RESERVED = frozenset({_ENCODING, _LINEAGES, _SHARED, "$table", "$lineage", "$shared"})


def _pointer_get(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("invalid context JSON pointer")
    value = document
    for part in pointer.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def _has_reserved(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(_RESERVED.intersection(value)) or any(_has_reserved(v) for v in value.values())
    return isinstance(value, list) and any(_has_reserved(v) for v in value)


def _encode_tables(value: Any, *, key: str = "", pointer: str = "", records: list[dict]) -> Any:
    if isinstance(value, dict):
        return {k: _encode_tables(v, key=k, pointer=pointer+"/"+k.replace("~", "~0").replace("/", "~1"), records=records)
                for k, v in value.items()}
    if not isinstance(value, list):
        return value
    rows = [_encode_tables(v, pointer=pointer+"/"+str(i), records=records) for i, v in enumerate(value)]
    if key not in _TABLE_KEYS or len(rows) < 2 or not all(isinstance(v, dict) for v in rows):
        return rows
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        return rows  # Missing fields and explicit null are not interchangeable.
    shared = {k: rows[0][k] for k in columns
              if all(canonical(row[k]) == canonical(rows[0][k]) for row in rows)}
    varying = [k for k in columns if k not in shared]
    packed = {"$table": {"shared": shared, "columns": varying,
                          "rows": [[row[k] for k in varying] for row in rows]}}
    before, after = len(canonical(rows)), len(canonical(packed))
    if after + 48 >= before:
        return rows
    records.append({"json_pointer": pointer, "rows": len(rows), "before_bytes": before,
                    "after_bytes": after, "row_order_preserved": True})
    return packed


def _share_repeated_values(view: dict) -> tuple[dict, dict]:
    """Intern repeated structures/opaque IDs, with their exact values IN the view.

    The dictionary is plain JSON, not a lossy text summary or an external lookup.
    Empty/false/null/numeric values are not collapsed into one another.
    """
    counts = Counter()
    values = {}
    def eligible(value):
        if isinstance(value, (dict, list)):
            return len(canonical(value)) >= 128
        return isinstance(value, str) and len(value) >= 20 and re.match(r"^(?:ED-|EF-|FD-|GQF-|GQB-|ADCS_)", value)
    def collect(value):
        if eligible(value):
            key = canonical(value)
            counts[key] += 1
            values[key] = value
        if isinstance(value, dict):
            for item in value.values(): collect(item)
        elif isinstance(value, list):
            for item in value: collect(item)
    collect(view)
    selected = {k for k, n in counts.items() if n >= 2 and n * (len(k)-20) - len(k)-12 > 48}
    catalogue = {}
    identities = {}
    def encode(value):
        if eligible(value):
            key = canonical(value)
            if key in selected:
                if key not in identities:
                    identity = "V"+str(len(identities)+1)
                    identities[key] = identity
                    catalogue[identity] = copy.deepcopy(value)
                return {"$shared": identities[key]}
        if isinstance(value, dict):return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):return [encode(v) for v in value]
        return value
    packed = encode(view)
    if len(canonical(packed)) + len(canonical(catalogue)) + 40 < len(canonical(view)):
        return packed, catalogue
    return view, {}


def project_diagnostic_context(payload: dict, *, full_payload: dict,
                               full_artifact: dict | None = None) -> tuple[dict, dict]:
    """Encode exact structure and losslessly referenced lineage, without selection.

    This function receives the Mission view AFTER optional source pools are
    separated; it is never called on the numeric input or BiRRF data structures.
    Without a persisted full artifact, lineage stays inline (fail closed later).
    """
    if _has_reserved(payload):
        raise ValueError("RESERVED_CONTEXT_ENCODING_KEY_IN_INPUT")
    import json
    original = json.loads(canonical(payload))
    view = copy.deepcopy(original)
    full_digest = sha(full_payload)
    if full_artifact is not None:
        name = full_artifact.get("name", "")
        if (full_artifact.get("sha256") != full_digest or not name or
                PurePath(name).name != name or "/" in name or "\\" in name):
            raise ValueError("CONTEXT_ARTIFACT_BINDING_INVALID")
    catalogue: dict[str, dict] = {}
    raw_lists: dict[str, list[str]] = {}
    digest_to_id: dict[str, str] = {}
    occurrences = 0
    ranking = (view.get("diagnostic_context") or {}).get("data_ranking", {})
    candidates = ranking.get("candidates", []) if isinstance(ranking, dict) else []
    if isinstance(candidates, list) and full_artifact is not None:
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            values = candidate.get("correlation_group_ids")
            if (not isinstance(values, list) or len(values) <= 32 or
                    not all(isinstance(v, str) for v in values)):
                continue
            pointer = f"/diagnostic_context/data_ranking/candidates/{i}/correlation_group_ids"
            if canonical(_pointer_get(full_payload, pointer)) != canonical(values):
                raise ValueError("LINEAGE_CHANGED_BEFORE_PROJECTION")
            digest = sha(values)
            if digest not in digest_to_id:
                if len(catalogue) >= 64:
                    raise ValueError("TOO_MANY_LINEAGE_SETS_FOR_EXACT_OVERLAP_VIEW")
                identity = "L" + str(len(catalogue) + 1)
                digest_to_id[digest] = identity
                raw_lists[identity] = values
                catalogue[identity] = {
                    "count": len(values), "unique_count": len(set(values)),
                    "sha256": digest, "membership_sha256": sha(sorted(set(values))),
                    "full_payload_pointer": pointer,
                }
            candidate["correlation_group_ids"] = {"$lineage": digest_to_id[digest]}
            occurrences += 1
    tables: list[dict] = []
    view = _encode_tables(view, records=tables)
    view, shared_values = _share_repeated_values(view)
    stats = {"shared_values": len(shared_values), "tables": tables, "lineage_occurrences": occurrences,
             "distinct_lineage_lists": len(catalogue), "lossless_round_trip_verified": False}
    if not tables and not catalogue and not shared_values:
        return view, stats
    encoding = {"version": "astrograph-context-lossless.v1", "full_payload_sha256": full_digest,
                "diagnostic_votes_added": False}
    if shared_values:
        view[_SHARED] = shared_values
    if catalogue:
        encoding["full_payload_artifact"] = full_artifact
        # An ordered-list identity differs from membership: overlaps preserve
        # dependencies even when different lists contain some/all of the same IDs.
        ids = list(catalogue)
        sets = {k: set(v) for k, v in raw_lists.items()}
        encoding["lineage_pairwise_overlap"] = [
            {"left": a, "right": b, "intersection_count": len(sets[a] & sets[b]),
             "union_count": len(sets[a] | sets[b])}
            for n, a in enumerate(ids) for b in ids[n+1:]
        ]
        view[_LINEAGES] = catalogue
    view[_ENCODING] = encoding
    restored = restore_diagnostic_context(view, full_payload=full_payload)
    if canonical(restored) != canonical(original):
        raise ValueError("CONTEXT_LOSSLESS_ROUND_TRIP_FAILED")
    stats["lossless_round_trip_verified"] = True
    stats["lineage_catalogue"] = copy.deepcopy(catalogue)
    return view, stats


def restore_diagnostic_context(view: dict, *, full_payload: dict) -> dict:
    """Offline audit decoder. No arbitrary file I/O; callers supply the artifact.

    A missing, mismatched or modified artifact is an error, not empty lineage.
    """
    encoding = view.get(_ENCODING)
    if encoding is None:
        return copy.deepcopy(view)
    if (encoding.get("version") != "astrograph-context-lossless.v1" or
            encoding.get("full_payload_sha256") != sha(full_payload)):
        raise ValueError("CONTEXT_FULL_ARTIFACT_HASH_MISMATCH")
    catalogue = view.get(_LINEAGES, {})
    shared = view.get(_SHARED, {})
    active = set()

    def restore(value):
        if isinstance(value, list):
            return [restore(v) for v in value]
        if not isinstance(value, dict):
            return value
        if "$shared" in value:
            identity = value["$shared"]
            if set(value) != {"$shared"} or identity not in shared or identity in active:
                raise ValueError("INVALID_OR_CYCLIC_SHARED_CONTEXT_REFERENCE")
            active.add(identity)
            try:
                return restore(shared[identity])
            finally:
                active.remove(identity)
        if "$lineage" in value:
            if set(value) != {"$lineage"} or value["$lineage"] not in catalogue:
                raise ValueError("INVALID_LINEAGE_REFERENCE")
            entry = catalogue[value["$lineage"]]
            raw = _pointer_get(full_payload, entry["full_payload_pointer"])
            if (not isinstance(raw, list) or not all(isinstance(v, str) for v in raw)
                    or sha(raw) != entry["sha256"] or len(raw) != entry["count"]
                    or len(set(raw)) != entry["unique_count"]
                    or sha(sorted(set(raw))) != entry["membership_sha256"]):
                raise ValueError("LINEAGE_REFERENCE_INTEGRITY_MISMATCH")
            return copy.deepcopy(raw)
        if "$table" in value:
            if set(value) != {"$table"}:
                raise ValueError("INVALID_CONTEXT_TABLE")
            table = value["$table"]
            columns, common, rows = table["columns"], table["shared"], table["rows"]
            if (not isinstance(columns, list) or not all(isinstance(k, str) for k in columns)
                    or len(columns) != len(set(columns)) or not isinstance(common, dict)
                    or set(columns).intersection(common) or not isinstance(rows, list)
                    or any(not isinstance(r, list) or len(r) != len(columns) for r in rows)):
                raise ValueError("INVALID_CONTEXT_TABLE")
            return [restore({**common, **dict(zip(columns, row))}) for row in rows]
        return {k: restore(v) for k, v in value.items()}

    return restore({k: v for k, v in view.items() if k not in {_ENCODING, _LINEAGES, _SHARED}})
