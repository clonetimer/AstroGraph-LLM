"""Generate non-binding, content-addressed suggestions for document KG review."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from astrograph.filesystem import publish_staged_directory
from .document_candidate_entrypoint import DocumentCandidateBuildReceipt
from .document_graph_review import REVIEW_FIELDS
from .document_graph_review_decisions import freeze_document_graph_review_decisions
from .schema_v3 import KnowledgeGraphV3


SUGGESTION_FIELDS = [
    "review_item_id",
    "dataset_id",
    "fault_class_id",
    "item_kind",
    "candidate_id",
    "node_type",
    "canonical_name",
    "canonical_id_hint",
    "machine_suggestion",
    "suggested_canonical_id",
    "reason_codes",
    "batch_status",
    "bundle_sha256",
    "base_graph_sha256",
]

COMPONENT_ARCHETYPE_SUGGESTION_FIELDS = [
    "review_item_id",
    "dataset_id",
    "fault_class_id",
    "candidate_id",
    "candidate_component_name",
    "machine_suggestion",
    "suggested_canonical_id",
    "channel_mapping_id",
    "reason_codes",
    "channel_map_catalog_sha256",
    "base_graph_sha256",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _validate_semantic_receipt(payload: dict[str, Any], *, label: str) -> None:
    expected = payload.get("receipt_sha256")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if not expected or _canonical_sha(body) != expected:
        raise ValueError(f"{label} semantic receipt hash mismatch")


def _normalise(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _base_names(node: Any) -> set[str]:
    properties = node.properties
    values = [node.node_id, properties.get("canonical_id", ""), properties.get("canonical_name", "")]
    values.extend(properties.get("aliases", []))
    return {_normalise(str(value)) for value in values if str(value).strip()}


def build_document_review_suggestions(
    *,
    candidate_pack_root: str | Path,
    review_root: str | Path,
    base_graph_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Suggest exact alignments without modifying decisions or publishing knowledge."""

    pack_root = Path(candidate_pack_root).resolve()
    review_root = Path(review_root).resolve()
    base_path = Path(base_graph_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite nonempty review suggestion directory")

    pack_receipt_path = pack_root / "receipt.json"
    review_receipt_path = review_root / "review_handoff_receipt.json"
    review_sheet_path = review_root / "human_review_decisions.csv"
    pack_receipt = json.loads(pack_receipt_path.read_text(encoding="utf-8"))
    review_receipt = json.loads(review_receipt_path.read_text(encoding="utf-8"))
    _validate_semantic_receipt(pack_receipt, label="candidate pack")
    _validate_semantic_receipt(review_receipt, label="review handoff")
    if review_receipt.get("primary_receipt_sha256") != _sha(pack_receipt_path):
        raise ValueError("review handoff is not bound to the supplied candidate pack")
    if review_receipt.get("review_sheet_sha256") != _sha(review_sheet_path):
        raise ValueError("review sheet changed after handoff")

    graph = KnowledgeGraphV3.model_validate_json(base_path.read_text(encoding="utf-8"))
    base_graph_sha = _canonical_sha(graph.model_dump(mode="json"))
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    nodes_by_type_name: dict[tuple[str, str], list[Any]] = {}
    for node in graph.nodes:
        for name in _base_names(node):
            nodes_by_type_name.setdefault((node.node_type.value, name), []).append(node)

    pack_results = {
        (str(item["dataset_id"]), str(item["fault_class_id"])): item
        for item in pack_receipt["results"]
    }
    candidate_items: dict[tuple[str, str, str], dict[str, Any]] = {}
    bundle_hashes: dict[tuple[str, str], str] = {}
    for key, result in pack_results.items():
        dataset_id, fault_id = key
        candidate_root = pack_root / dataset_id / fault_id / "candidate"
        receipt_path = candidate_root / "receipt.json"
        if _sha(receipt_path) != result["candidate_receipt_file_sha256"]:
            raise ValueError("candidate pack summary and per-document receipt differ")
        candidate_receipt = DocumentCandidateBuildReceipt.model_validate_json(
            receipt_path.read_text(encoding="utf-8")
        )
        for filename, expected in candidate_receipt.artifacts.items():
            artifact = candidate_root / filename
            if not artifact.is_file() or _sha(artifact) != expected:
                raise ValueError(f"candidate artifact is missing or changed: {dataset_id}/{fault_id}/{filename}")
        bundle = json.loads((candidate_root / "construction_bundle.json").read_text(encoding="utf-8"))
        bundle_hashes[key] = str(bundle["bundle_sha256"])
        for node in bundle["proposal"].get("nodes", []):
            candidate_items[(dataset_id, fault_id, str(node["node_id"]))] = node
        for relation in bundle["proposal"].get("relations", []):
            candidate_items[(dataset_id, fault_id, str(relation["edge_id"]))] = relation

    original_review_bytes = review_sheet_path.read_bytes()
    with review_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    if len(review_rows) != review_receipt.get("review_item_count"):
        raise ValueError("review item count differs from handoff receipt")

    suggestions: list[dict[str, str]] = []
    for row in review_rows:
        if any(row[field] for field in ("decision", "approved_canonical_id", "reviewer", "review_comment")):
            raise ValueError("machine suggestions require an unmodified blank human review sheet")
        key = (row["dataset_id"], row["fault_class_id"])
        if row["bundle_sha256"] != bundle_hashes.get(key):
            raise ValueError("review row bundle identity differs from candidate pack")
        candidate = candidate_items.get((*key, row["candidate_id"]))
        if candidate is None:
            raise ValueError("review row candidate is absent from candidate pack")

        hint = str(candidate.get("canonical_id_hint") or "")
        suggestion = "DEFER_CANONICAL_MAPPING"
        suggested_id = ""
        reasons = ["NO_UNIQUE_EXACT_BASE_MATCH"]
        if row["batch_status"] != "CANDIDATE_READY_FOR_HUMAN_REVIEW":
            suggestion = "REQUIRES_CANDIDATE_CORRECTION"
            reasons = ["SOURCE_CANDIDATE_PACKAGE_BLOCKED"]
        elif row["item_kind"] != "NODE":
            suggestion = "DEFER_RELATION_REVIEW"
            reasons = ["RELATION_REQUIRES_ENDPOINT_AND_SHAPE_REVIEW"]
        else:
            node_type = row["node_type"]
            exact = nodes_by_id.get(hint) if hint else None
            if exact is not None and exact.node_type.value == node_type:
                suggestion = "ALIGN_EXISTING_EXACT_ID"
                suggested_id = exact.node_id
                reasons = ["CANONICAL_ID_HINT_EXACT_TYPE_MATCH"]
            else:
                matches = nodes_by_type_name.get((node_type, _normalise(row["canonical_name"])), [])
                unique = {node.node_id: node for node in matches}
                if len(unique) == 1:
                    suggestion = "ALIGN_EXISTING_EXACT_NAME"
                    suggested_id = next(iter(unique))
                    reasons = ["NORMALIZED_NAME_OR_ALIAS_EXACT_TYPE_MATCH"]

        suggestions.append({
            "review_item_id": row["review_item_id"],
            "dataset_id": row["dataset_id"],
            "fault_class_id": row["fault_class_id"],
            "item_kind": row["item_kind"],
            "candidate_id": row["candidate_id"],
            "node_type": row["node_type"],
            "canonical_name": row["canonical_name"],
            "canonical_id_hint": hint,
            "machine_suggestion": suggestion,
            "suggested_canonical_id": suggested_id,
            "reason_codes": ";".join(reasons),
            "batch_status": row["batch_status"],
            "bundle_sha256": row["bundle_sha256"],
            "base_graph_sha256": base_graph_sha,
        })

    if review_sheet_path.read_bytes() != original_review_bytes:
        raise RuntimeError("human review sheet changed while generating suggestions")
    output.mkdir(parents=True, exist_ok=True)
    suggestion_path = output / "machine_review_suggestions.csv"
    with suggestion_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUGGESTION_FIELDS)
        writer.writeheader()
        writer.writerows(suggestions)
    counts = dict(sorted(Counter(row["machine_suggestion"] for row in suggestions).items()))
    body = {
        "schema_version": "astrograph-document-review-suggestions.v1",
        "candidate_pack_receipt_file_sha256": _sha(pack_receipt_path),
        "review_handoff_receipt_file_sha256": _sha(review_receipt_path),
        "review_sheet_sha256": _sha(review_sheet_path),
        "base_graph_file_sha256": _sha(base_path),
        "base_graph_sha256": base_graph_sha,
        "suggestion_count": len(suggestions),
        "suggestion_counts": counts,
        "suggestion_sheet_sha256": _sha(suggestion_path),
        "human_decision_modified": False,
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "Non-binding machine assistance based only on exact typed ID or normalized name/alias "
            "matches. It is not a human decision, approval, publication, or diagnostic fact."
        ),
    }
    receipt = {**body, "receipt_sha256": _canonical_sha(body)}
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def build_document_component_archetype_suggestions(
    *,
    review_root: str | Path,
    suggestion_root: str | Path,
    channel_map_catalog_path: str | Path,
    base_graph_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Suggest archetypes only for exact component IDs in governed channel maps."""

    review_root = Path(review_root).resolve()
    suggestion_root = Path(suggestion_root).resolve()
    catalog_path = Path(channel_map_catalog_path).resolve()
    base_path = Path(base_graph_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite component archetype suggestions")
    review_sheet_path = review_root / "human_review_decisions.csv"
    suggestion_receipt_path = suggestion_root / "receipt.json"
    suggestion_sheet_path = suggestion_root / "machine_review_suggestions.csv"
    suggestion_receipt = json.loads(
        suggestion_receipt_path.read_text(encoding="utf-8")
    )
    _validate_semantic_receipt(suggestion_receipt, label="review suggestion")
    if (
        suggestion_receipt.get("review_sheet_sha256") != _sha(review_sheet_path)
        or suggestion_receipt.get("suggestion_sheet_sha256")
        != _sha(suggestion_sheet_path)
        or suggestion_receipt.get("base_graph_file_sha256") != _sha(base_path)
    ):
        raise ValueError("component archetype inputs differ from review suggestions")
    with review_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        review_rows = list(csv.DictReader(stream))
    human_fields = ["decision", "approved_canonical_id", "reviewer", "review_comment"]
    if any(any(row[field].strip() for field in human_fields) for row in review_rows):
        raise ValueError("component archetype suggestions require a blank review sheet")
    with suggestion_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        suggestions = list(csv.DictReader(stream))
    review_by_id = {row["review_item_id"]: row for row in review_rows}

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "astrograph-channel-map-catalog.v1":
        raise ValueError("unsupported component channel map catalog")
    exact_component_maps: dict[str, list[tuple[str, str]]] = {}
    for channel_map in catalog.get("maps", []):
        mapping_id = str(channel_map.get("mapping_id", ""))
        marker = ".component."
        if marker not in mapping_id:
            continue
        component_token = mapping_id.split(marker, 1)[1].rsplit(".v", 1)[0]
        archetypes = sorted(
            {
                str(entry["component_archetype_id"])
                for entry in channel_map.get("entries", [])
                if entry.get("component_archetype_id")
            }
        )
        if len(archetypes) == 1:
            exact_component_maps.setdefault(_normalise(component_token), []).append(
                (mapping_id, archetypes[0])
            )
    base_graph = KnowledgeGraphV3.model_validate_json(base_path.read_text(encoding="utf-8"))
    base_graph_sha = _canonical_sha(base_graph.model_dump(mode="json"))
    base_nodes = {node.node_id: node for node in base_graph.nodes}
    rows: list[dict[str, str]] = []
    for suggestion in suggestions:
        if suggestion["machine_suggestion"] != "DEFER_CANONICAL_MAPPING":
            continue
        review = review_by_id.get(suggestion["review_item_id"])
        if review is None or review["node_type"] != "ComponentArchetype":
            continue
        matches = exact_component_maps.get(_normalise(review["canonical_name"]), [])
        status = "DEFER_NO_EXACT_CHANNEL_MAP_COMPONENT"
        suggested_id = ""
        mapping_id = ""
        reasons = ["NO_UNIQUE_EXACT_COMPONENT_CHANNEL_MAP"]
        if len(matches) == 1:
            mapping_id, archetype_id = matches[0]
            candidate_id = f"archetype:{archetype_id}"
            node = base_nodes.get(candidate_id)
            if node is not None and node.node_type.value == "ComponentArchetype":
                status = "ALIGN_ARCHETYPE_FROM_EXACT_CHANNEL_MAP_COMPONENT_ID"
                suggested_id = candidate_id
                reasons = [
                    "EXACT_NORMALIZED_COMPONENT_ID_MATCH",
                    "CHANNEL_MAP_HAS_SINGLE_ARCHETYPE",
                    "TARGET_EXISTS_IN_BASE_GRAPH_WITH_MATCHING_TYPE",
                ]
            else:
                status = "DEFER_ARCHETYPE_TARGET_NOT_IN_BASE_GRAPH"
                reasons = ["EXACT_CHANNEL_MAP_TARGET_ABSENT_OR_WRONG_TYPE"]
        rows.append(
            {
                "review_item_id": review["review_item_id"],
                "dataset_id": review["dataset_id"],
                "fault_class_id": review["fault_class_id"],
                "candidate_id": review["candidate_id"],
                "candidate_component_name": review["canonical_name"],
                "machine_suggestion": status,
                "suggested_canonical_id": suggested_id,
                "channel_mapping_id": mapping_id,
                "reason_codes": ";".join(reasons),
                "channel_map_catalog_sha256": _sha(catalog_path),
                "base_graph_sha256": base_graph_sha,
            }
        )
    rows.sort(key=lambda row: (row["dataset_id"], row["fault_class_id"], row["candidate_id"]))
    if not rows:
        raise ValueError("review suggestions contain no deferred component candidates")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent)
    )
    try:
        sheet_path = staging / "component_archetype_suggestions.csv"
        with sheet_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=COMPONENT_ARCHETYPE_SUGGESTION_FIELDS
            )
            writer.writeheader()
            writer.writerows(rows)
        counts = dict(sorted(Counter(row["machine_suggestion"] for row in rows).items()))
        body = {
            "schema_version": "astrograph-document-component-archetype-suggestions.v1",
            "status": "MACHINE_SUGGESTIONS_READY_FOR_HUMAN_USE",
            "review_sheet_file_sha256": _sha(review_sheet_path),
            "source_suggestion_receipt_file_sha256": _sha(suggestion_receipt_path),
            "channel_map_catalog_file_sha256": _sha(catalog_path),
            "base_graph_file_sha256": _sha(base_path),
            "suggestion_count": len(rows),
            "suggestion_counts": counts,
            "suggestion_sheet_file_sha256": _sha(sheet_path),
            "human_decisions_written": False,
            "knowledge_patch_proposal_allowed": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Exact component-ID joins to existing channel-map archetypes are non-binding "
                "review assistance. Channel maps are engineering contracts, not proof that a "
                "candidate component should be approved or that the archetype improves diagnosis."
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha(body)}
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_staged_directory(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_document_review_workbook(
    *,
    candidate_pack_root: str | Path,
    review_root: str | Path,
    suggestion_root: str | Path,
    output_dir: str | Path,
    component_suggestion_root: str | Path | None = None,
) -> dict[str, Any]:
    """Merge locked graph candidates and suggestions into a fillable priority workbook."""

    pack_root = Path(candidate_pack_root).resolve()
    review_root = Path(review_root).resolve()
    suggestion_root = Path(suggestion_root).resolve()
    component_root = (
        Path(component_suggestion_root).resolve() if component_suggestion_root else None
    )
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite document graph review workbook")

    pack_receipt_path = pack_root / "receipt.json"
    review_receipt_path = review_root / "review_handoff_receipt.json"
    review_sheet_path = review_root / "human_review_decisions.csv"
    suggestion_receipt_path = suggestion_root / "receipt.json"
    suggestion_sheet_path = suggestion_root / "machine_review_suggestions.csv"
    review_receipt = json.loads(review_receipt_path.read_text(encoding="utf-8"))
    suggestion_receipt = json.loads(
        suggestion_receipt_path.read_text(encoding="utf-8")
    )
    _validate_semantic_receipt(review_receipt, label="review handoff")
    _validate_semantic_receipt(suggestion_receipt, label="review suggestion")
    if (
        suggestion_receipt.get("candidate_pack_receipt_file_sha256")
        != _sha(pack_receipt_path)
        or suggestion_receipt.get("review_handoff_receipt_file_sha256")
        != _sha(review_receipt_path)
        or suggestion_receipt.get("review_sheet_sha256") != _sha(review_sheet_path)
        or suggestion_receipt.get("suggestion_sheet_sha256")
        != _sha(suggestion_sheet_path)
    ):
        raise ValueError("document graph suggestions are not bound to supplied inputs")

    with review_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        review_reader = csv.DictReader(stream)
        if list(review_reader.fieldnames or []) != REVIEW_FIELDS:
            raise ValueError("document graph review sheet columns changed")
        review_rows = list(review_reader)
    human_fields = ["decision", "approved_canonical_id", "reviewer", "review_comment"]
    if any(any(row[field].strip() for field in human_fields) for row in review_rows):
        raise ValueError("document graph workbook requires a blank human review sheet")
    with suggestion_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
        suggestion_reader = csv.DictReader(stream)
        if list(suggestion_reader.fieldnames or []) != SUGGESTION_FIELDS:
            raise ValueError("document graph suggestion sheet columns changed")
        suggestion_rows = list(suggestion_reader)
    suggestions_by_id = {row["review_item_id"]: row for row in suggestion_rows}
    if len(suggestions_by_id) != len(suggestion_rows) or len(review_rows) != len(
        suggestion_rows
    ):
        raise ValueError("document graph review and suggestion identities differ")

    component_by_id: dict[str, dict[str, str]] = {}
    component_receipt_path: Path | None = None
    if component_root is not None:
        component_receipt_path = component_root / "receipt.json"
        component_sheet_path = component_root / "component_archetype_suggestions.csv"
        component_receipt = json.loads(
            component_receipt_path.read_text(encoding="utf-8")
        )
        _validate_semantic_receipt(component_receipt, label="component archetype suggestion")
        if (
            component_receipt.get("review_sheet_file_sha256") != _sha(review_sheet_path)
            or component_receipt.get("source_suggestion_receipt_file_sha256")
            != _sha(suggestion_receipt_path)
            or component_receipt.get("suggestion_sheet_file_sha256")
            != _sha(component_sheet_path)
        ):
            raise ValueError("component archetype suggestions target different review inputs")
        with component_sheet_path.open(encoding="utf-8-sig", newline="") as stream:
            component_reader = csv.DictReader(stream)
            if list(component_reader.fieldnames or []) != COMPONENT_ARCHETYPE_SUGGESTION_FIELDS:
                raise ValueError("component archetype suggestion columns changed")
            component_rows = list(component_reader)
        component_by_id = {row["review_item_id"]: row for row in component_rows}
        if len(component_by_id) != len(component_rows):
            raise ValueError("component archetype suggestions contain duplicate identities")

    advisory_fields = [
        "canonical_id_hint",
        "machine_suggestion",
        "suggested_canonical_id",
        "reason_codes",
        "base_graph_sha256",
        "component_mapping_suggestion",
        "component_suggested_canonical_id",
        "component_channel_mapping_id",
        "component_mapping_reason_codes",
    ]
    machine_review_fields = REVIEW_FIELDS[:-4]
    workbook_rows: list[dict[str, str]] = []
    for review in review_rows:
        suggestion = suggestions_by_id.get(review["review_item_id"])
        if suggestion is None:
            raise ValueError("document graph suggestion is missing a review identity")
        for field in (
            "dataset_id",
            "fault_class_id",
            "item_kind",
            "candidate_id",
            "node_type",
            "canonical_name",
            "batch_status",
            "bundle_sha256",
        ):
            if suggestion[field] != review[field]:
                raise ValueError(f"document graph suggestion differs from review: {field}")
        machine_suggestion = suggestion["machine_suggestion"]
        component = component_by_id.get(review["review_item_id"])
        component_suggestion = component["machine_suggestion"] if component else ""
        if machine_suggestion == "REQUIRES_CANDIDATE_CORRECTION":
            priority = "P0_BLOCKED_CANDIDATE_CORRECTION"
        elif machine_suggestion == "DEFER_RELATION_REVIEW":
            priority = "P1_RELATION_ENDPOINT_AND_SHAPE_REVIEW"
        elif machine_suggestion == "DEFER_CANONICAL_MAPPING":
            priority = (
                "P2_EXACT_CHANNEL_MAP_ARCHETYPE_CHECK"
                if component_suggestion
                == "ALIGN_ARCHETYPE_FROM_EXACT_CHANNEL_MAP_COMPONENT_ID"
                else "P2_CANONICAL_MAPPING_REQUIRED"
            )
        else:
            priority = "P3_EXACT_ALIGNMENT_CHECK"
        workbook_rows.append(
            {
                "review_priority": priority,
                **{field: review[field] for field in machine_review_fields},
                **{
                    field: suggestion[field]
                    for field in advisory_fields
                    if field in suggestion
                },
                "component_mapping_suggestion": component_suggestion,
                "component_suggested_canonical_id": (
                    component["suggested_canonical_id"] if component else ""
                ),
                "component_channel_mapping_id": (
                    component["channel_mapping_id"] if component else ""
                ),
                "component_mapping_reason_codes": (
                    component["reason_codes"] if component else ""
                ),
                **{field: "" for field in human_fields},
            }
        )
    if set(suggestions_by_id) != {row["review_item_id"] for row in review_rows}:
        raise ValueError("document graph suggestions contain unknown review identities")
    workbook_rows.sort(
        key=lambda row: (
            row["review_priority"],
            row["dataset_id"],
            row["fault_class_id"],
            row["item_kind"],
            row["candidate_id"],
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent)
    )
    try:
        baseline_path = staging / "human_review_priority_queue.csv"
        fields = list(workbook_rows[0])
        with baseline_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(workbook_rows)
        working_path = staging / "human_review_working_copy.csv"
        shutil.copyfile(baseline_path, working_path)
        counts = dict(
            sorted(Counter(row["review_priority"] for row in workbook_rows).items())
        )
        body = {
            "schema_version": "astrograph-document-graph-review-workbook.v1",
            "status": "READY_FOR_NAMED_HUMAN_REVIEW",
            "candidate_pack_receipt_file_sha256": _sha(pack_receipt_path),
            "review_handoff_receipt_file_sha256": _sha(review_receipt_path),
            "suggestion_receipt_file_sha256": _sha(suggestion_receipt_path),
            "component_suggestion_receipt_file_sha256": (
                _sha(component_receipt_path) if component_receipt_path else None
            ),
            "review_item_count": len(workbook_rows),
            "priority_counts": counts,
            "priority_queue_file_sha256": _sha(baseline_path),
            "working_copy_initial_file_sha256": _sha(working_path),
            "human_decisions_written": False,
            "knowledge_patch_proposal_allowed": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "The workbook only reorders source-bound candidates and exact-match advice for "
                "human review. It does not prefill decisions or canonical IDs and grants no "
                "KnowledgePatch, publication, or runtime authority."
            ),
        }
        receipt = {**body, "receipt_sha256": _canonical_sha(body)}
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_staged_directory(staging, output)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_document_review_workbook(
    *,
    candidate_pack_root: str | Path,
    workbook_root: str | Path,
    workbook_path: str | Path,
    output_dir: str | Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Restore canonical graph-review order and run the existing review freeze."""

    pack_root = Path(candidate_pack_root).resolve()
    workbook_root = Path(workbook_root).resolve()
    workbook_path = Path(workbook_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite compiled document graph review workbook")
    receipt_path = workbook_root / "receipt.json"
    baseline_path = workbook_root / "human_review_priority_queue.csv"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _validate_semantic_receipt(receipt, label="document graph review workbook")
    if (
        receipt.get("candidate_pack_receipt_file_sha256")
        != _sha(pack_root / "receipt.json")
        or receipt.get("priority_queue_file_sha256") != _sha(baseline_path)
    ):
        raise ValueError("document graph review workbook is not bound to candidate pack")
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        baseline_reader = csv.DictReader(stream)
        fields = list(baseline_reader.fieldnames or [])
        baseline_rows = list(baseline_reader)
    human_fields = ["decision", "approved_canonical_id", "reviewer", "review_comment"]
    if fields[-4:] != human_fields or not baseline_rows:
        raise ValueError("document graph review workbook template is invalid")
    machine_fields = fields[:-4]
    with workbook_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != fields:
            raise ValueError("document graph review workbook columns changed")
        rows = list(reader)
    if len(rows) != len(baseline_rows):
        raise ValueError("document graph review workbook row count changed")
    for index, (expected, row) in enumerate(zip(baseline_rows, rows, strict=True)):
        for field in machine_fields:
            if row.get(field, "") != expected[field]:
                raise ValueError(
                    f"document graph review workbook machine field changed: {index}:{field}"
                )
    by_id = {row["review_item_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("document graph review workbook contains duplicate identities")

    # Canonical order is deterministic and independent of the priority presentation.
    canonical_rows = sorted(
        ({field: row[field] for field in REVIEW_FIELDS} for row in rows),
        key=lambda row: (
            row["dataset_id"],
            row["fault_class_id"],
            row["item_kind"],
            row["candidate_id"],
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent)
    )
    try:
        canonical_path = staging / "human_review_decisions.csv"
        with canonical_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(canonical_rows)
        freeze_path = staging / "review_freeze.json"
        freeze = freeze_document_graph_review_decisions(
            review_sheet_path=canonical_path,
            graph_roots=[pack_root],
            output_path=freeze_path,
            require_complete=require_complete,
        )
        ready = freeze["status"] == "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL"
        body = {
            "schema_version": "astrograph-document-graph-review-workbook-compile.v1",
            "status": freeze["status"],
            "candidate_pack_receipt_file_sha256": _sha(pack_root / "receipt.json"),
            "workbook_receipt_file_sha256": _sha(receipt_path),
            "source_workbook_file_sha256": _sha(workbook_path),
            "canonical_review_sheet_file_sha256": _sha(canonical_path),
            "review_freeze_file_sha256": _sha(freeze_path),
            "review_freeze_sha256": freeze["freeze_sha256"],
            "review_item_count": len(canonical_rows),
            "decision_counts": freeze["decision_counts"],
            "knowledge_patch_proposal_allowed": ready,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Compilation only restores canonical ordering and invokes the existing source-"
                "reconstructed freeze. READY permits a separate KnowledgePatch proposal, not "
                "publication, Active KG mutation, or runtime voting."
            ),
        }
        compiled = {**body, "receipt_sha256": _canonical_sha(body)}
        (staging / "receipt.json").write_text(
            json.dumps(compiled, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_staged_directory(staging, output)
        return compiled
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "SUGGESTION_FIELDS",
    "build_document_review_suggestions",
    "build_document_component_archetype_suggestions",
    "build_document_review_workbook",
    "compile_document_review_workbook",
]
