"""Create source-bound LLM rewrite candidates without admitting them to retrieval or KG."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field


T = TypeVar("T", bound=BaseModel)
_NUMBERED = re.compile(r"^(\d+\.\s+)(.+)$")
_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_CODE = re.compile(r"`([^`]+)`")


class StructuredRewriteBackend(Protocol):
    model_id: str
    model_version: str

    def generate_structured(
        self, *, agent_id: str, output_model: type[T], payload: dict[str, Any]
    ) -> T: ...


class ControlledRewriteBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    block_id: str = Field(pattern=r"^rewrite-block:[0-9a-f]{20}$")
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rewritten_text: str = Field(min_length=1, max_length=1200)


class ControlledRewriteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "astrograph-controlled-document-rewrite-response.v1"
    source_document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blocks: list[ControlledRewriteBlock] = Field(min_length=1, max_length=20)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_sha(value: object) -> str:
    return _sha_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _extract_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    in_claim_boundary = False
    for line_index, line in enumerate(text.splitlines()):
        if line.startswith("## "):
            in_claim_boundary = line.strip() == "## Claim boundary"
            continue
        numbered = _NUMBERED.match(line)
        if numbered:
            prefix, source = numbered.groups()
            kind = "NUMBERED_QUESTION"
        elif in_claim_boundary and line.strip() and not line.startswith(("#", ">", "-", "|")):
            prefix, source, kind = "", line, "CLAIM_BOUNDARY_PROSE"
        else:
            continue
        digest = _sha_bytes(source.encode("utf-8"))
        blocks.append({
            "block_id": f"rewrite-block:{_canonical_sha({'line_index': line_index, 'text': source})[:20]}",
            "line_index": line_index,
            "kind": kind,
            "prefix": prefix,
            "source_text": source,
            "source_text_sha256": digest,
        })
    if not blocks:
        raise ValueError("source document has no controlled-rewrite blocks")
    return blocks


def _validate_and_render(
    source_text: str, blocks: list[dict[str, Any]], response: ControlledRewriteResponse
) -> tuple[str, list[dict[str, str]]]:
    findings: list[dict[str, str]] = []
    expected = {block["block_id"]: block for block in blocks}
    received: dict[str, ControlledRewriteBlock] = {}
    for item in response.blocks:
        if item.block_id in received:
            findings.append({"severity": "BLOCKER", "code": "DUPLICATE_BLOCK_ID", "block_id": item.block_id})
        received[item.block_id] = item
    missing = sorted(set(expected) - set(received))
    extra = sorted(set(received) - set(expected))
    findings.extend({"severity": "BLOCKER", "code": "MISSING_BLOCK", "block_id": item} for item in missing)
    findings.extend({"severity": "BLOCKER", "code": "UNKNOWN_BLOCK", "block_id": item} for item in extra)

    lines = source_text.splitlines()
    for block_id, source in expected.items():
        item = received.get(block_id)
        if item is None:
            continue
        if item.source_text_sha256 != source["source_text_sha256"]:
            findings.append({
                "severity": "WARNING", "code": "SOURCE_BLOCK_HASH_ECHO_REPAIRED",
                "block_id": block_id,
            })
        rewritten = item.rewritten_text.strip()
        if "\n" in rewritten or rewritten.startswith(("#", ">", "- ", "|", "```")):
            findings.append({"severity": "BLOCKER", "code": "MARKDOWN_STRUCTURE_INJECTION", "block_id": block_id})
            continue
        if Counter(_CODE.findall(rewritten)) != Counter(_CODE.findall(source["source_text"])):
            findings.append({"severity": "BLOCKER", "code": "CODE_TOKEN_SET_CHANGED", "block_id": block_id})
            continue
        if Counter(_NUMBER.findall(rewritten)) != Counter(_NUMBER.findall(source["source_text"])):
            findings.append({"severity": "BLOCKER", "code": "NUMERIC_LITERAL_SET_CHANGED", "block_id": block_id})
            continue
        lines[source["line_index"]] = source["prefix"] + rewritten
    rendered = "\n".join(lines) + ("\n" if source_text.endswith("\n") else "")
    if Counter(_CODE.findall(rendered)) != Counter(_CODE.findall(source_text)):
        findings.append({"severity": "BLOCKER", "code": "DOCUMENT_CODE_TOKEN_SET_CHANGED", "block_id": "document"})
    if Counter(_NUMBER.findall(rendered)) != Counter(_NUMBER.findall(source_text)):
        findings.append({"severity": "BLOCKER", "code": "DOCUMENT_NUMERIC_LITERAL_SET_CHANGED", "block_id": "document"})
    source_structure = [line for line in source_text.splitlines() if line.startswith(("#", ">", "- ", "|"))]
    rendered_structure = [line for line in rendered.splitlines() if line.startswith(("#", ">", "- ", "|"))]
    if source_structure != rendered_structure:
        findings.append({"severity": "BLOCKER", "code": "LOCKED_MARKDOWN_STRUCTURE_CHANGED", "block_id": "document"})
    return rendered, findings


def build_controlled_rewrite_pack(
    *, manifest_path: str | Path, corpus_root: str | Path, output_root: str | Path,
    backend: StructuredRewriteBackend | None = None,
    frozen_proposal_root: str | Path | None = None,
    maximum_documents: int | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    corpus_root = Path(corpus_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite nonempty controlled rewrite pack")
    if (backend is None) == (frozen_proposal_root is None):
        raise ValueError("select exactly one controlled rewrite source: backend or frozen proposal pack")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("manifest_sha256")
    semantic = _canonical_sha({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if not declared or semantic != declared:
        raise ValueError("source manifest semantic hash mismatch")
    documents = list(manifest.get("documents", []))
    if maximum_documents is not None:
        if maximum_documents < 1:
            raise ValueError("maximum_documents must be positive")
        documents = documents[:maximum_documents]
    if not documents:
        raise ValueError("source manifest has no selected documents")

    frozen_root = Path(frozen_proposal_root).resolve() if frozen_proposal_root else None
    frozen_receipt: dict[str, Any] | None = None
    frozen_results: dict[tuple[str, str], dict[str, Any]] = {}
    frozen_trace: list[dict[str, Any]] = []
    if frozen_root is not None:
        frozen_receipt_path = frozen_root / "receipt.json"
        frozen_receipt = json.loads(frozen_receipt_path.read_text(encoding="utf-8"))
        frozen_body = {k: v for k, v in frozen_receipt.items() if k != "receipt_sha256"}
        if _canonical_sha(frozen_body) != frozen_receipt.get("receipt_sha256"):
            raise ValueError("frozen controlled rewrite receipt semantic hash mismatch")
        if (
            frozen_receipt.get("source_manifest_file_sha256") != _sha(manifest_path)
            or frozen_receipt.get("source_manifest_sha256") != declared
        ):
            raise ValueError("frozen controlled rewrite pack targets a different source manifest")
        frozen_results = {
            (str(item["dataset_id"]), str(item["fault_class_id"])): item
            for item in frozen_receipt.get("results", [])
        }
        trace_path = frozen_root / "raw_generation_trace.json"
        if _sha(trace_path) != frozen_receipt.get("raw_generation_trace_sha256"):
            raise ValueError("frozen controlled rewrite generation trace changed")
        frozen_trace = json.loads(trace_path.read_text(encoding="utf-8"))

    # Validate the entire selected source set before making the output directory.
    sources: list[tuple[dict[str, Any], Path, str, list[dict[str, Any]]]] = []
    for document in documents:
        path = (corpus_root / document["path"]).resolve()
        if not path.is_relative_to(corpus_root) or not path.is_file():
            raise ValueError("controlled rewrite source path is missing or outside corpus root")
        if _sha(path) != document["source_sha256"]:
            raise ValueError("controlled rewrite source hash mismatch")
        text = path.read_text(encoding="utf-8")
        sources.append((document, path, text, _extract_blocks(text)))

    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    review_rows: list[dict[str, str]] = []
    review_machine_fields = [
        "dataset_id", "fault_class_id", "source_document_sha256", "block_id",
        "source_text_sha256", "source_text", "rewritten_text", "machine_status",
    ]
    for document, _, source_text, blocks in sources:
        payload = {
                "task": "Paraphrase each supplied block without adding, deleting, strengthening, or weakening claims.",
                "source_document_sha256": document["source_sha256"],
                "blocks": [{k: block[k] for k in ("block_id", "kind", "source_text", "source_text_sha256")} for block in blocks],
                "rules": [
                    "Return exactly one item for every block_id and preserve source_text_sha256.",
                    "Keep every code token and numeric literal exactly; do not add either.",
                    "Use one plain-text line per block and do not emit Markdown structure.",
                    "Do not introduce a diagnosis, threshold, causal certainty, implementation claim, or runtime authority.",
                ],
            }
        if backend is not None:
            response = backend.generate_structured(
                agent_id="controlled-document-rewrite-agent",
                output_model=ControlledRewriteResponse,
                payload=payload,
            )
        else:
            key = (str(document["dataset_id"]), str(document["fault_class_id"]))
            frozen_result = frozen_results.get(key)
            if frozen_result is None:
                raise ValueError("frozen controlled rewrite proposal is missing a selected document")
            proposal_path = (frozen_root / frozen_result["proposal_path"]).resolve()
            if not proposal_path.is_relative_to(frozen_root):
                raise ValueError("frozen controlled rewrite proposal path escapes pack")
            if _sha(proposal_path) != frozen_result["proposal_sha256"]:
                raise ValueError("frozen controlled rewrite proposal changed")
            response = ControlledRewriteResponse.model_validate_json(
                proposal_path.read_text(encoding="utf-8")
            )
        response = ControlledRewriteResponse.model_validate(response)
        findings: list[dict[str, str]] = []
        if response.source_document_sha256 != document["source_sha256"]:
            findings.append({"severity": "BLOCKER", "code": "SOURCE_DOCUMENT_HASH_MISMATCH", "block_id": "document"})
        rendered, render_findings = _validate_and_render(source_text, blocks, response)
        findings.extend(render_findings)
        relative = Path(document["dataset_id"]) / f"{document['fault_class_id']}.md"
        target = output / "candidates" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        response_path = output / "proposals" / relative.with_suffix(".json")
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
        status = (
            "CANDIDATE_READY_FOR_HUMAN_REVIEW"
            if not any(item.get("severity") == "BLOCKER" for item in findings)
            else "BLOCKED"
        )
        results.append({
            "dataset_id": document["dataset_id"], "fault_class_id": document["fault_class_id"],
            "source_sha256": document["source_sha256"], "candidate_path": str(Path("candidates") / relative),
            "candidate_sha256": _sha(target), "proposal_path": str(Path("proposals") / relative.with_suffix(".json")),
            "proposal_sha256": _sha(response_path), "block_count": len(blocks), "status": status,
            "findings": findings, "rag_admission_allowed": False, "kg_staging_allowed": False,
        })
        by_id = {item.block_id: item for item in response.blocks}
        for block in blocks:
            item = by_id.get(block["block_id"])
            review_rows.append({
                "dataset_id": document["dataset_id"], "fault_class_id": document["fault_class_id"],
                "source_document_sha256": document["source_sha256"], "block_id": block["block_id"],
                "source_text_sha256": block["source_text_sha256"], "source_text": block["source_text"],
                "rewritten_text": item.rewritten_text if item else "", "machine_status": status,
                "decision": "", "corrected_text": "", "reviewer": "", "review_comment": "",
            })

    basis_path = output / "review_basis.json"
    basis_path.write_text(
        json.dumps(
            [{field: row[field] for field in review_machine_fields} for row in review_rows],
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    review_path = output / "human_rewrite_review.csv"
    fields = list(review_rows[0])
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(review_rows)
    trace = getattr(backend, "response_trace", []) if backend is not None else frozen_trace
    trace_path = output / "raw_generation_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = dict(sorted(Counter(item["status"] for item in results).items()))
    body = {
        "schema_version": "astrograph-controlled-document-rewrite-pack.v1",
        "status": "HUMAN_REVIEW_REQUIRED" if counts.get("BLOCKED", 0) == 0 else "BLOCKED",
        "source_manifest_file_sha256": _sha(manifest_path), "source_manifest_sha256": declared,
        "generation_mode": "LIVE_VLLM" if backend is not None else "FROZEN_PROPOSAL_REPLAY",
        "frozen_proposal_receipt_file_sha256": (
            _sha(frozen_root / "receipt.json") if frozen_root is not None else None
        ),
        "model_id": backend.model_id if backend is not None else frozen_receipt["model_id"],
        "model_version": backend.model_version if backend is not None else frozen_receipt["model_version"],
        "document_count": len(results), "block_count": len(review_rows), "status_counts": counts,
        "results": results, "human_review_sheet_sha256": _sha(review_path),
        "review_basis_sha256": _sha(basis_path),
        "raw_generation_trace_sha256": _sha(trace_path), "unsupported_claim_rate": None,
        "rag_admission_allowed": False, "kg_staging_allowed": False, "publication_allowed": False,
        "active_graph_modified": False,
        "claim_boundary": (
            "Structurally gated LLM paraphrase candidates only. Human claim review remains required; "
            "null unsupported_claim_rate must not be interpreted as zero."
        ),
    }
    receipt = {**body, "receipt_sha256": _canonical_sha(body)}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def freeze_controlled_rewrite_review(
    *, rewrite_pack_root: str | Path, review_sheet_path: str | Path,
    output_path: str | Path, require_complete: bool = True,
) -> dict[str, Any]:
    """Freeze human semantic judgments without admitting rewritten documents."""

    root = Path(rewrite_pack_root).resolve()
    sheet = Path(review_sheet_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite controlled rewrite review freeze")
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _canonical_sha(body) != receipt.get("receipt_sha256"):
        raise ValueError("controlled rewrite receipt semantic hash mismatch")
    for result in receipt["results"]:
        for path_key, sha_key in (
            ("candidate_path", "candidate_sha256"), ("proposal_path", "proposal_sha256")
        ):
            path = (root / result[path_key]).resolve()
            if not path.is_relative_to(root) or not path.is_file() or _sha(path) != result[sha_key]:
                raise ValueError(f"controlled rewrite artifact is missing or changed: {path_key}")
    trace_path = root / "raw_generation_trace.json"
    basis_path = root / "review_basis.json"
    if _sha(trace_path) != receipt["raw_generation_trace_sha256"]:
        raise ValueError("controlled rewrite generation trace changed")
    if _sha(basis_path) != receipt.get("review_basis_sha256"):
        raise ValueError("controlled rewrite review basis changed or is unavailable")
    basis_rows = json.loads(basis_path.read_text(encoding="utf-8"))
    machine_fields = list(basis_rows[0]) if basis_rows else []
    human_fields = ["decision", "corrected_text", "reviewer", "review_comment"]
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [*machine_fields, *human_fields]:
            raise ValueError("controlled rewrite review columns changed")
        rows = list(reader)
    if len(rows) != len(basis_rows):
        raise ValueError("controlled rewrite review row count changed")
    blockers: list[dict[str, str]] = []
    decision_counts: Counter[str] = Counter()
    frozen_decisions: list[dict[str, str]] = []
    allowed = {"SUPPORTED", "UNSUPPORTED", "PARTIAL"}
    for index, (expected, row) in enumerate(zip(basis_rows, rows, strict=True)):
        item_id = f"{row.get('dataset_id', '')}:{row.get('fault_class_id', '')}:{row.get('block_id', '')}"
        for field in machine_fields:
            if row.get(field, "") != str(expected[field]):
                raise ValueError(f"controlled rewrite machine review field changed: {index}:{field}")
        decision = row["decision"].strip().upper()
        reviewer = row["reviewer"].strip()
        comment = row["review_comment"].strip()
        corrected = row["corrected_text"].strip()
        if decision not in allowed:
            blockers.append({"code": "VALID_DECISION_REQUIRED", "item_id": item_id})
        if not reviewer or not comment:
            blockers.append({"code": "NAMED_REVIEWER_AND_COMMENT_REQUIRED", "item_id": item_id})
        if decision == "PARTIAL" and not corrected:
            blockers.append({"code": "PARTIAL_CORRECTION_REQUIRED", "item_id": item_id})
        if decision != "PARTIAL" and corrected:
            blockers.append({"code": "CORRECTION_ONLY_ALLOWED_FOR_PARTIAL", "item_id": item_id})
        if corrected and (
            Counter(_CODE.findall(corrected)) != Counter(_CODE.findall(row["source_text"]))
            or Counter(_NUMBER.findall(corrected)) != Counter(_NUMBER.findall(row["source_text"]))
        ):
            blockers.append({"code": "CORRECTION_TOKEN_OR_NUMBER_SET_CHANGED", "item_id": item_id})
        if decision in allowed:
            decision_counts[decision] += 1
        frozen_decisions.append({
            "dataset_id": row["dataset_id"], "fault_class_id": row["fault_class_id"],
            "block_id": row["block_id"], "decision": decision,
            "corrected_text": corrected, "reviewer": reviewer, "review_comment": comment,
        })
    complete = not any(item["code"] in {
        "VALID_DECISION_REQUIRED", "NAMED_REVIEWER_AND_COMMENT_REQUIRED",
        "PARTIAL_CORRECTION_REQUIRED", "CORRECTION_ONLY_ALLOWED_FOR_PARTIAL",
        "CORRECTION_TOKEN_OR_NUMBER_SET_CHANGED",
    } for item in blockers)
    if not require_complete and not complete:
        # The flag permits a durable partial freeze, never readiness promotion.
        pass
    unsupported_rate = (
        (decision_counts["UNSUPPORTED"] + decision_counts["PARTIAL"]) / len(rows)
        if complete and rows else None
    )
    ready = complete and unsupported_rate == 0.0 and receipt["status"] == "HUMAN_REVIEW_REQUIRED"
    freeze_body = {
        "schema_version": "astrograph-controlled-document-rewrite-review-freeze.v1",
        "status": "READY_FOR_CORPUS_RELEASE_CONSIDERATION" if ready else "BLOCKED",
        "rewrite_pack_receipt_file_sha256": _sha(receipt_path),
        "review_sheet_sha256": _sha(sheet), "review_basis_sha256": _sha(basis_path),
        "review_item_count": len(rows), "decision_counts": dict(sorted(decision_counts.items())),
        "blockers": blockers, "decisions": frozen_decisions,
        "unsupported_claim_rate": unsupported_rate,
        "rag_admission_allowed": False, "kg_staging_allowed": False,
        "publication_allowed": False, "active_graph_modified": False,
        "claim_boundary": (
            "A complete zero-unsupported human review only permits later corpus-release consideration. "
            "This freeze does not index, stage, publish, or grant runtime voting authority."
        ),
    }
    freeze = {**freeze_body, "freeze_sha256": _canonical_sha(freeze_body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return freeze


__all__ = [
    "ControlledRewriteBlock", "ControlledRewriteResponse", "build_controlled_rewrite_pack",
    "freeze_controlled_rewrite_review",
]
