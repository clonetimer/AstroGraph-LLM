"""Build non-binding review assistance for controlled document rewrites."""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from astrograph.filesystem import publish_staged_directory
from .document_controlled_rewrite import freeze_controlled_rewrite_review


T = TypeVar("T", bound=BaseModel)


class StructuredSuggestionBackend(Protocol):
    model_id: str
    model_version: str

    def generate_structured(
        self, *, agent_id: str, output_model: type[T], payload: dict[str, Any]
    ) -> T: ...


class RewriteReviewSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str = Field(pattern=r"^rewrite-block:[0-9a-f]{20}$")
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rewritten_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suggested_decision: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
    rationale: str = Field(min_length=1, max_length=600)
    risk_codes: list[
        Literal[
            "NONE",
            "NEGATION_CHANGED",
            "MODALITY_CHANGED",
            "CAUSAL_STRENGTH_CHANGED",
            "SCOPE_CHANGED",
            "ENTITY_OR_DIRECTION_CHANGED",
            "FACT_ADDED_OR_REMOVED",
            "AMBIGUOUS_PARAPHRASE",
        ]
    ] = Field(min_length=1, max_length=8)


class RewriteReviewSuggestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "astrograph-controlled-rewrite-review-suggestion-response.v1"
    ] = "astrograph-controlled-rewrite-review-suggestion-response.v1"
    suggestions: list[RewriteReviewSuggestion] = Field(min_length=1, max_length=20)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_sha(value: object) -> str:
    return _sha_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _deterministic_text_diff(source: str, rewritten: str) -> dict[str, Any]:
    source_tokens = re.findall(r"\w+|[^\w\s]", source.casefold())
    rewritten_tokens = re.findall(r"\w+|[^\w\s]", rewritten.casefold())
    matcher = difflib.SequenceMatcher(a=source_tokens, b=rewritten_tokens, autojunk=False)
    removed = 0
    added = 0
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            removed += left_end - left_start
        if tag in {"insert", "replace"}:
            added += right_end - right_start
    return {
        "literal_text_identical": source == rewritten,
        "token_sequence_similarity": round(matcher.ratio(), 6),
        "removed_token_count": removed,
        "added_token_count": added,
    }


def _suggestion_warning_codes(
    *, source: str, rewritten: str, suggestion: RewriteReviewSuggestion
) -> list[str]:
    warnings: list[str] = []
    literal_identity_claim = bool(
        re.search(
            r"\bidentical text\b|\bthe rewritten text is identical\b|"
            r"(?:^|[.!?]\s*)text is identical(?:[.!?]|$)",
            suggestion.rationale.casefold(),
        )
    )
    if source != rewritten and literal_identity_claim:
        warnings.append("RATIONALE_LITERAL_IDENTITY_CONTRADICTION")
    if source == rewritten and suggestion.suggested_decision != "SUPPORTED":
        warnings.append("IDENTICAL_TEXT_NOT_SUGGESTED_SUPPORTED")
    semantic_change_claim = bool(
        re.search(
            r"\bchanges? the meaning\b|\bmeaning (?:is|was) (?:changed|altered|not preserved)\b|"
            r"\bnot semantically equivalent\b",
            suggestion.rationale.casefold(),
        )
    )
    if suggestion.suggested_decision == "SUPPORTED" and semantic_change_claim:
        warnings.append("SUPPORTED_RATIONALE_FLAGS_SEMANTIC_CHANGE")
    return warnings


def build_rewrite_review_suggestions(
    *,
    rewrite_pack_root: str | Path,
    review_sheet_path: str | Path,
    output_root: str | Path,
    backend: StructuredSuggestionBackend | None = None,
    frozen_suggestion_root: str | Path | None = None,
) -> dict[str, Any]:
    """Suggest decisions beside, never inside, a blank human review sheet."""

    root = Path(rewrite_pack_root).resolve()
    sheet = Path(review_sheet_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite rewrite review suggestions")
    if (backend is None) == (frozen_suggestion_root is None):
        raise ValueError("provide exactly one suggestion backend or frozen suggestion root")

    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if _canonical_sha(receipt_body) != receipt.get("receipt_sha256"):
        raise ValueError("controlled rewrite receipt semantic hash mismatch")
    if receipt.get("status") != "HUMAN_REVIEW_REQUIRED":
        raise ValueError("controlled rewrite pack is not ready for review assistance")
    basis_path = root / "review_basis.json"
    if _sha(basis_path) != receipt.get("review_basis_sha256"):
        raise ValueError("controlled rewrite review basis changed")
    basis_rows = json.loads(basis_path.read_text(encoding="utf-8"))
    machine_fields = list(basis_rows[0]) if basis_rows else []
    human_fields = ["decision", "corrected_text", "reviewer", "review_comment"]
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [*machine_fields, *human_fields]:
            raise ValueError("controlled rewrite review columns changed")
        rows = list(reader)
    if len(rows) != len(basis_rows) or not rows:
        raise ValueError("controlled rewrite review row count changed or is empty")
    for index, (expected, row) in enumerate(zip(basis_rows, rows, strict=True)):
        if any(row.get(field, "") != str(expected[field]) for field in machine_fields):
            raise ValueError(f"controlled rewrite machine review field changed: {index}")
        if any(row.get(field, "").strip() for field in human_fields):
            raise ValueError("review suggestions require an untouched blank human decision sheet")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset_id"], row["fault_class_id"])].append(row)

    frozen_root = Path(frozen_suggestion_root).resolve() if frozen_suggestion_root else None
    frozen_receipt: dict[str, Any] | None = None
    frozen_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    frozen_trace: list[dict[str, Any]] = []
    if frozen_root is not None:
        frozen_receipt_path = frozen_root / "receipt.json"
        frozen_receipt = json.loads(frozen_receipt_path.read_text(encoding="utf-8"))
        frozen_body = {
            key: value for key, value in frozen_receipt.items() if key != "receipt_sha256"
        }
        if _canonical_sha(frozen_body) != frozen_receipt.get("receipt_sha256"):
            raise ValueError("frozen rewrite suggestion receipt semantic hash mismatch")
        if (
            frozen_receipt.get("rewrite_pack_receipt_file_sha256") != _sha(receipt_path)
            or frozen_receipt.get("review_sheet_sha256") != _sha(sheet)
        ):
            raise ValueError("frozen rewrite suggestions target different review inputs")
        frozen_by_key = {
            (str(item["dataset_id"]), str(item["fault_class_id"])): item
            for item in frozen_receipt.get("results", [])
        }
        trace_path = frozen_root / "raw_generation_trace.json"
        if _sha(trace_path) != frozen_receipt.get("raw_generation_trace_sha256"):
            raise ValueError("frozen rewrite suggestion generation trace changed")
        frozen_trace = json.loads(trace_path.read_text(encoding="utf-8"))

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent)
    )
    try:
        results: list[dict[str, Any]] = []
        flat: list[dict[str, Any]] = []
        for (dataset_id, fault_id), document_rows in sorted(grouped.items()):
            payload_blocks = [
                {
                    "block_id": row["block_id"],
                    "source_text": row["source_text"],
                    "source_text_sha256": row["source_text_sha256"],
                    "rewritten_text": row["rewritten_text"],
                    "rewritten_text_sha256": _sha_bytes(
                        row["rewritten_text"].encode("utf-8")
                    ),
                }
                for row in document_rows
            ]
            if backend is not None:
                response = backend.generate_structured(
                    agent_id="controlled-rewrite-review-suggestion-agent",
                    output_model=RewriteReviewSuggestionResponse,
                    payload={
                        "task": (
                            "Conservatively compare each rewrite to its source. Mark SUPPORTED only "
                            "when meaning, polarity, modality, entities, direction, scope and causal "
                            "strength are preserved exactly. This is advisory, not human approval."
                        ),
                        "blocks": payload_blocks,
                    },
                )
            else:
                frozen = frozen_by_key.get((dataset_id, fault_id))
                if frozen is None:
                    raise ValueError("frozen rewrite suggestions are missing a document")
                proposal_path = (frozen_root / str(frozen["proposal_path"])).resolve()
                if (
                    not proposal_path.is_relative_to(frozen_root)
                    or not proposal_path.is_file()
                    or _sha(proposal_path) != frozen.get("proposal_sha256")
                ):
                    raise ValueError("frozen rewrite suggestion proposal changed")
                response = RewriteReviewSuggestionResponse.model_validate_json(
                    proposal_path.read_text(encoding="utf-8")
                )
            response = RewriteReviewSuggestionResponse.model_validate(response)
            expected = {item["block_id"]: item for item in payload_blocks}
            actual = {item.block_id: item for item in response.suggestions}
            if len(actual) != len(response.suggestions) or set(actual) != set(expected):
                raise ValueError("rewrite suggestion block identities are incomplete or duplicated")
            for block_id, item in actual.items():
                source = expected[block_id]
                if (
                    item.source_text_sha256 != source["source_text_sha256"]
                    or item.rewritten_text_sha256 != source["rewritten_text_sha256"]
                ):
                    raise ValueError("rewrite suggestion source binding changed")
                risks = list(dict.fromkeys(item.risk_codes))
                if item.suggested_decision == "SUPPORTED" and risks != ["NONE"]:
                    raise ValueError("SUPPORTED rewrite suggestion must carry only NONE risk")
                if item.suggested_decision != "SUPPORTED" and risks == ["NONE"]:
                    raise ValueError("non-SUPPORTED rewrite suggestion must identify a risk")
                diff = _deterministic_text_diff(
                    source["source_text"], source["rewritten_text"]
                )
                warning_codes = _suggestion_warning_codes(
                    source=source["source_text"],
                    rewritten=source["rewritten_text"],
                    suggestion=item,
                )
                flat.append(
                    {
                        "dataset_id": dataset_id,
                        "fault_class_id": fault_id,
                        "block_id": block_id,
                        "source_text_sha256": item.source_text_sha256,
                        "rewritten_text_sha256": item.rewritten_text_sha256,
                        "suggested_decision": item.suggested_decision,
                        "risk_codes": risks,
                        "rationale": item.rationale,
                        **diff,
                        "suggestion_warning_codes": warning_codes,
                    }
                )
            proposal_relative = Path("proposals") / dataset_id / f"{fault_id}.json"
            proposal_path = staging / proposal_relative
            proposal_path.parent.mkdir(parents=True, exist_ok=True)
            proposal_path.write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
            results.append(
                {
                    "dataset_id": dataset_id,
                    "fault_class_id": fault_id,
                    "suggestion_count": len(actual),
                    "proposal_path": proposal_relative.as_posix(),
                    "proposal_sha256": _sha(proposal_path),
                }
            )

        flat.sort(key=lambda item: (item["dataset_id"], item["fault_class_id"], item["block_id"]))
        suggestion_path = staging / "review_suggestions.json"
        suggestion_path.write_text(
            json.dumps(flat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        source_by_key = {
            (row["dataset_id"], row["fault_class_id"], row["block_id"]): row
            for row in rows
        }
        priority_rows: list[dict[str, Any]] = []
        for item in flat:
            source_row = source_by_key[
                (item["dataset_id"], item["fault_class_id"], item["block_id"])
            ]
            if item["suggestion_warning_codes"]:
                priority = "P0_MODEL_INTERNAL_CONTRADICTION"
            elif item["suggested_decision"] != "SUPPORTED":
                priority = "P1_MODEL_FLAGS_SEMANTIC_RISK"
            elif not item["literal_text_identical"] and item["token_sequence_similarity"] < 0.8:
                priority = "P2_LOW_LITERAL_SIMILARITY"
            elif not item["literal_text_identical"]:
                priority = "P3_CHANGED_TEXT"
            else:
                priority = "P4_LITERAL_IDENTITY"
            priority_rows.append(
                {
                    "review_priority": priority,
                    "dataset_id": item["dataset_id"],
                    "fault_class_id": item["fault_class_id"],
                    "block_id": item["block_id"],
                    "source_text": source_row["source_text"],
                    "rewritten_text": source_row["rewritten_text"],
                    "literal_text_identical": item["literal_text_identical"],
                    "token_sequence_similarity": item["token_sequence_similarity"],
                    "removed_token_count": item["removed_token_count"],
                    "added_token_count": item["added_token_count"],
                    "suggested_decision": item["suggested_decision"],
                    "risk_codes": "|".join(item["risk_codes"]),
                    "suggestion_warning_codes": "|".join(
                        item["suggestion_warning_codes"]
                    ),
                    "rationale": item["rationale"],
                    "decision": "",
                    "corrected_text": "",
                    "reviewer": "",
                    "review_comment": "",
                }
            )
        priority_rows.sort(
            key=lambda item: (
                item["review_priority"],
                item["token_sequence_similarity"],
                item["dataset_id"],
                item["fault_class_id"],
                item["block_id"],
            )
        )
        queue_path = staging / "human_review_priority_queue.csv"
        with queue_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(priority_rows[0]))
            writer.writeheader()
            writer.writerows(priority_rows)
        working_copy_path = staging / "human_review_working_copy.csv"
        shutil.copyfile(queue_path, working_copy_path)
        trace = getattr(backend, "response_trace", []) if backend is not None else frozen_trace
        trace_path = staging / "raw_generation_trace.json"
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        counts = Counter(str(item["suggested_decision"]) for item in flat)
        warning_counts = Counter(
            code for item in flat for code in item["suggestion_warning_codes"]
        )
        priority_counts = Counter(str(item["review_priority"]) for item in priority_rows)
        receipt_body = {
            "schema_version": "astrograph-controlled-rewrite-review-suggestions.v1",
            "status": "MACHINE_SUGGESTIONS_READY_FOR_HUMAN_USE",
            "rewrite_pack_receipt_file_sha256": _sha(receipt_path),
            "review_sheet_sha256": _sha(sheet),
            "review_sheet_unchanged": _sha(sheet) == receipt.get("human_review_sheet_sha256"),
            "generation_mode": "LIVE_VLLM" if backend is not None else "FROZEN_REPLAY",
            "model_id": backend.model_id if backend is not None else frozen_receipt["model_id"],
            "model_version": (
                backend.model_version if backend is not None else frozen_receipt["model_version"]
            ),
            "document_count": len(results),
            "suggestion_count": len(flat),
            "suggestion_counts": dict(sorted(counts.items())),
            "suggestion_warning_counts": dict(sorted(warning_counts.items())),
            "suggestions_with_warnings": sum(
                bool(item["suggestion_warning_codes"]) for item in flat
            ),
            "review_priority_counts": dict(sorted(priority_counts.items())),
            "results": results,
            "review_suggestions_sha256": _sha(suggestion_path),
            "human_review_priority_queue_sha256": _sha(queue_path),
            "human_review_working_copy_initial_sha256": _sha(working_copy_path),
            "raw_generation_trace_sha256": _sha(trace_path),
            "human_decisions_written": False,
            "rag_admission_allowed": False,
            "kg_staging_allowed": False,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Suggestions are produced by an LLM that is not independent of the rewrite model. "
                "They may prioritize human review but cannot fill decisions, establish an "
                "unsupported-claim rate, admit documents, or approve graph knowledge."
            ),
        }
        receipt_out = {
            **receipt_body,
            "receipt_sha256": _canonical_sha(receipt_body),
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt_out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        publish_staged_directory(staging, output)
        return receipt_out
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_rewrite_review_workbook(
    *,
    rewrite_pack_root: str | Path,
    suggestion_root: str | Path,
    workbook_path: str | Path,
    output_root: str | Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Compile a priority-ordered workbook into the canonical review and freeze."""

    rewrite_root = Path(rewrite_pack_root).resolve()
    suggestion_root = Path(suggestion_root).resolve()
    workbook = Path(workbook_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("refusing to overwrite compiled rewrite review workbook")

    suggestion_receipt_path = suggestion_root / "receipt.json"
    suggestion_receipt = json.loads(
        suggestion_receipt_path.read_text(encoding="utf-8")
    )
    suggestion_body = {
        key: value
        for key, value in suggestion_receipt.items()
        if key != "receipt_sha256"
    }
    if _canonical_sha(suggestion_body) != suggestion_receipt.get("receipt_sha256"):
        raise ValueError("rewrite suggestion receipt semantic hash mismatch")
    rewrite_receipt_path = rewrite_root / "receipt.json"
    if suggestion_receipt.get("rewrite_pack_receipt_file_sha256") != _sha(
        rewrite_receipt_path
    ):
        raise ValueError("rewrite suggestions target a different rewrite pack")
    baseline_path = suggestion_root / "human_review_priority_queue.csv"
    if _sha(baseline_path) != suggestion_receipt.get(
        "human_review_priority_queue_sha256"
    ):
        raise ValueError("rewrite suggestion priority queue changed")

    human_fields = ["decision", "corrected_text", "reviewer", "review_comment"]
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        baseline_reader = csv.DictReader(stream)
        baseline_fields = list(baseline_reader.fieldnames or [])
        baseline_rows = list(baseline_reader)
    if baseline_fields[-4:] != human_fields or not baseline_rows:
        raise ValueError("rewrite suggestion workbook template is invalid")
    machine_fields = baseline_fields[:-4]
    with workbook.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if list(reader.fieldnames or []) != baseline_fields:
            raise ValueError("rewrite review workbook columns changed")
        rows = list(reader)
    if len(rows) != len(baseline_rows):
        raise ValueError("rewrite review workbook row count changed")
    for index, (expected, row) in enumerate(zip(baseline_rows, rows, strict=True)):
        for field in machine_fields:
            if row.get(field, "") != expected[field]:
                raise ValueError(
                    f"rewrite review workbook machine field changed: {index}:{field}"
                )

    basis_path = rewrite_root / "review_basis.json"
    rewrite_receipt = json.loads(rewrite_receipt_path.read_text(encoding="utf-8"))
    if _sha(basis_path) != rewrite_receipt.get("review_basis_sha256"):
        raise ValueError("controlled rewrite review basis changed")
    basis_rows = json.loads(basis_path.read_text(encoding="utf-8"))
    basis_machine_fields = list(basis_rows[0]) if basis_rows else []
    by_identity: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        identity = (row["dataset_id"], row["fault_class_id"], row["block_id"])
        if identity in by_identity:
            raise ValueError("rewrite review workbook contains duplicate identities")
        by_identity[identity] = row
    canonical_rows: list[dict[str, str]] = []
    for basis in basis_rows:
        identity = (
            str(basis["dataset_id"]),
            str(basis["fault_class_id"]),
            str(basis["block_id"]),
        )
        row = by_identity.get(identity)
        if row is None:
            raise ValueError("rewrite review workbook is missing a canonical identity")
        canonical_rows.append(
            {
                **{field: str(basis[field]) for field in basis_machine_fields},
                **{field: row[field] for field in human_fields},
            }
        )
    if len(by_identity) != len(canonical_rows):
        raise ValueError("rewrite review workbook contains unknown identities")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.astrograph-txn-", dir=output.parent)
    )
    try:
        canonical_path = staging / "human_rewrite_review.csv"
        with canonical_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=[*basis_machine_fields, *human_fields]
            )
            writer.writeheader()
            writer.writerows(canonical_rows)
        review_freeze_path = staging / "review_freeze.json"
        review_freeze = freeze_controlled_rewrite_review(
            rewrite_pack_root=rewrite_root,
            review_sheet_path=canonical_path,
            output_path=review_freeze_path,
            require_complete=require_complete,
        )
        ready = review_freeze["status"] == "READY_FOR_CORPUS_RELEASE_CONSIDERATION"
        receipt_body = {
            "schema_version": "astrograph-controlled-rewrite-review-workbook-compile.v1",
            "status": "READY_FOR_DUAL_USE_HANDOFF" if ready else "BLOCKED",
            "rewrite_pack_receipt_file_sha256": _sha(rewrite_receipt_path),
            "suggestion_receipt_file_sha256": _sha(suggestion_receipt_path),
            "source_workbook_file_sha256": _sha(workbook),
            "canonical_review_sheet_file_sha256": _sha(canonical_path),
            "review_freeze_file_sha256": _sha(review_freeze_path),
            "review_freeze_sha256": review_freeze["freeze_sha256"],
            "review_item_count": len(canonical_rows),
            "decision_counts": review_freeze["decision_counts"],
            "unsupported_claim_rate": review_freeze["unsupported_claim_rate"],
            "rag_corpus_build_allowed": ready,
            "kg_staging_allowed": ready,
            "publication_allowed": False,
            "runtime_vote_allowed": False,
            "active_graph_modified": False,
            "claim_boundary": (
                "Workbook compilation restores canonical review order and reuses the controlled "
                "rewrite freeze. READY permits only the later review-only RAG/candidate-KG handoff; "
                "it is not domain knowledge approval or runtime authority."
            ),
        }
        compiled = {
            **receipt_body,
            "receipt_sha256": _canonical_sha(receipt_body),
        }
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
    "RewriteReviewSuggestion",
    "RewriteReviewSuggestionResponse",
    "build_rewrite_review_suggestions",
    "compile_rewrite_review_workbook",
]
