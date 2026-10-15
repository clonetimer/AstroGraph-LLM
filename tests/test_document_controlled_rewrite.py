import csv
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.retrieval.document_controlled_rewrite import (
    ControlledRewriteResponse,
    build_controlled_rewrite_pack,
    freeze_controlled_rewrite_review,
)
from astrograph.retrieval.document_governance import GovernedDocumentDescriptor
from astrograph.retrieval.document_rewrite_handoff import (
    materialize_reviewed_rewrite_dual_use_pack,
)
from astrograph.retrieval.document_rewrite_review_suggestions import (
    RewriteReviewSuggestionResponse,
    build_rewrite_review_suggestions,
    compile_rewrite_review_workbook,
)
from astrograph.knowledge.document_ingestion import (
    ingest_governed_document_to_directory,
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _Backend:
    model_id = "rewrite-fixture"
    model_version = "1"
    response_trace = [{"response_id": "fixture"}]

    def __init__(self, *, inject_number: bool = False):
        self.inject_number = inject_number

    def generate_structured(self, *, agent_id, output_model, payload):
        blocks = []
        for item in payload["blocks"]:
            text = "Rephrased: " + item["source_text"]
            if self.inject_number:
                text += " 999"
            blocks.append({
                "block_id": item["block_id"], "source_text_sha256": item["source_text_sha256"],
                "rewritten_text": text,
            })
        return ControlledRewriteResponse(
            source_document_sha256=payload["source_document_sha256"], blocks=blocks
        )


class _SuggestionBackend:
    model_id = "suggestion-fixture"
    model_version = "1"
    response_trace = [{"response_id": "suggestion-fixture"}]

    def __init__(
        self,
        *,
        bad_hash: bool = False,
        false_identity: bool = False,
        semantic_contradiction: bool = False,
    ):
        self.bad_hash = bad_hash
        self.false_identity = false_identity
        self.semantic_contradiction = semantic_contradiction

    def generate_structured(self, *, agent_id, output_model, payload):
        suggestions = []
        for item in payload["blocks"]:
            suggestions.append(
                {
                    "block_id": item["block_id"],
                    "source_text_sha256": (
                        "0" * 64 if self.bad_hash else item["source_text_sha256"]
                    ),
                    "rewritten_text_sha256": item["rewritten_text_sha256"],
                    "suggested_decision": "SUPPORTED",
                    "rationale": (
                        "Identical text."
                        if self.false_identity
                        else (
                            "This changes the meaning, but the core meaning is preserved."
                            if self.semantic_contradiction
                            else "The fixture preserves the statement meaning."
                        )
                    ),
                    "risk_codes": ["NONE"],
                }
            )
        return RewriteReviewSuggestionResponse(suggestions=suggestions)


def _source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "corpus"
    path = root / "comparisons/d/f.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "# Candidate\n\n> Review state: REQUIRED\n\n## Discriminator questions\n\n"
        "1. Is signal `x` below 2?\n2. Is response absent?\n\n## Claim boundary\n\n"
        "This is not confirmed evidence.\n",
        encoding="utf-8",
    )
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    body = {
        "schema_version": "astrograph-fault-confusion-document-pack.v1", "documents": [{
            "dataset_id": "d", "fault_class_id": "f", "path": "comparisons/d/f.md",
            "source_sha256": source_sha,
            "descriptor": {
                "document_id": "document:fixture:f:r001",
                "document_type": "DESIGN_PROPOSAL",
                "visibility_class": "DESIGN_PUBLIC",
                "source_uri": "corpus://fixture/comparisons/d/f.md",
                "source_sha256": source_sha,
                "review_state": "REVIEW_REQUIRED",
                "authority": "NON_NORMATIVE",
                "provenance_group_id": "fixture-generated-source",
                "template_version": "fixture-r001",
                "synthetic_from_simulation": True,
                "runtime_vote_allowed": False,
            },
        }],
    }
    manifest = {**body, "manifest_sha256": _canonical_sha(body)}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path


def test_controlled_rewrite_preserves_locked_structure_and_stays_unadmitted(tmp_path: Path) -> None:
    root, manifest = _source(tmp_path)
    output = tmp_path / "rewrites"
    receipt = build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=output, backend=_Backend()
    )
    assert receipt["status"] == "HUMAN_REVIEW_REQUIRED"
    assert receipt["status_counts"] == {"CANDIDATE_READY_FOR_HUMAN_REVIEW": 1}
    assert receipt["block_count"] == 3
    assert receipt["unsupported_claim_rate"] is None
    assert receipt["rag_admission_allowed"] is False
    assert receipt["kg_staging_allowed"] is False
    candidate = (output / "candidates/d/f.md").read_text(encoding="utf-8")
    assert "`x`" in candidate and "2" in candidate
    with (output / "human_rewrite_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(not row["decision"] and not row["reviewer"] for row in rows)

    replay = build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=tmp_path / "replay",
        frozen_proposal_root=output,
    )
    assert replay["generation_mode"] == "FROZEN_PROPOSAL_REPLAY"
    assert replay["results"][0]["candidate_sha256"] == receipt["results"][0]["candidate_sha256"]

    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_controlled_rewrite_pack(
            manifest_path=manifest, corpus_root=root, output_root=output, backend=_Backend()
        )


def test_controlled_rewrite_blocks_added_numeric_claim_and_source_drift(tmp_path: Path) -> None:
    root, manifest = _source(tmp_path)
    receipt = build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=tmp_path / "blocked",
        backend=_Backend(inject_number=True),
    )
    assert receipt["status"] == "BLOCKED"
    assert receipt["status_counts"] == {"BLOCKED": 1}
    assert any(
        finding["code"] == "NUMERIC_LITERAL_SET_CHANGED"
        for finding in receipt["results"][0]["findings"]
    )

    root, manifest = _source(tmp_path / "drift")
    (root / "comparisons/d/f.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_controlled_rewrite_pack(
            manifest_path=manifest, corpus_root=root, output_root=tmp_path / "must-not-exist",
            backend=_Backend(),
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_controlled_rewrite_replay_rejects_changed_proposal(tmp_path: Path) -> None:
    root, manifest = _source(tmp_path)
    frozen = tmp_path / "frozen"
    build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=frozen, backend=_Backend()
    )
    proposal = frozen / "proposals/d/f.json"
    proposal.write_text(proposal.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="proposal changed"):
        build_controlled_rewrite_pack(
            manifest_path=manifest, corpus_root=root, output_root=tmp_path / "replay",
            frozen_proposal_root=frozen,
        )


def test_controlled_rewrite_review_freezes_only_complete_zero_unsupported_gold(tmp_path: Path) -> None:
    root, manifest = _source(tmp_path)
    pack = tmp_path / "pack"
    build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=pack, backend=_Backend()
    )
    blank = freeze_controlled_rewrite_review(
        rewrite_pack_root=pack, review_sheet_path=pack / "human_rewrite_review.csv",
        output_path=tmp_path / "blank-freeze.json",
    )
    assert blank["status"] == "BLOCKED"
    assert blank["unsupported_claim_rate"] is None

    sheet = tmp_path / "completed.csv"
    with (pack / "human_rewrite_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); rows = list(reader); fields = reader.fieldnames
    for row in rows:
        row.update({"decision": "SUPPORTED", "reviewer": "domain-reviewer",
                    "review_comment": "Meaning is preserved.", "corrected_text": ""})
    with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    completed = freeze_controlled_rewrite_review(
        rewrite_pack_root=pack, review_sheet_path=sheet,
        output_path=tmp_path / "completed-freeze.json",
    )
    assert completed["status"] == "READY_FOR_CORPUS_RELEASE_CONSIDERATION"
    assert completed["unsupported_claim_rate"] == 0.0
    assert completed["publication_allowed"] is False

    rows[0]["source_text"] = "tampered"
    tampered = tmp_path / "tampered.csv"
    with tampered.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="machine review field changed"):
        freeze_controlled_rewrite_review(
            rewrite_pack_root=pack, review_sheet_path=tampered,
            output_path=tmp_path / "tampered-freeze.json",
        )


def test_reviewed_rewrite_materializes_for_both_rag_and_candidate_kg(
    tmp_path: Path,
) -> None:
    root, manifest_path = _source(tmp_path)
    pack = tmp_path / "pack"
    build_controlled_rewrite_pack(
        manifest_path=manifest_path,
        corpus_root=root,
        output_root=pack,
        backend=_Backend(),
    )
    with (pack / "human_rewrite_review.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), reader.fieldnames
    for row in rows:
        row.update(
            {
                "decision": "SUPPORTED",
                "reviewer": "domain-reviewer",
                "review_comment": "Paraphrase preserves the canonical statement.",
                "corrected_text": "",
            }
        )
    reviewed_sheet = tmp_path / "reviewed.csv"
    with reviewed_sheet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    freeze_path = tmp_path / "review-freeze.json"
    freeze_controlled_rewrite_review(
        rewrite_pack_root=pack,
        review_sheet_path=reviewed_sheet,
        output_path=freeze_path,
    )

    handoff_root = tmp_path / "dual-use"
    receipt = materialize_reviewed_rewrite_dual_use_pack(
        source_manifest_path=manifest_path,
        rewrite_pack_root=pack,
        review_freeze_path=freeze_path,
        output_root=handoff_root,
        release_id="fixture-reviewed-r001",
    )
    assert receipt["status"] == "READY_FOR_REVIEW_ONLY_DUAL_PROCESSING"
    assert receipt["rag_corpus_build_allowed"] is True
    assert receipt["kg_staging_allowed"] is True
    assert receipt["runtime_retrieval_allowed"] is False
    assert receipt["knowledge_claims_human_reviewed"] is False
    assert receipt["publication_allowed"] is False

    manifest = json.loads((handoff_root / "manifest.json").read_text(encoding="utf-8"))
    row = manifest["documents"][0]
    assert row["descriptor"]["review_state"] == "REVIEW_REQUIRED"
    assert row["descriptor"]["runtime_vote_allowed"] is False
    assert row["source_sha256"] != row["canonical_source_sha256"]
    assert hashlib.sha256(
        (handoff_root / row["path"]).read_bytes()
    ).hexdigest() == row["source_sha256"]

    # The derived bytes go through the same one-parse dual route as a real source.
    ingestion = ingest_governed_document_to_directory(
        descriptor=GovernedDocumentDescriptor.model_validate(row["descriptor"]),
        source_path=handoff_root / row["path"],
        output_dir=tmp_path / "ingestion",
        independence_group="fixture-generated-source",
        license_id="fixture-research-only",
    )
    assert ingestion.retrieval_chunk_count > 0
    assert ingestion.knowledge_section_count > 0
    assert ingestion.crosswalk_complete is True
    assert ingestion.candidate_graph_built is False


def test_rewrite_dual_use_handoff_fails_closed_on_incomplete_or_tampered_review(
    tmp_path: Path,
) -> None:
    root, manifest_path = _source(tmp_path)
    pack = tmp_path / "pack"
    build_controlled_rewrite_pack(
        manifest_path=manifest_path,
        corpus_root=root,
        output_root=pack,
        backend=_Backend(),
    )
    blocked_freeze = tmp_path / "blocked-freeze.json"
    freeze_controlled_rewrite_review(
        rewrite_pack_root=pack,
        review_sheet_path=pack / "human_rewrite_review.csv",
        output_path=blocked_freeze,
    )
    with pytest.raises(ValueError, match="not ready"):
        materialize_reviewed_rewrite_dual_use_pack(
            source_manifest_path=manifest_path,
            rewrite_pack_root=pack,
            review_freeze_path=blocked_freeze,
            output_root=tmp_path / "must-not-exist",
            release_id="fixture-reviewed-r001",
        )
    assert not (tmp_path / "must-not-exist").exists()

    payload = json.loads(blocked_freeze.read_text(encoding="utf-8"))
    payload["status"] = "READY_FOR_CORPUS_RELEASE_CONSIDERATION"
    tampered = tmp_path / "tampered-freeze.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        materialize_reviewed_rewrite_dual_use_pack(
            source_manifest_path=manifest_path,
            rewrite_pack_root=pack,
            review_freeze_path=tampered,
            output_root=tmp_path / "tampered-output",
            release_id="fixture-reviewed-r001",
        )


def test_rewrite_review_suggestions_are_separate_replayable_and_non_admitting(
    tmp_path: Path,
) -> None:
    root, manifest = _source(tmp_path)
    pack = tmp_path / "pack"
    build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=pack, backend=_Backend()
    )
    sheet = pack / "human_rewrite_review.csv"
    before = hashlib.sha256(sheet.read_bytes()).hexdigest()
    output = tmp_path / "suggestions"
    receipt = build_rewrite_review_suggestions(
        rewrite_pack_root=pack,
        review_sheet_path=sheet,
        output_root=output,
        backend=_SuggestionBackend(),
    )
    assert receipt["status"] == "MACHINE_SUGGESTIONS_READY_FOR_HUMAN_USE"
    assert receipt["suggestion_counts"] == {"SUPPORTED": 3}
    assert receipt["human_decisions_written"] is False
    assert receipt["rag_admission_allowed"] is False
    assert receipt["kg_staging_allowed"] is False
    assert hashlib.sha256(sheet.read_bytes()).hexdigest() == before
    suggestions = json.loads((output / "review_suggestions.json").read_text(encoding="utf-8"))
    assert all(item["literal_text_identical"] is False for item in suggestions)
    assert all(0.0 < item["token_sequence_similarity"] < 1.0 for item in suggestions)
    assert receipt["suggestions_with_warnings"] == 0
    with (output / "human_review_priority_queue.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        queue = list(csv.DictReader(stream))
    assert len(queue) == 3
    assert all(item["review_priority"] == "P3_CHANGED_TEXT" for item in queue)
    assert receipt["review_priority_counts"] == {"P3_CHANGED_TEXT": 3}
    assert (output / "human_review_working_copy.csv").read_bytes() == (
        output / "human_review_priority_queue.csv"
    ).read_bytes()

    blocked_compile = compile_rewrite_review_workbook(
        rewrite_pack_root=pack,
        suggestion_root=output,
        workbook_path=output / "human_review_priority_queue.csv",
        output_root=tmp_path / "compiled-blank",
    )
    assert blocked_compile["status"] == "BLOCKED"
    assert blocked_compile["rag_corpus_build_allowed"] is False

    fields = list(queue[0])
    for row in queue:
        row.update(
            {
                "decision": "SUPPORTED",
                "reviewer": "domain-reviewer",
                "review_comment": "Meaning preserved after direct comparison.",
            }
        )
    completed_workbook = tmp_path / "completed-workbook.csv"
    with completed_workbook.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
    completed_compile = compile_rewrite_review_workbook(
        rewrite_pack_root=pack,
        suggestion_root=output,
        workbook_path=completed_workbook,
        output_root=tmp_path / "compiled-complete",
    )
    assert completed_compile["status"] == "READY_FOR_DUAL_USE_HANDOFF"
    assert completed_compile["rag_corpus_build_allowed"] is True
    assert completed_compile["kg_staging_allowed"] is True
    assert json.loads(
        (tmp_path / "compiled-complete/review_freeze.json").read_text(encoding="utf-8")
    )["status"] == "READY_FOR_CORPUS_RELEASE_CONSIDERATION"

    replay = build_rewrite_review_suggestions(
        rewrite_pack_root=pack,
        review_sheet_path=sheet,
        output_root=tmp_path / "suggestions-replay",
        frozen_suggestion_root=output,
    )
    assert replay["generation_mode"] == "FROZEN_REPLAY"
    assert replay["suggestion_counts"] == receipt["suggestion_counts"]
    assert replay["review_suggestions_sha256"] == receipt["review_suggestions_sha256"]


def test_rewrite_review_suggestions_reject_nonblank_sheet_and_binding_drift(
    tmp_path: Path,
) -> None:
    root, manifest = _source(tmp_path)
    pack = tmp_path / "pack"
    build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=pack, backend=_Backend()
    )
    sheet = pack / "human_rewrite_review.csv"
    with pytest.raises(ValueError, match="source binding changed"):
        build_rewrite_review_suggestions(
            rewrite_pack_root=pack,
            review_sheet_path=sheet,
            output_root=tmp_path / "bad-hash",
            backend=_SuggestionBackend(bad_hash=True),
        )
    assert not (tmp_path / "bad-hash").exists()

    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), reader.fieldnames
    rows[0]["decision"] = "SUPPORTED"
    nonblank = tmp_path / "nonblank.csv"
    with nonblank.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="untouched blank"):
        build_rewrite_review_suggestions(
            rewrite_pack_root=pack,
            review_sheet_path=nonblank,
            output_root=tmp_path / "nonblank-output",
            backend=_SuggestionBackend(),
        )

    suggestion_root = tmp_path / "suggestions"
    build_rewrite_review_suggestions(
        rewrite_pack_root=pack,
        review_sheet_path=sheet,
        output_root=suggestion_root,
        backend=_SuggestionBackend(),
    )
    with (suggestion_root / "human_review_priority_queue.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        workbook_rows, workbook_fields = list(reader), reader.fieldnames
    workbook_rows[0]["rationale"] = "tampered advisory field"
    tampered_workbook = tmp_path / "tampered-workbook.csv"
    with tampered_workbook.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=workbook_fields)
        writer.writeheader()
        writer.writerows(workbook_rows)
    with pytest.raises(ValueError, match="machine field changed"):
        compile_rewrite_review_workbook(
            rewrite_pack_root=pack,
            suggestion_root=suggestion_root,
            workbook_path=tampered_workbook,
            output_root=tmp_path / "tampered-compiled",
        )


def test_rewrite_review_suggestions_flag_false_literal_identity_rationale(
    tmp_path: Path,
) -> None:
    root, manifest = _source(tmp_path)
    pack = tmp_path / "pack"
    build_controlled_rewrite_pack(
        manifest_path=manifest, corpus_root=root, output_root=pack, backend=_Backend()
    )
    output = tmp_path / "suggestions"
    receipt = build_rewrite_review_suggestions(
        rewrite_pack_root=pack,
        review_sheet_path=pack / "human_rewrite_review.csv",
        output_root=output,
        backend=_SuggestionBackend(false_identity=True),
    )
    assert receipt["suggestion_warning_counts"] == {
        "RATIONALE_LITERAL_IDENTITY_CONTRADICTION": 3
    }
    assert receipt["suggestions_with_warnings"] == 3
    assert all(
        item["suggestion_warning_codes"]
        == ["RATIONALE_LITERAL_IDENTITY_CONTRADICTION"]
        for item in json.loads((output / "review_suggestions.json").read_text(encoding="utf-8"))
    )

    contradiction = tmp_path / "semantic-contradiction"
    contradiction_receipt = build_rewrite_review_suggestions(
        rewrite_pack_root=pack,
        review_sheet_path=pack / "human_rewrite_review.csv",
        output_root=contradiction,
        backend=_SuggestionBackend(semantic_contradiction=True),
    )
    assert contradiction_receipt["suggestion_warning_counts"] == {
        "SUPPORTED_RATIONALE_FLAGS_SEMANTIC_CHANGE": 3
    }
