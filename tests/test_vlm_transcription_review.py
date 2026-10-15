import csv
import hashlib
import json

from astrograph.retrieval.vlm_transcription_review import (
    build_vlm_transcription_review_handoff, freeze_vlm_transcription_review,
)


def _fixture(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"; source.write_bytes(b"pdf")
    transcription = tmp_path / "transcription"; transcription.mkdir()
    candidate_body = {"schema_version":"astrograph-vlm-page-transcription-candidate.v1",
        "experiment_id":"x","model_id":"m","source_pdf_sha256":hashlib.sha256(b"pdf").hexdigest(),
        "scan_image_sha256":"2"*64,"source_page":1,"prompt_sha256":"3"*64,
        "transcription":{"lines":["hello world"],"uncertain_spans":[]},
        "representation_reviewed":False,"rag_admission_allowed":False,"kg_extraction_allowed":False}
    candidate_body["candidate_sha256"] = hashlib.sha256(json.dumps(candidate_body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    candidate_path=transcription/"transcription_candidate.json";candidate_path.write_text(json.dumps(candidate_body))
    receipt={"candidate_file_sha256":hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "source_pdf_sha256":hashlib.sha256(b"pdf").hexdigest(),"scan_image_sha256":"2"*64,"source_page":1}
    (transcription/"receipt.json").write_text(json.dumps(receipt))
    class E: source_locator="pdf#page:1:block:1";text_representation="hello world"
    monkeypatch.setattr("astrograph.retrieval.vlm_transcription_review.SectionFirstDocumentIngestor.parse_pdf_layout",
                        lambda self,path:([E()],[],object()))
    handoff=tmp_path/"handoff";build_vlm_transcription_review_handoff(
        transcription_root=transcription,source_pdf_path=source,output_root=handoff)
    return source,transcription,handoff


def test_blank_vlm_review_is_blocked_and_human_approval_only_stages_text(tmp_path,monkeypatch):
    source,transcription,handoff=_fixture(tmp_path,monkeypatch)
    blocked=freeze_vlm_transcription_review(transcription_root=transcription,source_pdf_path=source,
        review_sheet_path=handoff/"human_review.csv",output_root=tmp_path/"blocked")
    assert blocked["status"]=="BLOCKED" and blocked["rag_admission_allowed"] is False
    sheet=handoff/"human_review.csv"
    with sheet.open(encoding="utf-8-sig",newline="") as s:
        reader=csv.DictReader(s);fields=list(reader.fieldnames or []);rows=list(reader)
    rows[0].update({"decision":"APPROVE_AS_IS","reviewer":"reviewer","review_comment":"checked image"})
    with sheet.open("w",encoding="utf-8-sig",newline="") as s:
        writer=csv.DictWriter(s,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    approved=freeze_vlm_transcription_review(transcription_root=transcription,source_pdf_path=source,
        review_sheet_path=sheet,output_root=tmp_path/"approved")
    assert approved["status"]=="APPROVED_REPRESENTATION"
    assert approved["rag_admission_allowed"] is True and approved["kg_staging_allowed"] is True
    assert approved["runtime_vote_allowed"] is False
    assert (tmp_path/"approved/reviewed_transcription.txt").read_text(encoding="utf-8")=="hello world\n"
