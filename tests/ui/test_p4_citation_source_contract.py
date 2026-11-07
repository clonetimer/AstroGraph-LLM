"""Render citation dialogs from actual synthetic backend graph/memory records.

This checks the browser presentation contract, not LLM claim accuracy or real
mission provenance. The fixture was captured from material-knowledge tests.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESOURCES = ROOT / "src" / "astrograph" / "resources"
FIXTURE = ROOT / "research-evaluations" / "p4-e5-browser-20261107" / "citation_sources_synthetic.json"
MODEL_REPORT = ROOT / "research-evaluations" / "p4-e3-local-pilot-20261107" / "platform_tc01_zh" / "report.json"


def test_graph_and_historical_source_dialogs_in_chrome() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    html = (RESOURCES / "mission_console.html").read_text(encoding="utf-8")
    match = re.search(r'<dialog class="modal evidence-detail-dialog".*?</dialog>', html, re.S)
    assert match
    script = (RESOURCES / "mission_console.js").read_text(encoding="utf-8")
    assert script.rstrip().endswith("main();")
    script = script[: script.rfind("main();")]
    samples = [
        ("uploaded_reference", data["uploaded_reference"]["candidate"]["reference_retrieval"]["hits"][0]["knowledge_id"],
         "本次上传资料", "不是当前卫星遥测", ""),
        ("graph", data["graph"]["candidate"]["graph_evidence_paths"][0]["graph_path_id"],
         "已审图谱关联", "不是本次任务的直接观测", data["graph"]["candidate"]["graph_evidence_paths"][0]["evidence_label"]),
        ("approved_history", next(x["knowledge_id"] for x in data["approved_history"]["candidate"]["reference_retrieval"]["hits"]
                                  if x.get("knowledge_validation_status") == "EXPERT_APPROVED_HISTORICAL_REFERENCE"),
         "历史已审核参考资料", "不是当前卫星遥测", ""),
        ("approved_history", data["approved_history"]["candidate"]["historical_confirmed_cases"]["hits"][0]["case_id"],
         "历史核验案例", "不能直接当作当前任务结论", data["approved_history"]["candidate"]["historical_confirmed_cases"]["hits"][0]["verified_fault_id"]),
        ("misclassification_history", data["misclassification_history"]["candidate"]["historical_confirmed_cases"]["hits"][0]["case_id"],
         "历史核验案例", "不能直接当作当前任务结论", "历史反例"),
    ]
    evidence_dir = Path(os.environ["ASTROGRAPH_UI_EVIDENCE_DIR"]) if os.environ.get("ASTROGRAPH_UI_EVIDENCE_DIR") else None
    if evidence_dir:
        evidence_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        launch = {"headless": True, "args": ["--no-sandbox"]}
        if os.environ.get("ASTROGRAPH_CHROMIUM_EXECUTABLE"):
            launch["executable_path"] = os.environ["ASTROGRAPH_CHROMIUM_EXECUTABLE"]
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1100, "height": 800}, locale="zh-CN")
        page.set_content(match.group(0))
        page.add_style_tag(content=(RESOURCES / "mission_console.css").read_text(encoding="utf-8"))
        page.add_script_tag(content=script)
        for index, (case, citation_id, role_label, boundary, excerpt) in enumerate(samples, start=1):
            result = page.evaluate(
                """({candidate, citationId}) => {
                    S.report = {analysis: {
                        result_log_review: {evidence: []},
                        knowledge_enrichment: {},
                        material_reasoning: {candidates: [candidate]}
                    }};
                    const source = citationSource(citationId);
                    if (!source) return null;
                    showCitationSource(source, '引用来源');
                    return {role: source.role, label: citationDisplay(citationId, source)};
                }""",
                arg={"candidate": data[case]["candidate"], "citationId": citation_id},
            )
            assert result is not None
            if case == "graph":
                assert result["label"] == "图谱路径"
            if case == "uploaded_reference":
                assert "reference.md" in result["label"]
                assert "upload:" not in result["label"]
            if case == "approved_history" and citation_id.startswith("approved:"):
                assert "reference.md" in result["label"]
                assert "approved:" not in result["label"]
            assert page.locator("#evidence-detail[open]").count() == 1
            assert role_label in page.locator("#evidence-source-meta").inner_text()
            assert citation_id in page.locator("#evidence-source-meta").inner_text()
            assert boundary in page.locator(".evidence-source-boundary").inner_text()
            quote = page.locator("#evidence-source-quote").inner_text()
            assert quote.strip()
            if excerpt:
                assert excerpt in quote
            if evidence_dir:
                page.locator("#evidence-detail").screenshot(path=str(evidence_dir / f"{index:02d}_{case}.png"))
            page.evaluate("document.querySelector('#evidence-detail').close()")
        model_report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
        labels = page.evaluate(
            """report => {
                S.report = report;
                const analysis = report.analysis;
                const box = document.createElement('div');
                box.innerHTML = renderModelExplanation(
                    report, analysis.knowledge_enrichment, analysis.material_reasoning
                );
                return [...box.querySelectorAll('[data-model-citation]')].map(button => ({
                    id: button.dataset.modelCitation,
                    text: button.textContent
                }));
            }""",
            arg=model_report,
        )
        uploaded_label = next(item for item in labels if item["id"].startswith("upload:"))
        assert "reference.md" in uploaded_label["text"]
        assert "upload:" not in uploaded_label["text"]
        browser.close()
