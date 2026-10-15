from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from astrograph.integrations.platform_knowledge import (
    Anything2OntologyCLIAdapter,
    PlatformKnowledgeConfig,
    PlatformKnowledgeEnrichmentService,
    UploadedDocumentKnowledgeIndex,
)
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.oag.paperseek_cli import PaperSeekCLIClient, PaperSeekCLIConfig


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "sat_sim_closure"


def _put_pipeline(store: UploadReportJobStore, job_id: str, note: bytes | None = None) -> None:
    store.put_file(job_id, "runtime.json", (FIXTURES / "runtime_rw_jam.json").read_bytes())
    store.put_file(job_id, "mapping_bundle.json", (FIXTURES / "mapping_bundle.json").read_bytes())
    if note is not None:
        store.put_file(job_id, "knowledge.md", note)


def test_no_llm_still_produces_reference_preserving_final_explanation(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("分析反作用轮卡滞，并解释证据来源")
    _put_pipeline(
        store,
        job["job_id"],
        b"# Reaction wheel FMEA\nADCS_RW_JAM rotor lock removes reaction wheel torque response.\n",
    )
    report = store.run(job["job_id"])
    knowledge = report["analysis"]["knowledge_enrichment"]

    assert knowledge["generation"]["status"] == "NOT_CONFIGURED"
    assert knowledge["final_explanation"]["status"] == "SUCCEEDED"
    assert knowledge["final_explanation"]["source"] == "DETERMINISTIC_REFERENCE_SYNTHESIS"
    explanation = knowledge["final_explanation"]["explanation"]
    assert explanation["candidate_fault_id"] == "ADCS_RW_JAM"
    assert any(claim["kind"] == "LOCAL_KNOWLEDGE" for claim in explanation["claims"])
    allowed = {item["knowledge_id"] for item in knowledge["local_retrieval"]["hits"]}
    for claim in explanation["claims"]:
        if claim["kind"] == "LOCAL_KNOWLEDGE":
            assert set(claim["citation_ids"]) <= allowed
    assert report["analysis"]["pipeline_result"]["fusion"]["top_fault_id"] == "ADCS_RW_JAM"


def test_local_retrieval_quality_receipt_reports_candidate_anchor(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("解释反作用轮卡滞")
    _put_pipeline(
        store,
        job["job_id"],
        "ADCS_RW_JAM 是反作用轮 rotor lock 的工程故障标识。".encode("utf-8"),
    )
    report = store.run(job["job_id"])
    quality = report["analysis"]["knowledge_enrichment"]["local_retrieval"]["quality_receipt"]
    assert quality["grounding_status"] == "DIRECT_CANDIDATE_ANCHOR"
    assert quality["candidate_anchor_hit_count"] >= 1
    assert quality["retrieved_hit_count"] >= 1
    assert quality["top_score"] > 0


def test_chinese_bigram_tokenization_retrieves_reaction_wheel_context(tmp_path: Path) -> None:
    relevant = tmp_path / "rw.md"
    irrelevant = tmp_path / "battery.md"
    relevant.write_text("# 姿控\n反作用轮卡滞会导致姿态执行能力下降。", encoding="utf-8")
    irrelevant.write_text("# 电源\n蓄电池容量衰减需要检查充放电循环。", encoding="utf-8")
    index = UploadedDocumentKnowledgeIndex(top_k=2)
    chunks, receipts = index.build([relevant, irrelevant])
    assert all(item["status"] == "PARSED" for item in receipts)
    hits = index.retrieve("反作用轮故障", chunks)
    assert hits
    assert hits[0]["filename"] == "rw.md"


def test_paperseek_source_tree_runs_without_installed_console_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "paperseek-src"
    package = root / "paperseek"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        """
import json, sys
query = sys.argv[sys.argv.index('search') + 1]
print(json.dumps({
  'question': query,
  'ranked': [
    {'id':'P1','title':'Reaction wheel jam mechanisms','abstract':'Rotor lock removes wheel torque response.','url':'https://example.org/p1'},
    {'id':'P2','title':'Duplicate','abstract':'Rotor lock removes wheel torque response.','url':'https://example.org/p2'}
  ]
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", "")
    client = PaperSeekCLIClient(PaperSeekCLIConfig(source_root=str(root), top_k=5, timeout_s=5))
    assert client.available()
    assert client.backend_mode() == "SOURCE_TREE_MODULE"
    result = client.search("reaction wheel jam")
    assert result["status"] == "SUCCEEDED"
    assert result["backend"] == "PAPERSEEK_CLI"
    assert len(result["documents"]) == 1  # identical abstract is not independent evidence
    assert result["documents"][0]["title"] == "Reaction wheel jam mechanisms"


def test_a2o_source_tree_runs_modules_without_installed_console_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "a2o-src"
    for package_name in ("anything2markdown", "markdown2chunks", "chunks2skus", "skus2ontology"):
        package = root / "src" / package_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "cli.py").write_text(
            f"""
import os, pathlib
name = {package_name!r}
out = pathlib.Path(os.environ['OUTPUT_DIR']); out.mkdir(parents=True, exist_ok=True)
if name == 'anything2markdown':
    (out/'source.md').write_text('ADCS_RW_JAM rotor lock', encoding='utf-8')
elif name == 'markdown2chunks':
    (out/'chunks').mkdir(parents=True, exist_ok=True)
elif name == 'chunks2skus':
    pathlib.Path(os.environ['SKUS_OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)
else:
    ontology = pathlib.Path(os.environ['ONTOLOGY_DIR']); ontology.mkdir(parents=True, exist_ok=True)
    (ontology/'mapping.md').write_text('# mapping\\nADCS_RW_JAM -> rotor lock', encoding='utf-8')
""".strip()
            + "\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("PATH", "")
    doc = tmp_path / "fmea.md"
    doc.write_text("ADCS_RW_JAM rotor lock", encoding="utf-8")
    adapter = Anything2OntologyCLIAdapter(source_root=root, timeout_s=5)
    assert adapter.available()
    assert adapter.discovery()["ready"] is True
    result = adapter.run([doc], tmp_path / "out")
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["active_kg_modified"] is False
    assert result["review_material"]
    assert all(item["diagnostic_vote_allowed"] is False for item in result["review_material"])
    assert {item["execution_mode"] for item in result["commands"]} == {"SOURCE_TREE_MODULE"}


def test_service_auto_selects_paperseek_source_tree_and_exposes_tool_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "paperseek-src"
    package = root / "paperseek"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text("print('{}')\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "")
    service = PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(paperseek_source_root=str(root))
    )
    capabilities = service.capabilities()
    assert capabilities["paperseek"] == "CONFIGURED"
    assert capabilities["paperseek_backend"] == "SOURCE_TREE_MODULE"
    assert capabilities["deterministic_grounded_synthesis"] is True
    assert capabilities["retrieval_quality_receipt"] is True


def test_stage_timings_are_finite_nonnegative_and_do_not_affect_candidate(tmp_path: Path) -> None:
    store = UploadReportJobStore(tmp_path / "jobs")
    job = store.create("解释诊断")
    _put_pipeline(store, job["job_id"], b"ADCS_RW_JAM rotor lock")
    report = store.run(job["job_id"])
    knowledge = report["analysis"]["knowledge_enrichment"]
    assert report["analysis"]["pipeline_result"]["fusion"]["top_fault_id"] == "ADCS_RW_JAM"
    assert {"local_parse_and_retrieval", "paperseek", "anything2ontology", "llm_generation", "total"} <= set(knowledge["timings_s"])
    assert all(isinstance(value, (int, float)) and value >= 0 for value in knowledge["timings_s"].values())


def test_paperseek_probe_reports_incomplete_configuration_without_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "paperseek-src"
    package = root / "paperseek"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        """
import json, sys
if 'doctor' not in sys.argv:
    raise SystemExit(17)
print(json.dumps({'checks':[{'id':'llm.api_key','status':'fail','severity':'error','ok':False}]}))
raise SystemExit(1)
""".strip() + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PATH", "")
    client = PaperSeekCLIClient(PaperSeekCLIConfig(source_root=str(root), timeout_s=5))
    result = client.probe()
    assert result["status"] == "CONFIGURATION_INCOMPLETE"
    assert result["backend_mode"] == "SOURCE_TREE_MODULE"
    assert result["exit_code"] == 1
    assert result["checks"][0]["id"] == "llm.api_key"


def test_a2o_probe_reports_import_error_without_running_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "a2o-src"
    for package_name in ("anything2markdown", "markdown2chunks", "chunks2skus", "skus2ontology"):
        package = root / "src" / package_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "cli.py").write_text(
            "raise SystemExit(3)\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("PATH", "")
    adapter = Anything2OntologyCLIAdapter(source_root=root, timeout_s=5)
    assert adapter.available() is True  # discoverable source is not the same as import-ready
    result = adapter.probe()
    assert result["status"] == "NOT_READY"
    assert {item["status"] for item in result["stages"]} == {"IMPORT_OR_CONFIG_ERROR"}
    assert all(item["exit_code"] == 3 for item in result["stages"])


def test_explicit_paperseek_source_tree_is_not_shadowed_by_global_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "paperseek-src"
    package = root / "paperseek"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(
        """
import json, sys
if 'search' in sys.argv:
    print(json.dumps({'ranked':[{'id':'S','title':'Pinned source','abstract':'Pinned revision result.','url':'https://example.org/pinned'}]}))
else:
    print(json.dumps({'checks':[]}))
""".strip() + "\n",
        encoding="utf-8",
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    executable = bindir / "paperseek"
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    client = PaperSeekCLIClient(PaperSeekCLIConfig(source_root=str(root), timeout_s=5))
    assert client.backend_mode() == "SOURCE_TREE_MODULE"
    assert client.search("test")["documents"][0]["title"] == "Pinned source"


def test_explicit_a2o_source_tree_is_not_shadowed_by_global_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "a2o-src"
    for package_name in ("anything2markdown", "markdown2chunks", "chunks2skus", "skus2ontology"):
        package = root / "src" / package_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "cli.py").write_text(
            """
import os, pathlib
if __name__ == '__main__':
    out=pathlib.Path(os.environ.get('OUTPUT_DIR','.')); out.mkdir(parents=True,exist_ok=True)
    onto=pathlib.Path(os.environ.get('ONTOLOGY_DIR', out/'ontology')); onto.mkdir(parents=True,exist_ok=True)
    (onto/'source.md').write_text('source-tree',encoding='utf-8')
""".strip() + "\n",
            encoding="utf-8",
        )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for command in ("anything2md", "md2chunks", "chunks2skus", "skus2ontology"):
        executable = bindir / command
        executable.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    doc = tmp_path / "doc.md"
    doc.write_text("source-tree", encoding="utf-8")
    adapter = Anything2OntologyCLIAdapter(source_root=root, timeout_s=5)
    result = adapter.run([doc], tmp_path / "out")
    assert result["status"] == "REVIEW_REQUIRED"
    assert {item["execution_mode"] for item in result["commands"]} == {"SOURCE_TREE_MODULE"}
