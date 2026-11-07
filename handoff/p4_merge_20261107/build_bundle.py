"""Build and verify a source snapshot for manual P4 integration.

This reads the current worktree without importing AstroGraph or changing source files.
Historical delivery hash files are replaced by a fresh package-local source inventory.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
HANDOFF = Path(__file__).resolve().parent
PACKAGE_ID = "AstroGraph_P4_Merge_Source_20261107_R4"
ZIP_PATH = ROOT / "dist" / f"{PACKAGE_ID}.zip"
PREFIX = f"{PACKAGE_ID}/"
SOURCE_EXCLUDES = {
    "SHA256SUMS.txt", "SOURCE_MANIFEST.json", "DELIVERY_FILES_SHA256.txt",
    "verify_delivery_hashes.py",
}
SOURCE_PREFIX_EXCLUDES = ("run_artifacts/",)
SOURCE_ADDITIONS = [
    ".gitattributes",
    "examples/result_log_review/operations.log",
    "tests/ui/test_p4_account_workflow_browser.py",
    "tests/ui/test_p4_citation_source_contract.py",
]
REQUIRED_FEATURE_FILES = {
    "platform_and_accounts": [
        "pyproject.toml", "run_platform_accounts_demo.ps1", "src/astrograph/api.py",
        "src/astrograph/integrations/platform_accounts.py",
        "src/astrograph/integrations/platform_state.py",
        "src/astrograph/integrations/platform_secrets.py",
    ],
    "upload_preflight_and_report": [
        "src/astrograph/integrations/upload_report.py",
        "src/astrograph/integrations/platform_worker.py",
        "src/astrograph/integrations/http_worker.py",
        "src/astrograph/integrations/result_log_review.py",
        "src/astrograph/integrations/maintenance_input_admission.py",
        "src/astrograph/integrations/manual_report_review.py",
    ],
    "llm_diagnostic_explanation": [
        "src/astrograph/integrations/platform_knowledge.py",
        "src/astrograph/llm/openai_compatible.py",
        "src/astrograph/llm/task_prompts.py",
        "src/astrograph/llm/candidate_selection.py",
        "src/astrograph/llm/token_budget.py",
        "src/astrograph/llm/trace_verification.py",
        "src/astrograph/llm/audit.py",
    ],
    "rag_graph_oag_and_review": [
        "src/astrograph/integrations/material_hierarchical_rag.py",
        "src/astrograph/integrations/material_multihop.py",
        "src/astrograph/integrations/material_reasoning.py",
        "src/astrograph/integrations/material_case_memory.py",
        "src/astrograph/integrations/material_graph_builder.py",
        "src/astrograph/integrations/material_graph_release.py",
        "src/astrograph/integrations/material_engine_settings.py",
        "src/astrograph/retrieval/structured_document.py",
        "src/astrograph/retrieval/document_corpus.py",
        "src/astrograph/retrieval/document_graph_bridge.py",
        "src/astrograph/retrieval/advanced_multimodal.py",
        "src/astrograph/retrieval/persistent_vectors.py",
        "src/astrograph/retrieval/qdrant_vectors.py",
        "src/astrograph/retrieval/paperseek.py",
        "src/astrograph/oag/platform.py",
        "src/astrograph/oag/paperseek_http.py",
        "src/astrograph/oag/paperseek_cli.py",
    ],
    "mineru_mission_agent_adapter": [
        "src/astrograph/agents/mission/documents.py",
        "src/astrograph/agents/mission/models.py",
        "src/astrograph/agents/mission/document_quality.py",
        "scripts/acceptance/stages_tools.py",
        "tests/implementation/mission/test_external_contracts.py",
    ],
    "mission_interface": [
        "src/astrograph/resources/mission_console.html",
        "src/astrograph/resources/mission_console.css",
        "src/astrograph/resources/mission_console.js",
        "src/astrograph/resources/mission_demo_cases.json",
        "src/astrograph/resources/public_spacecraft_seed.v1.json",
        "src/astrograph/resources/home.html",
        "src/astrograph/resources/home.css",
        "src/astrograph/resources/home.js",
        "src/astrograph/resources/home_scene.js",
        "src/astrograph/resources/home_three.js",
        "src/astrograph/resources/platform_ui.html",
        "src/astrograph/resources/platform_ui.css",
        "src/astrograph/resources/platform_ui.js",
    ],
    "acceptance_and_regression": [
        "tests/test_upload_report_platform.py",
        "tests/test_platform_knowledge_enrichment.py",
        "tests/test_openai_compatible_backend.py",
        "tests/test_result_log_review.py",
        "tests/test_material_knowledge_fusion.py",
        "tests/ui/test_p4_account_workflow_browser.py",
        "tests/ui/test_p4_citation_source_contract.py",
        "examples/platform_acceptance_cases_v1/README_使用说明.md",
        "examples/platform_acceptance_cases_v1/TC01_ADCS_反作用轮摩擦/operations.log",
        "examples/result_log_review/operations.log",
    ],
}
EVIDENCE = [
    "research-evaluations/paddleocr-qwen-20261107/README.md",
    "research-evaluations/paddleocr-qwen-20261107/default_parser_summary.json",
    "research-evaluations/paddleocr-qwen-20261107/synthetic_rw_table.png",
    "research-evaluations/paddleocr-qwen-20261107/synthetic_rw_table_truth.json",
    "research-evaluations/paddleocr-qwen-20261107/synthetic_qwen3_vl_4b_content.txt",
    "research-evaluations/paddleocr-qwen-20261107/synthetic_paddleocr_vl_1_6_content.txt",
    "research-evaluations/paddleocr-qwen-20261107/synthetic_paddle_footer_content.txt",
    "research-evaluations/paddleocr-qwen-20261107/mco_page_07_generic_content.txt",
    "research-evaluations/paddleocr-qwen-20261107/mco_page_11_generic_content.txt",
    "research-evaluations/mineru-local-20261107/compare_mco_page_07_qwen_qwen3-vl-4b_content.txt",
    "research-evaluations/mineru-local-20261107/compare_qwen_qwen3-vl-4b_content.txt",
    "research-evaluations/mineru-local-20261107/mco_page_07.png",
    "research-evaluations/mineru-local-20261107/mco_page_11.png",
    "research-evaluations/P4_STATUS_20261107.md",
    "research-evaluations/P4_E2_REVIEW_PROTOCOL_20261107.md",
    "research-evaluations/P4_E2_CALIBRATION_20261107.md",
    "research-evaluations/p4-e1-20261107/README.md",
    "research-evaluations/p4-e1-20261107/case_manifest.json",
    "research-evaluations/p4-e1-20261107/case_manifest.sha256",
    "research-evaluations/p4-e1-public-candidates-20261107/README.md",
    "research-evaluations/p4-e1-public-candidates-20261107/cases/NH2015/input.md",
    "research-evaluations/p4-e1-public-candidates-20261107/cases/GAL2017/input.md",
    "research-evaluations/p4-e3-local-pilot-20261107/README.md",
    "research-evaluations/p4-e3-local-pilot-20261107/host-validation-retest.md",
    "research-evaluations/p4-e3-local-pilot-20261107/run_public_pilot.py",
    "research-evaluations/p4-e3-local-pilot-20261107/run_public_semantic_retest.py",
    "research-evaluations/p4-e3-local-pilot-20261107/public_semantic_retest/README.md",
    "research-evaluations/p4-e3-local-pilot-20261107/public_semantic_retest/source_fact_anchors.json",
    "research-evaluations/p4-e3-local-pilot-20261107/platform_tc01_clean_guided_retry/raw_model_response_01.json",
    "research-evaluations/p4-e3-local-pilot-20261107/platform_tc01_clean_guided_retry/raw_model_response_02.json",
    "research-evaluations/p4-e3-local-pilot-20261107/platform_tc01_clean_guided_retry/summary.json",
    "research-evaluations/p4-e5-browser-20261107/README.md",
    "research-evaluations/p4-e5-browser-20261107/citation_sources_synthetic.json",
    "research-evaluations/p4-e5-browser-20261107/account_process_model_9b_citations_final/workflow_receipt.json",
]
for _path in sorted((ROOT / "research-evaluations/paddleocr-qwen-20261107").iterdir()):
    if _path.is_file() and _path.name not in {
        "llama_pid.txt", "llama_stdout.log", "llama_stderr.log",
    }:
        EVIDENCE.append(_path.relative_to(ROOT).as_posix())
for _case in ("NH2015", "GAL2017"):
    for _model in ("qwen3.5-0.8b", "qwen_qwen3.5-9b"):
        EVIDENCE.append(
            f"research-evaluations/p4-e3-local-pilot-20261107/public_semantic_retest/{_case}_{_model}.json"
        )
for _name in ("analysis_input.json", "diagnosis_result.json", "operations.txt", "reference.md"):
    EVIDENCE.append(f"research-evaluations/p4-e3-local-pilot-20261107/clean_synthetic_tc01/{_name}")
for _name in (
    "06a_新建分析输入页.png", "06_真实文件上传预检.png",
    "07_管理员查看分析报告.png", "07a_任务总览.png",
    "07b_模型引用来源.png", "07c_上传资料引用.png", "08_专家复核.png",
):
    EVIDENCE.append(
        "research-evaluations/p4-e5-browser-20261107/"
        f"account_process_model_9b_citations_final/{_name}"
    )
for _name in ("01_uploaded_reference.png", "02_graph.png", "03_approved_history.png",
              "04_approved_history.png", "05_misclassification_history.png"):
    EVIDENCE.append(f"research-evaluations/p4-e5-browser-20261107/citation_contract/{_name}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_paths() -> list[str]:
    raw = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"])
    return sorted(path.decode("utf-8") for path in raw.split(b"\0") if path)


def safe_file(relative: str) -> bytes:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe source path: {relative}")
    path = ROOT.joinpath(*pure.parts)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT) or not path.is_file() or path.is_symlink():
        raise ValueError(f"nonregular or escaping source path: {relative}")
    return path.read_bytes()


def build_entries() -> tuple[dict[str, bytes], str]:
    base_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    source_paths = [
        path for path in git_paths()
        if path not in SOURCE_EXCLUDES
        and not path.startswith(SOURCE_PREFIX_EXCLUDES)
    ]
    source_paths += SOURCE_ADDITIONS
    source_paths += [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "examples/platform_acceptance_cases_v1").glob("TC*/operations.log")
    ]
    source_paths = sorted(set(source_paths))
    entries: dict[str, bytes] = {}
    for relative in source_paths:
        entries[f"source/{relative}"] = safe_file(relative)

    filelist = "\n".join(source_paths) + "\n"
    entries["source/FILELIST.txt"] = filelist.encode("utf-8")
    source_manifest = {
        "schema_version": "astrograph-p4-merge-source-manifest.v1",
        "package_id": PACKAGE_ID,
        "snapshot_date": "2026-11-07",
        "source_repository_base_commit": base_commit,
        "source_kind": "CURRENT_WORKTREE_FOR_MANUAL_P4_MERGE",
        "content_file_count": len(source_paths),
        "content_total_bytes": sum(len(entries[f"source/{path}"]) for path in source_paths),
        "whole_project_qualification_recertified": False,
        "prior_delivery_hash_files_replaced": sorted(SOURCE_EXCLUDES),
    }
    entries["source/SOURCE_MANIFEST.json"] = (
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    source_checksums = "\n".join(
        f"{digest(data)}  {name.removeprefix('source/')}"
        for name, data in sorted(entries.items()) if name.startswith("source/")
    ) + "\n"
    entries["source/SHA256SUMS.txt"] = source_checksums.encode("utf-8")

    for path in sorted(HANDOFF.glob("*.md")):
        entries[f"handoff/{path.name}"] = path.read_bytes()
    for relative in EVIDENCE:
        entries[f"handoff/evidence/{relative.removeprefix('research-evaluations/')}"] = safe_file(relative)
    coverage = {}
    for feature, required in REQUIRED_FEATURE_FILES.items():
        covered = []
        for relative in required:
            name = f"source/{relative}"
            if name not in entries:
                raise RuntimeError(f"required feature file missing: {feature}: {relative}")
            data = entries[name]
            covered.append({"path": name, "bytes": len(data), "sha256": digest(data)})
        coverage[feature] = covered
    entries["handoff/PACKAGE_COVERAGE.json"] = (
        json.dumps({"schema_version": "astrograph-p4-feature-coverage.v1",
                    "package_id": PACKAGE_ID, "features": coverage},
                   ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    entries["handoff/P4_PATCH_FROM_REPO_BASE.diff"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--binary", "HEAD", "--", "src", "tests"]
    )
    return entries, base_commit


def validate_names(entries: dict[str, bytes]) -> None:
    folded: set[str] = set()
    reserved = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$", re.I)
    for name in entries:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or any(reserved.match(part) for part in pure.parts):
            raise ValueError(f"unsafe zip member: {name}")
        if len(PREFIX + name) > 240:
            raise ValueError(f"zip member exceeds portable path limit: {name}")
        key = name.casefold()
        if key in folded:
            raise ValueError(f"case-insensitive duplicate: {name}")
        folded.add(key)


def main() -> None:
    entries, base_commit = build_entries()
    validate_names(entries)
    manifest = {
        "schema_version": "astrograph-p4-merge-package-manifest.v1",
        "package_id": PACKAGE_ID,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": base_commit,
        "file_count_excluding_manifest": len(entries),
        "files": [
            {"path": path, "bytes": len(data), "sha256": digest(data)}
            for path, data in sorted(entries.items())
        ],
    }
    entries["PACKAGE_MANIFEST.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    validate_names(entries)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, data in sorted(entries.items()):
            info = zipfile.ZipInfo(PREFIX + path, date_time=(2026, 11, 7, 12, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("zip CRC verification failed")
        actual = {name.removeprefix(PREFIX): name for name in archive.namelist()}
        if set(actual) != set(entries) or len(actual) != len(entries):
            raise RuntimeError("zip inventory differs from built entries")
        for path, data in entries.items():
            if digest(archive.read(actual[path])) != digest(data):
                raise RuntimeError(f"zip member hash differs: {path}")
        listed = {row["path"]: row for row in manifest["files"]}
        if set(listed) != set(entries) - {"PACKAGE_MANIFEST.json"}:
            raise RuntimeError("package manifest inventory mismatch")
        for path, row in listed.items():
            if row["sha256"] != digest(archive.read(actual[path])):
                raise RuntimeError(f"package manifest hash mismatch: {path}")
    zip_hash = digest(ZIP_PATH.read_bytes())
    ZIP_PATH.with_suffix(ZIP_PATH.suffix + ".sha256").write_text(
        f"{zip_hash}  {ZIP_PATH.name}\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "zip": str(ZIP_PATH), "sha256": zip_hash,
        "zip_bytes": ZIP_PATH.stat().st_size,
        "source_files": sum(name.startswith("source/") for name in entries),
        "total_members": len(entries), "crc_and_hash_verified": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
