from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
CURRENT_INDEX = DOCS / "CURRENT_DOCS.json"
HISTORY_MANIFEST = DOCS / "history" / "MANIFEST.json"
RELEASE_READINESS = ROOT / "governance" / "implementation" / "post49" / "RELEASE_READINESS.json"
DELIVERY_PROFILE = ROOT / "DELIVERY_PROFILE.json"

HISTORICAL_NAME_RE = re.compile(
    r"(?:^|[_-])(?:DEV\d+|POST(?:\d+|48_DEV\d+)|20\d{6})(?:[_.-]|$)", re.IGNORECASE
)
CURRENT_ROOT_ALLOWED = {
    "LLM_ACTIVE_DIAGNOSIS_EXPLORATORY_RUNBOOK.md",
    "LLM_ACTIVE_DIAGNOSIS_RESEARCH_PLAN.md",
    "POST50_PROGRESSIVE_HIERARCHY_RESEARCH.md",
    "POST51_MULTISEED_NEURAL_BENCHMARK.md",
    "THESIS_METHOD_REMEDIATION.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_10.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_9.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_8.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_7.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_6.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_5.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_4.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_3.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_2.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35_1.md",
    "ASTROGRAPH_PRODUCT_EXPERIENCE_R35.md",
    "ASTROGRAPH_DIAGNOSTIC_WORKSPACE_R34_2.md",
    "ASTROGRAPH_DIAGNOSTIC_WORKSPACE_R34_1.md",
    "ASTROGRAPH_DIAGNOSTIC_WORKSPACE_R34.md",
    "ASTROGRAPH_OPERATIONS_CONSOLE_R33_2.md",
    "ASTROGRAPH_OPERATIONS_CONSOLE_R33_1.md",
    "ASTROGRAPH_OPERATIONS_CONSOLE_R33.md",
    "ASTROGRAPH_SPACECRAFT_SCOPE_CONVERGENCE_R32.md",
    "ASTROGRAPH_POST_R32_ROADMAP.md",
    "ASTROGRAPH_AGENT_TOOL_SPECIFICATION_R31.md",
    "ASTROGRAPH_R31_WATERPUMP_REDUCED_ORDER_SIMULATION.md",
    "ASTROGRAPH_AGENT_TOOL_SPECIFICATION_R30.md",
    "ASTROGRAPH_R30_MECHANISM_MODEL_AND_BLIND_SIMULATION.md",
    "ASTROGRAPH_R29_REDUCED_ORDER_SIMULATION.md",
    "ASTROGRAPH_R28_DXC_FORWARD_SIMULATION.md",
    "ASTROGRAPH_R27_DXC_EXPECTED_RESPONSE_ARCHITECTURE.md",
    "ASTROGRAPH_R26_DXC_DISCRETE_ESA_FREEZE.md",
    "ASTROGRAPH_R23_GRAPH_MODEL_SIMULATION_DIAGNOSIS.md",
    "ASTROGRAPH_R25_DXC_ESA_REMEDIATION.md",
    "ASTROGRAPH_R21_REAL_DATA_FAILURE_MECHANISMS.md",
    "ASTROGRAPH_RESEARCH_BACKLOG_R21.md",
    "ASTROGRAPH_R22_NEW_INDEPENDENT_REPLICATION.md",
    "ASTROGRAPH_RESEARCH_BACKLOG_R22.md",
    "ASTROGRAPH_R19_CALIBRATION_FITTING_AND_EXTERNAL_VALIDATION.md",
    "ASTROGRAPH_RESEARCH_BACKLOG_R19.md",
    "ASTROGRAPH_R18_CALIBRATION_TRANSFER_AND_FUSION_DEPENDENCE.md",
    "ASTROGRAPH_RESEARCH_BACKLOG_R18.md",
    "ASTROGRAPH_R17_SCORE_SEMANTICS_AND_CALIBRATION.md",
    "ASTROGRAPH_RESEARCH_BACKLOG_R16.md",
    "ASTROGRAPH_RESEARCH_BACKLOG_R17.md",
    "LLM_METHOD_CONFIRMATORY_R16_RUNBOOK.md",
    "ASTROGRAPH_R16_REPRODUCIBILITY_IDENTITY.md",
    "MAINTENANCE_SUPPLEMENTAL_EVIDENCE_R10.md",
    "LLM_METHOD_CONFIRMATORY_R6_RUNBOOK.md",
    "SUPPLEMENTAL_PROVENANCE_R11.md",
    "UPSTREAM_RECOMPUTE_PROTOCOL_R13.md",
    "MAINTENANCE_PRODUCTION_RERUN_R12.md",
    "AGENT_TOOL_PLUGINS.md",
    "LLM_METHOD_CONFIRMATORY_R8_RUNBOOK.md",
    "LLM_METHOD_CONFIRMATORY_R10_RUNBOOK.md",
    "LLM_METHOD_CONFIRMATORY_R7_RUNBOOK.md",
    "MAINTENANCE_GOVERNED_RERUN_R11.md",
    "MAINTENANCE_INPUT_CONTRACT_R9.md",
    "LLM_METHOD_CONFIRMATORY_R9_RUNBOOK.md",
    "ASTROGRAPH_RESEARCH_AUDIT_R14.md",
    "ASTROGRAPH_R15_EVIDENCE_AND_CONFIRMATORY_DESIGN.md",
    "ASTROGRAPH_UNIFIED_SOURCE_R14.md",
    "README.md",
    "CURRENT_DATA_DRIVEN_METRICS.md",
    "CURRENT_DELIVERY_SOURCE.md",
    "CURRENT_DOCS.json",
    "CURRENT_ENGINEERING_POLICY.md",
    "DIRECTORY_STRUCTURE.md",
    "DOCUMENT_STATUS_POLICY.md",
    "FRAMEWORK_MAP.md",
    "JOINT_DIAGNOSIS_ARCHITECTURE.md",
    "PROVIDER_QUALIFICATION.md",
    "PROVIDER_QUALIFICATION_VALIDATION.md",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_current_docs(index: dict) -> Iterable[Path]:
    for item in index.get("documents", []):
        yield ROOT / item["path"]


def validate() -> list[str]:
    errors: list[str] = []
    if not CURRENT_INDEX.is_file():
        return [f"missing current index: {CURRENT_INDEX}"]
    if not RELEASE_READINESS.is_file():
        return [f"missing release readiness: {RELEASE_READINESS}"]

    current = load_json(CURRENT_INDEX)
    readiness = load_json(RELEASE_READINESS)
    delivery = load_json(DELIVERY_PROFILE) if DELIVERY_PROFILE.is_file() else {}
    history_included = bool(delivery.get("historical_records_included", True))
    if history_included and not HISTORY_MANIFEST.is_file():
        return [f"missing history manifest: {HISTORY_MANIFEST}"]
    history = load_json(HISTORY_MANIFEST) if HISTORY_MANIFEST.is_file() else {"records": []}
    if not history_included and (DOCS / "history").exists():
        errors.append("current-source delivery must not bundle docs/history")

    if current.get("release") != "0.7.1.post49":
        errors.append("CURRENT_DOCS release must be 0.7.1.post49")
    if readiness.get("version") != "0.7.1.post49":
        errors.append("release readiness must be post49")

    # Current-document corpus integrity and status.
    seen: set[str] = set()
    for item in current.get("documents", []):
        rel = item.get("path", "")
        if not rel or rel in seen:
            errors.append(f"duplicate/empty current doc path: {rel!r}")
            continue
        seen.add(rel)
        if rel.startswith("docs/history/") or rel.startswith("research/") or rel.startswith("governance/evidence/"):
            errors.append(f"history/research/evidence cannot be CURRENT: {rel}")
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing current document: {rel}")
            continue
        if item.get("sha256") != _sha(path):
            errors.append(f"CURRENT_DOCS sha mismatch: {rel}")
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8", errors="replace")[:800]
            if "CURRENT_" not in text and rel not in {"README.md", "CHANGELOG.md"}:
                errors.append(f"current markdown lacks CURRENT status marker: {rel}")

    # No historical-looking top-level docs outside history. Current metrics are an explicit current name.
    for path in DOCS.iterdir():
        if not path.is_file():
            continue
        if path.name not in CURRENT_ROOT_ALLOWED:
            errors.append(f"unexpected top-level docs file outside history/current whitelist: docs/{path.name}")
        if HISTORICAL_NAME_RE.search(path.name) and path.name not in {
            "POST50_PROGRESSIVE_HIERARCHY_RESEARCH.md",
            "POST51_MULTISEED_NEURAL_BENCHMARK.md",
        }:
            errors.append(f"historical-looking file outside docs/history: docs/{path.name}")

    # Implementation surface has exactly the stable release note.
    impl = DOCS / "implementation"
    impl_files = sorted(p.name for p in impl.iterdir() if p.is_file()) if impl.is_dir() else []
    if impl_files != ["POST49_STABLE_RELEASE.md"]:
        errors.append(f"docs/implementation must contain only POST49_STABLE_RELEASE.md, found {impl_files}")

    # History manifest integrity is required only in a full archival checkout.
    if history_included:
        hpaths: set[str] = set()
        for rec in history.get("records", []):
            rel = rec.get("path", "")
            if not rel.startswith("docs/history/"):
                errors.append(f"history record outside archive: {rel}")
                continue
            if rel in hpaths:
                errors.append(f"duplicate history record: {rel}")
                continue
            hpaths.add(rel)
            path = ROOT / rel
            if not path.is_file():
                errors.append(f"missing history file: {rel}")
                continue
            if rec.get("sha256") != _sha(path):
                errors.append(f"history sha mismatch: {rel}")
            if rec.get("status") != "HISTORICAL_NON_AUTHORITATIVE":
                errors.append(f"history status mismatch: {rel}")

        actual_history = {
            p.relative_to(ROOT).as_posix()
            for p in (DOCS / "history").rglob("*")
            if p.is_file() and p.name != "MANIFEST.json"
        }
        missing_manifest = sorted(actual_history - hpaths)
        if missing_manifest:
            errors.append(f"history files missing from manifest: {missing_manifest[:10]}")

    # Current entry points must not claim a pre-post49 current version.
    entrypoints = [ROOT / "README.md", ROOT / "START_HERE.md", ROOT / "RUN_GUIDE.md", ROOT / "TESTING.md"]
    for path in entrypoints:
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"当前(?:开发)?版本.{0,30}post48|CURRENT.{0,30}post48", text, re.IGNORECASE):
            errors.append(f"pre-post49 current-version claim in {path.name}")
        if path.name in {"README.md", "START_HERE.md"}:
            if history_included and "docs/history/" not in text:
                errors.append(f"{path.name} must expose explicit history boundary")
            if not history_included and "历史" not in text and "history" not in text.lower():
                errors.append(f"{path.name} must state that historical records are not bundled")

    release = load_json(ROOT / "release.json")
    doc_meta = release.get("documentation", {})
    if doc_meta.get("current_index") != "docs/CURRENT_DOCS.json":
        errors.append("release.json documentation.current_index mismatch")
    if history_included:
        if doc_meta.get("history_root") != "docs/history/":
            errors.append("release.json documentation.history_root mismatch")
    else:
        if doc_meta.get("history_included") is not False or doc_meta.get("history_location") != "NOT_INCLUDED_IN_THIS_SOURCE_PACKAGE":
            errors.append("release.json must declare history excluded for current-source delivery")
    if doc_meta.get("history_may_override_current") is not False:
        errors.append("release.json must forbid history overriding current")
    if release.get("production_paths", {}).get("trusted_data_protocol") != "docs/CURRENT_ENGINEERING_POLICY.md":
        errors.append("release.json trusted_data_protocol must point to current engineering policy")

    return errors


def refresh_indexes() -> None:
    current = load_json(CURRENT_INDEX)
    for item in current.get("documents", []):
        path = ROOT / item["path"]
        if path.is_file():
            item["sha256"] = _sha(path)
    CURRENT_INDEX.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    delivery = load_json(DELIVERY_PROFILE) if DELIVERY_PROFILE.is_file() else {}
    if bool(delivery.get("historical_records_included", True)):
        hist_path = DOCS / "history"
        old = load_json(HISTORY_MANIFEST) if HISTORY_MANIFEST.exists() else {"records": []}
        by_path = {r.get("path"): r for r in old.get("records", [])}
        records = []
        for path in sorted(p for p in hist_path.rglob("*") if p.is_file() and p.name != "MANIFEST.json"):
            rel = path.relative_to(ROOT).as_posix()
            rec = dict(by_path.get(rel, {}))
            rec.update({"path": rel, "status": "HISTORICAL_NON_AUTHORITATIVE", "sha256": _sha(path)})
            rec.setdefault("original_path", "")
            rec.setdefault("category", "preexisting_history_or_snapshot")
            rec.setdefault("reason", "Historical/legacy documentation retained for provenance.")
            records.append(rec)
        history = {
            "schema_version": "astrograph-document-history-manifest.v1",
            "release": "0.7.1.post49",
            "archive_root": "docs/history",
            "claim_precedence": "HISTORY_NEVER_OVERRIDES_CURRENT_RELEASE_READINESS",
            "file_count": len(records),
            "records": records,
        }
        HISTORY_MANIFEST.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refresh CURRENT_DOCS/history manifest hashes first")
    args = parser.parse_args()
    if args.refresh:
        refresh_indexes()
    errors = validate()
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print("PASS: current/history documentation boundary is consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
