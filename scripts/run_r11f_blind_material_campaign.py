#!/usr/bin/env python3
"""Operator-owned R11-F blind campaign; no model contact without --allow-live.

Case manifest must contain inference-only fields. Gold labels and expert ratings
belong to separate post-run evaluation, never in this runner's input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from astrograph.evaluation.material_llm_blind_campaign import (
    ARM_ORDER, BlindMaterialCase, run_blind_material_campaign,
)
from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeConfig, PlatformKnowledgeEnrichmentService,
)


def load_blind_cases(manifest_path: Path) -> list[BlindMaterialCase]:
    manifest_path = manifest_path.resolve(strict=True)
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "cases"} or payload["schema_version"] != "astrograph-blind-inputs.v1":
        raise ValueError("BLIND_MANIFEST_SCHEMA_INVALID")
    rows = payload["cases"]
    if not isinstance(rows, list) or not (1 <= len(rows) <= 100):
        raise ValueError("BLIND_CASE_COUNT_INVALID")
    cases = []
    for row in rows:
        if not isinstance(row, dict) or not set(row) <= {
            "case_id", "group_id", "frozen_diagnosis", "reviewed_material", "document_paths", "task"
        } or not {"case_id", "group_id", "frozen_diagnosis"} <= set(row):
            raise ValueError("BLIND_CASE_SCHEMA_INVALID")
        paths = []
        for name in row.get("document_paths", []):
            if not isinstance(name, str) or not name:
                raise ValueError("INVALID_BLIND_DOCUMENT_PATH")
            item = (root / name).resolve(strict=True)
            if not item.is_relative_to(root):
                raise ValueError("BLIND_DOCUMENT_PATH_OUTSIDE_INPUT_ROOT")
            paths.append(item)
        case = BlindMaterialCase(
            case_id=row["case_id"], group_id=row["group_id"],
            frozen_diagnosis=row["frozen_diagnosis"],
            reviewed_material=row.get("reviewed_material"),
            document_paths=tuple(paths),
            task=row.get("task", "Explain this frozen diagnosis; cite evidence and uncertainty."),
        )
        case.verify()
        cases.append(case)
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("BLIND_CASE_IDS_DUPLICATED")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-live", action="store_true",
                        help="Explicit consent to run actual configured LLM inference.")
    args = parser.parse_args()
    cases = load_blind_cases(args.manifest)
    if not args.allow_live:
        print(json.dumps({
            "status": "PREFLIGHT_ONLY_NO_LLM_CALLS",
            "cases": len(cases), "arms": list(ARM_ORDER),
            "next_step": "Explicitly set model endpoint and pass --allow-live.",
        }, ensure_ascii=False))
        return 0
    base = PlatformKnowledgeConfig.from_environment()
    if not base.llm_base_url or not base.llm_model:
        raise ValueError("R11F_REAL_LLM_ENDPOINT_AND_MODEL_REQUIRED")
    def factory(arm: str) -> PlatformKnowledgeEnrichmentService:
        config = base.model_copy(update={
            "material_tool_calls_enabled": arm == "LLM_FULL_TOOLS",
            "paperseek_url": None, "paperseek_source_root": None,
            "a2o_enabled": False, "require_configured_services": False,
        })
        service = PlatformKnowledgeEnrichmentService(config=config)
        if service.llm_backend is None or service.paperseek_client is not None:
            raise RuntimeError("R11F_LLM_BACKEND_NOT_CONFIGURED")
        return service
    result = run_blind_material_campaign(
        cases, service_factory=factory, output_root=args.output,
    )
    print(json.dumps({
        "status": "RECORDED_UNSCORED",
        "case_count": len(cases),
        "cells": len(result["cells"]),
        "model_real_inference_verified": False,
        "gold_labels_loaded": False,
        "outcome_path": str(args.output / "outcomes.json"),
        "human_adjudication": "REQUIRED_AFTER_RUN",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
