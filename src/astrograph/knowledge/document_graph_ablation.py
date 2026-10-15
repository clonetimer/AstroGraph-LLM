"""Build a hash-checked diagnostic ablation for document-to-graph extraction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def build_document_graph_ablation_report(
    *, config_path: str | Path, project_root: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Compare actual pilot runs without treating candidate count as accuracy."""

    config_path = Path(config_path).resolve()
    project_root = Path(project_root).resolve()
    output_path = Path(output_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "astrograph-document-graph-ablation-config.v1":
        raise ValueError("unsupported document graph ablation config schema")
    variants: list[dict[str, Any]] = []
    common_identity: tuple[Any, ...] | None = None
    for variant in config["variants"]:
        receipt_path = (project_root / variant["receipt_path"]).resolve()
        if not receipt_path.is_relative_to(project_root) or not receipt_path.is_file():
            raise ValueError(f"receipt is missing or outside project root: {receipt_path}")
        if _sha(receipt_path) != variant["receipt_file_sha256"]:
            raise ValueError(f"receipt file hash mismatch: {variant['variant_id']}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        identity = (
            receipt["configuration_sha256"],
            receipt["source_manifest_sha256"],
            receipt["selected_document_count"],
            json.dumps(receipt["lane_counts"], sort_keys=True),
            json.dumps(receipt["deterministic_scaffold_counts"], sort_keys=True),
            json.dumps(receipt["deterministic_content_extraction_counts"], sort_keys=True),
        )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise ValueError("ablation variants do not share the same frozen comparison set")

        audit_summary: dict[str, Any]
        audit_relative = variant.get("grounding_audit_path")
        if audit_relative:
            audit_path = (project_root / audit_relative).resolve()
            if not audit_path.is_relative_to(project_root) or not audit_path.is_file():
                raise ValueError(f"grounding audit is missing: {variant['variant_id']}")
            if _sha(audit_path) != variant["grounding_audit_file_sha256"]:
                raise ValueError(f"grounding audit hash mismatch: {variant['variant_id']}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_summary = {
                "status": audit["status"],
                "finding_counts": audit["finding_counts"],
                "checked_explicit_field_count": sum(
                    row.get("checked_explicit_field_count", 0) for row in audit["results"]
                ),
                "grounded_explicit_field_count": sum(
                    row.get("grounded_explicit_field_count", 0) for row in audit["results"]
                ),
                "valid_cited_candidate_count": sum(
                    row.get("valid_cited_candidate_count", 0) for row in audit["results"]
                ),
                "audit_sha256": audit["audit_sha256"],
            }
        else:
            if any(receipt["llm_raw_candidate_counts"].values()):
                raise ValueError("an LLM-producing variant requires a grounding audit")
            audit_summary = {
                "status": "NOT_APPLICABLE_NO_LLM_CANDIDATES",
                "finding_counts": {},
                "checked_explicit_field_count": 0,
                "grounded_explicit_field_count": 0,
                "valid_cited_candidate_count": 0,
            }
        variants.append(
            {
                "variant_id": variant["variant_id"],
                "intervention": variant["intervention"],
                "receipt_sha256": receipt["receipt_sha256"],
                "pipeline_source_sha256": receipt.get("pipeline_source_sha256"),
                "candidate_ready_count": receipt["candidate_ready_count"],
                "blocked_count": receipt["blocked_count"],
                "llm_raw_candidate_counts": receipt["llm_raw_candidate_counts"],
                "grounding": audit_summary,
            }
        )

    body = {
        "schema_version": "astrograph-document-graph-ablation-report.v1",
        "ablation_id": config["ablation_id"],
        "configuration_sha256": _sha(config_path),
        "comparison_controls": config["comparison_controls"],
        "variants": variants,
        "excluded_variants": config.get("excluded_variants", []),
        "decision": "NO_PROMOTION_HUMAN_GOLD_REQUIRED",
        "interpretation": (
            "The mandatory local extraction objective increased non-empty LLM output, but its "
            "first run invented unsupported distinct counts. The corrected prompt and hard "
            "grounding gate retained one source-traceable candidate with four checked numeric "
            "fields. This is pipeline evidence, not extraction accuracy or diagnostic gain."
        ),
        "publication_allowed": False,
        "runtime_vote_allowed": False,
        "claim_boundary": (
            "Actual single-run development diagnostics on eight frozen documents. Candidate "
            "volume is not precision/recall; no human gold, repeated-run distribution, Active KG "
            "promotion or diagnostic accuracy claim is made."
        ),
    }
    report = {**body, "report_sha256": _canonical_sha(body)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["build_document_graph_ablation_report"]
