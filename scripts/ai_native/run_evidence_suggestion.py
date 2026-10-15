#!/usr/bin/env python3
"""AI-Native Phase 4: evidence-definition suggestion demo on real telemetry.

Runs one whole-spacecraft diagnosis on a frozen ADCS test sample, takes the
built episode evidence bundle, and generates governed evidence-definition
suggestions (READY_FOR_HUMAN_REVIEW only). Nothing modifies the active config.

Usage:
    ASTROGRAPH_ADCS_V522_DATA_ROOT=<root> \
        python scripts/ai_native/run_evidence_suggestion.py --class adcs_gyro_bias_step \
            --output-dir governance/evidence/ai_native/knowledge_candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.evidence.suggestion import suggest_evidence_definitions

from scripts.ai_native.run_llm_hybrid_ci_regression import (
    ADCS_DATASET,
    _invoke_chain,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class", dest="fault_class", default="adcs_gyro_bias_step")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (ADCS_DATASET / "sample_index.csv").is_file():
        print(f"[error] dataset not found: {ADCS_DATASET}", file=sys.stderr)
        return 1
    from scripts.integration import run_adcs_v522_regression as adcs
    trajectories = adcs.load_trajectories(ADCS_DATASET)
    sample = next(
        row for row in trajectories
        if row.split == "test" and row.diagnostic_class == args.fault_class
    )
    nominal = next(
        row for row in trajectories
        if row.diagnostic_class == "nominal" and row.trajectory_pair_id == sample.trajectory_pair_id
    )
    import tempfile
    import uuid
    work = Path(tempfile.gettempdir()) / ("sug-" + uuid.uuid4().hex[:6])
    result = _invoke_chain(
        work, "rule", None, sample, nominal,
        profile_id="adcs.v5_2_2",
        dataset_id="whole_spacecraft_adcs_paired_dataset_v5_2_2",
    )
    if result.status.value != "SUCCEEDED":
        print(f"[error] workflow failed: {result.error}", file=sys.stderr)
        return 1
    bundle = EpisodeEvidenceBundle.model_validate(
        (result.output_payload or {})["episode_evidence_bundle"]
    )
    report = suggest_evidence_definitions(bundle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"suggestions_{args.fault_class}.json"
    path.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(f"[episode] {bundle.episode.episode_id}")
    print(f"[suggestions] {len(report.suggestions)} -> {path}")
    for item in report.suggestions:
        print(f"  - {item.kind.value}: {item.feature_name} (threshold={item.suggested_threshold})")
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
