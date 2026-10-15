#!/usr/bin/env python3
"""AI-Native Phase 4: multi-episode aggregated evidence-definition suggestions.

Runs whole-spacecraft diagnosis over several frozen ADCS test episodes per fault
class, collects the evidence bundles, and emits threshold-recalibration
suggestions only when multiple episodes agree on the same unsatisfied
definition (>= minimum agreement). Nothing modifies the active config.

Usage:
    ASTROGRAPH_ADCS_V522_DATA_ROOT=<root> \
        python scripts/ai_native/run_evidence_suggestion_aggregate.py \
            --episodes-per-class 3 --output-dir governance/evidence/ai_native/knowledge_candidates
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.evidence.suggestion_aggregate import AggregationMode, aggregate_suggestions

from scripts.ai_native.run_llm_hybrid_ci_regression import ADCS_DATASET, _invoke_chain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes-per-class', type=int, default=3)
    parser.add_argument('--minimum-agreement', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-classes', type=int, default=0, help='0 = all fault classes')
    parser.add_argument('--classes', nargs='+', default=None, help='explicit fault-class list (overrides --max-classes)')
    parser.add_argument('--output-name', default='suggestions_adcs_aggregate.json')
    args = parser.parse_args()
    if not (ADCS_DATASET / 'sample_index.csv').is_file():
        print(f'[error] dataset not found: {ADCS_DATASET}', file=sys.stderr)
        return 1
    from scripts.integration import run_adcs_v522_regression as adcs
    fault_class_map = json.loads((ROOT / 'configs/datasets/adcs_v5_2_2/fault_class_map.json').read_text(encoding='utf-8'))['mappings']

    def registered_fault_id(fault_class: str) -> str:
        return fault_class_map.get(fault_class, fault_class)

    trajectories = adcs.load_trajectories(ADCS_DATASET)
    test_rows = [
        row for row in trajectories
        if row.split == 'test' and row.diagnostic_class != 'nominal'
    ]
    all_classes = sorted({row.diagnostic_class for row in test_rows})
    if args.classes:
        classes = args.classes
    elif args.max_classes:
        classes = all_classes[: args.max_classes]
    else:
        classes = all_classes
    bundles: list[EpisodeEvidenceBundle] = []
    provenance: list[dict] = []
    work_root = Path(tempfile.gettempdir()) / ('sug-agg-' + uuid.uuid4().hex[:6])
    work_root.mkdir(parents=True, exist_ok=True)
    try:
        for fault_class in classes:
            class_rows = [row for row in test_rows if row.diagnostic_class == fault_class]
            selected = class_rows[: args.episodes_per_class]
            for index, sample in enumerate(selected, start=1):
                nominal = next(
                    row for row in trajectories
                    if row.diagnostic_class == 'nominal'
                    and row.trajectory_pair_id == sample.trajectory_pair_id
                )
                work = work_root / f'{fault_class}_{index}'
                result = _invoke_chain(
                    work, 'rule', None, sample, nominal,
                    profile_id='adcs.v5_2_2',
                    dataset_id='whole_spacecraft_adcs_paired_dataset_v5_2_2',
                    episode_tag=f'{fault_class}_{sample.sample_id}',
                )
                if result.status.value != 'SUCCEEDED':
                    print(f'[warn] {fault_class}/{sample.sample_id} failed: {result.error}', file=sys.stderr)
                    continue
                bundle = EpisodeEvidenceBundle.model_validate(
                    (result.output_payload or {})['episode_evidence_bundle'],
                )
                bundles.append(bundle)
                provenance.append({
                    'fault_class': fault_class,
                    'registered_fault_id': registered_fault_id(fault_class),
                    'sample_id': sample.sample_id,
                    'episode_id': bundle.episode.episode_id,
                })
                print(f'[episode] {fault_class} #{index}: {bundle.episode.episode_id}', flush=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    if not bundles:
        print('[error] no successful episodes', file=sys.stderr)
        return 1
    bundle_classes = [item['registered_fault_id'] for item in provenance]
    report = aggregate_suggestions(
        bundles, dataset_id='whole_spacecraft_adcs_paired_dataset_v5_2_2',
        bundle_fault_classes=bundle_classes,
        minimum_episode_agreement=args.minimum_agreement,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / args.output_name
    payload = report.model_dump(mode='json')
    payload['provenance'] = provenance
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(f'[episodes] {len(bundles)}', flush=True)
    print(f'[suggestions] {len(report.suggestions)} -> {path}', flush=True)
    for item in report.suggestions:
        print(f'  - {item.target_fault_id}: {item.feature_name} (threshold={item.suggested_threshold})', flush=True)
    for item in report.skipped_definitions:
        print(f'  - skipped: {item["evidence_definition_id"]} ({item["reason"]})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())