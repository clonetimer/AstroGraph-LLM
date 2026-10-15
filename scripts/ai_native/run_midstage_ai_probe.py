#!/usr/bin/env python3
"""B2: mid-stage AI probe - constrained LLM hypothesis generation vs rule consistency."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from astrograph.services import ReferenceStructuredReasoningBackend
from scripts.ai_native.run_llm_hybrid_ci_regression import ADCS_DATASET, _invoke_chain, _decision

EVIDENCE_DIR = ROOT / 'governance/evidence/ai_native/governance_health'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-cases', type=int, default=12)
    args = parser.parse_args()
    from scripts.integration import run_adcs_v522_regression as adcs
    import tempfile, uuid, shutil
    trajectories = adcs.load_trajectories(ADCS_DATASET)
    test_rows = [
        row for row in trajectories
        if row.split == 'test' and row.diagnostic_class != 'nominal'
    ]
    # Sample across fault classes for coverage.
    by_class: dict[str, list] = {}
    for row in test_rows:
        by_class.setdefault(row.diagnostic_class, []).append(row)
    selected = []
    for cls_rows in by_class.values():
        if len(selected) >= args.max_cases:
            break
        selected.append(cls_rows[0])
    selected = selected[: args.max_cases]
    work_root = Path(tempfile.gettempdir()) / ('b2-' + uuid.uuid4().hex[:6])
    results = []
    agreed = 0
    try:
        for i, sample in enumerate(selected, start=1):
            nominal = next(
                row for row in trajectories
                if row.diagnostic_class == 'nominal'
                and row.trajectory_pair_id == sample.trajectory_pair_id
            )
            rule = _invoke_chain(
                work_root / f'rule-{i}', 'rule', None, sample, nominal,
                profile_id='adcs.v5_2_2',
                dataset_id='whole_spacecraft_adcs_paired_dataset_v5_2_2',
                episode_tag=f'rule-{sample.sample_id}',
            )
            hybrid = _invoke_chain(
                work_root / f'hybrid-{i}', 'hybrid', ReferenceStructuredReasoningBackend(), sample, nominal,
                profile_id='adcs.v5_2_2',
                dataset_id='whole_spacecraft_adcs_paired_dataset_v5_2_2',
                episode_tag=f'hybrid-{sample.sample_id}',
            )
            rule_dec = _decision(rule)
            hybrid_dec = _decision(hybrid)
            same = (rule_dec.get('accepted_fault_id') == hybrid_dec.get('accepted_fault_id'))
            agreed += int(same)
            results.append({
                'case': i, 'sample': sample.sample_id, 'fault_class': sample.diagnostic_class,
                'rule_decision': rule_dec, 'hybrid_decision': hybrid_dec, 'consistent': same,
            })
            print(f'[b2] case {i} {sample.diagnostic_class}: consistent={same}', flush=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    consistency = agreed / len(results) if results else 1.0
    report = {
        'schema_version': 'astrograph-midstage-ai-probe.v1',
        'probe_id': 'PROBE-20260824-B2-MIDSTAGE',
        'case_count': len(results),
        'decision_consistency_rule_vs_hybrid': round(consistency, 4),
        'gate_met': consistency >= 0.95,
        'results': results,
        'claim_boundary': 'Lab paired synthetic probe on frozen ADCS test; reference backend only (no external LLM).',
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / 'midstage_ai_probe_20260824.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(f'[b2] consistency={consistency:.2%} gate_met={consistency >= 0.95}', flush=True)
    print(f'[b2] evidence -> {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())