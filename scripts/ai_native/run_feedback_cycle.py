#!/usr/bin/env python3
"""AI-Native Phase 2: controlled training-feedback cycle (P5 fusion host).

Hosts the governed feedback loop on a real trained model: the Phase 3 unified
baseline (research channel) is evaluated on the frozen PHM v6.0 test split and
compared against the frozen independent three-stage baseline (E2E 0.5147). The
feedback gate then produces a human-gated upgrade proposal (never auto-promotes)
and appends the cycle to the append-only ledger.

This is the training-capable feedback-cycle host the roadmap Phase 2 requires:
evaluation -> selection -> human-gated upgrade proposal on real metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from astrograph.data_driven.feedback import (
    FeedbackCandidate, FeedbackCycleLedger, FeedbackCycleRecord, FeedbackCycleStatus, TaskMetrics,
    advance_cycle, build_upgrade_proposal, evaluate_candidate,
)
from astrograph.data_driven.unified_platform import DatasetTaskSpec, make_unified_platform_model

PHM_HIERARCHY = ROOT / 'configs/datasets/phm_v6_0/diagnosis_hierarchy.json'
LEDGER_ROOT = ROOT / 'governance/evidence/ai_native/feedback_cycles'
PHM_BASELINE_E2E = 0.5147
PHM_BASELINE_DET = 0.5301
PHM_BASELINE_LOC = 0.7120
PHM_BASELINE_CLS = 0.6187


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cycle-id', default='cycle-1-unified-phm')
    parser.add_argument('--phm-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        print(f'[error] unified baseline checkpoint missing: {checkpoint}', file=sys.stderr)
        return 1
    # Load frozen PHM windows (same pipeline as Phase 3).
    from scripts.data_driven import compare_phm_causal_windows as phm_deep
    from astrograph.data_driven import phm_v60
    import torch
    hierarchy = json.loads(PHM_HIERARCHY.read_text(encoding='utf-8'))
    records = phm_v60.load_records(args.phm_root.resolve())
    class_names = sorted(hierarchy['fault_to_component'])
    component_names = sorted(set(hierarchy['fault_to_component'].values()))
    selected, median = phm_deep.select_train_channels(records, class_names, 96)
    windows = phm_deep.scale_windows(
        phm_deep.build_windows(records, selected, class_names, component_names, hierarchy['fault_to_component']),
        median,
    )
    model = make_unified_platform_model({
        'adcs': DatasetTaskSpec(channels=29, components=5, classes=14, window_size=20),
        'phm': DatasetTaskSpec(channels=96, components=13, classes=17, window_size=60),
    })
    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state['model_state_dict'])
    model.to(args.device)
    print(f'[checkpoint] loaded best_epoch={state.get("best_epoch")}', flush=True)

    # Evaluate the unified baseline on frozen PHM test.
    from scripts.ai_native.run_unified_baseline import UnifiedAdapter, SEED
    import numpy as np
    phm_test = windows['test']
    phm_logistic = phm_deep.logistic_detector(windows['train'], phm_test, SEED)
    # Recompute best validation selection from Phase 3 report.
    report = json.loads((ROOT / 'governance/evidence/ai_native/unified_baseline_20260823/unified_baseline_report.json').read_text(encoding='utf-8'))
    best_validation = report['unified']['best_validation']['phm']
    eval_result = phm_deep.evaluate_test(
        UnifiedAdapter(model, 'phm'), phm_test, phm_logistic, best_validation,
        class_names, component_names, hierarchy['fault_to_component'], device=args.device,
    )
    det = float(eval_result['detection']['macro_f1'])
    loc = float(eval_result['localization']['macro_f1'])
    cls = float(eval_result['classification_predicted_component']['macro_f1'])
    e2e = float(eval_result['end_to_end_18_class']['macro_f1'])
    print(f'[unified phm test] det={det:.4f} loc={loc:.4f} cls={cls:.4f} e2e={e2e:.4f}', flush=True)

    baseline = FeedbackCandidate(
        candidate_id='baseline-phm-three-stage',
        model_id='phm-v60-three-stage-logistic-regression',
        model_version='2026.08.13',
        family='THREE_STAGE_LOGISTIC',
        metrics=TaskMetrics(
            detection_macro_f1=PHM_BASELINE_DET, localization_macro_f1=PHM_BASELINE_LOC,
            classification_macro_f1=PHM_BASELINE_CLS, end_to_end_macro_f1=PHM_BASELINE_E2E,
        ),
        rationale='Frozen PHM v6.0 three-stage regression baseline.',
    )
    candidate = FeedbackCandidate(
        candidate_id='candidate-unified-phm',
        model_id='unified-platform-phm-adapter',
        model_version='20260823-research',
        family='UNIFIED_PLATFORM_PHM_ADAPTER',
        metrics=TaskMetrics(
            detection_macro_f1=det, localization_macro_f1=loc,
            classification_macro_f1=cls, end_to_end_macro_f1=e2e,
        ),
        rationale='Phase 3 unified baseline PHM adapter (research channel) on frozen test.',
        metadata={'checkpoint': str(checkpoint), 'device': args.device},
    )
    selection = evaluate_candidate(baseline.metrics, candidate.metrics)
    print(f'[selection] {selection["decision"]} reasons={selection["reasons"]}', flush=True)
    ledger = FeedbackCycleLedger(LEDGER_ROOT)
    cycle = FeedbackCycleRecord.new_cycle(
        cycle_id=args.cycle_id, dataset_id='whole_spacecraft_phm_dataset_v6_0', baseline=baseline,
        split_audit_ref='frozen grouped split (sample_index % 5)',
    )
    if selection['decision'] == 'PROPOSE':
        proposal = build_upgrade_proposal(
            cycle_id=args.cycle_id, dataset_id='whole_spacecraft_phm_dataset_v6_0',
            candidate=candidate, selection=selection,
        )
        cycle = advance_cycle(
            cycle, candidate=candidate, selection=selection, upgrade_proposal=proposal,
            status=FeedbackCycleStatus.PENDING_HUMAN_GATE,
            evidence_refs=['governance/evidence/ai_native/unified_baseline_20260823/unified_baseline_report.json'],
        )
    else:
        cycle = advance_cycle(
            cycle, candidate=candidate, selection=selection, status=FeedbackCycleStatus.STOPPED,
            evidence_refs=['governance/evidence/ai_native/unified_baseline_20260823/unified_baseline_report.json'],
        )
    path = ledger.append(cycle)
    print(f'[ledger] {path}', flush=True)
    print(f'[cycle] {cycle.cycle_id} status={cycle.status.value}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
