#!/usr/bin/env python3
"""AI-Native Phase 2: PHM 5-fold grouped CV for the unified PHM adapter.

Before promoting cycle-1 (unified PHM adapter) to the production registry, the
single-split evidence must be validated across 5 grouped folds (sample_index % 5
rotations). Each fold trains the unified backbone (PHM-only, same architecture as
the Phase 3 unified baseline) and an independent residual_tcn control under the
same budget/seed/preprocessing, then evaluates the frozen fold-test once.

Outputs the per-fold E2E and the 5-fold mean/std delta to decide promotion.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from scripts.data_driven import compare_phm_causal_windows as phm_deep
from astrograph.data_driven import phm_v60
from astrograph.data_driven.unified_platform import DatasetTaskSpec, make_unified_platform_model

PHM_ROOT = Path('D:/Project/test_lunwen/whole_spacecraft_phm_dataset_v6_0')
PHM_HIERARCHY = ROOT / 'configs/datasets/phm_v6_0/diagnosis_hierarchy.json'
PHM_BASELINE_E2E = 0.5147

BATCH_SIZE = 768
MAX_EPOCHS = 40
MIN_EPOCHS = 10
PATIENCE = 10
BASE_SEED = 20260814


def fold_split(records: list[Any], fold: int, *, exclude_subsystems=None) -> list[Any]:
    """Rotate sample_index % 5; optionally exclude subsystems from TRAIN (C1)."""
    excluded = set(exclude_subsystems or [])
    result = []
    for record in records:
        rem = record.sample_index % 5
        if rem == fold:
            split = 'test'
        elif rem == (fold + 1) % 5:
            split = 'validation'
        else:
            split = 'train'
        if split == 'train' and record.subsystem in excluded:
            continue
        result.append(dataclasses.replace(record, split=split))
    return result


def evaluate_phm(model, windows: dict[str, Any], class_names, component_names, fault_to_component, seed: int, device: str) -> dict[str, Any]:
    from scripts.ai_native.run_unified_baseline import UnifiedAdapter
    test = windows['test']
    logistic = phm_deep.logistic_detector(windows['train'], test, seed)
    # Use a default selection with detector=deep, alpha=0.5 (mirrors validation-time search).
    selection = {
        'detector': 'deep',
        'detection_threshold': 0.5,
        'alarm_on_threshold': 0.5, 'alarm_off_threshold': 0.5, 'alarm_minimum_consecutive': 1,
        'hierarchy_blend_alpha': 0.5, 'conditioning_mode': 'hard',
    }
    result = phm_deep.evaluate_test(
        UnifiedAdapter(model, 'phm'), test, logistic, selection,
        class_names, component_names, fault_to_component, device=device,
    )
    return result


def train_unified_phm(windows, class_names, component_names, fault_to_component, seed: int, device: str, subsystem_epochs=None):
    import torch
    import numpy as np
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch_device = phm_deep.prepare_torch_device(device, seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = make_unified_platform_model({
        'phm': DatasetTaskSpec(channels=windows['train'].x.shape[-1], components=len(component_names),
                               classes=len(class_names), window_size=windows['train'].x.shape[1]),
    }).to(device)
    if subsystem_epochs:
        # C2 differentiated rounds: oversample windows whose subsystem is prioritized.
        import numpy as np
        extra = json.loads(subsystem_epochs) if isinstance(subsystem_epochs, str) else subsystem_epochs
        train_ids = np.asarray(windows['train'].sample_ids)
        keep = list(range(len(train_ids)))
        for sub, count in extra.items():
            mask = np.array([sid.split(':')[0] == sub for sid in train_ids])
            idx = np.flatnonzero(mask)
            if len(idx):
                keep.extend(idx.tolist() * int(count))
        tx = windows['train'].x[keep]
        td = windows['train'].detection[keep]
        tc = windows['train'].component[keep]
        ti = windows['train'].isolation[keep]
    else:
        tx, td, tc, ti = windows['train'].x, windows['train'].detection, windows['train'].component, windows['train'].isolation
    loader = DataLoader(
        TensorDataset(torch.from_numpy(tx), torch.from_numpy(td), torch.from_numpy(tc), torch.from_numpy(ti)),
        batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed),
        pin_memory=device.startswith('cuda'),
    )
    criteria = (
        nn.CrossEntropyLoss(weight=phm_deep.weights(windows['train'].detection, 2).to(device)),
        nn.CrossEntropyLoss(weight=phm_deep.weights(windows['train'].component, len(component_names)).to(device)),
        nn.CrossEntropyLoss(weight=phm_deep.weights(windows['train'].isolation, len(class_names)).to(device)),
    )
    component_index = {name: index for index, name in enumerate(component_names)}
    class_component = [component_index[fault_to_component[name]] for name in class_names]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for x, d, c, i in loader:
            x, d, c, i = x.to(device), d.to(device), c.to(device), i.to(device)
            optimizer.zero_grad(set_to_none=True)
            dlogits, llogits, clogits = model(x, 'phm')
            active = i >= 0
            losses = [criteria[0](dlogits, d)]
            if active.any():
                losses.extend([criteria[1](llogits[active], c[active]), criteria[2](clogits[active], i[active])])
                losses.append(0.1 * phm_deep.hierarchy_js_loss(llogits, clogits, class_component, active))
            sum(losses).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(losses[0].detach()) * len(x)
        scheduler.step()
        # validation selection
        from scripts.ai_native.run_unified_baseline import UnifiedAdapter
        val = windows['validation']
        val_logistic = phm_deep.logistic_detector(windows['train'], val, seed)
        val_score = phm_deep.select_validation_state(
            phm_deep.infer(UnifiedAdapter(model, 'phm'), val, device=device),
            val_logistic, val, class_names, component_names, fault_to_component,
        )
        score = float(val_score['selection_score'])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f'[fold unified] epoch={epoch} val={score:.4f} best={best_epoch}', flush=True)
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            print(f'[fold unified] early_stop at {epoch}', flush=True)
            break
    model.load_state_dict(best_state)
    return model, best_epoch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'governance/evidence/ai_native/phm_cv_20260824')
    parser.add_argument('--exclude-subsystems', nargs='*', default=None,
                        help='subsystems to exclude from training (C1 task filtering; test still evaluates all)')
    parser.add_argument('--subsystem-epochs', default=None,
                        help='JSON map subsystem->extra epochs, e.g. {"thermal":10} (C2 differentiated rounds)')
    args = parser.parse_args()
    hierarchy = json.loads(PHM_HIERARCHY.read_text(encoding='utf-8'))
    records = phm_v60.load_records(PHM_ROOT)
    class_names = sorted(hierarchy['fault_to_component'])
    component_names = sorted(set(hierarchy['fault_to_component'].values()))
    fault_to_component = hierarchy['fault_to_component']
    fold_results = []
    for fold in range(args.folds):
        print(f'=== fold {fold} ===', flush=True)
        fold_records = fold_split(records, fold, exclude_subsystems=args.exclude_subsystems)
        subsystem_epochs = None
        if args.subsystem_epochs:
            try:
                subsystem_epochs = json.loads(args.subsystem_epochs)
            except Exception:
                subsystem_epochs = {}
                for part in args.subsystem_epochs.split(','):
                    if ':' in part:
                        sub, count = part.split(':', 1)
                        subsystem_epochs[sub.strip()] = int(count)
        audit = phm_v60.audit_split(fold_records)
        if audit['group_overlap_count'] != 0:
            raise RuntimeError(f'fold {fold} split audit failed: {audit["group_overlap"]}')
        seed = BASE_SEED + fold
        selected, median = phm_deep.select_train_channels(fold_records, class_names, 96)
        unscaled = phm_deep.build_windows(fold_records, selected, class_names, component_names, fault_to_component)
        windows = phm_deep.scale_windows(unscaled, median)
        print(f'  windows train={windows["train"].x.shape[0]} val={windows["validation"].x.shape[0]} test={windows["test"].x.shape[0]}', flush=True)
        # Control: independent residual_tcn (same budget).
        control = phm_deep.train_one(
            'residual_tcn', windows['train'], windows['validation'], windows['test'],
            class_names, component_names, fault_to_component, seed, MAX_EPOCHS,
            selected_channels=selected, device=args.device, batch_size=BATCH_SIZE,
            minimum_epochs=MIN_EPOCHS, early_stop_patience=PATIENCE,
        )
        control_e2e = float(control['end_to_end_18_class']['macro_f1'])
        print(f'  control residual_tcn E2E={control_e2e:.4f} best_epoch={control["best_epoch"]}', flush=True)
        # Unified PHM adapter.
        model, best_epoch = train_unified_phm(windows, class_names, component_names, fault_to_component, seed, args.device, subsystem_epochs=subsystem_epochs)
        unified_result = evaluate_phm(model, windows, class_names, component_names, fault_to_component, seed, args.device)
        unified_e2e = float(unified_result['end_to_end_18_class']['macro_f1'])
        print(f'  unified PHM adapter E2E={unified_e2e:.4f} best_epoch={best_epoch}', flush=True)
        fold_results.append({
            'fold': fold, 'seed': seed,
            'control_residual_tcn_e2e': control_e2e,
            'unified_phm_e2e': unified_e2e,
            'delta_vs_control': unified_e2e - control_e2e,
            'control_metrics': {k: round(float(v['macro_f1']), 4) for k, v in control.items() if isinstance(v, dict) and 'macro_f1' in v},
            'unified_metrics': {k: round(float(v['macro_f1']), 4) for k, v in unified_result.items() if isinstance(v, dict) and 'macro_f1' in v},
        })
    # Summary
    control_means = np.mean([r['control_residual_tcn_e2e'] for r in fold_results])
    unified_means = np.mean([r['unified_phm_e2e'] for r in fold_results])
    control_std = float(np.std([r['control_residual_tcn_e2e'] for r in fold_results]))
    unified_std = float(np.std([r['unified_phm_e2e'] for r in fold_results]))
    deltas = [r['delta_vs_control'] for r in fold_results]
    summary = {
        'folds': args.folds, 'seed_base': BASE_SEED,
        'control_mean_e2e': float(control_means), 'control_std_e2e': control_std,
        'unified_mean_e2e': float(unified_means), 'unified_std_e2e': unified_std,
        'mean_delta_vs_control': float(np.mean(deltas)), 'std_delta': float(np.std(deltas)),
        'all_folds_positive': all(d > 0 for d in deltas),
        'all_folds_beat_baseline': all(r['unified_phm_e2e'] > PHM_BASELINE_E2E for r in fold_results),
        'promotion_recommendation': (
            'PROMOTION_SUPPORTED' if (all(d > 0 for d in deltas) and float(np.mean(deltas)) > 0.01)
            else 'PROMOTION_NOT_SUPPORTED'
        ),
    }
    report = {'schema_version': 'astrograph-phm-cv.v1', 'fold_results': fold_results, 'summary': summary,
              'baseline_frozen_e2e': PHM_BASELINE_E2E, 'claim_boundary': 'Lab paired synthetic evidence; not flight/safety qualification.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'phm_cv_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())