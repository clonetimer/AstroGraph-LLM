#!/usr/bin/env python3
"""AI-Native Phase 2: PHM 5-fold CV with decision-level ensemble."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
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

BATCH_SIZE = 768
MAX_EPOCHS = 40
MIN_EPOCHS = 10
PATIENCE = 10
BASE_SEED = 20260814


def fold_split(records, fold: int):
    result = []
    for record in records:
        rem = record.sample_index % 5
        split = 'test' if rem == fold else ('validation' if rem == (fold + 1) % 5 else 'train')
        result.append(dataclasses.replace(record, split=split))
    return result


def infer_window_probs(model, window_set, device: str, *, dataset_id: str = 'phm'):
    import torch
    outputs = [[], [], []]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(window_set.x), 1024):
            batch = torch.from_numpy(window_set.x[start:start + 1024]).to(device)
            if dataset_id == 'phm':
                # UnifiedAdapter already carries the dataset id.
                dlogits, llogits, clogits = model(batch)
            else:
                dlogits, llogits, clogits = model(batch)
            probs = [torch.softmax(v, 1).cpu().numpy() for v in (dlogits, llogits, clogits)]
            for target, value in zip(outputs, probs):
                target.append(value)
    return tuple(np.concatenate(v) for v in outputs)


def train_unified_phm(windows, class_names, component_names, fault_to_component, seed: int, device: str):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    phm_deep.prepare_torch_device(device, seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = make_unified_platform_model({
        'phm': DatasetTaskSpec(channels=windows['train'].x.shape[-1], components=len(component_names),
                               classes=len(class_names), window_size=windows['train'].x.shape[1]),
    }).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(windows['train'].x), torch.from_numpy(windows['train'].detection),
                      torch.from_numpy(windows['train'].component), torch.from_numpy(windows['train'].isolation)),
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
    from scripts.ai_native.run_unified_baseline import UnifiedAdapter
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
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
        scheduler.step()
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
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_epoch


def train_control_tcn(windows, class_names, component_names, fault_to_component, seed: int, device: str):
    """Train the independent residual_tcn control and return (model, best_epoch, result_dict)."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    phm_deep.prepare_torch_device(device, seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    component_index = {name: index for index, name in enumerate(component_names)}
    class_component = [component_index[fault_to_component[name]] for name in class_names]
    model = phm_deep.make_model('residual_tcn', windows['train'].x.shape[-1], len(component_names), len(class_names)).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(windows['train'].x), torch.from_numpy(windows['train'].detection),
                      torch.from_numpy(windows['train'].component), torch.from_numpy(windows['train'].isolation)),
        batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed),
        pin_memory=device.startswith('cuda'),
    )
    criteria = (
        nn.CrossEntropyLoss(weight=phm_deep.weights(windows['train'].detection, 2).to(device)),
        nn.CrossEntropyLoss(weight=phm_deep.weights(windows['train'].component, len(component_names)).to(device)),
        nn.CrossEntropyLoss(weight=phm_deep.weights(windows['train'].isolation, len(class_names)).to(device)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, d, c, i in loader:
            x, d, c, i = x.to(device), d.to(device), c.to(device), i.to(device)
            optimizer.zero_grad(set_to_none=True)
            dlogits, llogits, clogits = model(x)
            active = i >= 0
            losses = [criteria[0](dlogits, d)]
            if active.any():
                losses.extend([criteria[1](llogits[active], c[active]), criteria[2](clogits[active], i[active])])
                losses.append(0.1 * phm_deep.hierarchy_js_loss(llogits, clogits, class_component, active))
            sum(losses).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()
        val_logistic = phm_deep.logistic_detector(windows['train'], windows['validation'], seed)
        val_score = phm_deep.select_validation_state(
            phm_deep.infer(model, windows['validation'], device=device),
            val_logistic, windows['validation'], class_names, component_names, fault_to_component,
        )
        score = float(val_score['selection_score'])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    result = phm_deep.evaluate_test(
        model, windows['test'], phm_deep.logistic_detector(windows['train'], windows['test'], seed),
        {'detector': 'deep', 'detection_threshold': 0.5, 'alarm_on_threshold': 0.5, 'alarm_off_threshold': 0.5,
         'alarm_minimum_consecutive': 1, 'hierarchy_blend_alpha': 0.5, 'conditioning_mode': 'hard'},
        class_names, component_names, fault_to_component, device=device,
    )
    return model, best_epoch, result


def ensemble_metrics(unified_probs, control_probs, test: Any, class_names, component_names, fault_to_component):
    from astrograph.data_driven.alarm_policy import apply_alarm_policy, AlarmPolicy
    ud, uc, ucls = unified_probs
    cd, cc, ccls = control_probs
    det_ens = np.maximum(ud[:, 1], cd[:, 1])
    comp_ens = 0.5 * (uc + cc)
    cls_ens = 0.5 * (ucls + ccls)
    policy = AlarmPolicy(on_threshold=0.5, off_threshold=0.5, minimum_consecutive=1)
    det_pred = np.asarray(apply_alarm_policy(det_ens.tolist(), test.sample_ids, test.end_times_s.tolist(), policy))
    component_index = {name: index for index, name in enumerate(component_names)}
    active = test.isolation >= 0
    component_pred = comp_ens.argmax(1)
    conditioned = np.zeros_like(cls_ens)
    for row, comp in enumerate(component_pred):
        allowed = [i for i, name in enumerate(class_names) if fault_to_component[name] == component_names[comp]]
        vals = cls_ens[row, allowed]
        conditioned[row, allowed] = vals / vals.sum() if vals.sum() else 1.0 / len(allowed)
    class_pred = conditioned.argmax(1)
    cascade_truth = np.where(test.detection == 0, 0, test.isolation + 1)
    cascade_pred = np.where(det_pred == 0, 0, class_pred + 1)
    def macro_f1(truth, pred, labels):
        from sklearn.metrics import f1_score
        return float(f1_score(truth, pred, labels=labels, average='macro', zero_division=0))
    return {
        'detection_macro_f1': macro_f1(test.detection, det_pred, [0, 1]),
        'localization_macro_f1': macro_f1(test.component[active], component_pred[active], list(range(len(component_names)))),
        'classification_macro_f1': macro_f1(test.isolation[active], class_pred[active], list(range(len(class_names)))),
        'end_to_end_18_class': macro_f1(cascade_truth, cascade_pred, list(range(len(class_names) + 1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'governance/evidence/ai_native/phm_cv_20260824')
    args = parser.parse_args()
    hierarchy = json.loads(PHM_HIERARCHY.read_text(encoding='utf-8'))
    records = phm_v60.load_records(PHM_ROOT)
    class_names = sorted(hierarchy['fault_to_component'])
    component_names = sorted(set(hierarchy['fault_to_component'].values()))
    fault_to_component = hierarchy['fault_to_component']
    from scripts.ai_native.run_unified_baseline import UnifiedAdapter
    fold_results = []
    for fold in range(args.folds):
        print(f'=== fold {fold} ===', flush=True)
        fold_records = fold_split(records, fold)
        seed = BASE_SEED + fold
        selected, median = phm_deep.select_train_channels(fold_records, class_names, 96)
        unscaled = phm_deep.build_windows(fold_records, selected, class_names, component_names, fault_to_component)
        windows = phm_deep.scale_windows(unscaled, median)
        control_model, control_best, control_result = train_control_tcn(
            windows, class_names, component_names, fault_to_component, seed, args.device,
        )
        control_e2e = float(control_result['end_to_end_18_class']['macro_f1'])
        model, best_epoch = train_unified_phm(windows, class_names, component_names, fault_to_component, seed, args.device)
        unified_probs = infer_window_probs(UnifiedAdapter(model, 'phm'), windows['test'], args.device)
        control_probs = infer_window_probs(control_model, windows['test'], args.device, dataset_id='ctrl')
        unified_result = phm_deep.evaluate_test(
            UnifiedAdapter(model, 'phm'), windows['test'],
            phm_deep.logistic_detector(windows['train'], windows['test'], seed),
            {'detector': 'deep', 'detection_threshold': 0.5, 'alarm_on_threshold': 0.5,
             'alarm_off_threshold': 0.5, 'alarm_minimum_consecutive': 1, 'hierarchy_blend_alpha': 0.5,
             'conditioning_mode': 'hard'},
            class_names, component_names, fault_to_component, device=args.device,
        )
        unified_e2e = float(unified_result['end_to_end_18_class']['macro_f1'])
        ens = ensemble_metrics(unified_probs, control_probs, windows['test'], class_names, component_names, fault_to_component)
        ens_e2e = float(ens['end_to_end_18_class'])
        print(f'  control={control_e2e:.4f} unified={unified_e2e:.4f} ensemble={ens_e2e:.4f}', flush=True)
        fold_results.append({
            'fold': fold, 'seed': seed,
            'control_e2e': control_e2e, 'unified_e2e': unified_e2e, 'ensemble_e2e': ens_e2e,
            'ensemble_metrics': ens,
        })
    summary = {
        'control_mean': float(np.mean([r['control_e2e'] for r in fold_results])),
        'unified_mean': float(np.mean([r['unified_e2e'] for r in fold_results])),
        'ensemble_mean': float(np.mean([r['ensemble_e2e'] for r in fold_results])),
        'ensemble_std': float(np.std([r['ensemble_e2e'] for r in fold_results])),
        'ensemble_beats_control_all': all(r['ensemble_e2e'] >= r['control_e2e'] for r in fold_results),
        'ensemble_beats_unified_all': all(r['ensemble_e2e'] >= r['unified_e2e'] for r in fold_results),
        'promotion_recommendation': (
            'PROMOTION_SUPPORTED' if all(r['ensemble_e2e'] >= r['control_e2e'] for r in fold_results) and float(np.mean([r['ensemble_e2e'] for r in fold_results])) > 0.84
            else 'PROMOTION_NOT_SUPPORTED'
        ),
    }
    report = {'schema_version': 'astrograph-phm-cv-ensemble.v1', 'fold_results': fold_results, 'summary': summary,
              'claim_boundary': 'Lab paired synthetic evidence; not flight/safety qualification.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'phm_cv_ensemble_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())