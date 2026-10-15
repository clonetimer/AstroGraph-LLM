#!/usr/bin/env python3
"""C3: PEFT comparison - freeze shared backbone, train only PHM adapter (vs full training)."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

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


def train_peft(windows, class_names, component_names, fault_to_component, seed: int, device: str):
    """Freeze backbone + pool; train only the PHM adapter (PEFT/C3)."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    phm_deep.prepare_torch_device(device, seed)
    np.random.seed(seed)
    model = make_unified_platform_model({
        'phm': DatasetTaskSpec(channels=windows['train'].x.shape[-1], components=len(component_names),
                               classes=len(class_names), window_size=windows['train'].x.shape[1]),
    }).to(device)
    for name, param in model.named_parameters():
        if name.startswith('blocks') or name.startswith('pool_projection'):
            param.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
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
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=2e-3)
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
            sum(losses).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        val_logistic = phm_deep.logistic_detector(windows['train'], windows['validation'], seed)
        val_score = phm_deep.select_validation_state(
            phm_deep.infer(UnifiedAdapter(model, 'phm'), windows['validation'], device=device),
            val_logistic, windows['validation'], class_names, component_names, fault_to_component,
        )
        score = float(val_score['selection_score'])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    return model, best_epoch, trainable_count, total_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'governance/evidence/ai_native/phm_cv_20260824_peft')
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
        model, best_epoch, trainable, total = train_peft(
            windows, class_names, component_names, fault_to_component, seed, args.device,
        )
        result = phm_deep.evaluate_test(
            UnifiedAdapter(model, 'phm'), windows['test'],
            phm_deep.logistic_detector(windows['train'], windows['test'], seed),
            {'detector': 'deep', 'detection_threshold': 0.5, 'alarm_on_threshold': 0.5,
             'alarm_off_threshold': 0.5, 'alarm_minimum_consecutive': 1, 'hierarchy_blend_alpha': 0.5,
             'conditioning_mode': 'hard'},
            class_names, component_names, fault_to_component, device=args.device,
        )
        e2e = float(result['end_to_end_18_class']['macro_f1'])
        print(f'  peft unified E2E={e2e:.4f} best_epoch={best_epoch} trainable={trainable}/{total}', flush=True)
        fold_results.append({'fold': fold, 'seed': seed, 'peft_e2e': e2e, 'best_epoch': best_epoch, 'trainable_params': trainable, 'total_params': total})
    summary = {
        'peft_mean_e2e': float(np.mean([r['peft_e2e'] for r in fold_results])),
        'peft_std_e2e': float(np.std([r['peft_e2e'] for r in fold_results])),
        'trainable_fraction': round(fold_results[0]['trainable_params'] / fold_results[0]['total_params'], 4) if fold_results else 0.0,
        'full_training_mean_e2e_ref': 0.8578,
        'delta_peft_vs_full': round(float(np.mean([r['peft_e2e'] for r in fold_results])) - 0.8578, 4),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'phm_cv_peft_report.json').write_text(json.dumps({'fold_results': fold_results, 'summary': summary}, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())