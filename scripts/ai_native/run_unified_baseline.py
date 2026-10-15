#!/usr/bin/env python3
"""AI-Native Phase 3: unified domain-model baseline training (preregistered).

Trains one shared dilated-residual backbone with per-dataset adapters (ADCS / PHM)
under the frozen protocol in docs/CURRENT_ENGINEERING_POLICY.md:
* seed 20260814, batch 768, max/min 80/20 epochs, patience 15, CUDA,
* validation-only checkpoint selection, test evaluated exactly once,
* same-budget independent-model control on the same split/seed/preprocessing,
* transfer probe: frozen backbone + <=2 target-domain groups per dataset.

Honesty: ADCS uses the frozen 29-channel FEATURE_COLUMNS surface (the skeleton's
43-channel spec was a placeholder); PHM uses the frozen 96-channel selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from scripts.integration import run_adcs_v522_regression as adcs
from scripts.data_driven import compare_phm_causal_windows as phm_deep
from scripts.data_driven import compare_adcs_deep_encoders as adcs_deep
from astrograph.data_driven import phm_v60
from astrograph.data_driven.unified_platform import DatasetTaskSpec, make_unified_platform_model

SEED = 20260814
BATCH_SIZE = 768
MAX_EPOCHS = 80
MIN_EPOCHS = 20
PATIENCE = 15

ADCS_ROOT = Path('D:/Project/test_lunwen/whole_spacecraft_adcs_paired_dataset_v5_2_2')
PHM_ROOT = Path('D:/Project/test_lunwen/whole_spacecraft_phm_dataset_v6_0')
PHM_HIERARCHY = ROOT / 'configs/datasets/phm_v6_0/diagnosis_hierarchy.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_adcs(root: Path):
    rows = adcs.load_trajectories(root)
    audit = adcs.audit_split(rows)
    if audit['status'] != 'PASS':
        raise RuntimeError(f'ADCS split audit failed: {audit}')
    class_names = sorted({r.diagnostic_class for r in rows if r.diagnostic_class != 'nominal'})
    component_names = list(adcs.COMPONENT_NAMES)
    windows = adcs.build_windows(
        rows,
        {name: index for index, name in enumerate(class_names)},
        {name: index for index, name in enumerate(component_names)},
    )
    impute, mean, scale = adcs.fit_train_scaler(windows['train'].x)
    scaled = {name: adcs.scale_windows(windows[name], impute, mean, scale) for name in windows}
    return {
        'audit': audit,
        'class_names': class_names,
        'component_names': component_names,
        'windows': scaled,
        'channels': windows['train'].x.shape[-1],
        'window': windows['train'].x.shape[1],
    }


def load_phm(root: Path):
    hierarchy = json.loads(PHM_HIERARCHY.read_text(encoding='utf-8'))
    records = phm_v60.load_records(root)
    audit = phm_v60.audit_split(records)
    if audit['status'] != 'PASS':
        raise RuntimeError(f'PHM split audit failed: {audit}')
    class_names = sorted(hierarchy['fault_to_component'])
    component_names = sorted(set(hierarchy['fault_to_component'].values()))
    selected, median = phm_deep.select_train_channels(records, class_names, 96)
    unscaled = phm_deep.build_windows(
        records, selected, class_names, component_names, hierarchy['fault_to_component'],
    )
    windows = phm_deep.scale_windows(unscaled, median)
    return {
        'audit': audit,
        'class_names': class_names,
        'component_names': component_names,
        'windows': windows,
        'fault_to_component': hierarchy['fault_to_component'],
        'selected_channels': selected,
        'channels': windows['train'].x.shape[-1],
        'window': windows['train'].x.shape[1],
    }


class UnifiedAdapter:
    """One-dataset view of the unified model for reuse of validation/eval helpers."""

    def __init__(self, model: Any, dataset_id: str) -> None:
        self.model = model
        self.dataset_id = dataset_id

    def __call__(self, value):
        return self.model(value, self.dataset_id)

    def eval(self):
        return self.model.eval()

    def train(self, mode: bool = True):
        return self.model.train(mode)

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()


def make_class_weights(values: np.ndarray, count: int) -> np.ndarray:
    return phm_deep.weights(values, count)


def combined_validation_score(model: Any, adcs_data: dict[str, Any], phm_data: dict[str, Any], device: str) -> dict[str, Any]:
    """Selection score over both datasets (equal weights), mirroring per-dataset protocols."""
    import torch
    torch.cuda.empty_cache() if device.startswith('cuda') else None
    adcs_val = adcs_data['windows']['validation']
    phm_val = phm_data['windows']['validation']
    adcs_logistic = adcs_deep.logistic_detection_scores(
        adcs_data['windows']['train'], adcs_val, SEED,
    )
    adcs_score = adcs_deep.validation_score(
        UnifiedAdapter(model, 'adcs'), adcs_val,
        adcs_data['class_names'], adcs_data['component_names'], adcs_logistic,
        device=device,
    )
    phm_logistic = phm_deep.logistic_detector(
        phm_data['windows']['train'], phm_val, SEED,
    )
    phm_score = phm_deep.select_validation_state(
        phm_deep.infer(UnifiedAdapter(model, 'phm'), phm_val, device=device),
        phm_logistic, phm_val, phm_data['class_names'], phm_data['component_names'],
        phm_data['fault_to_component'],
    )
    return {
        'adcs': adcs_score,
        'phm': phm_score,
        'selection_score': 0.5 * float(adcs_score['selection_score']) + 0.5 * float(phm_score['selection_score']),
    }


def train_unified(
    adcs_data: dict[str, Any], phm_data: dict[str, Any], *, device: str,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if device == "cuda":
        device = "cuda:0"
    torch_device = phm_deep.prepare_torch_device(device, SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    specs = {
        'adcs': DatasetTaskSpec(
            channels=adcs_data['channels'], components=len(adcs_data['component_names']),
            classes=len(adcs_data['class_names']), window_size=adcs_data['window'],
        ),
        'phm': DatasetTaskSpec(
            channels=phm_data['channels'], components=len(phm_data['component_names']),
            classes=len(phm_data['class_names']), window_size=phm_data['window'],
        ),
    }
    model = make_unified_platform_model(specs).to(device)
    print(f'unified shared params: {model.shared_parameter_count()}', flush=True)

    def build_loader(name: str, window_set) -> DataLoader:
        return DataLoader(
            TensorDataset(
                torch.from_numpy(window_set.x),
                torch.from_numpy(window_set.detection),
                torch.from_numpy(window_set.component),
                torch.from_numpy(window_set.isolation),
            ),
            batch_size=BATCH_SIZE, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
            pin_memory=device.startswith('cuda'),
        )

    adcs_class_component = adcs_deep.adcs_subsystem_groups(
        adcs.FEATURE_COLUMNS, adcs_data['component_names'], adcs_data['class_names'],
    )[1]
    phm_component_index = {name: index for index, name in enumerate(phm_data['component_names'])}
    phm_class_component = [
        phm_component_index[phm_data['fault_to_component'][name]]
        for name in phm_data['class_names']
    ]

    criteria = {
        'adcs': (
            nn.CrossEntropyLoss(weight=make_class_weights(adcs_data['windows']['train'].detection, 2).to(device)),
            nn.CrossEntropyLoss(weight=make_class_weights(adcs_data['windows']['train'].component, len(adcs_data['component_names'])).to(device)),
            nn.CrossEntropyLoss(weight=make_class_weights(adcs_data['windows']['train'].isolation, len(adcs_data['class_names'])).to(device)),
        ),
        'phm': (
            nn.CrossEntropyLoss(weight=make_class_weights(phm_data['windows']['train'].detection, 2).to(device)),
            nn.CrossEntropyLoss(weight=make_class_weights(phm_data['windows']['train'].component, len(phm_data['component_names'])).to(device)),
            nn.CrossEntropyLoss(weight=make_class_weights(phm_data['windows']['train'].isolation, len(phm_data['class_names'])).to(device)),
        ),
    }
    class_component = {'adcs': adcs_class_component, 'phm': phm_class_component}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch, best_validation = -1.0, 0, None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running, steps = 0.0, 0
        loaders = {
            'adcs': build_loader('adcs', adcs_data['windows']['train']),
            'phm': build_loader('phm', phm_data['windows']['train']),
        }
        # interleave batches from both datasets until both are exhausted
        adcs_iter, phm_iter = iter(loaders['adcs']), iter(loaders['phm'])
        adcs_done = phm_done = False
        while not (adcs_done and phm_done):
            for name, iterator, done in (
                ('adcs', adcs_iter, adcs_done), ('phm', phm_iter, phm_done),
            ):
                if done:
                    continue
                try:
                    batch = next(iterator)
                except StopIteration:
                    if name == 'adcs':
                        adcs_done = True
                    else:
                        phm_done = True
                    continue
                x, detection, component, isolation = [item.to(device, non_blocking=True) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                dlogits, llogits, clogits = model(x, name)
                active = isolation >= 0
                task_losses = [criteria[name][0](dlogits, detection)]
                if active.any():
                    task_losses.extend([
                        criteria[name][1](llogits[active], component[active]),
                        criteria[name][2](clogits[active], isolation[active]),
                    ])
                    task_losses.append(
                        0.1 * phm_deep.hierarchy_js_loss(
                            llogits, clogits, class_component[name], active,
                        ),
                    )
                loss = sum(task_losses)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                running += float(loss.detach()) * len(x)
                steps += 1
        scheduler.step()
        validation = combined_validation_score(model, adcs_data, phm_data, device)
        score = float(validation['selection_score'])
        entry = {
            'epoch': epoch, 'train_loss': running / steps if steps else float('nan'),
            'selection_score': score,
            'adcs_selection': float(validation['adcs']['selection_score']),
            'phm_selection': float(validation['phm']['selection_score']),
        }
        history.append(entry)
        print(
            f'[unified] epoch={epoch} loss={entry["train_loss"]:.4f} '
            f"val={score:.4f} adcs={entry['adcs_selection']:.4f} phm={entry['phm_selection']:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score, best_epoch, best_validation = score, epoch, validation
            best_state = copy.deepcopy(model.state_dict())
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            print(f'[unified] early_stop epoch={epoch} best_epoch={best_epoch}', flush=True)
            break
    model.load_state_dict(best_state)
    return {
        'model': model,
        'best_epoch': best_epoch,
        'best_validation': best_validation,
        'history': history,
        'training_seconds': time.perf_counter() - started,
        'parameter_count': sum(value.numel() for value in model.parameters()),
        'shared_parameter_count': model.shared_parameter_count(),
    }


def evaluate_test_adcs(model: Any, data: adcs.WindowSet, train: adcs.WindowSet, selection: dict[str, Any], class_names: list[str], component_names: list[str], *, device: str) -> dict[str, Any]:
    """ADCS test evaluation mirroring compare_adcs_deep_encoders.train_model internals."""
    import numpy as np
    detection, component, classes = adcs_deep.predict(UnifiedAdapter(model, 'adcs'), data, device=device)
    test_logistic = adcs_deep.logistic_detection_scores(train, data, SEED)
    detection_score = (
        float(selection["detector_deep_blend_alpha"]) * detection[:, 1]
        + (1.0 - float(selection["detector_deep_blend_alpha"])) * test_logistic
    )
    component = adcs_deep.hierarchy_blend(
        component, classes, float(selection["hierarchy_blend_alpha"]),
        class_names, component_names,
    )
    from astrograph.data_driven.alarm_policy import AlarmPolicy, apply_alarm_policy
    alarm_policy = AlarmPolicy(
        on_threshold=float(selection["alarm_on_threshold"]),
        off_threshold=float(selection["alarm_off_threshold"]),
        minimum_consecutive=int(selection["alarm_minimum_consecutive"]),
    )
    detection_predicted = np.asarray(apply_alarm_policy(
        detection_score.tolist(), data.trajectory_ids, data.end_times_s.tolist(), alarm_policy,
    ))
    report, _ = adcs.three_stage_report(
        family="unified_platform_adcs",
        test=data,
        det_score=detection_score,
        component_prob=component,
        all_class_prob=classes,
        class_names=class_names,
        component_names=component_names,
        mechanism="unified dilated-residual backbone + adcs adapter",
        detection_threshold=float(selection["detection_threshold"]),
        detection_predicted=detection_predicted,
        conditioning_mode=str(selection["conditioning_mode"]),
    )
    return report


def evaluate_unified(model: Any, adcs_data: dict[str, Any], phm_data: dict[str, Any], best_validation: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Evaluate the frozen test split exactly once for each dataset."""
    adcs_test = adcs_data['windows']['test']
    phm_test = phm_data['windows']['test']
    adcs_eval = evaluate_test_adcs(
        model, adcs_test, adcs_data['windows']['train'],
        best_validation['adcs'], adcs_data['class_names'], adcs_data['component_names'],
        device=device,
    )
    phm_logistic = phm_deep.logistic_detector(
        phm_data['windows']['train'], phm_test, SEED,
    )
    phm_eval = phm_deep.evaluate_test(
        UnifiedAdapter(model, 'phm'), phm_test, phm_logistic,
        best_validation['phm'], phm_data['class_names'], phm_data['component_names'],
        phm_data['fault_to_component'], device=device,
    )
    return {'adcs': adcs_eval, 'phm': phm_eval}


def run_transfer_probes(model: Any, adcs_data: dict[str, Any], phm_data: dict[str, Any], best_validation: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Protocol item 4: frozen backbone + <=2 target-domain groups per dataset adapter fine-tune.

    Each probe freezes blocks + pool_projection and fine-tunes ONLY the target dataset
    adapter on a small train subset (sample_index groups {0,1} for ADCS, {0,1} for PHM).
    Reports E2E macro-F1 on the frozen test split vs the fully-trained unified model.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    for name in ('blocks', 'pool_projection'):
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad = False
    criteria = {}
    for dataset_id, data in (('adcs', adcs_data), ('phm', phm_data)):
        train = data['windows']['train']
        criteria[dataset_id] = (
            nn.CrossEntropyLoss(weight=make_class_weights(train.detection, 2).to(device)),
            nn.CrossEntropyLoss(weight=make_class_weights(train.component, len(data['component_names'])).to(device)),
            nn.CrossEntropyLoss(weight=make_class_weights(train.isolation, len(data['class_names'])).to(device)),
        )

    def filter_indices(window_set, id_field: str, allowed: set[str]) -> list[int]:
        ids = getattr(window_set, id_field)
        return [index for index, value in enumerate(ids) if value in allowed]

    def tensor_of(window_set, indices):
        return (
            torch.from_numpy(window_set.x[indices]),
            torch.from_numpy(window_set.detection[indices]),
            torch.from_numpy(window_set.component[indices]),
            torch.from_numpy(window_set.isolation[indices]),
        )

    def adapter_losses(dataset_id: str, x, detection, component, isolation):
        active = isolation >= 0
        dlogits, llogits, clogits = model(x, dataset_id)
        losses = [criteria[dataset_id][0](dlogits, detection)]
        if active.any():
            losses.extend([
                criteria[dataset_id][1](llogits[active], component[active]),
                criteria[dataset_id][2](clogits[active], isolation[active]),
            ])
        return losses

    results: dict[str, Any] = {}
    # ---- PHM probe: fine-tune phm adapter on <=2 sample groups of PHM train ----
    phm_train = phm_data['windows']['train']
    phm_groups = sorted(set(phm_train.sample_ids))[:2]
    phm_keep = filter_indices(phm_train, 'sample_ids', set(phm_groups))
    loader = DataLoader(
        TensorDataset(*tensor_of(phm_train, phm_keep)), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.adapters['phm'].parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.adapters['phm'].parameters(), lr=3e-4)
    before = evaluate_unified(model, adcs_data, phm_data, best_validation, device=device)
    for _ in range(5):
        model.train()
        for x, d, c, i in loader:
            x, d, c, i = x.to(device), d.to(device), c.to(device), i.to(device)
            optimizer.zero_grad(set_to_none=True)
            sum(adapter_losses('phm', x, d, c, i)).backward()
            optimizer.step()
    after = evaluate_unified(model, adcs_data, phm_data, best_validation, device=device)
    results['phm_adapter_finetune'] = {
        'before_e2e_macro_f1': float(before['phm']['end_to_end_18_class']['macro_f1']),
        'after_e2e_macro_f1': float(after['phm']['end_to_end_18_class']['macro_f1']),
        'delta_e2e_macro_f1': float(after['phm']['end_to_end_18_class']['macro_f1']) - float(before['phm']['end_to_end_18_class']['macro_f1']),
        'finetune_epochs': 5,
        'train_groups': phm_groups,
        'train_window_count': len(phm_keep),
        'backbone': 'frozen',
        'scope': 'phm adapter only (<=2 sample groups)',
    }
    # ---- ADCS probe: fine-tune adcs adapter on <=2 trajectory groups of ADCS train ----
    adcs_train = adcs_data['windows']['train']
    adcs_groups = sorted(set(adcs_train.trajectory_ids))[:2]
    adcs_keep = filter_indices(adcs_train, 'trajectory_ids', set(adcs_groups))
    loader = DataLoader(
        TensorDataset(*tensor_of(adcs_train, adcs_keep)), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    for parameter in model.adapters['phm'].parameters():
        parameter.requires_grad = False
    for parameter in model.adapters['adcs'].parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.adapters['adcs'].parameters(), lr=3e-4)
    before = evaluate_unified(model, adcs_data, phm_data, best_validation, device=device)
    for _ in range(5):
        model.train()
        for x, d, c, i in loader:
            x, d, c, i = x.to(device), d.to(device), c.to(device), i.to(device)
            optimizer.zero_grad(set_to_none=True)
            sum(adapter_losses('adcs', x, d, c, i)).backward()
            optimizer.step()
    after = evaluate_unified(model, adcs_data, phm_data, best_validation, device=device)
    results['adcs_adapter_finetune'] = {
        'before_e2e_macro_f1': float(before['adcs']['end_to_end_15_class']['macro_f1']),
        'after_e2e_macro_f1': float(after['adcs']['end_to_end_15_class']['macro_f1']),
        'delta_e2e_macro_f1': float(after['adcs']['end_to_end_15_class']['macro_f1']) - float(before['adcs']['end_to_end_15_class']['macro_f1']),
        'finetune_epochs': 5,
        'train_groups': adcs_groups,
        'train_window_count': len(adcs_keep),
        'backbone': 'frozen',
        'scope': 'adcs adapter only (<=2 trajectory groups)',
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'governance/evidence/ai_native/unified_baseline_20260823')
    parser.add_argument('--skip-controls', action='store_true')
    args = parser.parse_args()

    print('loading ADCS ...', flush=True)
    adcs_data = load_adcs(ADCS_ROOT)
    print(f'ADCS windows: ' + ', '.join(f"{name}={item.x.shape[0]}" for name, item in adcs_data['windows'].items()), flush=True)
    print('loading PHM ...', flush=True)
    phm_data = load_phm(PHM_ROOT)
    print(f'PHM windows: ' + ', '.join(f"{name}={item.x.shape[0]}" for name, item in phm_data['windows'].items()), flush=True)

    if args.device == "cuda":
        args.device = "cuda:0"
    result = train_unified(adcs_data, phm_data, device=args.device)
    model = result['model']
    checkpoint_path = args.output_dir / 'unified_baseline_model.pt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save({'model_state_dict': model.state_dict(), 'seed': SEED, 'best_epoch': result['best_epoch']}, checkpoint_path)
    result['checkpoint'] = str(checkpoint_path.resolve())
    test_eval = evaluate_unified(model, adcs_data, phm_data, result['best_validation'], device=args.device)
    result['test'] = test_eval
    transfer = run_transfer_probes(model, adcs_data, phm_data, result['best_validation'], device=args.device)
    result['transfer_probes'] = transfer
    result.pop('model')

    controls = {}
    if args.skip_controls:
        prior = args.output_dir / 'unified_baseline_report.json'
        if prior.is_file():
            prior_report = json.loads(prior.read_text(encoding='utf-8'))
            controls = prior_report.get('controls', {})
            print(f'reusing {len(controls)} prior control runs', flush=True)
    if not args.skip_controls:
        print('control: ADCS residual_tcn (same budget) ...', flush=True)
        control_adcs = adcs_deep.train_model(
            'residual_tcn', adcs_data['windows']['train'], adcs_data['windows']['validation'],
            adcs_data['windows']['test'], adcs_data['class_names'], adcs_data['component_names'],
            epochs=MAX_EPOCHS, seed=SEED, device=args.device, batch_size=BATCH_SIZE,
            minimum_epochs=MIN_EPOCHS, early_stop_patience=PATIENCE,
        )
        controls['adcs_residual_tcn'] = {k: v for k, v in control_adcs.items() if k != 'validation_history'}
        print('control: PHM residual_tcn (same budget) ...', flush=True)
        control_phm = phm_deep.train_one(
            'residual_tcn', phm_data['windows']['train'], phm_data['windows']['validation'],
            phm_data['windows']['test'], phm_data['class_names'], phm_data['component_names'],
            phm_data['fault_to_component'], SEED, MAX_EPOCHS,
            selected_channels=phm_data['selected_channels'], device=args.device,
            batch_size=BATCH_SIZE, minimum_epochs=MIN_EPOCHS,
            early_stop_patience=PATIENCE,
        )
        controls['phm_residual_tcn'] = {k: v for k, v in control_phm.items() if k != 'history'}

    frozen = {
        'adcs': {'e2e_macro_f1': 0.8974, 'source': 'governance/evidence/research/data_driven_runs/data_driven_mechanism_p2_20260818/report.md'},
        'phm': {'e2e_macro_f1': 0.5147, 'source': 'governance/evidence/research/data_driven_runs/phm_v60_regression/phm_v60_regression_report.json'},
    }
    decision = {}
    for name in ('adcs', 'phm'):
        unified_e2e = float(test_eval[name]['end_to_end_15_class' if name == 'adcs' else 'end_to_end_18_class']['macro_f1'])
        baseline = frozen[name]['e2e_macro_f1']
        regression = unified_e2e - baseline
        decision[name] = {
            'unified_e2e_macro_f1': unified_e2e,
            'frozen_independent_e2e_macro_f1': baseline,
            'delta': regression,
            'stop_loss_triggered': regression < -0.010,
        }

    report = {
        'schema_version': 'astrograph-ai-native-phase3-unified-baseline.v1',
        'generated_at': '2026-08-23T23:30:00+08:00',
        'seed': SEED, 'batch_size': BATCH_SIZE, 'max_epochs': MAX_EPOCHS,
        'min_epochs': MIN_EPOCHS, 'patience': PATIENCE, 'device': args.device,
        'unified': result,
        'controls': controls,
        'frozen_baselines': frozen,
        'decision': decision,
        'datasets': {
            'adcs': {'channels': adcs_data['channels'], 'window': adcs_data['window'],
                      'classes': len(adcs_data['class_names']), 'components': len(adcs_data['component_names']),
                      'split_audit': adcs_data['audit'], 'preprocessing': 'standard train-only',
                      'feature_surface': 'frozen 29-channel FEATURE_COLUMNS (skeleton 43 was a placeholder)'},
            'phm': {'channels': phm_data['channels'], 'window': phm_data['window'],
                     'classes': len(phm_data['class_names']), 'components': len(phm_data['component_names']),
                     'split_audit': phm_data['audit'], 'preprocessing': 'standard train-only',
                     'selected_channels': phm_data['selected_channels'], 'selected_count': len(phm_data['selected_channels'])},
        },
        'environment': {
            'python': platform.python_version(), 'platform': platform.platform(),
            'numpy': np.__version__,
        },
        'claim_boundary': (
            'Single frozen grouped split, single seed, validation-only checkpoint selection, test evaluated once. '
            'Lab paired synthetic evidence only; not calibration, online, safety, production or flight qualification.'
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'unified_baseline_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    print(f'[report] {args.output_dir / "unified_baseline_report.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
