"""Reproduce architecture counterexamples, not diagnostic performance estimates.

Uses the actual ADCS provider with controlled logits. No trained model, simulated
fault labels, or held-out data are used. Run from the repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

import torch


class ControlledHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, window):
        n = len(window)
        return tuple(torch.log(torch.tensor([p])).repeat(n, 1)
                     for p in ([0.1, 0.9], [0.99, 0.01], [0.49, 0.51]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fixture = runpy.run_path(str(root / 'tests/test_adcs_three_task_torch_provider.py'))
    provider = fixture['_provider']()
    provider.model = ControlledHeads()
    provider.class_map = {'adcs_other': 'IMU_OTHER', 'adcs_rw_jamming': 'ADCS_RW_JAM'}
    provider.runtime_fault_ids = frozenset(provider.class_map.values())
    provider.manifest = provider.manifest.model_copy(update={
        'supported_fault_ids': sorted(provider.runtime_fault_ids)})
    provider.__post_init__()
    output = provider.predict_raw(fixture['_request']())
    assert output.metadata['predicted_dataset_class'] == 'adcs_other'
    assert output.metadata['dataset_class_probabilities']['adcs_other'] == 1.0
    assert output.out_of_distribution_score == 0.0
    sources = [
        'src/astrograph/integrations/adcs_three_task_torch.py',
        'src/astrograph/oag/window_features.py',
        'src/astrograph/oag/summary.py',
        'src/astrograph/retrieval/experience_context.py',
        'src/astrograph/data_driven/memory_layers.py',
        'src/astrograph/integrations/platform_knowledge.py',
        'src/astrograph/retrieval/service.py',
    ]
    report = {
        'schema_version': 'astrograph-architecture-audit.v1',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        'experiment_type': 'CONTROLLED_LOGITS_COUNTEREXAMPLE_NOT_ACCURACY_BENCHMARK',
        'input_class_probabilities': {'adcs_other': 0.49, 'adcs_rw_jamming': 0.51},
        'input_direct_component_probabilities': {'imu': 0.99, 'reaction_wheel': 0.01},
        'hierarchy_blend_alpha': 0.5,
        'actual_output': output.model_dump(mode='json'),
        'findings': [
            'Hard component selection can eliminate the highest raw class probability.',
            'Renormalized probability 1.0 is conditional on the selected component, not calibrated correctness.',
            'With every class mapped, unsupported mass is zero; this is not an input-distribution novelty test.',
        ],
        'limits': 'This proves possible behavior in the current provider; no frequency, real-data harm, or fix benefit is estimated.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(args.output), 'probe_count': 3}))


if __name__ == '__main__':
    main()
