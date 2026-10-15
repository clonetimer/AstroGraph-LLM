"""Verify both receipt identities against the external receipt directory."""
import argparse
import json
import pathlib
import sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from astrograph.evaluation.dataset_admission_index import verify_dataset_admission_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=pathlib.Path, default=ROOT / 'PATCH_R20_DATASET_ADMISSION.json')
    parser.add_argument('--receipt-dir', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = verify_dataset_admission_index(json.loads(args.admission.read_text(encoding='utf-8-sig')), args.receipt_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
