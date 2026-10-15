"""Distinguish canonical receipt identity from exact serialized file identity."""
import json
from pathlib import Path
from .external_dataset_contract import ExternalDatasetReceipt, file_identity

DIGEST_SEMANTICS = 'CANONICAL_JSON_EXCLUDING_SELF_DIGEST_FIELD'


def verify_dataset_admission_index(admission, receipt_root):
    root = Path(receipt_root).resolve()
    if admission.get('receipt_digest_semantics') != DIGEST_SEMANTICS:
        raise ValueError('RECEIPT_DIGEST_SEMANTICS_REQUIRED')
    rows = admission.get('datasets', [])
    if not rows:
        raise ValueError('DATASET_INDEX_EMPTY')
    seen = set()
    verified = []
    for row in rows:
        name = row['receipt_file']
        if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('RECEIPT_PATH_ESCAPES_ROOT')
        if name in seen:
            raise ValueError('DUPLICATE_DATASET_INDEX_ENTRY')
        seen.add(name)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('RECEIPT_PATH_ESCAPES_ROOT')
        actual = file_identity(path)['sha256']
        if row.get('file_sha256') != actual:
            raise ValueError('DATASET_RECEIPT_FILE_BYTES_DRIFT:' + name)
        receipt = json.loads(path.read_text(encoding='utf-8-sig'))
        ExternalDatasetReceipt.model_validate(receipt)
        if row.get('receipt_sha256') != receipt['receipt_sha256']:
            raise ValueError('DATASET_RECEIPT_CANONICAL_INDEX_DRIFT:' + name)
        verified.append({'receipt_file': name, 'receipt_sha256': receipt['receipt_sha256'], 'file_sha256': actual})
    return {'status': 'PASS', 'verified_count': len(verified), 'datasets': verified}
