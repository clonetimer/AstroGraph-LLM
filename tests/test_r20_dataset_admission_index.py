import json
import pytest
from astrograph.evaluation.external_dataset_contract import digest, seal, file_identity
from astrograph.evaluation.dataset_admission_index import DIGEST_SEMANTICS, verify_dataset_admission_index


def fixture(tmp_path):
    semantics={'0':'nominal','1':'anomaly'};universe=['nominal','anomaly']
    receipt=seal({'schema_version':'astrograph-external-evaluation-dataset-receipt.v1','dataset_id':'TEST','dataset_family':'TEST',
        'source_artifact_sha256':'a'*64,'source_artifact_size_bytes':1,'evaluation_task_id':'TEST','task_kind':'TEST',
        'label_semantics':semantics,'label_semantics_sha256':digest(semantics),'target_universe':universe,'target_universe_sha256':digest(universe),
        'group_policy':{},'truth_provenance':{},'payload_manifest_sha256':'b'*64,'allowed_claims':[],'forbidden_claims':[]})
    path=tmp_path/'dataset_receipt.json';path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    admission={'receipt_digest_semantics':DIGEST_SEMANTICS,'datasets':[{'receipt_file':path.name,'receipt_sha256':receipt['receipt_sha256'],'file_sha256':file_identity(path)['sha256']}]}
    return path,admission


def test_r20_dataset_admission_index_matches_receipt_bytes(tmp_path):
    path,admission=fixture(tmp_path)
    row=admission['datasets'][0]
    assert row['file_sha256']!=row['receipt_sha256']
    assert verify_dataset_admission_index(admission,tmp_path)['verified_count']==1
    # A whitespace-only serialization change preserves canonical identity but
    # must fail exact-file binding until the file index is refreshed.
    path.write_text(path.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='FILE_BYTES_DRIFT'):
        verify_dataset_admission_index(admission,tmp_path)
    row['file_sha256']=file_identity(path)['sha256']
    assert verify_dataset_admission_index(admission,tmp_path)['status']=='PASS'


def test_file_hash_cannot_replace_canonical_identity(tmp_path):
    _,admission=fixture(tmp_path)
    admission['datasets'][0]['receipt_sha256']=admission['datasets'][0]['file_sha256']
    with pytest.raises(ValueError,match='CANONICAL_INDEX_DRIFT'):
        verify_dataset_admission_index(admission,tmp_path)
