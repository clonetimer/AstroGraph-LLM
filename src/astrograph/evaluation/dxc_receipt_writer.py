"""Seal the actual DXC offline runner outputs to their task and input bytes."""
import hashlib
import json
import zipfile
from .external_dataset_contract import (
    ExternalDatasetReceipt, ExternalValidationReceipt, build_snapshot, digest,
    file_identity, seal, verify_seal, verify_snapshot,
)


def bind_role_evaluation(data_path, output_dir, prefix, catalog):
    def read(name):
        return json.loads((output_dir / (prefix + name)).read_text(encoding='utf-8'))
    def write(name, obj):
        (output_dir / (prefix + name)).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    cases = read('_case_evaluation_receipt.json')
    report = read('_external_evaluation_report.json')
    predictions = read('_frozen_predictions.json')
    verify_seal(cases)
    verify_seal(report, 'report_sha256')
    verify_seal(predictions)
    if digest(predictions['predictions']) != report['predictions_sha256']:
        raise ValueError('PREDICTION_REPORT_BINDING_DRIFT')
    if digest(cases['cases']) != report['case_receipt_sha256']:
        raise ValueError('CASE_REPORT_BINDING_DRIFT')
    semantics = {
        'truth_source': 'explicit workbook component-mode sets; N/A is explicit nominal',
        'multi_fault': 'full set; no top-1 collapse',
        'time': 'workbook seconds; SCN ticks milliseconds independently verified',
        'primary_truth_policy': 'workbook primary; cross-source disagreements retained',
        'sample_policy': 'DEVELOPMENT_ONLY; aggregate performance withheld',
    }
    universe = sorted(catalog['target_universe'])
    task = {'evaluation_task_id': report['evaluation_task_id'], 'label_semantics_sha256': digest(semantics), 'target_universe_sha256': digest(universe)}
    scored = [c for c in cases['cases'] if c['status'] == 'SCORED']
    outside = sorted({f for c in scored for f in c['truth']['true_fault_ids'] if f not in universe})
    report.update(task, truth_fault_ids_outside_xml_universe=outside, classification_domain_status='XML_UNIVERSE_COVERED' if not outside else 'UNIVERSE_MISMATCH_RETAINED')
    if outside:
        report['status'] = 'EXECUTED_WITH_UNADMITTED_TARGETS'
    with zipfile.ZipFile(data_path) as z:
        manifest = [{'relative_path': i.filename, 'size': i.file_size, 'sha256': hashlib.sha256(z.read(i)).hexdigest()} for i in z.infolist() if not i.is_dir()]
    identity = file_identity(data_path)
    receipt = seal({
        'schema_version': 'astrograph-external-evaluation-dataset-receipt.v1', 'dataset_id': prefix.upper(), 'dataset_family': 'NASA_ADAPT_EPS',
        'source_artifact_sha256': identity['sha256'], 'source_artifact_size_bytes': identity['size'], **task, 'task_kind': 'MULTI_FAULT_SET_DIAGNOSIS',
        'label_semantics': semantics, 'target_universe': universe,
        'group_policy': {'role': report['role'], 'identity': 'SCN content SHA256', 'role_content_overlap': 0},
        'truth_provenance': {'workbook_member': cases['workbook_member'], 'workbook_sha256': cases['workbook_sha256'], 'source_catalog_xml_sha256': catalog['xml_sha256'],
            'explicit_annotations': True, 'cross_source_disagreements_retained': True, 'missing_truth_case_count': sum(c['status'] == 'SCORING_BLOCKED_TRUTH_UNAVAILABLE' for c in cases['cases'])},
        'payload_manifest_sha256': digest(manifest), 'allowed_claims': ['parser/unit/set comparison', 'frozen statistical baseline'] + ([] if report['role'] == 'DEVELOPMENT_ONLY' else ['external performance within admitted cases']),
        'forbidden_claims': ['sample publication performance', 'calibrated probabilities', 'formal fault evidence', 'flight qualification', 'validated complete physical model'],
    })
    ExternalDatasetReceipt.model_validate(receipt)
    groups = [c['group_id'] for c in cases['cases']]
    partitions = {g: report['role'] for g in groups}
    snapshot = build_snapshot(data_path.parent, [data_path], dataset_receipt=receipt, truth_paths=[data_path], partitions=partitions, groups=groups)
    verify_snapshot(data_path.parent, snapshot, dataset_receipt=receipt, partitions=partitions, groups=groups)
    report = seal(report, 'report_sha256')
    external = seal({'schema_version': 'astrograph-external-validation-receipt.v1', **task, 'dataset_receipt_sha256': receipt['receipt_sha256'],
        'snapshot_receipt_sha256': snapshot['receipt_sha256'], 'predictions_sha256': report['predictions_sha256'], 'evaluation_report_sha256': report['report_sha256'],
        'track_id': 'EPS_STATISTICAL_FULL_FAULT_SET', 'score_semantics': 'UNCALIBRATED_ANOMALY_SCORE',
        'status': report['status'] if report['role'] != 'DEVELOPMENT_ONLY' else 'DEVELOPMENT_ONLY', 'qualification': 'NOT_FLIGHT_QUALIFIED'})
    ExternalValidationReceipt.model_validate(external)
    for suffix, obj in [('dataset_receipt', receipt), ('snapshot_receipt', snapshot), ('external_validation_receipt', external), ('payload_manifest', manifest), ('partition_manifest', partitions), ('truth_receipt', cases)]:
        write('_' + suffix + '.json', obj)
    write('_external_evaluation_report.json', report)
    return report
