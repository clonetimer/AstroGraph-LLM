"""Frozen OPS-SAT experiment: prediction first, truth joined only for evaluation."""
from __future__ import annotations
import json, statistics
from pathlib import Path
from .external_dataset_contract import *
from .ops_sat_adapter import *
from .binary_anomaly_metrics import binary_metrics

def write(path, payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def feature_fit(online, nominal_training_ids):
    from astrograph.evaluation.data_exposure import guard_development
    guard_development('FIT', group_ids=nominal_training_ids)
    by={}
    for sid in nominal_training_ids:
        p=online[sid];validate_online_fields(p,track='feature');by.setdefault(p['channel'],[]).append(p)
    fitted={}
    for channel,rows in by.items():
        fitted[channel]={}
        for f in FEATURE_FIELDS:
            m=statistics.median(r[f] for r in rows);mad=statistics.median(abs(r[f]-m) for r in rows)
            fitted[channel][f]=(m,max(1.4826*mad,abs(m)*1e-9,1e-9))
    return fitted

def feature_score(payload, fitted):
    validate_online_fields(payload,track='feature')
    if payload['channel'] not in fitted:raise ValueError('UNSEEN_CHANNEL_BLOCKED')
    return max(abs(payload[f]-m)/scale for f,(m,scale) in fitted[payload['channel']].items())

def run(data_root: Path, output: Path, config_path: Path, provenance: dict):
    raise ValueError('R21_SEEN_EVALUATION_PREDICTION_REGENERATION_BLOCKED: use frozen R20 results')
    output.mkdir(parents=True,exist_ok=True);config=json.loads(config_path.read_text(encoding='utf-8'))
    identities={name:file_identity(data_root/name) for name in ('dataset.csv','segments.csv')}
    feature,truth=load_feature_track(data_root/'dataset.csv');raw,raw_truth=load_raw_track(data_root/'segments.csv')
    if raw_truth!=truth:raise ValueError('RAW_FEATURE_TRUTH_OR_ID_MISMATCH')
    partitions=build_partitions(truth,config);partition_by={r['group_id']:r['partition'] for r in partitions['groups']}
    semantics={'primary_field':'anomaly','0':'NOMINAL','1':'ANOMALY','train':'official holdout indicator; evaluator only',
      'auxiliary_label':'non-authoritative; constant anomaly in supplied raw CSV','official_code_provenance':provenance}
    universe=['NOMINAL','ANOMALY'];manifest=[{'relative_path':n,**v} for n,v in sorted(identities.items())]
    receipt=seal({'schema_version':'astrograph-external-evaluation-dataset-receipt.v1','dataset_id':'OPS_SAT_USER_SUPPLIED_CSV_V1','dataset_family':'OPS-SAT',
      'source_artifact_sha256':digest(manifest),'source_artifact_size_bytes':sum(v['size'] for v in identities.values()),
      'evaluation_task_id':config['evaluation_task_id'],'task_kind':'BINARY_ANOMALY_DETECTION','label_semantics':semantics,
      'label_semantics_sha256':digest(semantics),'target_universe':universe,'target_universe_sha256':digest(universe),
      'group_policy':{'group_id':'segment','config_sha256':digest(config),'source_identity_kind':'canonical multi-file manifest, not ZIP byte hash'},
      'truth_provenance':{'files':manifest,'authority':'numeric anomaly per official modeling code; local bytes user supplied, authenticity not independently certified'},
      'payload_manifest_sha256':digest(manifest),'allowed_claims':['segment anomaly detection on frozen supplied CSV'],
      'forbidden_claims':['root-cause fault isolation','calibrated probability','flight qualification']})
    ExternalDatasetReceipt.model_validate(receipt)
    snapshot=build_snapshot(data_root,[data_root/n for n in identities],dataset_receipt=receipt,truth_paths=[data_root/n for n in identities],partitions=partitions,groups=list(truth))
    write(output/'ops_sat_dataset_receipt.json',receipt);write(output/'ops_sat_partition_manifest.json',partitions)
    write(output/'ops_sat_external_snapshot_receipt.json',snapshot)
    nominal_training=[sid for sid in truth if partition_by[sid]=='MODEL_TRAIN' and truth[sid]['anomaly']==0]
    fitted=feature_fit(feature,nominal_training);write(output/'ops_sat_feature_fit.json',{'training_ids':sorted(nominal_training),'parameters':fitted,'config_sha256':digest(config)})
    tracks={};all_predictions={}
    for track,online,score in [('raw',raw,raw_segment_score),('feature',feature,lambda p:feature_score(p,fitted))]:
        predictions={}
        for sid,payload in sorted(online.items()):
            try:predictions[sid]={'status':'SCORED','score':score(payload)}
            except ValueError as e:predictions[sid]={'status':'BLOCKED','reason':str(e)}
        # Persist predictions before any holdout label join.
        write(output/f'ops_sat_{track}_frozen_predictions.json',predictions);all_predictions[track]=digest(predictions)
        dev=[sid for sid in predictions if partition_by[sid]=='DEVELOPMENT_SELECTION' and predictions[sid]['status']=='SCORED']
        if not dev:raise ValueError('DEVELOPMENT_PREDICTIONS_UNAVAILABLE')
        candidates=[]
        for t in config[track+'_threshold_grid']:
            m=binary_metrics([truth[s]['anomaly'] for s in dev],[predictions[s]['score'] for s in dev],t)
            candidates.append((m['F0.5'] or 0,-m['FP'],t))
        threshold=max(candidates)[2]
        summaries={}
        for partition in sorted(PARTITIONS):
            all_ids=[s for s in truth if partition_by[s]==partition];ids=[s for s in all_ids if predictions[s]['status']=='SCORED']
            metrics=binary_metrics([truth[s]['anomaly'] for s in ids],[predictions[s]['score'] for s in ids],threshold) if ids else {}
            strata={}
            for dimension in ('channel','sampling'):
                strata[dimension]={}
                for value in sorted({feature[s][dimension] for s in all_ids}):
                    subset=[s for s in ids if feature[s][dimension]==value]
                    strata[dimension][value]={'eligible_count':sum(feature[s][dimension]==value for s in all_ids),
                      'metrics':binary_metrics([truth[s]['anomaly'] for s in subset],[predictions[s]['score'] for s in subset],threshold) if subset else {}}
            summaries[partition]={'eligible_count':len(all_ids),'scored_count':len(ids),'blocked_count':len(all_ids)-len(ids),
              'nominal_count':sum(truth[s]['anomaly']==0 for s in all_ids),'anomaly_count':sum(truth[s]['anomaly']==1 for s in all_ids),
              'metrics_on_scored_segments':metrics,'blocked_segments':{s:predictions[s] for s in all_ids if s not in ids},'strata':strata}
        tracks[track]={'threshold':threshold,'selection_partition':'DEVELOPMENT_SELECTION','score_semantics':'UNCALIBRATED_ANOMALY_SCORE',
          'status':'EXECUTED_WITH_BLOCKED_SEGMENTS' if any(p['status']=='BLOCKED' for p in predictions.values()) else 'EXECUTED',
          'predictions_sha256':digest(predictions),'partitions':summaries}
    verify_snapshot(data_root,snapshot,dataset_receipt=receipt,partitions=partitions,groups=list(truth))
    report=seal({'schema_version':'astrograph-ops-sat-external-evaluation-report.v1',**task_identity(receipt),
      'dataset_receipt_sha256':receipt['receipt_sha256'],'snapshot_receipt_sha256':snapshot['receipt_sha256'],
      'eligible_segments':len(truth),'tracks':tracks,'train_test_group_overlap':0,'qualification':'NOT_FLIGHT_QUALIFIED',
      'warnings':['raw auxiliary label constant; numeric anomaly is authoritative','segment hash does not assert channel or temporal independence',
       'raw leading baseline may contain anomalies; segment score is retrospective, not online detection delay',
       'feature track is a separate simple statistical benchmark, not the raw AstroGraph detector','no probability metrics or qualification supplied']},'report_sha256')
    write(output/'ops_sat_external_evaluation_report.json',report)
    for track in tracks:
        validation=seal({'schema_version':'astrograph-external-validation-receipt.v1',**task_identity(receipt),
          'dataset_receipt_sha256':receipt['receipt_sha256'],'snapshot_receipt_sha256':snapshot['receipt_sha256'],
          'predictions_sha256':all_predictions[track],'evaluation_report_sha256':report['report_sha256'],'track_id':track,
          'score_semantics':'UNCALIBRATED_ANOMALY_SCORE','status':tracks[track]['status'],'qualification':'NOT_FLIGHT_QUALIFIED'})
        ExternalValidationReceipt.model_validate(validation);write(output/f'ops_sat_{track}_external_validation_receipt.json',validation)
    return report
