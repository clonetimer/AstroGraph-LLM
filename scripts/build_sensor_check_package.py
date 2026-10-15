"""Generate a RESEARCH knowledge candidate; never replace an Active KG release."""
from pathlib import Path
import json
from astrograph.oag.sensor_checks import PACKAGE_ID,VERSION,METRICS,feature_id,statistic_fields
from astrograph.oag.window_features import statistic_fields as rw_fields
from astrograph.evidence.configured_runtime import column_schema_sha256,ConfiguredRuntimeEvidenceBuilder
ROOT=Path(__file__).resolve().parents[1]

def build(output:Path):
    source=ROOT/'configs/datasets/adcs_v5_2_2/runtime_window_integrated_evidence.json'
    d=json.loads(source.read_text());d['package_id']=PACKAGE_ID;d['version']='1.1.0'
    canonical=json.loads((source.parent/'telemetry_projection.json').read_text())['canonical_columns']
    d['source_schema_sha256']=column_schema_sha256(canonical+rw_fields()+statistic_fields())
    refs=['src/astrograph/oag/physical_diagnosis.py','src/astrograph/oag/sensor_checks.py']
    claim='Research paired-window sensor effects, not fine root causes, arbitrary-device calibration, independent ground truth or compound-fault exclusion.'
    d['claim_boundary']=claim
    for key,unit in METRICS.items():
        field='asset.adcs.sensor_check.'+key
        d['channels'].append(dict(source_field='derived.adcs.sensor_check.'+key,semantic_channel_id=field,
            component_instance='ADCS',source_unit=unit,canonical_unit=unit,required=False,max_missing_fraction=1.0))
        d['feature_definitions'].append(dict(feature_definition_id=feature_id(key),name=key,source_field_template=field,
            aggregation='MAX_VALUE',unit=unit,window_scope='episode',algorithm_version=VERSION,source_refs=refs,claim_boundary=claim))
    def condition(key,op,threshold):return dict(feature_definition_id=feature_id(key),comparison=op,threshold=threshold,threshold_unit=METRICS[key])
    def rule(name,fault,key,op,threshold,other=()):
        d['evidence_definitions'].append(dict(evidence_definition_id='ED-SENSOR-'+name+'-V1',target_fault_id=fault,
            name=name,role='REQUIRED',critical=True,**condition(key,op,threshold),and_feature_conditions=list(other),source_refs=refs,claim_boundary=claim))
    # Replace the broad instantaneous gyro envelope as a knowledge candidate;
    # keep old package intact for explicit ablation. Thresholds below are copied
    # from the existing post47 time-aligned signature, NOT fitted to new labels.
    d['evidence_definitions']=[e for e in d['evidence_definitions'] if e['target_fault_id']!='ADCS_GYRO_BIAS_DRIFT']
    rule('ALIGNED-GYRO-BIAS','ADCS_GYRO_BIAS_DRIFT','gyro_aligned_bias_norm','GREATER_OR_EQUAL',4e-5,
         [condition('gyro_aligned_spread_ratio','LESS_OR_EQUAL',.25),condition('gyro_zero_axes','EQUAL',0),
          condition('star_output_frozen','EQUAL',0),condition('star_quality_valid','EQUAL',1),
          condition('actuator_response_consistent','EQUAL',1)])
    rule('SINGLE-GYRO-AXIS-ZERO','ADCS_IMU_AXIS_DROPOUT','gyro_zero_axes','EQUAL',1,
         [condition('gyro_reference_excited_axes','EQUAL',3)])
    rule('ALL-IMU-OUTPUT-ZERO','ADCS_IMU_OUTPUT_ZERO','imu_all_output_zero','EQUAL',1,
         [condition('gyro_reference_excited_axes','EQUAL',3)])
    rule('STAR-FROZEN-OUTPUT','ADCS_STAR_TRACKER_STUCK_AT_LAST','star_output_frozen','EQUAL',1)
    output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    ConfiguredRuntimeEvidenceBuilder(output)
    return output

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'configs/datasets/adcs_v5_2_2/runtime_sensor_checks.research.json');a=p.parse_args();print(build(a.output))
