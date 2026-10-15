from __future__ import annotations
import hashlib, json
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astrograph.data_driven import DataDrivenPrediction, DataDrivenModelManifest, ModelCapability, ProbabilityCalibrationReceipt, ProbabilityCalibrationQualificationReceipt
from astrograph.fusion import FaultRanking, BiRRFResult, ScoreSemantics

class ScoreSemanticsAudit(BaseModel):
    model_config=ConfigDict(extra='forbid', frozen=True)
    schema_version:str='astrograph-score-semantics-audit.v1'
    model_id:str
    model_version:str
    output_semantics:str
    class_value_semantics:str
    calibrated_probabilities:bool
    calibration_id:str|None=None
    calibration_receipt_sha256:str|None=Field(default=None, pattern=r'^[0-9a-f]{64}$')
    calibration_evidence_status:str
    probability_threshold_eligible:bool
    external_validation_qualified:bool=False
    ranking_score_semantics:str
    fusion_score_semantics:str|None=None
    warnings:list[str]=[]
    claim_boundary:str=(
        'Score-semantics consistency audit only. It does not establish calibration transfer, '
        'OOD reliability, HIL/flight qualification, or diagnostic correctness.'
    )
    audit_sha256:str=''


def _sha(payload:dict[str,Any])->str:
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def audit_score_semantics(*, prediction:DataDrivenPrediction, manifest:DataDrivenModelManifest, ranking:FaultRanking, fusion:BiRRFResult|None=None, calibration_receipt:ProbabilityCalibrationReceipt|ProbabilityCalibrationQualificationReceipt|None=None, external_task_identity:dict[str,str]|None=None)->ScoreSemanticsAudit:
    if external_task_identity is not None or (isinstance(calibration_receipt,ProbabilityCalibrationQualificationReceipt) and calibration_receipt.evaluation_task_id is not None):
        if not isinstance(calibration_receipt,ProbabilityCalibrationQualificationReceipt):raise ValueError('CROSS_TASK_CALIBRATION_REUSE_BLOCKED')
        calibration_receipt.require_external_task(external_task_identity or {})
    if prediction.model_id!=manifest.model_id or prediction.model_version!=manifest.model_version:
        raise ValueError('prediction/manifest model identity mismatch')
    if ranking.producer_id!=prediction.model_id:
        raise ValueError('ranking/prediction producer mismatch')
    expected=ScoreSemantics(prediction.class_value_semantics)
    if ranking.score_semantics!=expected:
        raise ValueError('ranking score semantics do not match prediction class-value semantics')
    calibrated=(ModelCapability.CALIBRATED_PROBABILITY in manifest.capabilities and prediction.calibrated_probabilities)
    external_qualified=False
    if calibrated and calibration_receipt is not None and manifest.calibration_receipt_sha256:
        if calibration_receipt.receipt_sha256 != manifest.calibration_receipt_sha256:
            raise ValueError('calibration receipt hash does not match model manifest')
        if (calibration_receipt.calibration_id, calibration_receipt.model_id, calibration_receipt.model_version) != (manifest.calibration_id, manifest.model_id, manifest.model_version):
            raise ValueError('calibration receipt identity does not match model manifest')
        if isinstance(calibration_receipt, ProbabilityCalibrationQualificationReceipt):
            external_qualified = calibration_receipt.qualification_status == 'PASS_FOR_ENGINEERING_CALIBRATION_REVIEW'
            evidence='EXTERNAL_VALIDATION_RECEIPT_VERIFIED' if external_qualified else 'EXTERNAL_VALIDATION_RECEIPT_BLOCKED'
        else:
            evidence='CONTENT_BOUND_RECEIPT_VERIFIED_LEGACY_NO_EXTERNAL_VALIDATION'
    elif calibrated and manifest.calibration_receipt_sha256:
        evidence='CONTENT_BOUND_RECEIPT_DECLARED_NOT_VERIFIED'
    elif calibrated:
        evidence='LEGACY_ID_ONLY_NOT_CONTENT_BOUND'
    else:
        evidence='NOT_CALIBRATED'
    warnings=[]
    if prediction.class_value_semantics=='NORMALIZED_RANKING_MASS': warnings.append('NORMALIZED_RANKING_MASS_NOT_PROBABILITY')
    if prediction.class_value_semantics=='UNCALIBRATED_CLASS_PROBABILITY': warnings.append('UNCALIBRATED_PROBABILITY_NOT_THRESHOLD_ELIGIBLE')
    if calibrated and evidence not in {'CONTENT_BOUND_RECEIPT_VERIFIED_LEGACY_NO_EXTERNAL_VALIDATION','EXTERNAL_VALIDATION_RECEIPT_VERIFIED'}: warnings.append('CALIBRATION_ID_NOT_CONTENT_BOUND')
    if calibrated and not external_qualified: warnings.append('EXTERNAL_CALIBRATION_VALIDATION_NOT_QUALIFIED')
    if fusion is not None and fusion.fused_score_semantics!='RECIPROCAL_RANK_SCORE_NOT_PROBABILITY':
        raise ValueError('unexpected BiRRF fused-score semantics')
    if fusion is not None: warnings.append('BIRRF_SCORE_NOT_PROBABILITY')
    payload=dict(schema_version='astrograph-score-semantics-audit.v1',model_id=prediction.model_id,model_version=prediction.model_version,
        output_semantics=prediction.output_semantics,class_value_semantics=prediction.class_value_semantics,
        calibrated_probabilities=prediction.calibrated_probabilities,calibration_id=prediction.calibration_id,
        calibration_receipt_sha256=manifest.calibration_receipt_sha256,calibration_evidence_status=evidence,
        probability_threshold_eligible=bool(calibrated and evidence in {'CONTENT_BOUND_RECEIPT_VERIFIED_LEGACY_NO_EXTERNAL_VALIDATION','EXTERNAL_VALIDATION_RECEIPT_VERIFIED'}),external_validation_qualified=external_qualified,ranking_score_semantics=ranking.score_semantics.value,
        fusion_score_semantics=(fusion.fused_score_semantics if fusion else None),warnings=warnings,
        claim_boundary='Score-semantics consistency audit only. It does not establish calibration transfer, OOD reliability, HIL/flight qualification, or diagnostic correctness.',audit_sha256='')
    payload['audit_sha256']=_sha({k:v for k,v in payload.items() if k!='audit_sha256'})
    return ScoreSemanticsAudit.model_validate(payload)

__all__=['ScoreSemanticsAudit','audit_score_semantics']
