"""Content-addressed evaluator-run receipts bound to an exact challenger seal."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.agents.mission.storage import sha
from astrograph.data_driven.feedback import FeedbackCandidate
from .layer_evaluation import LayerEvaluationArtifact

Layer=Literal["planning","memory","tool","knowledge","model","execution"]
ResultKind=Literal["MODEL_TASK_METRICS","LAYER_CHECKS"]


class EvaluationRunReceipt(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    schema_version: Literal["astrograph-evaluation-run-receipt.v1"]="astrograph-evaluation-run-receipt.v1"
    run_id: str=Field(min_length=1)
    improvement_candidate_id: str=Field(min_length=1)
    feedback_id: str=Field(min_length=1)
    target_layer: Layer
    work_order_sha256: str=Field(pattern=r"^[0-9a-f]{64}$")
    challenger_seal_sha256: str=Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_id: str=Field(min_length=1)
    evaluator_version: str=Field(min_length=1)
    result_kind: ResultKind
    result_payload: dict[str,Any]
    result_payload_sha256: str=Field(pattern=r"^[0-9a-f]{64}$")
    source_refs: list[str]=Field(min_length=1)
    truth_used_for_tuning: Literal[False]=False
    production_modified: Literal[False]=False
    automatic_promotion_allowed: Literal[False]=False

    @field_validator("source_refs")
    @classmethod
    def unique_sources(cls,value:list[str])->list[str]:
        if len(value)!=len(set(value)): raise ValueError("source_refs must be unique")
        return value

    @model_validator(mode="after")
    def payload_hash_matches(self):
        if sha(self.result_payload)!=self.result_payload_sha256:
            raise ValueError("evaluation result payload hash mismatch")
        return self


def _file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def write_evaluation_run_receipt(*, root:str|Path, improvement_candidate_id:str, feedback_id:str, target_layer:Layer,
                                 work_order_sha256:str, challenger_seal_sha256:str, evaluator_id:str,evaluator_version:str,
                                 result_kind:ResultKind,result_payload:dict[str,Any],source_refs:list[str])->tuple[EvaluationRunReceipt,Path,str]:
    identity={"candidate_id":improvement_candidate_id,"feedback_id":feedback_id,"target_layer":target_layer,
              "work_order_sha256":work_order_sha256,"challenger_seal_sha256":challenger_seal_sha256,
              "evaluator_id":evaluator_id,"evaluator_version":evaluator_version,"result_kind":result_kind,
              "result_payload_sha256":sha(result_payload),"source_refs":source_refs}
    receipt=EvaluationRunReceipt(run_id="ERUN-"+sha(identity)[:20].upper(),improvement_candidate_id=improvement_candidate_id,
        feedback_id=feedback_id,target_layer=target_layer,work_order_sha256=work_order_sha256,
        challenger_seal_sha256=challenger_seal_sha256,evaluator_id=evaluator_id,evaluator_version=evaluator_version,
        result_kind=result_kind,result_payload=result_payload,result_payload_sha256=sha(result_payload),source_refs=source_refs)
    rootp=Path(root).expanduser().resolve();rootp.mkdir(parents=True,exist_ok=True)
    path=rootp/f"{receipt.run_id}.json";payload=receipt.model_dump_json(indent=2)
    if path.exists() and path.read_text(encoding="utf-8")!=payload:
        raise ValueError("existing evaluation run receipt differs for deterministic run identity")
    path.write_text(payload,encoding="utf-8")
    return receipt,path,_file_sha(path)


def load_and_validate_evaluation_run_receipt(*, path:str|Path,expected_sha256:str,improvement_candidate_id:str,
        feedback_id:str,target_layer:Layer,work_order_sha256:str,challenger_seal_sha256:str,
        expected_result_kind:ResultKind,expected_result_payload:dict[str,Any],required_source_ref:str|None=None,
        expected_evaluator_id:str|None=None,expected_evaluator_version:str|None=None)->EvaluationRunReceipt:
    raw=Path(path).expanduser()
    if raw.is_symlink(): raise ValueError("evaluation run receipt must not be a symlink")
    p=raw.resolve(strict=True)
    if not p.is_file(): raise ValueError("evaluation run receipt must be a regular file")
    if _file_sha(p)!=expected_sha256: raise ValueError("evaluation run receipt SHA-256 mismatch")
    receipt=EvaluationRunReceipt.model_validate_json(p.read_text(encoding="utf-8"))
    exact={
        "improvement_candidate_id":improvement_candidate_id,"feedback_id":feedback_id,"target_layer":target_layer,
        "work_order_sha256":work_order_sha256,"challenger_seal_sha256":challenger_seal_sha256,
        "result_kind":expected_result_kind,
    }
    for field,value in exact.items():
        if getattr(receipt,field)!=value: raise ValueError(f"evaluation run receipt {field} binding mismatch")
    if sha(receipt.result_payload)!=sha(expected_result_payload):
        raise ValueError("evaluation run receipt result payload does not match evaluator input")
    if required_source_ref and required_source_ref not in receipt.source_refs:
        raise ValueError("evaluation run receipt is missing required source reference")
    if expected_evaluator_id and receipt.evaluator_id!=expected_evaluator_id:
        raise ValueError("evaluation run receipt evaluator_id mismatch")
    if expected_evaluator_version and receipt.evaluator_version!=expected_evaluator_version:
        raise ValueError("evaluation run receipt evaluator_version mismatch")
    return receipt


def model_metric_payload(challenger:FeedbackCandidate)->dict[str,Any]:
    return challenger.metrics.model_dump(mode="json")


def layer_check_payload(artifact:LayerEvaluationArtifact)->dict[str,Any]:
    return {"checks":artifact.checks,"metrics":artifact.metrics}


def evaluation_artifact_ref(digest:str)->str:
    return f"evaluation-artifact-sha256:{digest}"


__all__=["EvaluationRunReceipt","write_evaluation_run_receipt","load_and_validate_evaluation_run_receipt",
         "model_metric_payload","layer_check_payload","evaluation_artifact_ref"]
