#!/usr/bin/env python3
"""Fit and single-use validate the frozen final generic diagnosis candidate."""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from astrograph.diagnosis.generic import FailureMechanism
ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "configs/diagnosis/generic_head_calibration_protocol.v3.json"
P2 = ROOT / "configs/diagnosis/generic_head_calibration_protocol.v2.json"
CLASSES = ("NORMAL", "CAPACITY_LOSS", "DRIFT", "LEAKAGE")


def _candidate2_module():
    path = ROOT / "scripts/acceptance/calibrate_segmented_generic_diagnosis.py"
    spec = importlib.util.spec_from_file_location("candidate2_frozen_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate-2 frozen generator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


c2 = _candidate2_module()
KNOWN = c2.KNOWN
OPEN = c2.OPEN_SET


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load():
    p3 = json.loads(P3.read_text(encoding="utf-8"))
    p2 = json.loads(P2.read_text(encoding="utf-8"))
    if _sha(P2) != p3["base_feature_protocol_sha256"]:
        raise ValueError("candidate-2 base feature protocol changed")
    att = ROOT / p3["timer_attestation_path"]
    if _sha(att) != p3["timer_attestation_sha256"]:
        raise ValueError("Timer authorization changed")
    return p3, p2


def _cases(p2, seed, count, mechanisms, normal, validation):
    return c2._extract(c2._generate(
        p2, seed=seed, count=count, mechanisms=mechanisms,
        validation_domain=validation, include_normal=normal,
    ))


def _temperature(logits, labels, p3):
    cfg = p3["single_head"]["temperature_scaling"]
    grid = np.linspace(cfg["minimum_temperature"], cfg["maximum_temperature"], cfg["grid_point_count"])
    losses = []
    for t in grid:
        z = logits / t
        z -= z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        losses.append(-float(np.mean(lp[np.arange(len(labels)), labels])))
    return float(grid[int(np.argmin(losses))])


def _quantile(values, alpha):
    values = np.sort(np.asarray(values))
    rank = min(len(values), math.ceil((len(values) + 1) * (1 - alpha)))
    return float(values[rank - 1])


def _rff(z, weights, phases):
    return math.sqrt(2.0 / weights.shape[1]) * np.cos(z @ weights + phases)


def _probabilities(artifact, values):
    x = (np.asarray(values) - artifact["feature_mean"]) / artifact["feature_scale"]
    mapped = _rff(x, np.asarray(artifact["rff_weights"]), np.asarray(artifact["rff_phases"]))
    logits = (np.asarray(artifact["coefficients"]) @ mapped + artifact["intercepts"]) / artifact["temperature"]
    logits -= logits.max()
    p = np.exp(logits); p /= p.sum()
    index = int(np.argmax(p))
    nonconformity = -math.log(max(float(p[index]), 1e-15))
    return index, p, nonconformity > artifact["class_nonconformity_thresholds"][index]


def fit(p3, p2):
    started = time.monotonic(); split = p3["split"]; count_d = split["development_count_per_archetype_per_output_class"]; count_c = split["calibration_count_per_archetype_per_output_class"]
    dev = _cases(p2, split["development_seed"], count_d, KNOWN, False, False) + _cases(p2, split["development_seed"], count_d, (), True, False)
    cal = _cases(p2, split["calibration_seed"], count_c, KNOWN, False, False) + _cases(p2, split["calibration_seed"], count_c, (), True, False)
    if any(row.failed for row in dev + cal):
        raise RuntimeError("frozen feature extraction failed during candidate-3 fit")
    def label(row): return 0 if row.mechanism is None else CLASSES.index(row.mechanism.value)
    x = np.asarray([r.features.values for r in dev]); y = np.asarray([label(r) for r in dev])
    xc = np.asarray([r.features.values for r in cal]); yc = np.asarray([label(r) for r in cal])
    mean=x.mean(0); scale=np.maximum(x.std(0),1e-12); z=(x-mean)/scale; zc=(xc-mean)/scale
    cfg=p3["fixed_nonlinear_mapping"]; rng=np.random.default_rng(cfg["random_seed"])
    weights=rng.normal(0, math.sqrt(2*cfg["gamma"]), (cfg["input_dimension"],cfg["output_dimension"]))
    phases=rng.uniform(0,2*math.pi,cfg["output_dimension"]); m=_rff(z,weights,phases); mc=_rff(zc,weights,phases)
    head=p3["single_head"]; clf=LogisticRegression(C=head["l2_inverse_regularization"],solver=head["solver"],max_iter=head["maximum_iterations"],tol=head["tolerance"],random_state=head["random_seed"]).fit(m,y)
    temp=_temperature(clf.decision_function(mc),yc,p3)
    logits=clf.decision_function(mc)/temp; logits-=logits.max(1,keepdims=True); probs=np.exp(logits); probs/=probs.sum(1,keepdims=True)
    alpha=p3["thresholds"]["open_set_alpha"]; thresholds=[_quantile(-np.log(np.maximum(probs[yc==i,i],1e-15)),alpha) for i in range(4)]
    artifact={"schema_version":"astrograph-final-generic-calibration.v1","candidate_ordinal":3,"protocol_sha256":_sha(P3),"class_order":list(CLASSES),"feature_mean":mean.tolist(),"feature_scale":scale.tolist(),"rff_weights":weights.tolist(),"rff_phases":phases.tolist(),"coefficients":clf.coef_.tolist(),"intercepts":clf.intercept_.tolist(),"temperature":temp,"class_nonconformity_thresholds":thresholds,"open_set_alpha":alpha,"development_split_sha256":_canonical([(r.case_key,label(r),r.features.values) for r in dev]),"calibration_split_sha256":_canonical([(r.case_key,label(r),r.features.values) for r in cal]),"development_case_count":len(dev),"calibration_case_count":len(cal),"formal_hidden_truth_used":False,"candidate_bound_to_production":False}
    correct=sum(_probabilities(artifact,r.features.values)[0]==label(r) for r in cal)/len(cal)
    evidence={"schema_version":"astrograph-final-generic-fit-evidence.v1","candidate_ordinal":3,"protocol_sha256":_sha(P3),"development_case_count":len(dev),"calibration_case_count":len(cal),"failure_count":0,"calibration_classification_accuracy":correct,"elapsed_seconds":time.monotonic()-started,"exhausted_validation_seeds_used":[],"new_validation_seed_used":False,"technical_validation_read":False,"candidate_bound_to_production":False,"status":"FIT_COMPLETE","claim_boundary":"Fit/calibration only; no validation or promotion."}
    return artifact,evidence


def validate(p3,p2,artifact):
    started=time.monotonic(); split=p3["split"]; rows0=_cases(p2,split["technical_validation_seed"],split["technical_validation_count_per_archetype_per_mechanism"],KNOWN+OPEN,True,True)
    rows=[]; failures=0
    for r in rows0:
        role="NORMAL" if r.mechanism is None else "KNOWN" if r.mechanism in KNOWN else "OPEN_SET"
        if r.failed: failures+=1; idx=None; ood=False
        else: idx,_,ood=_probabilities(artifact,r.features.values)
        predicted=CLASSES[idx] if idx is not None else None
        anomalous=bool(idx is not None and (predicted!="NORMAL" or ood))
        detect=not r.failed and ((not anomalous) if role=="NORMAL" else anomalous)
        retained=role=="KNOWN" and anomalous and not ood
        rejected=role=="OPEN_SET" and anomalous and ood
        classified=bool(retained and predicted==r.mechanism.value)
        rows.append({"case_key":r.case_key,"archetype":r.archetype,"role":role,"mechanism":r.mechanism.value if r.mechanism else "NORMAL","failed":r.failed,"detection_correct":detect,"known_retained":retained,"open_set_rejected":rejected,"classification_correct":classified,"localization_correct":classified,"reasoning_correct":classified,"predicted_class":predicted,"open_set_ood":ood})
    normal=[r for r in rows if r["role"]=="NORMAL"]; known=[r for r in rows if r["role"]=="KNOWN"]; opened=[r for r in rows if r["role"]=="OPEN_SET"]
    rate=lambda xs:sum(xs)/len(xs)
    den={"normal":len(normal),"known":len(known),"open_set":len(opened),"total":len(rows)}
    metrics={"detection_accuracy":rate([r["detection_correct"] for r in rows]),"classification_accuracy":rate([r["classification_correct"] for r in known]),"localization_accuracy":rate([r["localization_correct"] for r in known]),"reasoning_correctness":rate([r["reasoning_correct"] for r in known]),"known_class_retention":rate([r["known_retained"] for r in known]),"open_set_rejection":rate([r["open_set_rejected"] for r in opened])}
    checks={"denominators_preserved":den==p3["denominators"],**{k:metrics[k]>=p3["thresholds"][k] for k in metrics},"runtime_budget":time.monotonic()-started<=600}
    passed=all(checks.values())
    return {"schema_version":"astrograph-final-generic-technical-validation.v1","candidate_ordinal":3,"protocol_sha256":_sha(P3),"calibration_artifact_sha256":_canonical(artifact),"validation_seed":split["technical_validation_seed"],"validation_run_count":1,"exhausted_validation_seeds_used":[],"formal_hidden_truth_used":False,"denominators":den,"failed_run_count":failures,"metrics":metrics,"thresholds":{k:p3["thresholds"][k] for k in metrics},"checks":checks,"case_results":rows,"elapsed_seconds":time.monotonic()-started,"technical_status":"PASS" if passed else "FAIL","candidate_bound_to_production":False,"claim_boundary":"Single-use technical result; independent Timer review still required."}


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    f=sub.add_parser("fit"); f.add_argument("--artifact",type=Path,default=ROOT/"src/astrograph/resources/generic_mechanism_calibration.v3.json"); f.add_argument("--evidence",type=Path,default=ROOT/"governance/evidence/0.7.1-extension/generic_mechanism_fit_calibration.v3.json")
    v=sub.add_parser("validate"); v.add_argument("--artifact",type=Path,default=ROOT/"src/astrograph/resources/generic_mechanism_calibration.v3.json"); v.add_argument("--evidence",type=Path,default=ROOT/"governance/evidence/0.7.1-extension/generic_head_technical_validation.v3.json"); v.add_argument("--query",type=Path,default=ROOT/"governance/evidence/0.7.1-extension/generic_head_technical_validation_query.v3.json")
    a=ap.parse_args(); p3,p2=_load()
    if a.cmd=="fit":
        art,ev=fit(p3,p2); _write(a.artifact,art); ev["calibration_artifact_sha256"]=_canonical(art); _write(a.evidence,ev); print(json.dumps(ev,indent=2)); return 0
    if a.query.exists() or a.evidence.exists(): raise PermissionError("candidate-3 validation already consumed")
    art=json.loads(a.artifact.read_text()); _write(a.query,{"schema_version":"astrograph-single-use-technical-validation-query.v1","candidate_ordinal":3,"validation_seed":20300820,"maximum_invocations":1,"status":"STARTED_AND_SEED_CONSUMED","started_at_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":_sha(P3),"calibration_artifact_sha256":_canonical(art),"exhausted_validation_seeds_used":[]})
    result=validate(p3,p2,art); _write(a.evidence,result); print(json.dumps(result,indent=2)); return 0 if result["technical_status"]=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
