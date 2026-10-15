"""Non-inferiority-first selection of fusion policies on development groups only."""
from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass, asdict
from typing import Sequence

@dataclass(frozen=True)
class GroupPolicyMetrics:
    policy_id:str; group_id:str; identification_accuracy:float; selective_risk:float; nll:float; brier:float; ece:float
    def __post_init__(self):
        for name in ("identification_accuracy","selective_risk","nll","brier","ece"):
            v=getattr(self,name)
            if not math.isfinite(v) or v<0: raise ValueError(f"{name} must be finite non-negative")
        if self.identification_accuracy>1 or self.selective_risk>1 or self.ece>1: raise ValueError("bounded metrics must be <=1")


def select_policy_noninferiority_first(
    rows:Sequence[GroupPolicyMetrics], *, baseline_policy_id:str, development_groups:set[str], forbidden_groups:set[str],
    max_risk_increase:float, max_nll_increase:float, max_brier_increase:float, max_ece_increase:float,
    max_accuracy_drop:float,
)->dict:
    if development_groups & forbidden_groups: raise ValueError("development groups overlap forbidden groups")
    by_policy={}
    for r in rows: by_policy.setdefault(r.policy_id,{})[r.group_id]=r
    if baseline_policy_id not in by_policy: raise ValueError("baseline policy missing")
    expected=set(development_groups)
    for pid,groups in by_policy.items():
        if set(groups)!=expected: raise ValueError(f"policy {pid} does not contain exactly development groups")
        if set(groups)&forbidden_groups: raise ValueError("post-hoc group present in selection rows")
    def avg(pid,name): return sum(getattr(r,name) for r in by_policy[pid].values())/len(expected)
    b={m:avg(baseline_policy_id,m) for m in ("identification_accuracy","selective_risk","nll","brier","ece")}
    feasible=[]; audit={}
    for pid in sorted(by_policy):
        m={k:avg(pid,k) for k in b}
        checks={
            "accuracy": m["identification_accuracy"] >= b["identification_accuracy"]-max_accuracy_drop,
            "risk": m["selective_risk"] <= b["selective_risk"]+max_risk_increase,
            "nll": m["nll"] <= b["nll"]+max_nll_increase,
            "brier": m["brier"] <= b["brier"]+max_brier_increase,
            "ece": m["ece"] <= b["ece"]+max_ece_increase,
        }
        ok=all(checks.values()); audit[pid]={"group_macro":m,"noninferiority":checks,"feasible":ok}
        if ok: feasible.append(pid)
    selected=None
    if feasible:
        # Benefit second: accuracy first, then probabilistic quality. Deterministic tie break by policy_id.
        selected=sorted(feasible,key=lambda pid:(-audit[pid]["group_macro"]["identification_accuracy"],audit[pid]["group_macro"]["nll"],audit[pid]["group_macro"]["brier"],audit[pid]["group_macro"]["ece"],pid))[0]
    payload={
        "schema_version":"astrograph-development-policy-selection-receipt.v1","baseline_policy_id":baseline_policy_id,
        "development_groups":sorted(development_groups),"forbidden_groups":sorted(forbidden_groups),
        "margins":{"max_risk_increase":max_risk_increase,"max_nll_increase":max_nll_increase,"max_brier_increase":max_brier_increase,"max_ece_increase":max_ece_increase,"max_accuracy_drop":max_accuracy_drop},
        "audit":audit,"feasible_policy_ids":feasible,"selected_policy_id":selected,
    }
    payload["receipt_sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return payload

__all__=["GroupPolicyMetrics","select_policy_noninferiority_first"]
