from __future__ import annotations
from typing import Any, Iterable
from .profiles import AgentProfile, PROFILE_BOUNDARIES
from .protocol import CapabilityProtocolError

PERMANENT_POSTHOC_GROUPS={"4","9"}
DEVELOPMENT_GROUPS={"3","8"}
TRUTH_TOKENS={"ground_truth","true_label","fault_label","y_true","target"}

def _walk_group_values(value: Any, key_path: str="") -> list[tuple[str,str]]:
    out=[]
    if isinstance(value,dict):
        for k,v in value.items():
            kp=f"{key_path}.{k}" if key_path else str(k)
            if "group" in str(k).lower():
                if isinstance(v,(str,int)): out.append((kp,str(v)))
                elif isinstance(v,list): out.extend((kp,str(x)) for x in v if isinstance(x,(str,int)))
            out.extend(_walk_group_values(v,kp))
    elif isinstance(value,list):
        for i,v in enumerate(value): out.extend(_walk_group_values(v,f"{key_path}[{i}]"))
    return out

def _walk_keys(value: Any) -> set[str]:
    out=set()
    if isinstance(value,dict):
        for k,v in value.items(): out.add(str(k).lower()); out |= _walk_keys(v)
    elif isinstance(value,list):
        for v in value: out |= _walk_keys(v)
    return out

def authorize(definition: Any, profile: str, arguments: dict[str,Any]) -> None:
    try: AgentProfile(profile)
    except ValueError: raise CapabilityProtocolError("UNKNOWN_PROFILE", f"unknown profile {profile!r}")
    if profile not in definition.allowed_profiles:
        raise CapabilityProtocolError("CAPABILITY_NOT_ALLOWED", f"{definition.name} is not available in profile {profile}")
    groups={g for _,g in _walk_group_values(arguments)}
    if profile==AgentProfile.DEVELOPMENT.value:
        if groups & PERMANENT_POSTHOC_GROUPS:
            raise CapabilityProtocolError("POSTHOC_GROUP_SELECTION_FORBIDDEN","groups 4/9 cannot be used for development or policy selection")
        if definition.policy_selection and groups and not groups <= DEVELOPMENT_GROUPS:
            raise CapabilityProtocolError("DEVELOPMENT_GROUP_CONTRACT_VIOLATION","policy selection groups must be a subset of {3,8}")
    if profile==AgentProfile.POSTHOC_AUDIT.value and definition.side_effect not in {"NONE","ARTIFACT_WRITE"}:
        raise CapabilityProtocolError("POSTHOC_MUTATION_FORBIDDEN","post-hoc profile cannot mutate model, evidence, or policy")
    if profile in {AgentProfile.DIAGNOSIS.value,AgentProfile.SHADOW.value,AgentProfile.EVIDENCE_CURATION.value,AgentProfile.MAINTAINER.value}:
        leaked=_walk_keys(arguments)&TRUTH_TOKENS
        if leaked:
            raise CapabilityProtocolError("LABEL_ACCESS_FORBIDDEN",f"truth-like fields are forbidden in {profile}: {sorted(leaked)}")
    if profile==AgentProfile.CONFIRMATORY_SEALED.value and definition.policy_selection:
        raise CapabilityProtocolError("CONFIRMATORY_POLICY_SELECTION_FORBIDDEN","sealed confirmatory profile cannot select or modify policy")
