from __future__ import annotations
from enum import Enum

class AgentProfile(str, Enum):
    DIAGNOSIS = "DIAGNOSIS"
    DEVELOPMENT = "DEVELOPMENT"
    POSTHOC_AUDIT = "POSTHOC_AUDIT"
    EVIDENCE_CURATION = "EVIDENCE_CURATION"
    CONFIRMATORY_SEALED = "CONFIRMATORY_SEALED"
    MAINTAINER = "MAINTAINER"
    SHADOW = "SHADOW"

PROFILE_BOUNDARIES = {
    AgentProfile.DIAGNOSIS.value: {"truth_access": False, "policy_mutation": False, "allowed_groups": []},
    AgentProfile.DEVELOPMENT.value: {"truth_access": True, "policy_mutation": True, "allowed_groups": ["3", "8"]},
    AgentProfile.POSTHOC_AUDIT.value: {"truth_access": True, "policy_mutation": False, "allowed_groups": ["4", "9"]},
    AgentProfile.EVIDENCE_CURATION.value: {"truth_access": False, "policy_mutation": False, "allowed_groups": []},
    AgentProfile.CONFIRMATORY_SEALED.value: {"truth_access": True, "policy_mutation": False, "allowed_groups": []},
    AgentProfile.MAINTAINER.value: {"truth_access": False, "policy_mutation": False, "allowed_groups": []},
    AgentProfile.SHADOW.value: {"truth_access": False, "policy_mutation": False, "allowed_groups": []},
}
