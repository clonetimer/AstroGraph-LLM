import json
from pathlib import Path
import pytest
from astrograph.agentization.capabilities import CapabilityEnvironment, build_registry
from astrograph.agentization.protocol import CapabilityRequest, CapabilityProtocolError, PROTOCOL_VERSION

@pytest.fixture
def registry(tmp_path):
    project=Path(__file__).resolve().parents[2]
    env=CapabilityEnvironment(project,(project,tmp_path),tmp_path/'artifacts',project/'configs/agentization/workflows',(project,))
    return build_registry(env)

def test_protocol_rejects_wrong_version():
    with pytest.raises(CapabilityProtocolError,match='expected'):
        CapabilityRequest.from_mapping({'protocol_version':'v0','request_id':'r1','method':'health','profile':'MAINTAINER','params':{}})

def test_manifest_is_content_addressed(registry):
    m=registry.manifest(); assert len(m['manifest_sha256'])==64; assert m['project_status']=='B014_TEMPORAL_STACK_MERGED_CONFIRMATION_NOT_EXECUTED'

def test_development_rejects_groups_4_9(registry):
    with pytest.raises(CapabilityProtocolError) as exc:
        registry.invoke('research.route_a.selection_gate','DEVELOPMENT',{'selection_groups':[4,9],'required_artifacts':[]})
    assert exc.value.code=='POSTHOC_GROUP_SELECTION_FORBIDDEN'

def test_diagnosis_has_no_truth_access(registry):
    with pytest.raises(CapabilityProtocolError) as exc:
        registry.invoke('artifact.verify','DIAGNOSIS',{'path':'x','expected_sha256':'0'*64,'ground_truth':'fault'})
    assert exc.value.code=='LABEL_ACCESS_FORBIDDEN'

def test_confirmatory_has_no_policy_selector(registry):
    with pytest.raises(CapabilityProtocolError):
        registry.invoke('research.route_a.selection_gate','CONFIRMATORY_SEALED',{'selection_groups':[3,8],'required_artifacts':[]})

def test_posthoc_cannot_select(registry):
    with pytest.raises(CapabilityProtocolError):
        registry.invoke('research.route_a.selection_gate','POSTHOC_AUDIT',{'selection_groups':[4,9],'required_artifacts':[]})
