from pathlib import Path
from astrograph.agentization.capabilities import CapabilityEnvironment,build_registry

def test_sealed_campaign_status_is_runtime_blocked(tmp_path):
 project=Path(__file__).resolve().parents[2]
 r=build_registry(CapabilityEnvironment(project,(project,tmp_path),tmp_path/'a',project/'configs/agentization/workflows',(project,)))
 out=r.invoke('confirmatory.temporal_campaign.status','CONFIRMATORY_SEALED',{})
 assert out['status']=='FROZEN_RUNTIME_BLOCKED'
 assert out['groups_4_9_read'] is False
 assert out['generation_consumed'] is False
