import hashlib,json,pytest
from pathlib import Path
from astrograph.agentization.capabilities import CapabilityEnvironment,build_registry
from astrograph.agentization.protocol import CapabilityProtocolError
def reg(root):
    return build_registry(CapabilityEnvironment(root,(root,),root/"a",root/"w",(root,)))
def test_description(tmp_path):
    r=reg(tmp_path).invoke("temporal.mechanism_evidence.describe","DEVELOPMENT",{})
    assert r["groups_4_9_read"] is False and r["fault_identification_established"] is False
def test_verifier(tmp_path):
    p={"development_groups":[3,8],"excluded_groups":[4,9],"calibration_groups":[0,1,2,5,6,7],
       "groups_4_9_read":False,"truth_access_during_generation":False,
       "outcome_data_read_during_generation":False,"candidate_selections_frozen_before_truth_join":True,
       "candidate_count":600,"index_sha256":"x"}
    f=tmp_path/"i.json";f.write_text(json.dumps(p));s=hashlib.sha256(f.read_bytes()).hexdigest()
    assert reg(tmp_path).invoke("temporal.mechanism_evidence.verify_trace_index","DEVELOPMENT",{"path":str(f),"expected_sha256":s})["status"]=="VERIFIED"
    p["truth_access_during_generation"]=True;f.write_text(json.dumps(p));s=hashlib.sha256(f.read_bytes()).hexdigest()
    with pytest.raises(CapabilityProtocolError):reg(tmp_path).invoke("temporal.mechanism_evidence.verify_trace_index","DEVELOPMENT",{"path":str(f),"expected_sha256":s})
