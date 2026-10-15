import hashlib
import pytest
from astrograph.agentization.shadow import compare_plan
from astrograph.agentization.evidence_agent import propose_card, validate_card
from astrograph.agentization.protocol import CapabilityProtocolError

def test_shadow_detects_order_violation():
    r=compare_plan(['b','a'],['a','b'],{'a','b'}); assert not r.valid and r.order_violations
def test_shadow_never_has_execution_authority():
    assert compare_plan(['a'],['a'],{'a'}).to_mapping()['execution_authority']=='REFERENCE_WORKFLOW_ONLY'
def test_evidence_agent_cannot_freeze():
    card=propose_card({'fault_id':'F1','evidence_type':'REQUIRED','telemetry_fields':['x'],'predicate':{'op':'gt','value':1},'applicability_envelope':{'mode':'NOMINAL'}},{'verified':True,'sha256':'0'*64})
    with pytest.raises(CapabilityProtocolError) as exc: validate_card(card,'FROZEN')
    assert exc.value.code=='HUMAN_SCIENTIFIC_APPROVAL_REQUIRED'
def test_evidence_requires_pinned_source():
    with pytest.raises(CapabilityProtocolError): validate_card({'status':'PROPOSED','source_receipt':{'verified':False}})
