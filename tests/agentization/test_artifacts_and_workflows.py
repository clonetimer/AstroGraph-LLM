import hashlib, json, zipfile
from pathlib import Path
import pytest
from astrograph.agentization.artifacts import verify_file, scan_zip
from astrograph.agentization.protocol import CapabilityProtocolError
from astrograph.agentization.workflows import validate_workflow

def test_hash_verification(tmp_path):
    p=tmp_path/'a.txt'; p.write_text('x'); h=hashlib.sha256(b'x').hexdigest(); assert verify_file(str(p),h,[tmp_path])['verified']
def test_path_scope(tmp_path):
    with pytest.raises(CapabilityProtocolError): verify_file('/etc/hosts','',[tmp_path])
def test_zip_traversal_rejected(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../x','x')
    assert scan_zip(str(p),[tmp_path])['admission']=='REJECT'
def test_workflow_cycle_rejected():
    with pytest.raises(CapabilityProtocolError): validate_workflow({'nodes':[{'id':'a'},{'id':'b'}],'edges':[{'from':'a','to':'b'},{'from':'b','to':'a'}]})
def test_workflow_topological_order():
    r=validate_workflow({'nodes':[{'id':'a'},{'id':'b'}],'edges':[{'from':'a','to':'b'}]}); assert r['topological_order']==['a','b']
