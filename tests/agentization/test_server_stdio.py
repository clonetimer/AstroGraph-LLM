import io, json
from pathlib import Path
from astrograph.agentization.server import serve

def test_stdio_health_and_manifest(tmp_path):
    root=Path(__file__).resolve().parents[2]
    requests=[{'protocol_version':'astrograph-capability-protocol.v1','request_id':'r1','method':'health','profile':'MAINTAINER','params':{}},{'protocol_version':'astrograph-capability-protocol.v1','request_id':'r2','method':'capabilities.list','profile':'MAINTAINER','params':{}},{'protocol_version':'astrograph-capability-protocol.v1','request_id':'r3','method':'shutdown','profile':'MAINTAINER','params':{}}]
    inp=io.StringIO(''.join(json.dumps(x)+'\n' for x in requests)); out=io.StringIO(); serve(root,[root,tmp_path],tmp_path/'artifacts',[root],inp,out)
    rows=[json.loads(x) for x in out.getvalue().splitlines()]; assert rows[0]['status']=='OK'; assert rows[1]['result']['manifest_sha256']; assert rows[2]['result']['status']=='SHUTTING_DOWN'
