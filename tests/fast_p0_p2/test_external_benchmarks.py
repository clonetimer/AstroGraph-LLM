import zipfile
from pathlib import Path
from astrograph.integrations.external_benchmarks import build_smap_msl_manifest,build_dxc_manifest,build_esa_quarantine_plan

def test_smap_adapter_excludes_bundled_predictions(tmp_path:Path):
 p=tmp_path/'s.zip'
 with zipfile.ZipFile(p,'w') as z:
  z.writestr('data/labeled_anomalies.csv','chan_id,spacecraft\nA-1,SMAP\n'); z.writestr('data/train/A-1.npy',b'x'); z.writestr('data/test/A-1.npy',b'y'); z.writestr('data/models/A-1.h5',b'z')
 r=build_smap_msl_manifest(p); assert r['channel_count']==1 and r['excluded_contamination_members']==['data/models/A-1.h5']

def test_dxc_groups_representations_by_experiment(tmp_path:Path):
 p=tmp_path/'d.zip'
 with zipfile.ZipFile(p,'w') as z:
  z.writestr('Exp_1_pb_t2.txt','');z.writestr('Exp_1_pb_t2.mat','');z.writestr('Exp_2_pb_t2.scn','')
 r=build_dxc_manifest([p]); assert r['experiment_group_count']==2

def test_esa_nested_pickle_is_quarantined(tmp_path:Path):
 inner=tmp_path/'inner.zip'
 with zipfile.ZipFile(inner,'w') as z:z.writestr('x.pkl',b'pickle')
 outer=tmp_path/'e.zip'
 with zipfile.ZipFile(outer,'w') as z:z.writestr('root/channels/channel_1.zip',inner.read_bytes())
 r=build_esa_quarantine_plan(outer); assert r['channel_archives'][0]['unsafe_serialized_members']==['x.pkl'] and not r['normal_runtime_deserialization_allowed']
