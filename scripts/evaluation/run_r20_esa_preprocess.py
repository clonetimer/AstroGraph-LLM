import pathlib,pickle,io,zipfile,sys,json
import argparse
p=argparse.ArgumentParser();p.add_argument('--workspace',type=pathlib.Path,required=True);p.add_argument('--data-dir',type=pathlib.Path,required=True);args=p.parse_args()
raise ValueError('R21_SEEN_EVALUATION_PREDICTION_REGENERATION_BLOCKED: historical R20 runners disabled in R21; read frozen results')
ROOT=args.workspace.resolve();SOURCE=pathlib.Path(__file__).resolve().parents[2];DATA=args.data_dir.resolve()/'ESA-Mission1-lightweight.zip';OUT=ROOT/'work/esa_preprocessed'
if DATA.is_relative_to(SOURCE) or ROOT.is_relative_to(SOURCE):raise ValueError('EXTERNAL_DATA_OR_WORKSPACE_INSIDE_SOURCE')
OUT.mkdir(parents=True,exist_ok=True)
ALLOWED={('pandas.core.frame','DataFrame'),('pandas.core.internals.managers','BlockManager'),('pandas._libs.internals','_unpickle_block'),
 ('numpy.core.numeric','_frombuffer'),('numpy._core.numeric','_frombuffer'),('numpy','dtype'),('builtins','slice'),
 ('pandas.core.indexes.base','_new_Index'),('pandas.core.indexes.base','Index'),('numpy.core.multiarray','_reconstruct'),
 ('numpy._core.multiarray','_reconstruct'),('numpy','ndarray'),('pandas.core.indexes.datetimes','_new_DatetimeIndex'),
 ('pandas.core.indexes.datetimes','DatetimeIndex'),('pandas._libs.arrays','__pyx_unpickle_NDArrayBacked'),
 ('pandas.core.arrays.datetimes','DatetimeArray')}
class Restricted(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name) not in ALLOWED:raise ValueError('UNADMITTED_PICKLE_GLOBAL:'+module+':'+name)
  return super().find_class(module,name)
import pandas as pd,numpy as np
metadata=[]
with zipfile.ZipFile(DATA) as outer:
 for channel in range(41,47):
  name=next(n for n in outer.namelist() if n.endswith('/channels/channel_'+str(channel)+'.zip'))
  with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
   member=next(i for i in inner.infolist() if not i.is_dir())
   with inner.open(member) as content:frame=Restricted(content).load()
  if not isinstance(frame,pd.DataFrame) or list(frame.columns)!=['channel_'+str(channel)] or not isinstance(frame.index,pd.DatetimeIndex):raise ValueError('ESA_CHANNEL_SCHEMA_UNADMITTED')
  if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:raise ValueError('ESA_TIMESTAMP_DRIFT')
  series=frame.iloc[:,0]
  if not np.isfinite(series.to_numpy()).all():raise ValueError('ESA_NONFINITE_OBSERVATION')
  result=series.resample('5min').median()
  np.savez_compressed(OUT/('channel_'+str(channel)+'.npz'),time_ns=result.index.as_unit('ns').asi8,value=result.to_numpy(),observed=result.notna().to_numpy())
  metadata.append({'channel':'channel_'+str(channel),'raw_sample_count':len(frame),'start_ns':int(frame.index.as_unit('ns').asi8[0]),'end_ns':int(frame.index.as_unit('ns').asi8[-1]),
   'binned_count':len(result),'observed_bins':int(result.notna().sum()),'transform':'retrospective nonoverlapping 5min median; empty bins have no prediction'})
  print(metadata[-1],flush=True);del frame,series,result
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
