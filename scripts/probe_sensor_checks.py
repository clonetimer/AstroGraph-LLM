"""Developer illustration only; no inference path reads this example's truth."""
from pathlib import Path
import zipfile,io,hashlib,json,argparse
import pandas as pd
import torch
from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
from astrograph.simulation import SimulationRunCompleted
from astrograph.integrations.adcs_three_task_torch import load_adcs_three_task_torch_package,build_registry_from_adcs_three_task_provider
from astrograph.oag.native_bridge import from_native_frames,write_package_graph
from astrograph.oag.sensor_checks import sensor_check_tools
from astrograph.oag.kernel import EvidenceDiagnosisKernel
from astrograph.oag.agent import AutonomousDiagnosisService
from astrograph.orchestration.models import WorkflowContext,PermissionScope

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
 path=root/'models/adcs_v5_2_2/p2_innovation_v2.pt';model=load_adcs_three_task_torch_package(path,expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),device='cpu')
 with zipfile.ZipFile(args.dataset_zip) as z:
  dataset='whole_spacecraft_adcs_paired_dataset_v5_2_2/'
  name=dataset+'scenarios/attitude_acquisition_post_settle/adcs_imu_axis_dropout/sample_03/telemetry.csv';n=dataset+'nominal_baseline/attitude_acquisition/sample_03/telemetry.csv'
  a=pd.read_csv(io.BytesIO(z.read(name)),low_memory=False);b=pd.read_csv(io.BytesIO(z.read(n)),low_memory=False);a=a[a.time_s<=150];b=b[b.time_s<=150]
 event=SimulationRunCompleted(event_id='E',run_id='R',run_dir='/admitted/r',nominal_run_dir='/admitted/n',runtime_id='dev3-smoke')
 pkg=root/'configs/datasets/adcs_v5_2_2/runtime_sensor_checks.research.json';builder=ConfiguredRuntimeEvidenceBuilder(pkg)
 data=from_native_frames(builder=builder,event=event,source=a,nominal=b,provider=model,registry=build_registry_from_adcs_three_task_provider(model))
 graph=write_package_graph(pkg,args.output/'graph')
 out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=sensor_check_tools(data,source=a,nominal=b)).run(data,context=WorkflowContext(workflow_id='w',run_id='r',actor='diagnostic-agent',permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE]))
 print(out['diagnosis']['status'],out['diagnosis']['selected_fault_id'],out['checks_executed'],out['planning_trace'])
 (args.output/'output.json').write_text(json.dumps(out,indent=2))
 (args.output/'scope.json').write_text(json.dumps({'post_hoc_development_illustration':True,'not_accuracy_sample':True,'source':name,'endpoint_s':150},indent=2))
if __name__=='__main__':main()
