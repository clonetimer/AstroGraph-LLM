import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_synthetic_cli_uses_isolated_workers_and_canonical_runtime(tmp_path):
    output=tmp_path/'candidate.json'
    command=[sys.executable,'scripts/run_diagnostic_dag.py','--request','examples/implementation_dag/request.json',
             '--plugins','examples/implementation_dag','--approved-config','examples/implementation_dag/approved_config.json',
             '--output',str(output),'--ledger',str(tmp_path/'events.jsonl')]
    result=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stderr
    value=json.loads(output.read_text(encoding="utf-8"));assert value['pipeline_id']=='diagnosis.dual_source.research.v1'
    assert value['output']['tool_calls']==5
    assert value['output']['planning_trace'][0]['data_top']=='FIXTURE_F1'
    assert value['output']['planning_trace'][1]['data_top']=='FIXTURE_F2'
    assert value['output']['formal_decision'] is None
