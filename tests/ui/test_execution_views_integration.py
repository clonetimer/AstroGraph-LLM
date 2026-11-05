"""Execute each production report surface and verify it mounts the common view."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]/'src/astrograph/resources'

@pytest.mark.parametrize('surface',['mission','legacy'])
def test_external_reports_mount_shared_execution_view(surface):
    if not shutil.which('node'):pytest.skip('Node required for UI behavior test')
    harness=r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const [root,surface]=process.argv.slice(1),nodes=new Map(),calls=[];
function make(){return {innerHTML:'',textContent:'',hidden:false,isConnected:true,dataset:{},style:{},
  classList:{add(){},remove(){},toggle(){}},append(x){this.child=x},setAttribute(){},addEventListener(){},querySelectorAll:()=>[]};}
const $=s=>{if(!nodes.has(s))nodes.set(s,make());return nodes.get(s)};
const fixture={job_id:'a'.repeat(32),status:'SUCCEEDED',artifact_kind:'RESULT_LOG_REVIEW',
  analysis:{analysis_mode:'EXTERNAL_DIAGNOSIS_ANALYSIS',knowledge_enrichment:{generation:{status:'FAILED'},model_explanation:{status:'FAILED',sources:[]}}}};
const context={console,fixture,$,$$:()=>[],state:{},api:async()=>({}),clearTimeout(){},setTimeout(){},
  document:{querySelector:$,querySelectorAll:()=>[],createElement:make},
  AstroGraphExecution:{mount:(host,opts)=>calls.push({host,opts})},
  artifactKindZh:{},modeZh:{},focusManualReportReview(){},nav(){},enhanceOperationsResult(){},localStorage:{}};
vm.createContext(context);
if(surface==='mission'){
  let src=fs.readFileSync(root+'/mission_console.js','utf8');src=src.slice(0,src.lastIndexOf('main();'));
  vm.runInContext(src,context);vm.runInContext('S.report=fixture;S.reportTab="conclusion";renderReport()',context);
}else{
  const core=fs.readFileSync(root+'/platform_core.js','utf8');
  vm.runInContext(core.split('\n').find(l=>l.startsWith('function esc(')),context);
  let src=fs.readFileSync(root+'/platform_report.js','utf8');
  src=src.split('\n').filter(l=>!l.startsWith('import ')).join('\n').split('\nexport {')[0];
  vm.runInContext(src,context);vm.runInContext('renderReport(fixture)',context);
}
assert.equal(calls.length,1,'production external report must mount execution receipts');
assert.equal(calls[0].opts.baseUrl,'/v1/platform/jobs/'+'a'.repeat(32));
assert.equal(calls[0].opts.current(),true);
if(surface==='mission'){
  assert.equal(typeof calls[0].opts.download,'function','bearer-aware download required');
  vm.runInContext('S.report=null',context);
}else vm.runInContext('state.report=null',context);
assert.equal(calls[0].opts.current(),false,'late responses must not affect a new selected task');
'''
    result=subprocess.run(['node','-e',harness,str(ROOT),surface],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
