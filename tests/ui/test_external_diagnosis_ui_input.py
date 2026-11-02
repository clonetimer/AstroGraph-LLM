"""Execute production input handlers in Node; exercise their API request contract.

Only the DOM and HTTP boundary are substituted. These tests catch a return to
three mandatory roles, dropped empty-role arrays, and optional-file bypasses.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


RESOURCES = Path(__file__).resolve().parents[2] / "src/astrograph/resources"


@pytest.mark.parametrize("surface", ["mission", "legacy"])
def test_single_external_result_submits_and_optional_files_stay_validated(surface):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript input contract regression")
    harness = r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const [root,surface]=process.argv.slice(1),nodes=new Map(),calls=[],timers=[];
const $=key=>{if(!nodes.has(key))nodes.set(key,{value:'',textContent:'',innerHTML:'',disabled:true,
  style:{},dataset:{},classList:{add(){},remove(){}},addEventListener(){},setAttribute(){}});return nodes.get(key)};
const state={files:[],inputRoles:{},inputFlow:'review',agentMode:'normal',preflight:null};
const S={files:{diagnosis_results:[],log_reports:[],references:[]},uploadRevision:0,busy:false,
  analysisEngineId:'llm:configured',preflight:null,preflightId:''};
let allowed=true;
const context={console,Blob,JSON,Array,Set,Map,String,Number,encodeURIComponent,
  $,$$:()=>[],state,S,can:()=>allowed,html:String,esc:String,bytes:String,
  clearTimeout(){},setTimeout(callback){timers.push(callback);return timers.length},
  toast(){},showToast(){},remember(){},renderOperationsWizard(){},renderOperationsWorkflow(){},
  activeModeInfo:()=>({}),currentModeToolIds:()=>[],modeZh:{normal:'普通'},artifactKindZh:{},
  badgeFor:()=>'',localStorage:{setItem(){}},
  api:async(path,options={})=>{calls.push({path,options});
    if(path==='/v1/platform/jobs')return {job_id:'JOB-INPUT'};
    if(path.endsWith('/preflight'))return {can_start:true,artifact_kind:'RESULT_LOG_REVIEW',blockers:[]};
    return {};
  }
};
vm.createContext(context);
if(surface==='mission'){
  const source=fs.readFileSync(root+'/mission_console.js','utf8');
  const input=source.slice(source.indexOf('function getRoleFiles('),source.indexOf('async function fetchDemos('));
  const submit=source.slice(source.indexOf('function fileList('),source.indexOf('function remember('));
  vm.runInContext('const qs=encodeURIComponent;'+input+'\n'+submit,context);
}else{
  let source=fs.readFileSync(root+'/platform_job.js','utf8');
  source=source.split('\n').filter(line=>!line.startsWith('import ')).join('\n').split('\nexport {')[0];
  vm.runInContext(source,context);
}
const diagnosis={name:'diagnosis.txt',size:42,lastModified:1};
const setInputs=(results,logs=[],references=[])=>{
  S.files={diagnosis_results:results,log_reports:logs,references};
  state.files=[...results,...logs,...references];state.inputRoles={};
  for(const [role,files] of Object.entries(S.files))for(const file of files)state.inputRoles[file.name]=role;
};
const acceptable=()=>surface==='mission'?vm.runInContext('validUploads().length>0',context):vm.runInContext('canSubmitInputs()',context);
(async()=>{
  setInputs([diagnosis]);
  assert.equal(acceptable(),true,'one external diagnosis must be sufficient');
  if(surface==='mission'){
    vm.runInContext('schedulePreflight()',context);
    assert.equal(timers.length,1,'single diagnosis should schedule automatic preflight');
    await vm.runInContext('preflight()',context);
    assert.equal($('#mission-start').disabled,false);
  }else{
    await $('#preflight-btn').onclick();
    assert.equal($('#start-btn').disabled,false);
    context.badgeFor=()=>['','已完成'];context.renderOperationsWorkflow=()=>{};
    vm.runInContext('renderJobDiagnostics({job_id:"JOB-1",status:"SUCCEEDED",stage:"COMPLETED",llm_call_status:"NOT_CONFIGURED"})',context);
    assert.ok(!$('#jobdiag-llm').textContent.includes('核心诊断'),'external import must not claim a core diagnosis ran');
    assert.ok($('#jobdiag-llm').textContent.includes('未配置'));
  }
  const manifestCall=calls.find(call=>call.path.endsWith('/files/analysis_input.json'));
  assert.ok(manifestCall,'the input handler must upload its role manifest');
  const body=manifestCall.options.body;
  const manifest=JSON.parse(body instanceof Blob?await body.text():body);
  assert.deepEqual(manifest,{schema_version:'astrograph-result-log-review.v1',analysis_mode:'EXTERNAL_DIAGNOSIS_ANALYSIS',
    diagnosis_results:['diagnosis.txt'],log_reports:[],references:[]});
  assert.ok(calls.some(call=>call.path.endsWith('/files/diagnosis.txt')),'diagnosis file must be uploaded');
  setInputs([],[{name:'operations.log',size:3}]);
  if(surface==='mission')assert.throws(acceptable,/诊断/);else assert.equal(acceptable(),false);
  setInputs([diagnosis],[],[{name:'reference.json',size:3}]);
  if(surface==='mission')assert.throws(acceptable,/格式/);else assert.equal(acceptable(),false);
  if(surface==='mission'){
    setInputs([diagnosis]);allowed=false;
    assert.throws(acceptable,/权限/,'optional inputs must not bypass authorization');
  }
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    result = subprocess.run(
        [node, "-e", harness, str(RESOURCES), surface],
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
