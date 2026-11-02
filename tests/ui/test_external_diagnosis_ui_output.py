"""Run both production report surfaces against the external-LLM output contract."""
from pathlib import Path
import shutil
import subprocess

import pytest


RESOURCES = Path(__file__).resolve().parents[2] / "src/astrograph/resources"


@pytest.mark.parametrize("status", [
    "AVAILABLE_REQUIRES_EXPERT_REVIEW", "FAILED", "NOT_CONFIGURED", "INPUT_TOO_LARGE",
    "EXTRACTIVE_REVIEW_ONLY", "GENERATION_FAILED", "BAD_CITATION",
])
def test_both_report_surfaces_show_actual_model_status_and_escaped_complete_output(status):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the JavaScript report regression")
    harness = r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const [root,status]=process.argv.slice(1);
const legacy=status==='EXTRACTIVE_REVIEW_ONLY',success=status==='AVAILABLE_REQUIRES_EXPERT_REVIEW'||legacy;
const quote='SOURCE_BEGIN '+('完整原文 '.repeat(1200))+' SOURCE_END <img src=x onerror=alert(1)>';
const modelStatus=['GENERATION_FAILED','BAD_CITATION'].includes(status)?'AVAILABLE_REQUIRES_EXPERT_REVIEW':status;
const model={schema_version:legacy?undefined:'astrograph-external-diagnosis-analysis.v1',status:modelStatus,
  source:legacy?'LLM_EXTRACTIVE_REVIEW':'LLM_EXTERNAL_DIAGNOSIS',
  summary:'外部未知候选的分析 <script>alert(1)</script>',
  findings:Array.from({length:24},(_,i)=>({text:'分析依据 '+(i+1),citation_ids:['SRC-1']})),
  hypotheses:Array.from({length:12},(_,i)=>({title:'待核查假设 '+(i+1),analysis:'未知故障不需要内置故障编码',citation_ids:['SRC-1']})),
  claims:[{text:'历史来源摘录',citation_ids:['SRC-1']}],
  information_gaps:['缺少负载状态'],recommended_review_actions:['请人工核对记录'],
  sources:[{source_id:'SRC-1',role:'diagnosis_results',filename:'diagnosis<script>.txt',locator:'line:7',sha256:'a'.repeat(64),text:quote},
    {source_id:'SRC-2',role:'log_reports',filename:'unquoted<img>.log',locator:'line:8',sha256:'b'.repeat(64),text:'未被模型引用的完整原文 <script>source()</script>'}],
  citation_reference_validation:status==='BAD_CITATION'?'FAILED':(success||status==='GENERATION_FAILED'?'PASS':'NOT_EVALUATED'),semantic_hallucination_assessment:'NOT_EVALUATED'};
const generation={status:status==='GENERATION_FAILED'?'FAILED':(success||status==='BAD_CITATION'?'SUCCEEDED':status)};
const fixture={job_id:'JOB-OUTPUT',status:'SUCCEEDED',artifact_kind:'RESULT_LOG_REVIEW',analysis_engine:{kind:'rules'},
  analysis:{analysis_mode:legacy?'MATERIAL_REVIEW':'EXTERNAL_DIAGNOSIS_ANALYSIS',
    result_log_review:{candidates:[],evidence:[],source_manifest:[]},
    knowledge_enrichment:{generation,model_explanation:model}}};
const outputs=[];
for(const surface of ['mission','legacy']){
  const nodes=new Map(),$=key=>{if(!nodes.has(key))nodes.set(key,{textContent:'',innerHTML:'',hidden:false,
    dataset:{},style:{},classList:{add(){},remove(){},toggle(){}},setAttribute(){},addEventListener(){},querySelectorAll:()=>[]});return nodes.get(key)};
  const context={console,fixture,clearTimeout(){},setTimeout(){},document:{querySelector:$,querySelectorAll:()=>[]},
    $,$$:()=>[],state:{},api:async()=>({}),artifactKindZh:{},modeZh:{},can:()=>true,
    focusManualReportReview(){},nav(){},enhanceOperationsResult(){},localStorage:{},
  };
  vm.createContext(context);
  if(surface==='mission'){
    let source=fs.readFileSync(root+'/mission_console.js','utf8');source=source.slice(0,source.lastIndexOf('main();'));
    vm.runInContext(source,context);
    vm.runInContext('S.report=fixture;S.reportTab="conclusion";renderReport()',context);
    outputs.push($('#report-content').innerHTML);
    if(status==='AVAILABLE_REQUIRES_EXPERT_REVIEW'){
      vm.runInContext("S.analysisEngines=[{engine_id:'rules',kind:'rules',name:'规则引擎',description:'运行规则诊断',available:true},{engine_id:'llm:configured',kind:'llm',name:'已配置模型',available:true}];S.analysisEngineId='rules';renderAnalysisEngines()",context);
      assert.equal($('#analysis-engine-label').textContent,'仅导入材料');
      assert.ok(!$('#analysis-engine-options').innerHTML.includes('运行规则诊断'));
      vm.runInContext("selectAnalysisEngine('llm:configured')",context);
      assert.equal($('#analysis-engine-label').textContent,'已配置模型');
      vm.runInContext("S.reportTab='evidence';renderReport()",context);
      assert.ok($('#report-content').innerHTML.includes('诊断报告'));
      vm.runInContext("S.reportTab='reasoning';renderReport()",context);
      assert.ok($('#report-content').innerHTML.includes('本次为外部文本分析，未执行图谱推理'));
    }
    if(!legacy){vm.runInContext('renderWorkbench()',context);assert.ok(!$('#workbench-timeline').innerHTML.includes('未形成候选'));
      if(!success)assert.ok(!$('#workbench-actions').innerHTML.includes('请人工核对记录'),'failed model must not leak stale suggested actions');}
  }else{
    const core=fs.readFileSync(root+'/platform_core.js','utf8');
    vm.runInContext(core.split('\n').find(line=>line.startsWith('function esc(')),context);
    let source=fs.readFileSync(root+'/platform_report.js','utf8');
    source=source.split('\n').filter(line=>!line.startsWith('import ')).join('\n').split('\nexport {')[0];
    vm.runInContext(source,context);vm.runInContext('renderReport(fixture)',context);
    outputs.push($('#result-content').innerHTML);
  }
}
const messages={FAILED:'大模型分析未完成',NOT_CONFIGURED:'尚未配置大模型',INPUT_TOO_LARGE:'输入材料超出分析容量',GENERATION_FAILED:'大模型分析未完成',BAD_CITATION:'大模型分析未完成'};
for(const output of outputs){
  assert.ok(!output.includes('已回退到规则引擎'),'model status must not imply a fabricated fallback');
  assert.ok(!output.includes('<script>')&&!output.includes('<img '),'model and source HTML must be escaped');
  if(legacy){assert.ok(output.includes('历史来源摘录'),'historical extractive reports remain readable');continue}
  assert.ok(!output.includes('没有形成可信的故障方向')&&!output.includes('无法确诊'),'no candidate IDs does not mean failed text analysis');
  for(const text of ['外部导入材料','SRC-1','SRC-2','diagnosis&lt;script&gt;.txt','unquoted&lt;img&gt;.log','line:8','SOURCE_BEGIN','SOURCE_END','未被模型引用的完整原文','&lt;script&gt;source()&lt;/script&gt;'])
    assert.ok(output.includes(text),'all imported sources remain accessible: '+text);
  if(status==='INPUT_TOO_LARGE'){assert.ok(output.includes('60000'));assert.ok(!output.includes('更大上下文'));}
  if(success){
    for(const text of ['外部未知候选的分析','&lt;script&gt;','分析依据 24','待核查假设 12',
      '缺少负载状态','请人工核对记录','diagnosis&lt;script&gt;.txt','line:7','SOURCE_BEGIN','SOURCE_END'])
      assert.ok(output.includes(text),'missing complete output or traceable source: '+text);
  }else{
    assert.ok(output.includes(messages[status]),'distinct status guidance required: '+status);
    assert.ok(!output.includes('外部未知候选的分析'),'failed generation must not display stale model prose');
  }
}
"""
    result = subprocess.run(
        [node, "-e", harness, str(RESOURCES), status],
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
