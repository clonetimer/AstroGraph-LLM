import { $, $$, state, esc, faultLabel, badgeFor } from './platform_core.js';

let operationsHooks={};
function setupOperationsHooks(nextHooks={}){operationsHooks={...operationsHooks,...nextHooks}}

const probeLabels={
  EVIDENCE_THRESHOLD_MARGIN:'阈值余量复核',
  PRE_POST_CHANGE:'前后变化检查',
  COMMAND_RESPONSE_RESIDUAL:'命令响应检查',
  LAG_CORRELATION:'时滞相关检查',
  OSCILLATION_SIGNATURE:'振荡特征检查',
  SATURATION_LIMIT_CHECK:'饱和/限幅检查',
  MINIMAL_DIAGNOSIS_REVIEW:'最小诊断比较',
  SIMULATION_COUNTERFACTUAL_REVIEW:'反事实仿真复核',
  DXC_EXPECTED_RESPONSE_REVIEW:'期望响应复核',
  DXC_FORWARD_SIMULATION_REVIEW:'前向机理仿真复核',
  DXC_REDUCED_ORDER_SIMULATION_REVIEW:'低阶电气仿真复核',
  DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW:'泵机理仿真复核',
};

function probeDisplayName(probeId){return probeLabels[String(probeId||'')]||String(probeId||'受控调查步骤').replaceAll('_',' ')}
function probeRequestBody(probeId){return {probe_id:String(probeId||'')}}

function operatorStatusViewModel(job={},maintenance={}){
  const rawReview=maintenance?.requires_human_review??job?.requires_human_review;
  return {
    executionStatus:job?.execution_status??job?.status??'UNAVAILABLE',
    diagnosisDisposition:maintenance?.diagnosis_disposition??maintenance?.diagnosis_status??job?.diagnosis_disposition??'UNAVAILABLE',
    releaseStatus:maintenance?.release_status??job?.release_status??'UNAVAILABLE',
    requiresHumanReview:rawReview===true?true:rawReview===false?false:null,
  };
}

function operatorAttentionViewModel(job={}){
  const status=operatorStatusViewModel(job);
  const next=job.next_diagnostic_check?.display_name||job.next_diagnostic_check?.label||job.next_diagnostic_check?.probe_id||job.next_diagnostic_check||null;
  const needsReview=status.requiresHumanReview||/REVIEW|ABSTAIN|UNKNOWN|INSUFFICIENT|AMBIGUOUS/i.test(status.diagnosisDisposition);
  const releaseBlocked=/BLOCK/i.test(status.releaseStatus);
  return {status,nextDiagnosticCheck:typeof next==='string'?probeDisplayName(next):null,needsAttention:status.executionStatus==='FAILED'||needsReview||releaseBlocked||Boolean(next)};
}

function preflightRecognizedInputsHtml(preflight){
  if(!preflight)return '<div class="ops-empty">等待服务器识别输入；浏览器不会猜测数据类型。</div>';
  const raw=preflight.recognized_inputs||preflight.input_recognition||preflight.inputs||[];
  const items=Array.isArray(raw)?raw:Object.entries(raw||{}).map(([kind,value])=>value&&typeof value==='object'&&!Array.isArray(value)?{kind,...value}:{kind,status:value});
  if(!items.length){const kind=preflight.artifact_kind||'UNAVAILABLE';return `<div class="ops-recognized-row"><span>服务器识别的工件合同</span><strong>${esc(kind)}</strong></div>`}
  return items.map(item=>{const label=item.label||item.display_name||item.kind||item.type||item.name||'输入';const status=item.status??item.present??item.available??'UNKNOWN';const ok=status===true||/PRESENT|AVAILABLE|PASS|RECOGNIZED/i.test(String(status));const display=typeof status==='boolean'?(status?'已识别':'缺失'):String(status);return `<div class="ops-recognized-row"><span>${ok?'✓':'⚠'} ${esc(label)}</span><strong>${esc(display)}</strong></div>`}).join('');
}

function renderOperationsWizard(){
  const steps=$$('[data-ops-step]');if(!steps.length)return;
  const hasFiles=state.files.length>0,current=Boolean(state.preflight&&state.preflightSignature),canStart=Boolean(current&&state.preflight?.can_start);
  const blocked=Boolean(current&&!canStart),active=!hasFiles?1:(!current||blocked)?2:3;
  steps.forEach(el=>{const n=Number(el.dataset.opsStep);el.classList.toggle('active',n===active);el.classList.toggle('blocked',n===2&&blocked);el.classList.toggle('complete',n<active&&!blocked||(n===3&&canStart))});
  const recognized=$('#ops-recognized-inputs');if(recognized)recognized.innerHTML=preflightRecognizedInputsHtml(current?state.preflight:null);
  const modes=$('#ops-mode-aliases');if(modes){
    const supported=state.agentCapabilities?.modes||[];
    const normal=supported.find(x=>x.id==='normal');const enhanced=supported.find(x=>x.id==='thinking');
    const choices=[];
    if(normal)choices.push({id:normal.id,label:'标准诊断',description:'当前默认确定性核心路径'});
    if(enhanced)choices.push({id:enhanced.id,label:'增强诊断',description:'核心路径 + 已批准的 Agent 调查'});
    modes.innerHTML=choices.length?choices.map(x=>`<button type="button" class="ops-mode-choice ${state.agentMode===x.id?'active':''}" data-ops-mode="${esc(x.id)}"><strong>${esc(x.label)}</strong><span>${esc(x.description)}</span></button>`).join(''):'<div class="ops-empty">等待服务器返回可用诊断模式。</div>';
    $$('[data-ops-mode]').forEach(b=>b.onclick=()=>operationsHooks.setMode?.(b.dataset.opsMode));
  }
}

function actualWorkflowStages(job={}){
  const raw=Array.isArray(job.stage_history)?job.stage_history:Array.isArray(job.stages)?job.stages:[];
  if(raw.length)return raw.map(x=>typeof x==='string'?{id:x,status:x===job.stage?'RUNNING':'COMPLETED'}:{id:x.id||x.stage||x.name,status:x.status||'UNKNOWN',detail:x.detail||x.stage_detail||''});
  const id=job.stage||job.status;return id?[{id,status:job.status==='SUCCEEDED'?'COMPLETED':job.status||'UNKNOWN',detail:job.stage_detail||''}]:[];
}
function renderOperationsWorkflow(job={}){
  const host=$('#ops-runtime-workflow');if(!host)return;
  const stages=actualWorkflowStages(job);
  host.innerHTML=stages.length?stages.map(x=>`<div class="ops-workflow-step ${/COMPLETE|SUCCEEDED|PASS/i.test(x.status)?'complete':/RUNNING|ACTIVE/i.test(x.status)?'active':''}"><span>${esc(x.id)}</span><strong>${esc(x.status)}</strong>${x.detail?`<small>${esc(x.detail)}</small>`:''}</div>`).join(''):'<div class="ops-empty">后端尚未返回执行阶段。</div>';
}

function formalEvidenceHtml(evidence){
  if(!evidence||evidence.status==='UNAVAILABLE'||evidence.available===false)return '<div class="ops-evidence-unavailable">正式证据摘要不可用</div>';
  const groups=[['支持证据',evidence.supporting_items||evidence.supporting||evidence.supporting_evidence],['反证',evidence.contradicting_items||evidence.contradicting||evidence.counter_evidence],['缺失证据',evidence.missing_items||evidence.missing||evidence.missing_evidence]];
  return `<div class="ops-evidence-list">${groups.map(([label,items])=>`<section><h3>${label}</h3>${Array.isArray(items)&&items.length?items.map(x=>`<div>${esc(x.text||x.summary||x.label||x.evidence_definition_id||x.fact_id||x)}</div>`).join(''):'<div class="ops-muted">无</div>'}</section>`).join('')}</div>`;
}

function agentTimelineHtml(agent){
  if(!agent)return '<div class="ops-empty">当前报告没有 Agent 调查回执。</div>';
  const findings=agent.investigation?.findings||[];
  const required=agent.investigation?.required_observations||[];
  const steps=[`<div class="ops-workflow-step complete"><span>受控 Agent 调查</span><strong>${esc(agent.status||agent.execution_status||'已记录')}</strong></div>`];
  findings.forEach(f=>{const technical=f.probe_id||f.kind||'FINDING';steps.push(`<details class="ops-workflow-step complete"><summary>${esc(probeDisplayName(technical))}</summary><div>${esc(f.text||f.summary||'已完成')}</div><small class="mono">${esc(technical)}${f.result_hash?' · '+esc(f.result_hash):''}</small></details>`)});
  required.forEach(x=>steps.push(`<div class="ops-workflow-step active"><span>待补充：${esc(probeDisplayName(x.probe_id||x))}</span><strong>等待</strong></div>`));
  return `<div class="ops-investigation">${steps.join('')}</div>`;
}

function enhanceOperationsResult(report,maintenance,job=state.job,visualization=state.visualization,diagnosticWorkspace=null){
  const container=$('#result-content');if(!container)return;
  container.querySelector('#ops-result-navigation')?.remove();
  state.maintenance=maintenance||null;
  const s=operatorStatusViewModel(job||{},maintenance||{}),reviewText=s.requiresHumanReview===true?'是':s.requiresHumanReview===false?'否':'未提供',candidate=maintenance?.fault_id||maintenance?.candidate||report?.candidate||report?.analysis?.candidate||'UNAVAILABLE';
  const component=maintenance?.component||maintenance?.component_id||report?.analysis?.component||'UNAVAILABLE';
  const evidence=maintenance?.evidence_summary;
  const graph=maintenance?.mechanism_graph;
  const agent=report?.agent_execution||report?.analysis?.agent_execution;
  const detection=maintenance?.detection_status||maintenance?.temporal_localization?.status||'UNAVAILABLE';
  const root=document.createElement('section');root.id='ops-result-navigation';root.className='ops-result-shell';
  root.innerHTML=`<div class="ops-result-tabs" role="tablist">${[['conclusion','结论'],['evidence','证据'],['mechanism','图谱与机理'],['simulation','仿真'],['agent','Agent 调查'],['technical','技术详情']].map(([id,label],i)=>`<button type="button" role="tab" class="ops-result-tab ${i===0?'active':''}" data-ops-result-tab="${id}">${label}</button>`).join('')}</div>
  <div class="ops-result-panel active" data-ops-result-panel="conclusion"><div class="ops-diagnosis-summary">${[['检测状态',detection],['定位组件',component],['故障模式',candidate==='UNAVAILABLE'?candidate:faultLabel(candidate)],['诊断状态',s.diagnosisDisposition],['人工复核要求',reviewText],['发布状态',s.releaseStatus]].map(([a,b])=>`<div><span>${a}</span><strong>${esc(b)}</strong></div>`).join('')}</div><p class="ops-boundary">执行状态：${esc(s.executionStatus)}。执行完成不会被转换为诊断成功；操作员需关注不等于后端已正式要求人工复核。</p></div>
  <div class="ops-result-panel" data-ops-result-panel="evidence">${formalEvidenceHtml(evidence)}<p class="ops-boundary">只显示正式 EpisodeEvidenceBundle；波形、RAG、Agent 推理和图表不补造正式证据。</p></div>
  <div class="ops-result-panel" data-ops-result-panel="mechanism"><h3>${esc(graph?.title||'图谱与机理')}</h3><p>${esc(graph?.query_backed?'本次内容来自受治理的只读图谱查询。':'当前没有本次只读图谱查询结果；下方详细工作区仅显示报告内已有机理材料。')}</p></div>
  <div class="ops-result-panel" data-ops-result-panel="simulation"><h3>仿真复核</h3><p>${maintenance?.simulation_review?esc(maintenance.simulation_review.summary||maintenance.simulation_review.status||'已有受治理仿真回执'):'当前报告没有受治理的仿真复核回执。'}</p><p class="ops-boundary">仿真拟合分不作为故障概率。</p></div>
  <div class="ops-result-panel" data-ops-result-panel="agent">${agentTimelineHtml(agent)}<p class="ops-boundary">Probe 回执不会改写正式 Evidence、诊断或发布状态。</p></div>
  <div class="ops-result-panel" data-ops-result-panel="technical"><div class="ops-advanced-details"><div><span>Job</span><strong class="mono">${esc(report?.job_id||job?.job_id||'UNAVAILABLE')}</strong></div><div><span>Execution</span><strong>${esc(s.executionStatus)}</strong></div><div><span>Diagnosis</span><strong>${esc(s.diagnosisDisposition)}</strong></div><div><span>Release</span><strong>${esc(s.releaseStatus)}</strong></div></div></div>`;
  container.prepend(root);
  $$('[data-ops-result-tab]').forEach(b=>b.onclick=()=>{$$('[data-ops-result-tab]').forEach(x=>x.classList.toggle('active',x===b));$$('[data-ops-result-panel]').forEach(x=>x.classList.toggle('active',x.dataset.opsResultPanel===b.dataset.opsResultTab))});
  operationsHooks.renderDiagnosticWorkspace?.({report,maintenance,job,visualization,diagnosticWorkspace});
}

function renderKnowledgeOperations(){
  const host=$('#knowledge-operations');if(!host)return;
  const m=state.maintenance||{},graph=m.mechanism_graph||{},candidate=m.fault_id||m.candidate||'UNAVAILABLE',component=m.component||m.component_id||'UNAVAILABLE';
  const tabs=[['system','系统图谱'],['propagation','故障传播'],['candidates','候选诊断'],['simulation','仿真复核'],['related','相关知识']];
  host.innerHTML=`<div class="ops-result-tabs">${tabs.map(([id,label],i)=>`<button type="button" class="ops-result-tab ${i===0?'active':''}" data-knowledge-tab="${id}">${label}</button>`).join('')}</div>
  <div class="ops-result-panel active" data-knowledge-panel="system"><div class="ops-diagnosis-summary"><div><span>组件</span><strong>${esc(component)}</strong></div><div><span>候选故障</span><strong>${esc(candidate)}</strong></div><div><span>图谱来源</span><strong>${graph.query_backed?'受治理只读查询':'报告内参考机理'}</strong></div></div></div>
  <div class="ops-result-panel" data-knowledge-panel="propagation"><p>${esc(graph.title||'当前任务没有故障传播图谱结果。')}</p></div>
  <div class="ops-result-panel" data-knowledge-panel="candidates"><p>${esc(candidate)}</p></div>
  <div class="ops-result-panel" data-knowledge-panel="simulation"><p>${m.simulation_review?esc(m.simulation_review.summary||m.simulation_review.status||'已有受治理仿真回执'):'当前任务没有受治理仿真回执。'}</p><p class="ops-boundary">这里只显示现有治理接口的结果，不提供任意函数或参数执行。</p></div>
  <div class="ops-result-panel" data-knowledge-panel="related"><p>相关文档显示在下方任务资料中；检索材料不参与核心诊断投票。</p></div>`;
  $$('[data-knowledge-tab]').forEach(b=>b.onclick=()=>{$$('[data-knowledge-tab]').forEach(x=>x.classList.toggle('active',x===b));$$('[data-knowledge-panel]').forEach(x=>x.classList.toggle('active',x.dataset.knowledgePanel===b.dataset.knowledgeTab))});
}

export { agentTimelineHtml, enhanceOperationsResult, operatorAttentionViewModel, operatorStatusViewModel, preflightRecognizedInputsHtml, probeDisplayName, probeRequestBody, renderKnowledgeOperations, renderOperationsWizard, renderOperationsWorkflow, setupOperationsHooks };
