import { $, $$, state, modeZh, artifactKindZh, esc, bytes, showToast, api, badgeFor } from './platform_core.js';
import { loadVisualization, renderReport } from './platform_report.js';
import { activeModeInfo, currentModeToolIds } from './platform_agent.js';
import { renderOperationsWizard, renderOperationsWorkflow } from './platform_operations.js';

const jobHooks={nav:()=>{},loadDashboard:async()=>{}};
function setupJobHooks(hooks={}){Object.assign(jobHooks,hooks)}

/* files: material-first role mapping, with legacy upload preserved */
const input=$('#file-input'),dz=$('#dropzone');
dz.onclick=()=>input.click();dz.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();input.click()}};
input.onchange=()=>addFiles(input.files);
['dragenter','dragover'].forEach(n=>dz.addEventListener(n,e=>{e.preventDefault();dz.classList.add('drag')}));
['dragleave','drop'].forEach(n=>dz.addEventListener(n,e=>{e.preventDefault();dz.classList.remove('drag')}));
dz.addEventListener('drop',e=>addFiles(e.dataTransfer.files));
const roleLabels={diagnosis_results:'诊断结果',log_reports:'日志报告',references:'参考文献'};
const roleShortLabels={diagnosis_results:'结果',log_reports:'日志',references:'文献'};
const roleSuffixes={diagnosis_results:/\.(json|csv|txt|md|pdf|docx)$/i,log_reports:/\.(log|json|csv|txt|md|pdf|docx)$/i,references:/\.(txt|md|pdf|docx)$/i};
function isDoc(name){return /\.(pdf|docx|txt|md)$/i.test(name)}
function isExecCandidate(name){return /\.(json|csv|zip|ya?ml)$/i.test(name)}
function guessRole(name){
  if(/(log|日志|event|告警|运行记录|报错)/i.test(name))return 'log_reports';
  if(/(reference|manual|paper|guideline|文献|手册|参考|规程)/i.test(name)||/\.(pdf|docx)$/i.test(name))return 'references';
  if(/(result|diagnos|分类|诊断|判定|检测)/i.test(name))return 'diagnosis_results';
  return isDoc(name)?'references':'diagnosis_results';
}
function fileSignature(){return state.files.map(f=>`${f.name}:${f.size}:${f.lastModified||0}:${state.inputRoles[f.name]||''}`).sort().join('|')+'::'+($('#equipment-scope')?.value||'')+'::'+state.inputFlow+'::'+state.agentMode+'::'+($('#task')?.value||'')}
function invalidatePreflight(){state.preflight=null;state.preflightJobId=null;state.preflightSignature=null;renderPreflight();renderInputChecks();renderOperationsWizard()}
function addFiles(list){
  let changed=false;
  for(const f of list){
    if(f.name==='analysis_input.json'&&state.inputFlow==='review'){showToast('系统会自动生成 analysis_input.json，请勿手动上传');continue}
    if(state.files.some(x=>x.name.toLowerCase()===f.name.toLowerCase())){showToast('已存在同名文件：'+f.name);continue}
    state.files.push(f);state.inputRoles[f.name]=guessRole(f.name);changed=true;
  }
  if(changed)invalidatePreflight();renderFiles();renderInputChecks();input.value='';
}
function reviewRoles(){const grouped={diagnosis_results:[],log_reports:[],references:[]};state.files.forEach(f=>{const role=state.inputRoles[f.name]||guessRole(f.name);if(grouped[role])grouped[role].push(f)});return grouped}
function canSubmitInputs(){
  if(state.inputFlow==='legacy')return state.files.some(f=>isExecCandidate(f.name));
  const groups=reviewRoles();return Object.entries(groups).every(([role,items])=>items.length>0&&items.every(f=>roleSuffixes[role].test(f.name)));
}
function renderFiles(){
  const box=$('#file-list');if(!state.files.length){box.innerHTML='';renderOperationsWizard();return}
  box.innerHTML=state.files.map((f,i)=>{
    const role=state.inputRoles[f.name]||guessRole(f.name),roleUi=state.inputFlow==='review'?`<label class="file-role-label">文件角色 <select class="field" data-file-role="${i}" aria-label="${esc(f.name)} 文件角色">${Object.entries(roleLabels).map(([id,label])=>`<option value="${id}" ${id===role?'selected':''}>${label}</option>`).join('')}</select></label>`:'';
    return `<div class="file-row"><div class="file-type ${isDoc(f.name)?'doc':''}">${state.inputFlow==='review'?esc(roleShortLabels[role]):(isDoc(f.name)?'资料':'数据')}</div><div><div class="file-name">${esc(f.name)}</div><div class="file-meta">${bytes(f.size)} · ${state.inputFlow==='review'?'确认材料角色后用于证据复核':(isDoc(f.name)?'参考文档':isExecCandidate(f.name)?'待后端识别的诊断工件':'非诊断附件')}</div>${roleUi}</div><button class="icon-btn" type="button" data-remove-file="${i}" aria-label="移除 ${esc(f.name)}">×</button></div>`;
  }).join('');
  $$('[data-file-role]').forEach(el=>el.onchange=()=>{const f=state.files[Number(el.dataset.fileRole)];state.inputRoles[f.name]=el.value;invalidatePreflight();renderFiles();renderInputChecks()});
  $$('[data-remove-file]').forEach(b=>b.onclick=()=>{const [removed]=state.files.splice(Number(b.dataset.removeFile),1);if(removed)delete state.inputRoles[removed.name];invalidatePreflight();renderFiles();renderInputChecks()});
  renderOperationsWizard();
}
$('#input-flow').onchange=e=>{state.inputFlow=e.target.value;invalidatePreflight();renderFiles();renderInputChecks()};
function renderPreflight(){const box=$('#preflight-result');if(!box)return;const p=state.preflight;if(!p){box.innerHTML='<div class="empty">尚未校验输入。</div>';renderOperationsWizard();return}const kind=artifactKindZh[p.artifact_kind]||p.artifact_kind||'未识别';const status=p.status==='PASS'?'通过':p.status==='WARNING'?'通过（有提醒）':'阻断';const cls=p.status==='BLOCKED'?'block':p.status==='WARNING'?'warn':'ok';const caps=(p.capability_matrix||[]).map(c=>`<div class="preflight-cap"><strong>${esc(c.label)} · ${esc(c.status==='AVAILABLE'?'可用':c.status==='DEGRADED'?'受限':'不可用')}</strong><span>${esc(c.reason||'')}</span></div>`).join('');const blockers=(p.blockers||[]).map(x=>`<div class="actual-check"><div class="actual-icon block">!</div><div><strong>${esc(x.code||'输入阻断')}</strong><span>${esc(x.message||'')}</span></div></div>`).join('');const warnings=(p.warnings||[]).map(x=>`<div class="actual-check"><div class="actual-icon warn">!</div><div><strong>${esc(x.code||'提醒')}</strong><span>${esc(x.message||'')}</span></div></div>`).join('');box.innerHTML=`<div class="preflight-head"><strong>${esc(kind)}</strong><span class="preflight-status ${cls}">${esc(status)}</span></div>${blockers}${warnings}${caps}`;renderOperationsWizard()}
function renderInputChecks(){
  const box=$('#input-checks');if(!box)return;
  const preflightCurrent=!!state.preflight&&state.preflightSignature===fileSignature();
  let rows=[];const mode=activeModeInfo();
  if(state.inputFlow==='review'){
    const groups=reviewRoles();
    for(const [role,files] of Object.entries(groups)){
      const invalid=files.filter(f=>!roleSuffixes[role].test(f.name));
      rows.push({ok:files.length>0&&!invalid.length,title:`${roleLabels[role]} ${files.length} 份${invalid.length?'（格式不符）':''}`,sub:files.length?files.map(f=>f.name).slice(0,3).join('、'):'至少需要 1 份材料'});
    }
  }else{
    const docs=state.files.filter(f=>isDoc(f.name)),execs=state.files.filter(f=>isExecCandidate(f.name));
    rows=[{ok:execs.length>0,title:execs.length?`旧版诊断工件 ${execs.length} 个`:'缺少诊断工件候选',sub:execs.length?execs.map(f=>f.name).slice(0,3).join('、'):'JSON / CSV / ZIP / YAML；具体格式以后端预检为准'},
      {ok:true,title:`补充参考资料 ${docs.length} 份`,sub:docs.length?docs.map(f=>f.name).slice(0,2).join('、'):'可选；不参与模型投票'}];
  }
  rows.push({ok:!!mode,title:mode?mode.display_name:'模式未加载',sub:mode?(state.inputFlow==='review'?'材料分析不执行 Agent 调查或航天器指令；模式仅保留任务元数据':`工具 ${currentModeToolIds().length} 个`):'等待后端'});
  rows.push({ok:preflightCurrent&&!!state.preflight?.can_start,title:preflightCurrent?(state.preflight?.can_start?'后端预检已通过':'后端预检已阻断'):'尚未执行后端预检',sub:preflightCurrent?(artifactKindZh[state.preflight?.artifact_kind]||state.preflight?.artifact_kind||'未识别'):'点击“校验输入”'});
  box.innerHTML=rows.map(r=>`<div class="actual-check"><div class="actual-icon ${r.ok?'ok':''}">${r.ok?'✓':'·'}</div><div><strong>${esc(r.title)}</strong><span>${esc(r.sub)}</span></div></div>`).join('');
  $('#preflight-btn').disabled=state.files.length===0;$('#start-btn').disabled=!(preflightCurrent&&state.preflight?.can_start&&state.preflightJobId);
  $('#run-note').textContent=!canSubmitInputs()?(state.inputFlow==='review'?'请补齐诊断结果、日志报告和参考文献，并校正文件格式':'请选择至少一组旧版诊断工件'):!preflightCurrent?'先执行后端预检，再启动分析':state.preflight?.can_start?'预检通过；将生成故障原因、处置措施、证据与待核查事项':'输入未通过预检，请修正阻断项';
}
function log(text){const box=$('#status-log');box.textContent+=`[${new Date().toLocaleTimeString('zh-CN')}] ${text}\n`;box.scrollTop=box.scrollHeight}
function setProgress(n,title,sub){$('#progress-bar').style.width=Math.max(0,Math.min(100,n))+'%';$('#progress-title').textContent=title;$('#progress-sub').textContent=sub}
function taskErrorText(message){const raw=String(message||'任务未成功完成');if(/Bad file descriptor|Errno 9/i.test(raw))return '检测到文件持久化层的描述符兼容错误。v0.8 已将核心诊断计算与审计工件持久化分阶段，并对有效普通文件的 Windows/CRT fsync EBADF 做受限兼容；若仍出现，请查看失败阶段是否为 CORE_DIAGNOSIS_PERSIST。原始错误：'+raw;return raw}
function rememberJob(job){if(!job?.job_id)return;state.job=job;state.currentJobId=job.job_id;localStorage.setItem('astrograph-current-job-v06',job.job_id);$('#current-job').textContent=job.job_id}
const stageZh={QUEUED:'排队中',RUNNING:'执行中',EXPANDING_INPUTS:'展开并校验输入',CLASSIFYING_ARTIFACT:'识别诊断工件',CORE_DIAGNOSIS:'核心诊断',CORE_DIAGNOSIS_PERSIST:'保存诊断审计结果',MATERIAL_KNOWLEDGE_FUSION:'RAG/图谱/案例联合复核',POST_DIAGNOSIS_KNOWLEDGE:'知识与大模型后处理',AGENT_INVESTIGATION:'增强诊断 / 扩展排查',VISUALIZATION:'生成维护可视化',REPORTING:'生成报告',COMPLETED:'已完成',FAILED:'失败'};
const executorZh={THREAD:'本机后台任务',PROCESS:'独立后台进程',INLINE:'当前进程'};
function llmStageText(d){const x=String(d?.llm_call_status||'NOT_STARTED');if(x==='NOT_STARTED')return '尚未需要';if(x==='PENDING')return '正在补充调查';if(x==='SUCCEEDED')return '调查增强已完成';if(x==='FAILED')return '增强调用失败';if(x==='NOT_CONFIGURED')return '未配置 · 使用核心诊断';return x}
function renderJobDiagnostics(d){const panel=$('#active-job-panel');if(!panel||!d)return;panel.hidden=false;$('#active-job-id').textContent=d.job_id||'—';$('#jobdiag-status').textContent=badgeFor(d.status||'UNKNOWN')[1];$('#jobdiag-stage').textContent=[stageZh[d.stage]||d.stage,d.stage_detail].filter(Boolean).join(' · ')||'—';$('#jobdiag-executor').textContent=executorZh[d.execution_backend]||d.execution_backend||'—';$('#jobdiag-llm').textContent=llmStageText(d);const err=$('#jobdiag-error');if(d.error){err.hidden=false;err.textContent=`${d.error.failed_stage?'失败阶段 '+d.error.failed_stage+' · ':''}${d.error.type||'Error'}: ${taskErrorText(d.error.message||'')}`}else{err.hidden=true;err.textContent=''}const logBox=$('#jobdiag-log');if(d.log_tail){logBox.hidden=false;logBox.textContent=d.log_tail}else{logBox.hidden=true;logBox.textContent=''}$('#retry-job').hidden=d.status!=='FAILED';$('#open-job-result').hidden=d.status!=='SUCCEEDED';renderOperationsWorkflow(d)}
async function loadJobDiagnostics(jobId=state.currentJobId){if(!jobId)return null;try{const d=await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}/diagnostics`);state.currentJobId=jobId;localStorage.setItem('astrograph-current-job-v06',jobId);$('#current-job').textContent=jobId;renderJobDiagnostics(d);return d}catch(e){if($('#active-job-panel'))$('#active-job-panel').hidden=true;return null}}
function stageProgress(job){const s=job?.stage||job?.status||'';const map={QUEUED:57,EXPANDING_INPUTS:62,CLASSIFYING_ARTIFACT:66,CORE_DIAGNOSIS:72,CORE_DIAGNOSIS_PERSIST:77,MATERIAL_KNOWLEDGE_FUSION:79,POST_DIAGNOSIS_KNOWLEDGE:82,AGENT_INVESTIGATION:88,VISUALIZATION:93,REPORTING:97,COMPLETED:100,FAILED:100};return map[s]??(job?.status==='RUNNING'?70:60)}
async function pollJobUntilDone(jobId){let current=null,lastStage='';for(let n=0;n<1200;n++){current=await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}`);rememberJob(current);const s=current.status,stage=current.stage||s,detail=current.stage_detail||'';setProgress(stageProgress(current),stage==='CORE_DIAGNOSIS'?'核心诊断':stage==='CORE_DIAGNOSIS_PERSIST'?'保存诊断审计工件':stage==='POST_DIAGNOSIS_KNOWLEDGE'?'知识 / LLM 后处理':stage==='AGENT_INVESTIGATION'?'Agent 调查':`执行状态：${s}`,detail||'等待任务状态更新');if(stage!==lastStage){log(`${stage}${detail?' · '+detail:''}`);lastStage=stage}if(s==='SUCCEEDED'||s==='FAILED')break;await new Promise(r=>setTimeout(r,700))}return current}
async function finishSuccessfulJob(jobId){setProgress(98,'读取报告','整理分析结论');const report=await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}/report`);state.report=report;setProgress(100,'分析完成','打开故障原因与建议');log(`任务模式：${report.agent_execution?.work_mode||report.work_mode}`);if(report.artifact_kind!=='RESULT_LOG_REVIEW')await loadVisualization(jobId);renderReport(report);await Promise.all([jobHooks.loadDashboard(true),loadJobDiagnostics(jobId)]);$('#progress-actions').hidden=true;jobHooks.nav('result')}
async function retryCurrentJob(){const jobId=state.currentJobId;if(!jobId)return;try{jobHooks.nav('progress',false);$('#status-log').textContent='';$('#progress-actions').hidden=true;setProgress(52,'重新启动任务','使用已上传输入重新执行');await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}/start`,{method:'POST'});log('任务已重新进入后台执行');const current=await pollJobUntilDone(jobId);if(current?.status!=='SUCCEEDED')throw new Error(taskErrorText(current?.error?.message||'任务未成功完成'));await finishSuccessfulJob(jobId)}catch(e){const msg=taskErrorText(e.message);setProgress(100,'任务未完成',msg);log('错误：'+msg);$('#progress-actions').hidden=false;await loadJobDiagnostics(jobId);showToast('诊断失败：'+msg)}}
$('#equipment-scope')?.addEventListener('input',()=>{if(state.preflight)invalidatePreflight()});$('#task').addEventListener('input',()=>{if(state.preflight)invalidatePreflight()});$('#refresh-job-diagnostics').onclick=()=>loadJobDiagnostics();$('#retry-job').onclick=retryCurrentJob;$('#progress-retry').onclick=retryCurrentJob;$('#progress-back').onclick=async()=>{await loadJobDiagnostics();jobHooks.nav('new')};$('#open-job-result').onclick=async()=>{if(!state.currentJobId)return;try{const report=await api(`/v1/platform/jobs/${encodeURIComponent(state.currentJobId)}/report`);state.report=report;await loadVisualization(state.currentJobId);renderReport(report);jobHooks.nav('result')}catch(e){showToast('结果读取失败：'+e.message)}};
$('#preflight-btn').onclick=async()=>{if(!canSubmitInputs()){showToast('请检查材料角色：需要诊断结果、日志报告、参考文献各至少一份，并确认格式');renderInputChecks();return;}const btn=$('#preflight-btn');btn.disabled=true;const original=btn.textContent;btn.textContent='校验中…';try{state.preflight=null;state.preflightJobId=null;state.preflightSignature=null;renderPreflight();renderInputChecks();const job=await api('/v1/platform/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:$('#task').value,work_mode:state.agentMode})});let done=0;for(const f of state.files){await api(`/v1/platform/jobs/${encodeURIComponent(job.job_id)}/files/${encodeURIComponent(f.name)}`,{method:'PUT',body:f});done++}
  if(state.inputFlow==='review'){
    const grouped=reviewRoles();
    const manifest={schema_version:'astrograph-result-log-review.v1',diagnosis_results:grouped.diagnosis_results.map(f=>f.name),log_reports:grouped.log_reports.map(f=>f.name),references:grouped.references.map(f=>f.name)};
    const scope=($('#equipment-scope')?.value||'').trim();if(scope)manifest.equipment_scope=scope;
    await api(`/v1/platform/jobs/${encodeURIComponent(job.job_id)}/files/analysis_input.json`,{method:'PUT',body:new Blob([JSON.stringify(manifest)],{type:'application/json'})});
  }
  const preflight=await api(`/v1/platform/jobs/${encodeURIComponent(job.job_id)}/preflight`,{method:'POST'});state.preflight=preflight;state.preflightJobId=job.job_id;state.preflightSignature=fileSignature();rememberJob({...job,artifact_kind:preflight.artifact_kind,preflight});renderPreflight();renderInputChecks();await loadJobDiagnostics(job.job_id);showToast(preflight.can_start?'输入预检通过':'输入预检已阻断')}catch(e){state.preflight=null;state.preflightJobId=null;state.preflightSignature=null;renderPreflight();renderInputChecks();showToast('输入校验失败：'+e.message)}finally{btn.textContent=original;renderInputChecks()}};
$('#start-btn').onclick=async()=>{if(!state.preflight?.can_start||!state.preflightJobId||state.preflightSignature!==fileSignature())return;const jobId=state.preflightJobId;try{jobHooks.nav('progress',false);$('#status-log').textContent='';$('#progress-actions').hidden=true;setProgress(53,'启动诊断',`${artifactKindZh[state.preflight.artifact_kind]||state.preflight.artifact_kind} · 已通过后端预检`);await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}/start`,{method:'POST'});log(`任务 ${jobId} 已通过预检并进入后台执行`);const current=await pollJobUntilDone(jobId);if(current?.status!=='SUCCEEDED')throw new Error(taskErrorText(current?.error?.message||'任务未成功完成'));await finishSuccessfulJob(jobId)}catch(e){const msg=taskErrorText(e.message);setProgress(100,'任务未完成',msg);log('错误：'+msg);$('#progress-actions').hidden=false;await loadJobDiagnostics(jobId);showToast('诊断失败：'+msg)}};

export { invalidatePreflight, renderInputChecks, renderFiles, renderPreflight, loadJobDiagnostics, rememberJob, setupJobHooks };
