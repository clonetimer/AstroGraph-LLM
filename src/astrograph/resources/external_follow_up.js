/* Task-bound external report questions. Model prose and questions are never evidence. */
(function(global){
'use strict';
const drafts=new Map();
let draftEpoch=0;
function clearDrafts(){drafts.clear();draftEpoch++;}
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const reasons={MODEL_GENERATION_FAILED:'模型调用未完成',MODEL_OUTPUT_INVALID:'模型输出结构检查未通过',CITATION_REFERENCE_INVALID:'引用检查未通过',LLM_NOT_CONFIGURED:'当前任务没有配置模型',INPUT_CHARACTER_LIMIT_EXCEEDED:'源文本超过分析容量',FOLLOW_UP_BUSY:'本任务已有追问正在处理，请稍后重试',FOLLOW_UP_REQUEST_CONFLICT:'请求编号与内容不一致，请重新载入当前对话',FOLLOW_UP_INPUT_CHANGED:'报告或输入已变化，请重新打开报告',FOLLOW_UP_LIMIT_REACHED:'已达到本任务追问次数上限',CONTEXT_TOO_LARGE:'历史上下文超出容量，请新建分析任务',MODEL_CONFIGURATION_UNAVAILABLE:'当前任务的模型配置不可用，请核对服务配置',CONTEXT_CHARACTER_LIMIT_EXCEEDED:'历史上下文超出容量，请新建分析任务',FOLLOW_UP_HISTORY_INVALID:'追问历史完整性检查未通过，请重新打开报告',FOLLOW_UP_REPORT_NOT_READY:'原始报告尚未准备好，请等待任务完成'};
function reason(code){return reasons[code]||'本次追问未完成，请核对模型状态或重新打开报告后重试';}
function turnsHtml(state){
 const sources=new Map((state?.sources||[]).map(s=>[s.source_id,s]));
 return (state?.turns||[]).map(t=>{
  const ok=t.status==='SUCCEEDED'&&t.citation_reference_validation==='PASS';
  const cites=ids=>(ids||[]).map(id=>{const s=sources.get(id);return s?`<details class="model-citation-source"><summary>${esc(s.filename)} · ${esc(s.locator)} · ${esc(id)}</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">${esc(s.text)}</pre><small>材料角色 ${esc(s.role)} · SHA-256 ${esc(s.sha256)}</small></details>`:'<p>引用来源不可用，请重新打开报告。</p>'}).join('');
  return `<article class="panel card card-pad external-follow-up-turn"><h3>追问</h3><p style="white-space:pre-wrap">${esc(t.question)}</p><h4>大模型追问 · 需专家复核</h4>${ok?(t.answer||[]).map(p=>`<section><p style="white-space:pre-wrap">${esc(p.text)}</p>${cites(p.citation_ids)}</section>`).join(''):`<p role="status">${esc(reason(t.reason_code))} · ${esc(t.status)}</p>`}${ok&&t.information_gaps?.length?`<h4>仍缺少的信息</h4><ul>${t.information_gaps.map(g=>`<li>${esc(g)}</li>`).join('')}</ul>`:''}</article>`;
 }).join('');
}
function requestId(){
 if(typeof global.crypto?.randomUUID==='function')return global.crypto.randomUUID();
 const bytes=new Uint8Array(16);global.crypto.getRandomValues(bytes);
 return 'question-'+Array.from(bytes,v=>v.toString(16).padStart(2,'0')).join('');
}
const defaultLimits={max_turns:12,max_question_characters:4000,max_history_characters:32000,max_source_characters:60000};
function create(options){
 const key=JSON.stringify([options.identity,options.jobId,options.inputHash]);
 const saved=drafts.get(key)||{},epoch=draftEpoch;
 let disposed=false,attempt=saved.attempt||null;
 const c={state:null,draft:saved.draft||'',busy:false,message:'',canAsk:options.canAsk};
 Object.defineProperty(c,'limits',{get:()=>Object.fromEntries(Object.entries(defaultLimits).map(([k,v])=>[k,Number.isInteger(c.state?.limits?.[k])&&c.state.limits[k]>0?c.state.limits[k]:v]))});
 Object.defineProperty(c,'limitsText',{get:()=>`每个问题最多 ${c.limits.max_question_characters} 字；本任务最多 ${c.limits.max_turns} 次记录；历史容量 ${c.limits.max_history_characters} 字；源文本容量 ${c.limits.max_source_characters} 字。`});
 const alive=()=>!disposed&&epoch===draftEpoch&&options.current();
 const remember=()=>{if(epoch!==draftEpoch)return;drafts.set(key,{draft:c.draft,attempt});if(drafts.size>24)drafts.delete(drafts.keys().next().value)};
 const emit=()=>{if(alive())options.onChange(c)};
 const path='/v1/platform/jobs/'+encodeURIComponent(options.jobId)+'/follow-up';
 const accept=data=>{
  if(!data||data.job_id!==options.jobId||data.input_sha256!==options.inputHash||!Array.isArray(data.turns)||!Array.isArray(data.sources))throw new Error('FOLLOW_UP_INPUT_CHANGED');
  c.state=data;
 };
 c.setDraft=value=>{c.draft=String(value);remember()};
 c.load=async()=>{
  if(c.busy||!alive())return;c.busy=true;c.message='正在读取追问记录…';emit();
  try{const data=await options.api(path);if(!alive())return;accept(data);c.message=c.canAsk?'可针对本报告继续追问；回答不会改写原报告。':'当前身份可查看追问记录；发送问题需要分析人员权限。'}
  catch(e){if(alive())c.message=reason(e.message)}
  finally{c.busy=false;emit()}
 };
 c.submit=async()=>{
  if(!alive()||c.busy||!c.canAsk)return;
  const question=c.draft.trim();if(!question||question.length>c.limits.max_question_characters){c.message=`请输入 1 至 ${c.limits.max_question_characters} 字的问题。`;emit();return}
  if(c.state?.turns.length>=c.limits.max_turns){c.message=reason('FOLLOW_UP_LIMIT_REACHED');emit();return}
  if(!attempt||attempt.question!==question)attempt={question,input_sha256:options.inputHash,request_id:requestId()};
  const submitted=attempt;remember();c.busy=true;c.message='大模型正在回答，请等待；可以继续编辑下一条问题。';emit();
  try{
   const data=await options.api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(submitted)});
   if(!alive())return;accept(data);
   const turn=data.turns.find(t=>t.request_id===submitted.request_id);
   if(!turn)throw new Error('FOLLOW_UP_INPUT_CHANGED');
   attempt=null;
   if(turn.status==='SUCCEEDED'&&turn.citation_reference_validation==='PASS'){
    if(c.draft.trim()===submitted.question)c.draft='';c.message='回答已保存；引用仅验证来源归属，仍需专家复核。';
   }else c.message=reason(turn.reason_code)+'；问题与历史记录已保留。';
   remember();
  }catch(e){if(alive())c.message=(reasons[e.message]?reason(e.message):'提交结果尚未确认；问题已保留。重试会复用请求编号。')}
  finally{c.busy=false;emit()}
 };
 c.dispose=()=>{remember();disposed=true};
 return c;
}
function mount(options){
 const host=options.host;
 host.innerHTML='<section class="panel card card-pad external-follow-up"><h2>针对本报告继续分析</h2><p>追问使用本任务的外部材料；问题、历史回答和模型分析都不作为新增证据，回答不改写原报告。</p><div data-follow-up-turns></div><form data-follow-up-form><label>分析问题<textarea data-follow-up-input rows="3" maxlength="4000" style="width:100%" placeholder="说明想比较的原因或需要核查的信息"></textarea></label><p><button type="submit" class="btn primary" data-follow-up-send>发送问题</button> <button type="button" class="btn" data-follow-up-reload>重新读取记录</button></p><p>Ctrl / ⌘ + Enter 发送</p><p data-follow-up-limits></p><p role="status" aria-live="polite" data-follow-up-status></p></form></section>';
 const input=host.querySelector('[data-follow-up-input]'),send=host.querySelector('[data-follow-up-send]'),reload=host.querySelector('[data-follow-up-reload]'),turns=host.querySelector('[data-follow-up-turns]');
 let rendered=null;
 const c=create({...options,onChange:value=>{
  if(input.value!==value.draft)input.value=value.draft;
  send.disabled=value.busy||!value.canAsk;send.textContent=value.busy?'正在处理…':'发送问题';reload.disabled=value.busy;
  host.querySelector('[data-follow-up-status]').textContent=value.message;
  input.maxLength=value.limits.max_question_characters;host.querySelector('[data-follow-up-limits]').textContent=value.limitsText;
  if(rendered!==value.state){turns.innerHTML=turnsHtml(value.state);rendered=value.state}
 }});
 input.value=c.draft;input.oninput=()=>c.setDraft(input.value);
 host.querySelector('[data-follow-up-form]').onsubmit=e=>{e.preventDefault();c.submit()};
 input.onkeydown=e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&!e.isComposing){e.preventDefault();c.submit()}};
 reload.onclick=()=>c.load();c.load();return c;
}
global.ExternalDiagnosisFollowUp={create,turnsHtml,mount,clearDrafts};
})(window);
