/* AstroGraph Mission Console · live API only, no simulated diagnosis or LLM text.
 * Authentication: in-memory bearer credential; HttpOnly expert cookie + CSRF.
 * Evidence-grounded chat: deterministic views over the immutable job report.
 */
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const html = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const qs = value => encodeURIComponent(String(value));
const statusZH = {SUCCEEDED:'已完成',FAILED:'失败',RUNNING:'执行中',QUEUED:'排队中',REVIEW_REQUIRED:'需复核',ABSTAIN:'无法确诊',PENDING:'待复核',IN_PROGRESS:'复核中',REVIEWED:'已记录复核意见',REVIEWED_WITH_GAPS:'已复核·待补证'};
const domainZH={ADCS:'姿态控制',EPS:'电源',THERMAL:'热控',PROP:'推进',COMM:'通信数传',CDH:'星载计算',PAYLOAD:'有效载荷'};
const kindZH={COMPONENT:'部件',FAULT_MODE:'故障',FAILURE_MECHANISM:'机理',SYMPTOM:'现象',VERIFICATION:'核查',ACTION:'措施',CONDITION:'条件'};
const roleRights={viewer:['viewer'],operator:['viewer','operator'],reviewer:['viewer','reviewer'],admin:['viewer','operator','reviewer','admin']};
const S={overview:null,demos:null,selectedEvent:null,token:'',identity:null,expert:null,engine:null,jobId:'',report:null,reportTab:'conclusion',graph:null,graphCode:'',files:{diagnosis_results:[],log_reports:[],references:[]},preflight:null,preflightId:'',busy:false,authMode:'local',knowledgeTab:'graph',graphEdgeId:'',accountInit:false,autoTimer:null,uploadRevision:0,openRequest:0,commandRequest:0,analysisEngines:[],analysisEngineId:'rules',analysisEngineLoaded:false};
let toastTimeout;
let welcomeMarkup='';
function resetConversation(){const host=$('#conversation');if(host)host.innerHTML=welcomeMarkup;}
function toast(message){const e=$('#toast');e.textContent=String(message);e.classList.add('show');clearTimeout(toastTimeout);toastTimeout=setTimeout(()=>e.classList.remove('show'),3000)}
async function api(path,init={}){
  const headers=new Headers(init.headers||{});
  if(S.token)headers.set('Authorization','Bearer '+S.token);
  if(init.expertWrite && S.expert?.csrf)headers.set('x-material-csrf',S.expert.csrf);
  const options={...init,headers,credentials:'same-origin'};delete options.expertWrite;
  const response=await fetch(path,options);
  let data;const mime=response.headers.get('content-type')||'';
  try{data=mime.includes('json')?await response.json():await response.text()}catch{data=null}
  if(!response.ok){const detail=data?.detail?.message||data?.detail||data?.error?.message||response.statusText;throw new Error(typeof detail==='string'?detail:JSON.stringify(detail))}
  return data;
}
function can(level){return (S.identity?.roles||[]).some(r=>(roleRights[r]||[]).includes(level))}
function useView(view){
  const min={'upload':'operator','review':'reviewer','graph-build':'reviewer','knowledge-review':'reviewer','settings':'admin','users':'admin'}[view];
  if(min&&!can(min)){toast('当前身份无权访问该页面');return}
  if(!$('#screen-'+view))return;
  $$('.screen').forEach(el=>el.classList.toggle('active',el.id==='screen-'+view));
  $$('.nav-link').forEach(el=>el.classList.toggle('active',el.dataset.route===view));
  const labels={command:'事件中心',workbench:'事件研判',assistant:'证据问答',upload:'新建分析',report:'决策报告',history:'任务档案',graph:'知识图谱',review:'待复核事项','graph-build':'文献建图','knowledge-review':'知识审核',settings:'模型与引擎',users:'人员与访问'};
  $('#page-label').textContent=labels[view]||'智能工作台';
  history.replaceState(null,'','#'+view);
  $('#sidebar').classList.remove('mobile-open');
  window.scrollTo(0,0);
  if(view==='command')loadCommand();if(view==='workbench')renderWorkbench();if(view==='upload'){loadDemoCards();loadAnalysisEngines()}if(view==='history')loadHistory();if(view==='graph')loadGraph();if(view==='report')renderReport();
  if(view==='review')loadExpert();if(view==='knowledge-review')loadKnowledgeReview();if(view==='settings')loadSettings();if(view==='users')loadUsers();
  document.title=`${labels[view]} · AstroGraph`;
}
function applyTheme(value){
  const preference=['dark','light','system'].includes(value)?value:'dark';
  const effective=preference==='system'?(window.matchMedia?.('(prefers-color-scheme: light)').matches?'light':'dark'):preference;
  document.body.dataset.theme=effective;$('#theme-select').value=preference;
  try{localStorage.setItem('astrograph-mission-theme-v1',preference)}catch{}
}
function applySize(value){const size=['compact','standard','large'].includes(value)?value:'standard';document.body.dataset.size=size;$('#font-select').value=size;try{localStorage.setItem('astrograph-mission-size-v1',size)}catch{}}
function roleLabel(){if(!S.identity)return '需要登录';const mode=S.identity.mode;if(mode==='local'||mode==='disabled')return '本机管理员';if(can('admin'))return '平台管理员';if(can('reviewer')&&!can('admin'))return '领域审核专家';if(can('operator'))return '分析人员';return '只读用户'}
function renderIdentity(){
  const yes=!!S.identity;$('#side-status').textContent=yes?'查看平台服务状态':'需要登录';$('#connection-light').classList.toggle('connected',yes);
  $('#identity-label').textContent=roleLabel();$('#identity-avatar').textContent=can('admin')?'管':can('operator')?'析':can('viewer')?'访':'?';
  $('#identity-info').textContent=yes?`当前身份：${roleLabel()} · ${S.identity.subject||'当前会话'}`:'请登录平台';
  $('#token-login-area').hidden=!!yes||S.authMode==='accounts';$('#identity-logout').hidden=!S.token;
  $$('[data-min-role]').forEach(el=>el.hidden=!can(el.dataset.minRole)||(el.dataset.accountOnly==='true'&&S.authMode!=='accounts'));
  const more=$('#nav-secondary');if(more&&!can('viewer'))more.open=false;
  const allowed=['command','workbench','assistant','report','history','graph',...(can('operator')?['upload']:[]),...(can('reviewer')?['review','graph-build','knowledge-review']:[]),...(can('admin')?['settings',...(S.authMode==='accounts'?['users']:[])]:[])];
  const active=$$('.screen').find(x=>x.classList.contains('active'))?.id?.replace('screen-','');
  if(active&&!allowed.includes(active))useView('command');
  if(!yes&&S.token)toast('访问令牌无效或权限不足');
}
function clearPrivateWorkspace(){
  // Switching accounts must not display the previous account's report or conversation.
  S.openRequest++; // Invalidates any older asynchronous report response across accounts.
  const citation=$('#evidence-detail');if(citation?.open)citation.close();
  S.commandRequest++; // Prevents stale event queues crossing account sessions.
  S.jobId=''; S.report=null; S.overview=null;S.selectedEvent=null; S.expert=null; S.engine=null; S.preflight=null; S.preflightId='';S.analysisEngines=[];S.analysisEngineId='rules';S.analysisEngineLoaded=false;
  const extra=$('#nav-secondary');if(extra)extra.open=false;
  S.files={diagnosis_results:[],log_reports:[],references:[]};S.uploadRevision++;clearTimeout(S.autoTimer);
  try{localStorage.removeItem('astrograph-mission-job-id')}catch{}
  resetConversation();
  const textarea=$('#chat-text');if(textarea)textarea.value='';
  const input=$('#mission-task');if(input)input.value='分析故障原因、相关证据、处置措施和待核查项目';
  for(const id of ['mission-diagnosis','mission-logs','mission-references','expert-credential','llm-key']){
    const node=$('#'+id);if(node)node.value='';
  }
  const scope=$('#mission-scope');if(scope)scope.value='';
  const start=$('#mission-start');if(start)start.disabled=true;
  const review=$('#review-queue');if(review)review.hidden=true;
  // Never display a previous account's queue, assignments or review details.
  for(const id of ['command-items','history-items','users-list','review-queue-items','review-form-box','knowledge-review-items','workbench-timeline','workbench-evidence','workbench-actions']){
    const node=$('#'+id);if(node)node.replaceChildren();
  }
  for(const id of ['metric-attention','metric-running','metric-done','metric-failed']){const node=$('#'+id);if(node)node.textContent='—'}
  const action=$('#command-action-content');if(action)action.textContent='请先选择事件';
  const next=$('#command-next-content');if(next)next.textContent='请先选择事件';
  renderFileSummary();renderReport();
}
async function refreshIdentity(){
  const previous=S.identity?.subject;
  try{
    const identity=await api('/v1/platform/mission/identity');
    if(previous && (previous!==identity.subject || S.identity?.mode!==identity.mode))clearPrivateWorkspace();
    S.identity=identity;renderIdentity();$('#identity-modal').open&&$('#identity-modal').close();
  }catch{
    clearPrivateWorkspace();S.identity=null;renderIdentity();
    if(S.authMode==='accounts'){if(!$('#account-modal').open)$('#account-modal').showModal()}else if(!$('#identity-modal').open)$('#identity-modal').showModal();
  }
}
async function loginToken(){const token=$('#identity-token').value.trim();if(!token)return toast('请输入访问令牌');if(token!==S.token)clearPrivateWorkspace();S.token=token;$('#identity-token').value='';await refreshIdentity();if(S.identity)useView('command')}
async function clearToken(){
  if(S.authMode==='accounts'&&S.token){try{await api('/v1/platform/auth/logout',{method:'POST'})}catch{}}
  clearPrivateWorkspace();S.token='';S.identity=null;try{sessionStorage.removeItem('astrograph-account-session')}catch{}
  renderIdentity();await refreshIdentity();
}
async function accountLogin(event){
  event.preventDefault();const msg=$('#account-message');msg.textContent='正在验证…';
  try{
    const username=$('#account-name').value.trim(),password=$('#account-pass').value;
    if(S.accountInit){
      await api('/v1/platform/auth/bootstrap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,password,display_name:$('#account-display').value.trim(),setup_secret:$('#account-setup-key').value})});
      S.accountInit=false;
      $('#account-modal-title').textContent='登录 AstroGraph';
      $('#account-submit').textContent='登录';
      $('#account-display-box').hidden=true;
      $('#account-setup-box').hidden=true;
    }
    const data=await api('/v1/platform/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,password})});
    clearPrivateWorkspace();S.token=data.token;try{sessionStorage.setItem('astrograph-account-session',S.token)}catch{}
    $('#account-pass').value='';$('#account-setup-key').value='';await refreshIdentity();
    if(S.identity){if($('#account-modal').open)$('#account-modal').close();useView('command');toast('登录成功')}else throw Error('未能读取登录身份');
  }catch(e){msg.textContent='登录失败：'+e.message}
}
async function openServiceStatus(){
  $('#service-modal').showModal();$('#service-details').textContent='正在读取…';
  try{const data=await api('/v1/platform/mission/services');renderServiceData(data)}
  catch(e){$('#service-details').textContent='连接失败：'+e.message}
}
function renderServiceData(data){
  const m=data.llm||{},labels={NOT_PROBED:'尚未检测',NOT_CONFIGURED:'未配置',AVAILABLE:'连接正常',DEGRADED:'连接受限',FAILED:'探测失败'};
  $('#service-details').innerHTML=`<div class="service-row"><span>LLM 状态</span><strong>${html(labels[m.status]||m.status||'未知')}</strong></div>
    <div class="service-row"><span>模型名称</span><strong>${html(m.model||'未设置')}</strong></div>
    <div class="service-row"><span>服务类型</span><strong>${html(m.provider||'—')}</strong></div>
    <div class="service-row"><span>最近探测</span><strong>${html(m.time||'从未检测')}</strong></div>
    ${m.reason?`<div class="service-row"><span>异常说明</span><strong>${html(m.reason)}</strong></div>`:''}`;
  $('#side-status').textContent=m.status==='AVAILABLE'?'服务正常':'服务 · '+(labels[m.status]||'待检测');$('#service-status-open').classList.toggle('attention',m.status!=='AVAILABLE');
}
async function probeServices(){const btn=$('#service-probe');if(!can('admin'))return;btn.disabled=true;
  try{const data=await api('/v1/platform/mission/services/probe',{method:'POST'});renderServiceData(data);toast('探测已完成')}
  catch(e){$('#service-details').textContent='探测失败：'+e.message}finally{btn.disabled=false}}
async function loadUsers(){
  if(!can('admin'))return;
  const host=$('#users-list');host.textContent='正在加载用户…';
  try{const data=await api('/v1/platform/accounts');const roles={viewer:'只读用户',operator:'分析人员',reviewer:'领域审核专家',admin:'平台管理员'};
    host.innerHTML=(data.users||[]).map(u=>`<div class="user-row" data-user-row="${Number(u.id)}"><div><strong>${html(u.display_name||u.username)}</strong><p>${html(u.username)} · ${html(u.active?'正常':'已停用')}</p></div><select aria-label="${html(u.username)}的角色" data-user-role="${Number(u.id)}">${Object.entries(roles).map(([role,label])=>`<option value="${role}" ${role===u.role?'selected':''}>${label}</option>`).join('')}</select><button type="button" class="btn" data-user-save="${Number(u.id)}">保存角色</button><button type="button" class="btn" data-user-toggle="${Number(u.id)}" data-active="${u.active?'1':'0'}">${u.active?'停用':'启用'}</button><button type="button" class="btn" data-user-password="${Number(u.id)}">重置密码</button></div>`).join('')||'<div class="empty">无用户</div>';
    $$('[data-user-save]').forEach(b=>b.onclick=()=>modifyUser(Number(b.dataset.userSave),{role:$(`[data-user-role="${b.dataset.userSave}"]`).value}));
    $$('[data-user-toggle]').forEach(b=>b.onclick=()=>{const active=b.dataset.active!=='1';if(confirm(`确定${active?'启用':'停用'}该账号？`))modifyUser(Number(b.dataset.userToggle),{active})});
    $$('[data-user-password]').forEach(b=>b.onclick=()=>{const password=prompt('输入至少12位的新密码（账号现有会话将失效）');if(password)modifyUser(Number(b.dataset.userPassword),{password})});
  }catch(e){host.textContent='读取失败：'+e.message}
}
async function modifyUser(id,payload){try{await api(`/v1/platform/accounts/${id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});toast('用户信息已更新');await loadUsers()}catch(e){toast('操作失败：'+e.message)}}
async function createUser(event){event.preventDefault();try{
  const payload={username:$('#new-user-name').value.trim(),display_name:$('#new-user-display').value.trim(),role:$('#new-user-role').value,password:$('#new-user-password').value};
  await api('/v1/platform/accounts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  $('#users-message').textContent='账号创建成功';$('#new-user-password').value='';await loadUsers();
}catch(e){$('#users-message').textContent='创建失败：'+e.message}}
async function loadKnowledgeReview(){
  if(!can('reviewer'))return;
  const box=$('#knowledge-review-items');box.textContent='正在加载审核候选…';
  try{const data=await api('/v1/platform/material-knowledge/pending?limit=50');const k=S.knowledgeTab;
    const items=k==='graph'?data.graph_proposals:k==='documents'?data.documents:data.cases;
    box.innerHTML=(items||[]).map((item,i)=>{const id=String(item.proposal_id||item.document_id||item.case_id||item.id||'');
      return `<form class="knowledge-card" data-review-id="${html(id)}"><h3>${html(item.title||item.name||item.fault_id||item.subject_name||id)}</h3><div class="knowledge-detail">${html(item.evidence_quote||item.source_summary_zh||item.source_locator||item.filename||'请对照原始材料核验')}</div><div class="field"><label>独立核验依据</label><input name="source" placeholder="原文页码、工单或正式规程编号" minlength="3" required></div><div class="knowledge-actions"><button type="submit" class="btn primary" name="decision" value="CONFIRM">审核通过</button><button type="submit" class="btn" name="decision" value="REJECT">驳回候选</button></div></form>`;
    }).join('')||'<div class="empty">暂无待审核条目。</div>';
    $$('[data-review-id]').forEach(f=>f.addEventListener('submit',async e=>{
      e.preventDefault();const id=f.dataset.reviewId,btn=e.submitter,decision=btn?.value||'REJECT';
      const source=f.querySelector('[name="source"]').value.trim();if(source.length<3)return toast('请填写原始依据');
      const base=k==='graph'?`graph/${qs(id)}`:k==='documents'?`documents/${qs(id)}`:`cases/${qs(id)}`;
      const body={decision,truth_source_ref:source,expected_snapshot:data.snapshot};
      if(k==='cases')body.verified_fault_id=String((items||[]).find(x=>x.case_id===id)?.fault_id||'')||null;
      const tail=k==='cases'?'/decision':'/decision';
      try{await api(`/v1/platform/material-knowledge/${base}${tail}`,{method:'POST',expertWrite:true,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});toast('审核记录已保存');await loadKnowledgeReview()}
      catch(err){toast('审核失败：'+err.message)}
    }));
  }catch(e){box.textContent='审核队列不可用：'+e.message}
}

function renderAnalysisEngines(){
  const host=$('#analysis-engine-options'),label=$('#analysis-engine-label');if(!host||!label)return;
  const selected=S.analysisEngines.find(x=>x.engine_id===S.analysisEngineId)||S.analysisEngines.find(x=>x.available)||{engine_id:'rules',name:'规则引擎',kind:'rules'};
  S.analysisEngineId=selected.engine_id;label.textContent=selected.name||selected.model||'规则引擎';
  host.innerHTML=S.analysisEngines.map(item=>`<button type="button" class="analysis-engine-option" role="option" data-analysis-engine="${html(item.engine_id)}" aria-selected="${item.engine_id===S.analysisEngineId?'true':'false'}" ${item.available?'':'disabled'}><span class="engine-kind-mark">${item.kind==='llm'?'AI':'R'}</span><span><strong>${html(item.name||item.model||'分析引擎')}</strong><small>${html(item.description||'')}</small></span><span class="engine-selected-mark">${item.engine_id===S.analysisEngineId?'已选':''}</span></button>`).join('')||'<div class="engine-menu-state">当前没有可用处理引擎</div>';
  $$('[data-analysis-engine]').forEach(button=>button.onclick=()=>selectAnalysisEngine(button.dataset.analysisEngine));
}
function selectAnalysisEngine(engineId){
  const engine=S.analysisEngines.find(x=>x.engine_id===engineId&&x.available);if(!engine)return;
  S.analysisEngineLoaded=true;if(S.analysisEngineId!==engineId){S.analysisEngineId=engineId;S.uploadRevision++;S.preflight=null;S.preflightId='';$('#mission-start').disabled=true;$('#preflight-results').textContent='';setUploadMessage(`已选择“${engine.name}”，需要重新检查材料。`);schedulePreflight()}
  renderAnalysisEngines();$('#analysis-engine-menu').hidden=true;$('#analysis-engine-toggle').setAttribute('aria-expanded','false');
}
async function loadAnalysisEngines(){
  const host=$('#analysis-engine-options');if(host)host.innerHTML='<div class="engine-menu-state">正在读取可用引擎…</div>';
  try{const data=await api('/v1/platform/analysis-engines');S.analysisEngines=(data.engines||[]).filter(x=>x.available!==false);const current=S.analysisEngineLoaded&&S.analysisEngines.some(x=>x.engine_id===S.analysisEngineId)?S.analysisEngineId:data.default_engine_id;S.analysisEngineId=current||'rules';S.analysisEngineLoaded=true;renderAnalysisEngines()}
  catch(error){S.analysisEngines=[{engine_id:'rules',kind:'rules',name:'规则引擎',description:'大模型配置不可用时的可靠兜底',available:true}];S.analysisEngineId='rules';renderAnalysisEngines();if(host)host.title=error.message}
}
function getRoleFiles(role){return S.files[role]||[]}
function schedulePreflight(){clearTimeout(S.autoTimer);const keys=['diagnosis_results','log_reports','references'];
  if(keys.every(k=>getRoleFiles(k).length)&&can('operator')){setUploadMessage('材料已齐，正在准备检查…');setAnalysisProcess('working','正在检查材料','确认文件能否支持本次分析');S.autoTimer=setTimeout(()=>preflight(),650)}
}
function setAnalysisProcess(mode,title,summary){
  const box=$('#analysis-process');if(!box)return;
  box.classList.remove('is-ready','is-working','is-error');if(mode)box.classList.add('is-'+mode);
  if(title)$('#analysis-process-title').textContent=title;if(summary)$('#analysis-process-summary').textContent=summary;
}
function renderAttachmentTray(){
  const tray=$('#attachment-tray');if(!tray)return;
  const labels={diagnosis_results:'诊断',log_reports:'日志',references:'资料'};
  const items=[];for(const role of ['diagnosis_results','log_reports','references'])for(const file of getRoleFiles(role))items.push(`<span class="attachment-chip"><b>${labels[role]}</b><span title="${html(file.name)}">${html(file.name)}</span></span>`);
  tray.innerHTML=items.join('')||'<span class="attachment-empty">尚未添加材料</span>';
}
function setFiles(role,files){S.uploadRevision++;S.files[role]=Array.from(files);S.preflight=null;S.preflightId='';$('#preflight-results').textContent='';$('#mission-start').disabled=true;$('#analysis-attach-menu').hidden=true;$('#analysis-attach-toggle').setAttribute('aria-expanded','false');renderFileSummary();schedulePreflight()}
function renderFileSummary(){const keys=['diagnosis_results','log_reports','references'];const targetIds={diagnosis_results:'diagnosis-summary',log_reports:'logs-summary',references:'references-summary'};for(const [k,target] of Object.entries(targetIds))$('#'+target).textContent=getRoleFiles(k).map(f=>f.name).join('、')||'尚未选择';$('#upload-file-count').textContent=Object.values(S.files).reduce((s,a)=>s+a.length,0);
  renderAttachmentTray();
  const complete=keys.every(k=>getRoleFiles(k).length);
  const step1=$('[data-upload-step="1"]'),step2=$('[data-upload-step="2"]'),step3=$('[data-upload-step="3"]');
  [step1,step2,step3].forEach(x=>x?.classList.remove('active','done'));
  if(complete){step1?.classList.add('done');step2?.classList.add(S.preflight?.can_start?'done':'active');if(S.preflight?.can_start)step3?.classList.add('active')}
  else step1?.classList.add('active');
  if(!complete){const labels={diagnosis_results:'诊断结果',log_reports:'运行日志',references:'参考资料'};
    const missing=keys.filter(k=>!getRoleFiles(k).length).map(k=>labels[k]);$('#upload-state').textContent='还需添加：'+missing.join('、');setAnalysisProcess('','等待材料',`还差 ${missing.join('、')}`);}
}
async function fetchDemos(){
  if(S.demos)return S.demos;
  const obj=await api('/platform/mission/assets/mission_demo_cases.json');
  S.demos=obj.cases||[];return S.demos;
}
async function loadDemoCards(){
  const el=$('#demo-cards');if(!el)return;
  const friendly={TC01:'反作用轮电流和温度同时升高，想确认是否存在摩擦异常',TC02:'加载时电池电压下降，想排查电池内阻是否增大',TC04:'推进系统出现陌生故障提示，想判断现有材料是否足够',TC06:'姿态与通信同时异常，想确认两者是否有关联'};
  try{const demos=await fetchDemos();el.innerHTML=demos.slice(0,2).map(c=>`<button type="button" class="demo-card" data-demo-inline="${html(c.id)}"><span>${html(friendly[c.id]||c.title)}</span><em>使用这个示例 →</em></button>`).join('');
    $$('[data-demo-inline]').forEach(b=>b.onclick=()=>{const c=demos.find(d=>d.id===b.dataset.demoInline);selectDemo(c)});
  }catch(e){el.innerHTML=`<button type="button" class="btn" id="demo-cards-retry">案例加载失败，点击重试</button>`;$('#demo-cards-retry').onclick=loadDemoCards;}
}
async function loadDemos(){try{const demos=await fetchDemos();$('#demo-list').innerHTML=demos.map(c=>`<button type="button" class="history-item" data-demo-case="${html(c.id)}"><strong>${html(c.id)} · ${html(c.title)}</strong><span>选择案例</span></button>`).join('');$$('[data-demo-case]').forEach(b=>b.onclick=()=>selectDemo(demos.find(c=>c.id===b.dataset.demoCase)))}catch(e){$('#demo-list').textContent='无法加载演示材料：'+e.message}}
function selectDemo(c){if(!c)return;S.uploadRevision++;const prompts={TC01:'反作用轮最近出现电流升高、绕组温度上升，但控制指令没有明显变化。请结合这些材料判断最值得排查的原因，告诉我现有证据够不够，以及下一步应该先检查什么。',TC02:'电池在负载升高时出现明显压降，卸载后又恢复。请判断是否可能与电池内阻增大有关，并说明还需要排除哪些情况。',TC04:'推进系统出现了一个资料库里没有的故障提示。请判断现有日志和资料能支持到什么程度，不要在证据不足时给出确定结论。',TC06:'姿态与通信分系统在接近的时间段都出现异常。请分别分析两个现象，并判断现有材料能否证明它们之间存在因果关系。'};const roles=['diagnosis_results','log_reports','references'];for(const role of roles){S.files[role]=(c.files?.[role]||[]).map(f=>new File([f.content],f.name,{type:'text/plain'}))}$('#mission-task').value=prompts[c.id]||$('#mission-task').value;$('#mission-scope').value=c.equipment_scope||'';for(const id of ['mission-diagnosis','mission-logs','mission-references'])$('#'+id).value='';S.preflight=null;S.preflightId='';$('#mission-start').disabled=true;$('#preflight-results').textContent='';renderFileSummary();$('#upload-state').textContent='示例材料已经放入输入框，正在自动检查。';setAnalysisProcess('working','正在阅读示例材料','确认材料是否齐全并理解问题');if($('#demo-modal').open)$('#demo-modal').close();schedulePreflight();toast('示例已载入')}
function fileList(){const all=[];for(const role of ['diagnosis_results','log_reports','references'])for(const f of getRoleFiles(role))all.push({role,file:f});return all}
function validUploads(){
  if(!can('operator'))throw new Error('当前身份没有新建任务权限');
  const files=fileList();if(files.length<3)throw new Error('请为诊断结果、运行日志和参考文献分别选择文件');
  for(const role of ['diagnosis_results','log_reports','references'])if(!getRoleFiles(role).length)throw new Error('缺少'+({diagnosis_results:'诊断结果',log_reports:'运行日志',references:'参考文献'}[role]));
  if(files.length>30)throw new Error('一次最多上传 30 份材料');
  const suffix={diagnosis_results:/\.(json|csv|txt|md|pdf|docx)$/i,log_reports:/\.(log|json|csv|txt|md|pdf|docx)$/i,references:/\.(txt|md|pdf|docx)$/i};
  for(const {role,file} of files){if(!suffix[role].test(file.name))throw new Error('格式不支持：'+file.name);if(file.size>32*1024*1024)throw new Error('文件大于 32 MiB：'+file.name)}
  if(files.reduce((s,item)=>s+item.file.size,0)>90*1024*1024)throw new Error('单次任务材料总量不得超过 90 MiB');
  return files;
}
function safeName(name){return name.normalize('NFKC').replace(/[\\/\x00-\x1F]/g,'_').slice(0,160)}
function mappedFiles(files){
  const entries=[],used=new Set(['analysis_input.json']);
  for(const {role,file} of files){let name=safeName(file.name);if(!name||name==='.'||name==='..')throw new Error('无效的上传文件名');const dot=name.lastIndexOf('.');const base=dot>0?name.slice(0,dot):name,ext=dot>0?name.slice(dot):'';let num=0;while(used.has(name.toLowerCase()))name=`${base.slice(0,125)}_${++num}${ext}`;used.add(name.toLowerCase());entries.push({role,file,name})}
  return entries;
}
function setUploadMessage(msg){$('#upload-state').textContent=msg}
async function preflight(){if(S.busy)return;const revision=S.uploadRevision;S.busy=true;$('#mission-preflight').disabled=true;$('#mission-start').disabled=true;try{
  const files=mappedFiles(validUploads());setUploadMessage('正在安全保存你添加的材料…');setAnalysisProcess('working','正在整理材料','先保存文件，再确认每类材料是否可读');
  const job=await api('/v1/platform/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:$('#mission-task').value.trim()||'分析故障原因与处置建议',work_mode:'normal',analysis_engine_id:S.analysisEngineId||'rules'})});
  const id=job.job_id;for(const item of files){setUploadMessage(`正在读取：${item.name}`);await api(`/v1/platform/jobs/${qs(id)}/files/${qs(item.name)}`,{method:'PUT',body:item.file})}
  const manifest={schema_version:'astrograph-result-log-review.v1',diagnosis_results:files.filter(f=>f.role==='diagnosis_results').map(f=>f.name),log_reports:files.filter(f=>f.role==='log_reports').map(f=>f.name),references:files.filter(f=>f.role==='references').map(f=>f.name)};
  const scope=$('#mission-scope').value.trim();if(scope)manifest.equipment_scope=scope;
  setAnalysisProcess('working','正在核对材料','检查诊断、日志和参考资料是否能够互相对应');await api(`/v1/platform/jobs/${qs(id)}/files/analysis_input.json`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(manifest)});
  const check=await api(`/v1/platform/jobs/${qs(id)}/preflight`,{method:'POST'});if(S.uploadRevision!==revision){setUploadMessage('材料已更新，正在重新检查最新输入。');return;}S.preflight=check;S.preflightId=id;
  $('#preflight-results').innerHTML=`<div class="check-row"><strong>${check.can_start?'输入预检通过':'输入预检未通过'}</strong></div>`;
  const problem=check.blockers||check.issues||check.errors||check.reason;setUploadMessage(check.can_start?'材料检查完成，可以开始分析。':(Array.isArray(problem)?problem.join('；'):String(problem||'有些材料还不能用于分析，请根据提示补充。')));
  setAnalysisProcess(check.can_start?'ready':'error',check.can_start?'材料已经准备好':'还需要补充材料',check.can_start?'点击“开始分析”，我会梳理证据和建议':'请查看下方提示后重新添加或检查材料');
  $('#mission-start').disabled=!check.can_start; S.jobId=id;remember(id);renderFileSummary();
}catch(e){if(S.uploadRevision===revision){setUploadMessage('材料检查没有完成：'+e.message);setAnalysisProcess('error','材料检查没有完成','请按提示检查文件后重试');toast('材料检查没有完成')}}finally{S.busy=false;$('#mission-preflight').disabled=false;if(S.uploadRevision!==revision)schedulePreflight()}}
function remember(id){S.jobId=id;try{if(['local','disabled'].includes(S.identity?.mode))localStorage.setItem('astrograph-mission-job-id',id);else localStorage.removeItem('astrograph-mission-job-id')}catch{}}
async function startJob(){if(S.busy||!S.preflight?.can_start||!S.preflightId)return;S.busy=true;$('#mission-start').disabled=true;$('#mission-progress').hidden=false;$('#mission-progress-inner').style.width='12%';try{
  const id=S.preflightId;setAnalysisProcess('working','正在开始分析','系统将按证据强弱逐步形成判断');await api(`/v1/platform/jobs/${qs(id)}/start`,{method:'POST'});
  const stagePct={QUEUED:18,EXPANDING_INPUTS:30,CLASSIFYING_ARTIFACT:42,MATERIAL_KNOWLEDGE_FUSION:64,POST_DIAGNOSIS_KNOWLEDGE:81,REPORTING:95,COMPLETED:100};
  const stageText={QUEUED:['分析任务已排队','马上开始读取材料'],EXPANDING_INPUTS:['正在阅读全部材料','把文件内容整理成可核对的证据'],CLASSIFYING_ARTIFACT:['正在识别异常线索','区分诊断判断、运行现象和参考依据'],MATERIAL_KNOWLEDGE_FUSION:['正在交叉核对证据','比较日志、诊断结果与参考资料是否一致'],POST_DIAGNOSIS_KNOWLEDGE:['正在判断证据边界','确认哪些结论有依据、哪些仍需人工核查'],REPORTING:['正在整理分析结果','用通俗语言汇总结论、风险和下一步'],COMPLETED:['分析完成','正在打开结果']};
  let completed=false;for(let tries=0;tries<900;tries++){
    const status=await api(`/v1/platform/jobs/${qs(id)}`);const label=status.stage_detail||status.stage||status.status;
    const spoken=stageText[status.stage]||['正在分析','正在处理当前材料'];setAnalysisProcess('working',spoken[0],spoken[1]);setUploadMessage(spoken[1]);
    $('#mission-progress-inner').style.width=`${stagePct[status.stage]||75}%`;
    if(status.status==='SUCCEEDED'){setAnalysisProcess('ready','分析完成','结论、证据和下一步已经整理好');await openJob(id);toast('分析结果已生成');completed=true;break}
    if(status.status==='FAILED')throw new Error(status.error?.message||'后端任务失败');
    await new Promise(resolve=>setTimeout(resolve,900));
  }
  if(!completed)throw new Error('任务等待超时，请到任务记录查询最新状态');
}catch(e){setUploadMessage('分析没有完成：'+e.message);setAnalysisProcess('error','分析没有完成','可以保留当前材料并稍后重试');toast('分析没有完成')}finally{S.busy=false;$('#mission-progress').hidden=true}}
async function openJob(id,destination='report'){
  const request=++S.openRequest;
  const report=await api(`/v1/platform/jobs/${qs(id)}/report`);
  // A slow response must never replace the event or the account selected afterward.
  if(request!==S.openRequest)return null;
  if(S.report?.job_id&&S.report.job_id!==report.job_id)resetConversation();
  S.report=report;remember(id);S.selectedEvent=id;S.reportTab='conclusion';
  $$('[data-report-tab]').forEach(b=>b.classList.toggle('active',b.dataset.reportTab==='conclusion'));
  renderReport();renderWorkbench();
  // Event queue already navigated synchronously. Do not pull users back if they left the workbench.
  if(destination==='report')useView('report');
  return report;
}
function enterEvent(item){
  if(!item?.job_id)return;
  const citation=$('#evidence-detail');if(citation?.open)citation.close();
  const id=item.job_id;
  ++S.openRequest; // Invalidate an outstanding request for any previous event.
  if(S.report?.job_id!==id){S.report=null;resetConversation();}
  S.selectedEvent=id;remember(id);renderReport();
  // Navigation is immediate; report I/O may still be in flight.
  useView('workbench');
  if(item.status!=='SUCCEEDED')return; // Failed/queued/running jobs still have a workbench.
  $('#workbench-evidence').innerHTML='<div class="loading-event" role="status">正在读取该事件的分析报告…</div>';
  openJob(id,'workbench').catch(err=>{
    if(S.selectedEvent!==id)return;
    const panel=$('#workbench-evidence');
    panel.innerHTML=`<div class="event-error" role="alert"><strong>无法加载此事件的报告</strong><p>${html(err.message)}</p><button type="button" class="btn" id="event-retry">重新加载报告</button></div>`;
    $('#event-retry').onclick=()=>enterEvent(item);
  });
}
// Highlight only literal diagnostic cues and observed key/value tokens from the source quote.
// Never calculate abnormal thresholds, infer a root cause or treat a highlight as verification.
function emphasizeEvidence(raw){
 const value=String(raw||'').slice(0,900);
 const pattern=/(\b(?:ERROR|WARN|WARNING|CRITICAL|FAIL(?:ED)?|ALARM)\b|[A-Za-z][A-Za-z0-9_.-]{0,46}[A-Za-z_][A-Za-z0-9_.-]{0,12}\s*[:=]\s*[-+]?(?:\d+(?:\.\d+)?)(?:\s*(?:°C|degC|mA|A|V|rpm|Hz|%))?|(?:异常|告警|升高|上升|超限|欠压|过压|失锁|饱和|泄漏|下降))/gi;
 let cursor=0,output='';
 for(const match of value.matchAll(pattern)){
   if(match.index<cursor)continue;
   output+=html(value.slice(cursor,match.index));
   output+='<mark class="evidence-spot">'+html(match[0])+'</mark>';
   cursor=match.index+match[0].length;
 }
 return output+html(value.slice(cursor));
}
function inspectEvidence(index){
 const e=findings()[index];if(!e)return;
 const entries=[['证据编号',e.evidence_id],['材料角色',({diagnosis_results:'诊断结果',log_reports:'运行日志',references:'参考文献'})[e.role]||e.role],['来源文件',e.filename],['原文定位',e.locator],['文件 SHA-256',e.sha256]];
 $('#evidence-source-meta').innerHTML=entries.map(([label,val])=>`<div><dt>${html(label)}</dt><dd title="${html(val||'未提供')}">${html(val||'未提供')}</dd></div>`).join('');
 $('#evidence-source-quote').innerHTML=emphasizeEvidence(e.excerpt||'');
 const dialog=$('#evidence-detail');if(!dialog.open)dialog.showModal();
}
const candidates=()=>S.report?.analysis?.result_log_review?.candidates||[];
const findings=()=>S.report?.analysis?.result_log_review?.evidence||[];
function renderModelExplanation(report,knowledge,review){
  const engine=report.analysis_engine||{},model=knowledge.model_explanation||{};
  if(engine.kind!=='llm')return '<div class="engine-result-note"><span>本次处理引擎</span><strong>规则引擎</strong><p>依据诊断结果、日志、上传文献和内置已审核图谱生成可复核结论。</p></div>';
  const engineName=engine.name||engine.model||'大模型';
  if(model.status!=='AVAILABLE_REQUIRES_EXPERT_REVIEW')return `<section class="panel model-result model-result-unavailable"><div class="model-result-head"><div><span>大模型处置结果</span><h2>${html(engineName)}</h2></div><span class="chip">已回退到规则引擎</span></div><p>所选模型没有返回通过结构与引用检查的结果。下方仍显示规则引擎结论，避免把失败或无法核验的文本当成处置依据。</p></section>`;
  const claims=(model.claims||[]).filter(x=>x.text);const actions=model.recommended_review_actions||[];
  const uploaded=((knowledge.local_retrieval||{}).hits||[]).length;
  const graphPaths=(review.candidates||[]).reduce((sum,item)=>sum+((item.graph_evidence_paths||[]).length),0);
  const claimRows=claims.slice(0,8).map(x=>`<li><p>${html(x.text)}</p>${(x.citation_ids||[]).length?`<div class="model-citations">${x.citation_ids.map(id=>`<code>${html(id)}</code>`).join('')}</div>`:''}</li>`).join('');
  const actionRows=actions.slice(0,8).map(x=>`<li>${html(x)}</li>`).join('');
  const limitations=(model.limitations||[]).slice(0,6).map(x=>`<li>${html(x)}</li>`).join('');
  return `<section class="panel model-result"><div class="model-result-head"><div><span>大模型综合处置结果</span><h2>${html(engineName)}</h2></div><span class="chip">需专家复核</span></div><div class="model-source-strip"><span>本次上传文献 <b>${uploaded}</b> 条命中</span><span>内置已审核图谱 <b>${graphPaths}</b> 条路径</span><span>引用检查 <b>${html(model.citation_reference_validation||'未评估')}</b></span></div><div class="model-summary"><span>综合判断</span><p>${html(model.summary||'模型未提供摘要。')}</p></div>${claimRows?`<div class="model-result-section"><h3>原因与依据</h3><ol>${claimRows}</ol></div>`:''}${actionRows?`<div class="model-result-section"><h3>建议处置与核查</h3><ol>${actionRows}</ol></div>`:''}${limitations?`<details class="technical-details"><summary>查看模型局限与使用边界</summary><ul class="list">${limitations}</ul></details>`:''}<div class="model-safety-line">这是基于证据的辅助研判，不构成自动控制授权；涉及设备操作时仍需人工审批。</div></section>`;
}
function renderReport(){const host=$('#report-content');for(const id of ['report-export-json','report-export-md'])$('#'+id).hidden=!S.report;if(!S.report){updateActionSummary();host.innerHTML='<div class="empty">请选择已完成的任务，或上传材料生成报告。</div>';$('#report-job-label').textContent='未选择任务';return}
  const report=S.report,cases=candidates(),review=report.analysis?.material_reasoning||{},knowledge=report.analysis?.knowledge_enrichment||{};
  $('#report-job-label').textContent=report.job_id||S.jobId;updateActionSummary();
  if(S.reportTab==='conclusion'){
    const cards=cases.length?cases.map(c=>{const causes=(c.cause_hypotheses||[]),measures=(c.handling_measures||[]),checks=(c.verification_checks||[]);return `<article class="panel report-block report-narrative"><div class="result-lead"><span>目前最值得核查的方向</span><strong>${html(c.fault_name||'尚未形成明确故障方向')}</strong></div><section class="report-section"><h3>为什么会这样判断</h3>${causes.map(x=>`<p>${html(x.text)}</p>`).join('')||'<p>现有材料还不足以支持具体的故障原因，因此系统没有补造解释。</p>'}</section><section class="report-section"><h3>建议先做什么</h3><ul class="list">${measures.map(x=>`<li><span class="measure-tag">${html(/检查|核对|查看|记录|比对|读取/.test(x.action||'')&&!/重启|切换|复位|卸载|执行|发送指令/.test(x.action||'')?'可以先核查':/重启|切换|复位|卸载|执行|发送指令/.test(x.action||'')?'需要审批':'分析建议')}</span>${html(x.action)}</li>`).join('')||'<li>先补充独立证据，再请领域专家复核。</li>'}</ul></section>${checks.length?`<section class="report-section"><h3>怎样确认</h3><ol class="list">${checks.map(x=>`<li>${html(x)}</li>`).join('')}</ol></section>`:''}<details class="detail-row technical-details"><summary>查看技术标识与系统状态</summary><div class="report-meta"><span class="chip">故障标识：${html(c.fault_id)}</span><span class="chip">${html(c.status==='PROVISIONAL_LOG_CUE'?'日志线索仍需核验':'还缺少独立日志佐证')}</span></div></details></article>`}).join(''):'<div class="panel"><div class="empty">现有材料没有形成可信的故障方向。系统不会仅凭零散日志或参考资料编造结论。</div></div>';
    const abstain=review.answer_policy?.all_candidates_abstained!==false;
    const modelNarrative=renderModelExplanation(report,knowledge,review);
    host.innerHTML=`${modelNarrative}<div class="warning-strip">${abstain?'先说结论：现在只能把它当作重点排查方向，还不能确认就是根因。':'材料已经给出较明确的排查方向，但仍要由专家结合实际设备状态确认。'}</div>${cards}`;
  }else if(S.reportTab==='evidence'){
    const rows=findings().map(e=>`<div class="evidence-row"><strong>${html(e.evidence_id||'')} · ${html(e.filename||'')} · ${html(e.locator||'')}</strong><p>${html(e.excerpt||'')}</p></div>`).join('');
    const ingest=knowledge.document_ingestion||[];
    host.innerHTML=`<div class="panel report-block"><h2>日志与文献证据</h2>${rows||'<div class="empty">尚无可定位的日志或文献片段。</div>'}</div><div class="panel report-block"><h2>文档解析与检索</h2>${ingest.map(d=>`<div class="evidence-row"><strong>${html(d.filename)}</strong><small>${html(d.parser||'')} · ${html(d.chunk_count||0)} 片段</small></div>`).join('')||'<div class="empty">当前没有文档解析信息。</div>'}</div>`;
  }else if(S.reportTab==='reasoning'){
    const rs=review.candidates||[];
    const gapNames={NO_APPROVED_MODEL_KG:'缺少同型号已审图谱',MISSING_INDEPENDENT_SYMPTOM_OBSERVATION:'缺少独立日志佐证',LOG_CONTRADICTION_REQUIRES_REVIEW:'日志中存在反证',PUBLIC_SEED_REQUIRES_MODEL_SPECIFIC_REVIEW:'公开图谱尚需型号级审核',CONFLICTED_NEGATED_CONDITIONAL_OR_REVERSE_PATH:'图谱存在争议或不能正向推断的路径',INVALID_APPROVED_RELATION_PROVENANCE:'已审关系的来源不完整'};
    const chains=rs.map(item=>{const hop=item.multi_hop||{};
      const paths=(hop.approved_paths||[]).slice(0,8).map((path,i)=>`<div class="hop-path"><div class="report-meta"><strong>知识路径 ${i+1}</strong><span class="chip">${html(path.hop_count)} 跳</span>${path.contains_reverse_association?'<span class="chip">含逆向关联 · 不能作因果证明</span>':''}</div><div class="hop-steps">${(path.hops||[]).map(step=>`<div class="hop-step"><div class="hop-step-top"><strong>${html(step.source?.name||'未知实体')} → ${html(step.target?.name||'未知实体')}</strong><span class="chip">${html(step.relation_type||'关系')}</span></div><div class="hop-reference">${html(step.evidence_quote||step.source_summary_zh||'来源有待补充')}</div><small>${html(step.edge_id||'')} · ${html(step.direction==='FORWARD'?'正向关系':'仅关联查询')}</small></div>`).join('')}</div></div>`).join('');
      const gaps=(hop.knowledge_gap_codes||[]).slice(0,8).map(code=>`<li>${html(gapNames[code]||code)}</li>`).join('');
      const decision=String(hop.decision||'');
      return `<section class="hop-candidate"><h3>${html(item.fault_name||item.fault_id)}</h3><span class="chip">${decision.startsWith('ABSTAIN')?`证据不足 · ${html(decision)}`:'仅支持后续核查'}</span>${paths||'<p>当前没有可用于确诊的完整已审核知识路径。</p>'}${gaps?`<details class="detail-row"><summary>待补充的证据</summary><ul class="list">${gaps}</ul></details>`:''}</section>`
    }).join('');
    host.innerHTML=`<div class="panel"><h2>多跳推理与证据链</h2>${chains||'<div class="empty">没有已审核的型号级图谱路径。</div>'}<div class="warning-strip">图谱路径属于待核查的知识关联，不构成工程确诊，也不授权自动控制。</div></div>`;
  }else{
    host.innerHTML=`<div class="panel"><h2>复核与溯源</h2><div class="evidence-row"><strong>任务编号</strong>${html(report.job_id)}</div><div class="evidence-row"><strong>报告状态</strong>${html(report.status||'SUCCEEDED')}</div><div class="evidence-row"><strong>材料来源</strong>${html(report.artifact_kind||'—')}</div><div class="evidence-row"><strong>自动控制授权</strong>否</div><div id="report-review-status" class="evidence-row">读取专家复核状态…</div>${can('reviewer')?'<button type="button" class="btn primary" id="audit-to-review">进入人工复核</button>':''}</div>`;
    if($('#audit-to-review'))$('#audit-to-review').onclick=()=>useView('review');api(`/v1/platform/jobs/${qs(S.jobId)}/manual-review/status`).then(x=>$('#report-review-status').textContent='人工复核：'+(statusZH[x.status]||x.status)+' · '+x.reviewed_count+'/'+x.total_count).catch(()=>$('#report-review-status').textContent='暂无人工复核信息');
  }
}
function updateActionSummary(){
  const host=$('#action-summary');if(!host)return;
  const report=S.report,cases=candidates();
  if(!report){host.hidden=true;return}host.hidden=false;
  const rows=report.analysis?.material_reasoning?.candidates||[];
  const decisions=rows.map(x=>x.multi_hop?.decision||'').filter(Boolean);
  const abstain=decisions.find(x=>x.startsWith('ABSTAIN'))||
    (report.analysis?.material_reasoning?.answer_policy?.all_candidates_abstained!==false?'ABSTAIN':'REVIEW_REQUIRED');
  $('#action-judgment').textContent=cases.length?(cases.length===1?(cases[0].fault_name||'发现一个重点排查方向'):`发现 ${cases.length} 个需要核查的方向`):'还不能判断具体故障';
  const code=decisions.every(x=>x&&!x.startsWith('ABSTAIN'))&&decisions.length?'REVIEW_REQUIRED':abstain;
  const evidence=$('#action-evidence');evidence.dataset.decisionCode=code;
  evidence.innerHTML=code.startsWith('ABSTAIN')?`还不够，不能据此确诊<code>${html(code)}</code>`:`已有线索，还需要人工确认<code>${html(code)}</code>`;
  $('#action-risk').textContent='如果直接按当前方向处置，仍有误判风险';
  $('#action-next').textContent=can('reviewer')?'请领域专家核对证据和原因':'补充独立证据后提交复核';$('#action-approval').textContent='未授权自动执行；处置前需要人工审批';
}

async function exportReport(format){
  if(!S.report||!S.jobId)return toast('请先选择已完成的分析报告');
  try{
    const id=S.jobId;
    const payload= format==='md'? await api(`/v1/platform/jobs/${qs(id)}/report.md`): S.report;
    const blob=new Blob([format==='md'?payload:JSON.stringify(payload,null,2)],{type:format==='md'?'text/markdown;charset=utf-8':'application/json;charset=utf-8'});
    const url=URL.createObjectURL(blob);const link=document.createElement('a');
    link.href=url;link.download=`${id}-report.${format==='md'?'md':'json'}`;
    document.body.append(link);link.click();link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1500);
  }catch(error){toast('导出失败：'+error.message)}
}
function materialReply(input){
  const prompt=input.trim(),report=S.report,cases=candidates();if(!report)return {title:'请选择分析报告',text:'目前没有可引用的分析报告。请先上传诊断结果、运行日志与参考文献，或从任务记录中选择已完成的分析。',route:'upload'};
  const cause=/(原因|根因|为什么|故障|机理)/.test(prompt),actions=/(措施|处置|怎么做|检查|核查|建议|步骤)/.test(prompt),evidence=/(证据|日志|文献|引文|来源)/.test(prompt),uncertainty=/(未知|拒答|确诊|不足|风险|反证|缺口)/.test(prompt),graph=/(图谱|多跳|推理|关联)/.test(prompt);
  if(![cause,actions,evidence,uncertainty,graph].some(Boolean))return {title:'当前问题没有足够的报告依据',text:'目前的对话仅用于查询已生成报告的原因、措施、证据、知识图谱和不确定性。未检索到相关字段时，我不会自行补充根因。',route:'report'};
  if(graph){const rows=report.analysis?.material_reasoning?.candidates||[];return {title:'图谱推理',text:rows.length?rows.map(r=>`${r.fault_id||'候选'}：${r.multi_hop?.decision||r.multi_hop?.status||'无已审完整路径'}`).join('\n'):'未形成已审多跳因果路径。',route:'report',tab:'reasoning'}}
  if(evidence){const lines=findings().slice(0,6).map(e=>`${e.evidence_id||''} ${e.filename||''} ${e.locator||''}：${String(e.excerpt||'').slice(0,110)}`);return {title:'可追溯的证据',text:lines.join('\n')||'报告没有可定位的引用条目，无法补造证据。',route:'report',tab:'evidence'}}
  if(uncertainty){const abstain=report.analysis?.material_reasoning?.answer_policy?.all_candidates_abstained!==false;return {title:'证据与确诊边界',text:abstain?'当前没有充足的同型号、已审核知识路径及独立日志验证，因此拒绝确诊。':'报告存在候选机理路径，但这不是实际工程确诊；仍需复核时序、设备状态和替代原因。',route:'report',tab:'reasoning'}}
  if(!cases.length)return {title:'尚无故障候选',text:'上游诊断没有给出可识别的候选。系统不会依据日志或参考文献自行编造故障编码。',route:'report'};
  const lines=cases.slice(0,8).map(c=>`${c.fault_name||c.fault_id}：${(actions?(c.handling_measures||[]).map(x=>x.action):(c.cause_hypotheses||[]).map(x=>x.text)).slice(0,3).join('；')||'证据不足'}`);
  return {title:actions?'建议措施（待工程审核）':'可能原因（未确诊）',text:lines.join('\n'),route:'report',tab:'conclusion'};
}
function say(kind,title,text,link){const box=document.createElement('div');box.className='message '+(kind==='user'?'user':'assistant');
  const who=document.createElement('div');who.className='who';who.textContent=kind==='user'?'我':'AstroGraph · 报告检索';
  const bubble=document.createElement('div');bubble.className='bubble';if(title){const bold=document.createElement('strong');bold.textContent=title;bubble.append(bold)}for(const t of String(text||'').split('\n')){const p=document.createElement('p');p.textContent=t;bubble.append(p)}box.append(who,bubble);
  if(link){const jump=document.createElement('button');jump.className='resource';jump.type='button';jump.textContent='查看报告与完整出处 →';jump.onclick=()=>{if(link.tab){S.reportTab=link.tab;$$('[data-report-tab]').forEach(b=>b.classList.toggle('active',b.dataset.reportTab===link.tab))}useView(link.route)};box.append(jump)}
  $('#conversation').append(box);box.scrollIntoView({block:'nearest',behavior:window.matchMedia?.('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});
}
function sendChat(){const q=$('#chat-text').value.trim();if(!q)return;$('#assistant-welcome')?.remove();say('user','',q);$('#chat-text').value='';const reply=materialReply(q);say('assistant',reply.title,reply.text,reply)}

const eventState = item => {
  const status=item.status||'UNKNOWN';
  if(status==='SUCCEEDED'&&item.manual_review_status==='REVIEWED')return {tone:'complete',label:'复核意见已记录',next:'查看决策报告',rank:4};
  if(status==='SUCCEEDED'&&item.manual_review_status==='REVIEWED_WITH_GAPS')return {tone:'attention',label:'复核后待补证',next:'补齐证据',rank:2};
  if(status==='FAILED')return {tone:'blocked',label:'执行失败',next:'核查失败原因',rank:0};
  if(['RUNNING','QUEUED'].includes(status))return {tone:'progress',label:statusZH[status]||status,next:'查看任务进度',rank:1};
  if(status==='SUCCEEDED'&&(item.requires_human_review||/ABSTAIN|REVIEW|INSUFFICIENT|UNKNOWN|BLOCK/.test(String(item.diagnosis_disposition||''))))return {tone:'attention',label:'待人工复核',next:'复核证据与原因',rank:2};
  if(status==='SUCCEEDED')return {tone:'complete',label:'已生成报告',next:'查看决策报告',rank:3};
  return {tone:'attention',label:'待启动',next:'检查材料',rank:1};
};
function renderCommand(){
  const o=S.overview,host=$('#command-items');if(!host)return;
  const guide=$('#command-first-run'),metrics=$('#command-metrics'),layout=$('#command-layout');
  if(!o){guide.hidden=true;metrics.hidden=true;layout.hidden=false;host.innerHTML='<div class="empty">事件队列暂不可用。<button type="button" class="btn" id="command-try">重试加载</button></div>';$('#command-try').onclick=loadCommand;return;}
  const counts=o.status_counts||{};
  $('#metric-attention').textContent=String(o.review_or_attention_count||0);
  $('#metric-running').textContent=String(o.active_job_count||0);
  $('#metric-done').textContent=String(counts.SUCCEEDED||0);
  $('#metric-failed').textContent=String(counts.FAILED||0);
  const filter=$('#command-filter').value;
  const events=(o.recent_jobs||[]).map(item=>({...item,state:eventState(item)}));events.sort((a,b)=>a.state.rank-b.state.rank);
  const firstRun=events.length===0;
  guide.hidden=!firstRun;metrics.hidden=firstRun;layout.hidden=firstRun;
  $('#command-create').hidden=firstRun; // The guide owns the sole primary new-analysis action.
  $('#command-first-viewer').hidden=can('operator');
  if(firstRun)return; // A single actionable guide replaces empty KPIs, table and current event.
  let shown=events;
  if(filter==='attention')shown=events.filter(item=>item.state.rank<3);
  if(filter==='running')shown=events.filter(item=>item.state.tone==='progress');
  if(filter==='completed')shown=events.filter(item=>item.status==='SUCCEEDED');
  if(filter==='attention'&&!shown.length)shown=events.slice(0,8); // never hide completed events in otherwise empty queue
  host.innerHTML=shown.map(item=>`<button type="button" class="command-event" data-command-job="${html(item.job_id)}" aria-label="打开任务 ${html(item.job_id)}"><span class="event-name"><strong>${html(item.fault_name||item.task||item.job_id)}</strong><small title="${html(item.job_id)}">${html(item.job_id)}</small></span><span class="event-status ${item.state.tone}">${html(item.state.label)}</span><span class="event-chevron" aria-hidden="true">→</span></button>`).join('')||`<div class="empty-state"><strong>本筛选条件下没有事件</strong><button type="button" class="btn primary" id="command-empty-create">新建分析 →</button></div>`;
  if($('#command-empty-create'))$('#command-empty-create').onclick=()=>useView('upload');
  $$('[data-command-job]').forEach(b=>{
    const item=events.find(x=>x.job_id===b.dataset.commandJob);
    b.onclick=()=>enterEvent(item);
    b.onfocus=()=>renderSelectedEvent(item);
    b.onpointerenter=()=>renderSelectedEvent(item);
  });
  // Do not duplicate the queue's status in three panels. Show details only for a focused/selected event.
  const selected=shown.find(item=>item.job_id===S.selectedEvent);
  renderSelectedEvent(selected||null);
}
function renderSelectedEvent(item){
 const el=$('#command-action-content'),next=$('#command-next-content');if(!el||!next)return;
 if(!item){el.innerHTML='<div class="event-choose">选择一条事件，查看证据与处置状态。</div>';next.innerHTML='<div class="next-step-empty">点击事件直接进入研判</div>';return;}
 const state=item.state||eventState(item);
 const abstain=String(item.diagnosis_disposition||'').startsWith('ABSTAIN');
 const evidence=abstain?'证据不足 · 无法确诊':item.status==='SUCCEEDED'?'报告需复核':'证据待获取';
 el.innerHTML=`<div class="event-focus"><h3>${html(item.fault_name||item.task||item.job_id)}</h3><div class="event-focus-id" title="${html(item.job_id)}">${html(item.job_id)}</div><div class="event-focus-grid"><div><span>证据</span><strong>${html(evidence)}</strong></div><div><span>风险</span><strong>未评定</strong></div><div><span>责任人</span><strong>未指派</strong></div></div></div>`;
 next.innerHTML=`<div class="next-priority"><span>下一步</span><strong>${html(state.next)}</strong></div><div class="next-action-bar"><span class="execution-lock">处置未授权自动执行</span><button type="button" class="btn primary" id="next-open-event">进入研判 →</button></div>`;
 $('#next-open-event').onclick=()=>enterEvent(item);
}
async function loadCommand(){
  const host=$('#command-items');if(!host)return;
  const request=++S.commandRequest,subject=S.identity?.subject;
  host.innerHTML='<div class="empty">正在获取任务…</div>';
  try{
    const overview=await api('/v1/platform/overview?limit=100');
    if(request!==S.commandRequest || subject!==S.identity?.subject)return;
    S.overview=overview;renderCommand();
  }catch(e){
    if(request!==S.commandRequest || subject!==S.identity?.subject)return;
    S.overview=null;renderCommand();toast('任务队列加载失败：'+e.message);
  }
}
function renderWorkbench(){
 const host=$('#workbench-timeline'),evidence=$('#workbench-evidence'),actions=$('#workbench-actions');if(!host||!evidence||!actions)return;
 const item=(S.overview?.recent_jobs||[]).find(x=>x.job_id===S.selectedEvent);
 const state=item?eventState(item):null;
 const report=S.report&&(!S.selectedEvent||S.report.job_id===S.selectedEvent)?S.report:null;
 const context=$('#workbench-context');
 const evidenceState=report?(report.analysis?.material_reasoning?.answer_policy?.all_candidates_abstained===false?'待核验':'证据不足 / 待核验'):(item?.status==='SUCCEEDED'?'报告读取中':'证据未就绪');
 if(context){context.innerHTML=`<div><span>任务阶段</span><strong>${html(state?.label|| (report?'报告已生成':'尚未选择事件'))}</strong></div><div><span>风险</span><strong>未评定</strong></div><div><span>责任人</span><strong>未指派</strong></div><div><span>证据状态</span><strong>${html(evidenceState)}</strong></div><div><span>下一动作</span><strong>${html(state?.next|| (report?'查看决策包':'选择事件'))}</strong></div>`;}
 if(!report){
   $('#workbench-job').textContent=S.selectedEvent||'未选择事件';
   host.innerHTML=item?`<ol class="case-timeline"><li>任务创建 <b>完成</b></li><li>当前阶段 <b>${html(state.label)}</b></li><li>报告 <b>${item.status==='FAILED'?'任务失败':'尚未就绪'}</b></li></ol>`:'<div class="empty">请先从任务指挥台选择事件</div>';
   const info=item? (item.status==='SUCCEEDED'?'正在读取事件报告…':`当前任务：${state.label}。${state.next}。`):'还没有可研判的事件。';
   evidence.innerHTML=`<div class="loading-event" role="status">${html(info)}</div>`;
   actions.innerHTML=`<h3>下一步</h3><p>${html(state?.next||'新建研判')}</p>${can('operator')?'<button id="workbench-create" type="button" class="btn primary">新建补充研判</button>':''}<button id="workbench-back" type="button" class="btn">返回任务队列</button>`;
   if($('#workbench-create'))$('#workbench-create').onclick=()=>useView('upload');$('#workbench-back').onclick=()=>useView('command');return;
 }
 $('#workbench-job').textContent=report.job_id||S.jobId;
 const cases=candidates(),proof=findings();const reason=report.analysis?.material_reasoning||{};
 const decisions=(reason.candidates||[]).map(c=>c.multi_hop?.decision||'');const abstain=decisions.some(d=>d.startsWith('ABSTAIN'))||reason.answer_policy?.all_candidates_abstained!==false;
 host.innerHTML=`<div class="section-label">任务状态</div><ol class="case-timeline"><li>上传与预检 <b>完成</b></li><li>证据检索 <b>完成</b></li><li>机理研判 <b>${abstain?'证据不足':'待核验'}</b></li><li>专家核验 <b>待复核</b></li></ol><div class="section-label">故障候选</div>${cases.map(c=>`<div class="fault-chip"><strong>${html(c.fault_name||c.fault_id)}</strong><small>${html(c.fault_id)}</small></div>`).join('')||'<div class="empty">未形成候选</div>'}`;
 const category=e=>{const name=String(e.filename||'').toLowerCase();const id=String(e.evidence_id||'').toUpperCase();if(id.startsWith('L')||/\.(log|csv)$/.test(name))return 'log';if(id.startsWith('R')||/\.(pdf|docx|md|txt)$/.test(name))return 'reference';return 'diagnosis'};
 const cueType={all:'全部证据',diagnosis:'诊断结论',log:'运行日志',reference:'参考文献'};
 const tabs=Object.entries(cueType).map(([type,label])=>`<button type="button" class="evidence-filter ${type==='all'?'active':''}" data-evidence-filter="${type}" aria-pressed="${type==='all'}">${label}</button>`).join('');
 const rows=proof.map((e,i)=>`<article class="evidence-tile" data-evidence-type="${category(e)}"><div class="evidence-meta-line"><strong>${html(e.evidence_id||'证据')}</strong><code title="${html(e.filename||'')}">${html(e.filename||'')}</code><span class="source-locator" title="原文定位">${html(e.locator||'定位待补充')}</span></div><p class="evidence-excerpt">${emphasizeEvidence(e.excerpt||'')}</p><button class="evidence-inspect" type="button" data-evidence-index="${i}" aria-label="查看证据 ${html(e.evidence_id||i)} 的原文定位">查看原文定位 →</button></article>`).join('')||'<div class="empty">暂无可定位的证据片段</div>';
 evidence.innerHTML=`<div class="decision-head"><span class="event-status ${abstain?'attention':'progress'}">${abstain?'ABSTAIN · 不能确诊':'候选分析 · 未确认'}</span><h3>${cases.length?html(cases[0].fault_name||cases[0].fault_id):'未识别到故障候选'}</h3><p>${cases.length?html(cases[0].cause_hypotheses?.[0]?.text||'需要独立证据验证可能机理'):'需补充上游诊断结果。'}</p></div><div class="evidence-focus"><div class="panel-head"><h3>证据链</h3><button class="btn small" type="button" id="workbench-evidence-open">查看完整出处 →</button></div><div class="evidence-filters" role="group" aria-label="按来源筛选证据">${tabs}</div><div class="evidence-list" id="workbench-evidence-list">${rows}</div><div class="evidence-none" id="workbench-evidence-none" hidden>该来源暂无可引用证据</div></div><div class="graph-workbench-link"><span>${abstain?'当前没有可证明根因的已审因果路径':'图谱关系待结合设备信息复核'}</span><button class="btn small" id="workbench-reasoning-open" type="button">查看多跳证据 →</button></div>`;
 $('#workbench-evidence-open').onclick=()=>showReportTab('evidence');$('#workbench-reasoning-open').onclick=()=>showReportTab('reasoning');
 $$('[data-evidence-filter]').forEach(button=>button.onclick=()=>{
   const active=button.dataset.evidenceFilter;$$('[data-evidence-filter]').forEach(x=>{x.classList.toggle('active',x===button);x.setAttribute('aria-pressed',String(x===button))});
   let visible=0;$$('#workbench-evidence-list [data-evidence-type]').forEach(row=>{row.hidden=active!=='all'&&row.dataset.evidenceType!==active;if(!row.hidden)visible++});$('#workbench-evidence-none').hidden=visible>0;
 });
 $$('[data-evidence-index]').forEach(button=>button.onclick=()=>inspectEvidence(Number(button.dataset.evidenceIndex)));
 const measures=cases.flatMap(c=>c.handling_measures||[]).slice(0,5);
 actions.innerHTML=`<div class="approval-block"><span>当前处置授权</span><strong>未批准自动执行</strong></div><h3>待完成事项</h3><ul class="workbench-todos">${measures.map(m=>`<li><span class="measure-tag">${html(/检查|核对|查看|记录|比对|读取/.test(m.action||'')&&!/重启|切换|复位|卸载|执行|发送指令/.test(m.action||'')?'可立即核验':/重启|切换|复位|卸载|执行|发送指令/.test(m.action||'')?'需审批后执行':'仅供研判')}</span>${html(m.action)}</li>`).join('')||'<li>补充日志或设备规程，完成专家核查。</li>'}</ul><button type="button" class="btn primary" id="workbench-review-open" ${can('reviewer')?'':'hidden'}>进入专家复核 →</button><button type="button" class="btn" id="workbench-report-open">查看决策包 →</button>${can('operator')?'<button type="button" class="btn" id="workbench-supplement">补充材料并新建研判</button>':''}`;
 if($('#workbench-review-open'))$('#workbench-review-open').onclick=()=>useView('review');$('#workbench-report-open').onclick=()=>useView('report');if($('#workbench-supplement'))$('#workbench-supplement').onclick=()=>{useView('upload');$('#mission-task').value='基于任务 '+(S.selectedEvent||S.jobId)+' 追加证据，重新分析故障原因与处置建议';};
}
function showReportTab(tab){S.reportTab=tab;$$('[data-report-tab]').forEach(b=>b.classList.toggle('active',b.dataset.reportTab===tab));useView('report')}

async function loadHistory(){const host=$('#history-items');try{const items=(await api('/v1/platform/overview?limit=60')).recent_jobs||[];host.innerHTML=items.map(i=>`<button type="button" class="history-item" data-history-job="${html(i.job_id)}"><div><strong>${html(i.task||i.job_id)}</strong><span>${html(i.job_id)} · ${html(i.artifact_kind||'分析任务')}</span></div><span>${html(statusZH[i.status]||i.status)}</span></button>`).join('')||'<div class="panel empty">还没有历史任务。</div>';$$('[data-history-job]').forEach(b=>b.onclick=async()=>{try{await openJob(b.dataset.historyJob)}catch(e){toast('报告不可用：'+e.message)}})}catch(e){host.textContent='读取失败：'+e.message}}
function graphSvg(nodes,edges,focusedId=''){const pos={FAULT_MODE:[120,163],COMPONENT:[120,68],FAILURE_MECHANISM:[360,68],SYMPTOM:[605,68],VERIFICATION:[605,259],ACTION:[365,259],CONDITION:[845,259]};
  const visible=nodes.filter(n=>pos[n.entity_type]);
  const box=new Set(visible.map(n=>n.id));
  const lines=edges.filter(r=>box.has(r.source)&&box.has(r.target)).map(r=>{const a=nodes.find(n=>n.id===r.source),b=nodes.find(n=>n.id===r.target),pa=pos[a.entity_type],pb=pos[b.entity_type];return `<path class="graph-edge ${focusedId&&r.id===focusedId?'focused':focusedId?'dimmed':''}" d="M${pa[0]},${pa[1]} Q${(pa[0]+pb[0])/2},${(pa[1]+pb[1])/2-19} ${pb[0]},${pb[1]}"/>`}).join('');
  const blocks=visible.map(n=>{const [x,y]=pos[n.entity_type];return `<g><title>${html(n.name)}</title><rect class="graph-node" x="${x-91}" y="${y-34}" width="182" height="68" rx="12"/><text class="graph-label" x="${x}" y="${y-7}" font-size="12" text-anchor="middle">${html(kindZH[n.entity_type]||n.entity_type)}</text><text class="graph-label" x="${x}" y="${y+18}" text-anchor="middle">${html(String(n.name).slice(0,12))}</text></g>`}).join('');
  return `<svg viewBox="0 0 955 330" role="img" aria-label="候选故障知识关系图">${lines}${blocks}</svg>`;
}
function renderGraph(){const data=S.graph;if(!data)return;const domain=$('#graph-domain').value,q=$('#graph-search').value.trim().toLowerCase();const entity=new Map(data.entities.map(x=>[x.id,x]));const relation=new Map(data.relations.map(x=>[x.id,x]));const sources=new Map(data.sources.map(x=>[x.id,x]));
  const clusters=data.clusters.filter(c=>(!domain||c.domain===domain)&&(!q||[c.title,c.code,...(c.edge_ids||[]).flatMap(id=>{const r=relation.get(id);return r?[entity.get(r.source)?.name,entity.get(r.target)?.name]:[]})].some(v=>String(v||'').toLowerCase().includes(q))));
  if(!clusters.some(c=>c.code===S.graphCode))S.graphCode=clusters[0]?.code||'';
  $('#graph-topics').innerHTML=clusters.map(c=>`<button class="graph-topic ${c.code===S.graphCode?'active':''}" data-graph-code="${html(c.code)}"><strong>${html(c.title)}</strong><span>${html(domainZH[c.domain]||c.domain)} · ${html(c.code)}</span></button>`).join('')||'<div class="empty">没有匹配主题。</div>';
  $$('[data-graph-code]').forEach(b=>b.onclick=()=>{S.graphCode=b.dataset.graphCode;renderGraph()});
  const cluster=clusters.find(c=>c.code===S.graphCode);if(!cluster){$('#graph-detail').innerHTML='<div class="empty">请选择一个主题。</div>';return}
  const raw=(cluster.edge_ids||[]).map(id=>relation.get(id)).filter(Boolean);
  const selected=$('#graph-evidence')?.value||'all';const edges=raw.filter(e=>selected==='all'||(selected==='supported'?(e.source_ids||[]).length>0:e.review_status!=='APPROVED'));
  const ids=new Set(edges.flatMap(e=>[e.source,e.target])),nodes=[...ids].map(id=>entity.get(id)).filter(Boolean);
  const src=new Set(edges.flatMap(e=>e.source_ids||[]));
  const row=edges.map(e=>`<div class="graph-relation ${S.graphEdgeId===e.id?'selected':''}"><button type="button" class="graph-relation-button" data-focus-edge="${html(e.id)}">查看路径 · ${html(entity.get(e.source)?.name||'未知')} → ${html(entity.get(e.target)?.name||'未知')}</button><div class="graph-source">${html(e.source_summary_zh||'文献归纳，尚未审核')} · ${html(e.source_locator||'')}</div></div>`).join('');
  $('#graph-detail').innerHTML=`<div class="report-meta"><h2>${html(cluster.title)}</h2><span class="chip">${html(cluster.code)}</span></div><div class="warning-strip">公开来源整理的参考图谱，尚未经型号级确认，不参与自动确诊。</div>${graphSvg(nodes,edges,S.graphEdgeId)}<div>${row}</div><h2>参考来源</h2>${[...src].map(id=>{const s=sources.get(id);if(!s)return '';let href=null;try{const u=new URL(s.url);if(['https:','http:'].includes(u.protocol))href=u.href}catch{}return `<div class="graph-source">${html(s.title||id)} ${href?`<a target="_blank" rel="noopener noreferrer" href="${html(href)}">查看原文 ↗</a>`:''}</div>`}).join('')}`;
  $$('[data-focus-edge]').forEach(b=>b.onclick=()=>{S.graphEdgeId=b.dataset.focusEdge;renderGraph();$('#graph-detail').scrollIntoView({behavior:'smooth',block:'nearest'})});
}
async function loadGraph(){if(S.graph){renderGraph();return}try{const data=await api('/v1/platform/knowledge/public-seed');if(data.graph_kind!=='UNREVIEWED_PUBLIC_REFERENCE_SEED'||data.is_flight_qualified!==false)throw new Error('图谱来源与审核状态不符合预期');S.graph=data;renderGraph()}catch(e){$('#graph-detail').textContent='读取知识图谱失败：'+e.message}}
async function loadExpert(){if(!can('reviewer'))return;try{S.expert=await api('/v1/platform/material-knowledge/review-session');$('#expert-status').textContent=S.expert.authenticated?'已登录：'+S.expert.reviewer:'专家未登录';$('#expert-login-panel').hidden=S.authMode==='accounts';$('#expert-credential').disabled=S.expert.authenticated;$('#expert-login').hidden=S.expert.authenticated;$('#expert-logout').hidden=!S.expert.authenticated||S.authMode==='accounts';$('#review-queue').hidden=!S.expert.authenticated;if(S.expert.authenticated){await loadReviewQueue();await renderReview()}else $('#review-form-box').innerHTML='<div class="empty">请先登录专家身份。</div>'}catch(e){$('#expert-status').textContent='专家会话不可用';$('#review-queue').hidden=true;$('#review-form-box').textContent=e.message}}
async function loadReviewQueue(){
  const host=$('#review-queue-items');
  try{const data=await api('/v1/platform/material-knowledge/report-reviews?limit=40');
    host.innerHTML=(data.items||[]).map(x=>`<button class="history-item" data-review-job="${html(x.job_id)}" type="button"><strong>${html(x.job_id)}</strong><span>${html(statusZH[x.status]||x.status)} · ${html(x.reviewed_count||0)}/${html(x.total_count||0)}</span></button>`).join('')||'<div class="empty">暂无待复核报告。</div>';
    $$('[data-review-job]').forEach(b=>b.onclick=()=>{remember(b.dataset.reviewJob);renderReview()});
  }catch(e){host.textContent='无法载入待办：'+e.message}
}

async function expertLogin(){const credential=$('#expert-credential').value;if(!credential)return toast('请输入专家凭据');try{await api('/v1/platform/material-knowledge/review-session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({credential})});$('#expert-credential').value='';await loadExpert();toast('专家登录成功')}catch(e){toast('登录失败：'+e.message)}}
async function expertLogout(){try{await api('/v1/platform/material-knowledge/review-session',{method:'DELETE',expertWrite:true});S.expert=null;await loadExpert()}catch(e){toast(e.message)}}
async function renderReview(){const host=$('#review-form-box');if(!S.jobId){host.innerHTML='<div class="empty">请从任务记录选择需要复核的报告。</div>';return}
  try{const detail=await api(`/v1/platform/material-knowledge/report-reviews/${qs(S.jobId)}`);host.innerHTML=`<h2>${html(S.jobId)} · ${html(statusZH[detail.status]||detail.status)}</h2>${(detail.candidates||[]).map((c,i)=>`<form class="review-candidate" data-review-fault="${html(c.fault_id)}"><h3>${html(c.fault_name||c.fault_id)}</h3><div class="field"><label for="review-choice-${i}">复核意见</label><select id="review-choice-${i}" name="decision"><option value="EVIDENCE_INSUFFICIENT">证据不足</option><option value="FOLLOW_UP_HYPOTHESIS">支持继续排查</option><option value="NOT_SUPPORTED">不支持候选</option></select></div><div class="field"><label for="review-evidence-${i}">独立核验依据</label><input id="review-evidence-${i}" name="evidence_ref" placeholder="文献编号、检查单或任务记录" /></div><div class="field"><label for="review-note-${i}">复核说明</label><textarea id="review-note-${i}" name="note" rows="3" required minlength="8" placeholder="填写至少 8 个字的具体意见"></textarea></div><button class="btn primary" type="submit">保存复核记录</button></form>`).join('')}`;
  $$('[data-review-fault]').forEach(f=>f.onsubmit=async e=>{e.preventDefault();const form=new FormData(f);const body={fault_id:f.dataset.reviewFault,decision:form.get('decision'),evidence_ref:String(form.get('evidence_ref')||''),note:String(form.get('note')||''),report_sha256:detail.report_sha256,expected_revision:detail.revision};try{await api(`/v1/platform/material-knowledge/report-reviews/${qs(S.jobId)}/decisions`,{method:'POST',expertWrite:true,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});toast('已记录复核意见');await loadReviewQueue();await renderReview()}catch(err){toast('保存失败：'+err.message)}})
  }catch(e){host.textContent='无法载入报告复核：'+e.message}}
async function runGraphBuild(){if(!S.expert?.authenticated){await loadExpert();if(!S.expert?.authenticated)return toast('需要专家身份才能构建候选图谱')};const files=[...$('#graph-build-files').files],scope=$('#graph-build-scope').value.trim();if(!scope||files.length===0)return toast('请填写设备构型并选择参考文献');if(files.length>8)return toast('最多 8 份文献');const body=new FormData();for(const file of files)body.append('reference_files',file,file.name);body.append('equipment_scope',scope);body.append('max_chunks',String($('#graph-build-chunks').value||6));$('#graph-build-run').disabled=true;$('#graph-build-status').textContent='正在抽取候选实体与关系…';
  try{
    const res=await api('/v1/platform/material-knowledge/graph/build',{method:'POST',expertWrite:true,body});
    const statusLabels={SUCCEEDED:'候选关系已生成',PARTIAL:'部分片段处理失败',MODEL_NOT_CONFIGURED:'未配置图谱抽取模型',DISABLED:'图谱构建已停用',NO_VALID_RELATIONS:'未抽取到可审核的关系'};
    const count=Number(res.proposal_count||0),processed=Number(res.chunks_processed||0),total=Number(res.chunks_considered||0);
    const summary=[statusLabels[res.status]||'构建状态：'+res.status,`候选关系 ${count} 条 · 已处理片段 ${processed}/${total}`];
    if(res.chunks_truncated)summary.push(`另有 ${res.chunks_truncated} 个片段未分析`);
    if(res.extraction_errors?.length)summary.push(`部分片段抽取异常 ${res.extraction_errors.length} 条`);
    if(count)summary.push('已进入待审核区，不会自动写入正式图谱。');
    $('#graph-build-status').textContent=summary.join('。');
    if(count)toast('待审核候选关系已生成');
    else if(res.status==='MODEL_NOT_CONFIGURED')toast('请先在系统设置中配置图谱抽取模型');
  }catch(e){$('#graph-build-status').textContent='构建失败：'+e.message}
  finally{$('#graph-build-run').disabled=false}}
async function loadSettings(){if(!can('admin'))return;try{const [llm,engine]=await Promise.all([api('/v1/platform/settings/llm'),api('/v1/platform/settings/knowledge-engine')]);$('#llm-provider').value=llm.provider||'local-openai-compatible';$('#llm-url').value=llm.base_url||'';$('#llm-model').value=llm.model||'';$('#llm-key').value='';$('#llm-persist').checked=false;$('#llm-profile-name-field').hidden=true;$('#llm-mode-state').textContent=llm.persists_across_restart?'当前配置已设为持久化默认服务':'当前为进程配置；如需重启后继续使用，请勾选保存为默认服务';S.engine=engine;fillEngine(engine);}catch(e){$('#llm-result').textContent='配置读取失败：'+e.message}}
function fillEngine(record){
  const settings=record?.settings||{};
  const ids={embedding_model:'engine-embedding',vector_backend:'engine-backend',qdrant_url:'engine-qdrant-url',qdrant_collection:'engine-qdrant-collection',ocr_engine:'engine-ocr',ocr_languages:'engine-ocr-lang',graph_llm_model:'engine-graph-model',vision_endpoint:'engine-vision-url',vision_model:'engine-vision-model',vision_pages:'engine-vision-pages',rag_documents:'engine-docs',rag_sections:'engine-sections',rag_hits:'engine-hits'};
  for(const [field,id] of Object.entries(ids))$('#'+id).value=String(settings[field]??'');
  for(const [field,id] of [['community_llm_enabled','engine-community'],['graph_build_enabled','engine-graph-build'],['vision_enabled','engine-vision']])$('#'+id).checked=!!settings[field];
  $('#engine-state').textContent='向量存储：'+(settings.vector_backend||'sqlite')+' · 配置版本：'+(record?.revision||'默认');
}
function enginePayload(){
  if(!S.engine)throw new Error('请先读取当前引擎配置');
  const settings={...S.engine.settings};
  for(const [field,id] of Object.entries({embedding_model:'engine-embedding',vector_backend:'engine-backend',qdrant_url:'engine-qdrant-url',qdrant_collection:'engine-qdrant-collection',ocr_engine:'engine-ocr',ocr_languages:'engine-ocr-lang',graph_llm_model:'engine-graph-model',vision_endpoint:'engine-vision-url',vision_model:'engine-vision-model'}))settings[field]=$('#'+id).value.trim();
  for(const [field,id] of Object.entries({community_llm_enabled:'engine-community',graph_build_enabled:'engine-graph-build',vision_enabled:'engine-vision'}))settings[field]=$('#'+id).checked;
  for(const [field,id] of Object.entries({vision_pages:'engine-vision-pages',rag_documents:'engine-docs',rag_sections:'engine-sections',rag_hits:'engine-hits'}))settings[field]=Number($('#'+id).value||0);
  return settings;
}
async function saveEngine(e){e.preventDefault();if(!can('admin'))return;$('#engine-result').textContent='正在保存配置…';try{
    const data=await api('/v1/platform/settings/knowledge-engine',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({settings:enginePayload(),expected_revision:S.engine.revision})});
    S.engine=data;fillEngine(data);$('#engine-result').textContent='知识引擎配置已保存；后续任务使用最新版本。';toast('配置已更新');
  }catch(err){$('#engine-result').textContent='保存失败：'+err.message}
}
async function probeEngine(){if(!can('admin'))return;$('#engine-result').textContent='正在检测本地依赖…';try{
  const data=await api('/v1/platform/settings/knowledge-engine/probe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(enginePayload())});
  $('#engine-result').textContent=JSON.stringify(data).slice(0,1500)
}catch(err){$('#engine-result').textContent='检测失败：'+err.message}}
async function saveLLM(e){
  e.preventDefault();if(!can('admin'))return;
  const body={enabled:true,provider:$('#llm-provider').value,base_url:$('#llm-url').value.trim(),model:$('#llm-model').value.trim(),api_key:$('#llm-key').value||null};
  const persist=$('#llm-persist').checked;
  try{
    if(persist){
      const name=$('#llm-profile-name').value.trim();if(!name)throw new Error('请为默认模型配置填写名称');
      const result=await api('/v1/platform/settings/llm/profiles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,settings:body,make_default:true,activate:true})});
      $('#llm-mode-state').textContent='已保存为默认配置 · 重新启动仍可读取';
      $('#llm-result').textContent=result.secret_persisted===false?'模型已激活，但密钥未安全持久化；重启后可能需要重新配置密钥。':'模型已检测通过，并已设置为默认配置。';
    }else{
      await api('/v1/platform/settings/llm',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      $('#llm-mode-state').textContent='仅当前服务进程生效，重启可能丢失';$('#llm-result').textContent='临时模型配置已生效。';
    }
    $('#llm-key').value='';toast('模型配置已更新');
  }catch(err){$('#llm-result').textContent='保存失败：'+err.message}
}
async function testLLM(){const body={enabled:true,provider:$('#llm-provider').value,base_url:$('#llm-url').value.trim(),model:$('#llm-model').value.trim(),api_key:$('#llm-key').value||null};$('#llm-result').textContent='正在检测模型…';try{const data=await api('/v1/platform/settings/llm/test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('#llm-result').textContent=JSON.stringify(data).slice(0,1200)}catch(e){$('#llm-result').textContent='检测失败：'+e.message}}
function bind(){
  welcomeMarkup=$('#assistant-welcome')?.outerHTML||'';
  $('#conversation').addEventListener('click',e=>{const button=e.target.closest('[data-chat-suggest]');if(button){$('#chat-text').value=button.dataset.chatSuggest;sendChat()}});
  let pTheme='light',pFont='standard';try{pTheme=localStorage.getItem('astrograph-mission-theme-v1')||pTheme;pFont=localStorage.getItem('astrograph-mission-size-v1')||pFont}catch{}applyTheme(pTheme);applySize(pFont);
  $('#theme-select').onchange=e=>applyTheme(e.target.value);$('#font-select').onchange=e=>applySize(e.target.value);window.matchMedia?.('(prefers-color-scheme: light)')?.addEventListener?.('change',()=>{if($('#theme-select').value==='system')applyTheme('system')});
  $$('.nav-link').forEach(b=>b.onclick=()=>useView(b.dataset.route));$('#mobile-nav').onclick=()=>$('#sidebar').classList.toggle('mobile-open');
  $('#identity-trigger').onclick=()=>{if(S.authMode==='accounts'){if(S.identity)$('#identity-modal').showModal();else $('#account-modal').showModal()}else $('#identity-modal').showModal()};$('#identity-login').onclick=loginToken;$('#identity-logout').onclick=clearToken;$('#account-form').onsubmit=accountLogin;
  $('#command-first-create').onclick=()=>useView('upload');$('#command-first-demo').onclick=()=>{useView('upload');$('#demo-cards').scrollIntoView({block:'center'});};$('#command-refresh').onclick=loadCommand;$('#command-filter').onchange=renderCommand;$('#command-create').onclick=()=>useView('upload');$('#command-open-history').onclick=()=>useView('history');$('#command-open-review').onclick=()=>useView('review');$('#workbench-open-report').onclick=()=>useView('report');$('#identity-open-users').onclick=()=>{if($('#identity-modal').open)$('#identity-modal').close();useView('users')};
  $('#chat-form').onsubmit=e=>{e.preventDefault();sendChat()};$('#chat-text').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();sendChat()}};
  $('#chat-add-material').onclick=()=>useView('upload');$('#chat-open-report').onclick=()=>useView('report');
  $('#analysis-thinking-toggle').onclick=()=>{const body=$('#analysis-thinking-body'),expanded=!body.hidden;body.hidden=expanded;$('#analysis-thinking-toggle').setAttribute('aria-expanded',String(!expanded))};
  $('#analysis-attach-toggle').onclick=e=>{e.stopPropagation();const menu=$('#analysis-attach-menu'),open=!menu.hidden;menu.hidden=open;$('#analysis-attach-toggle').setAttribute('aria-expanded',String(!open))};
  $('#analysis-engine-toggle').onclick=e=>{e.stopPropagation();const menu=$('#analysis-engine-menu'),open=!menu.hidden;menu.hidden=open;$('#analysis-engine-toggle').setAttribute('aria-expanded',String(!open));if(!open&&!S.analysisEngines.length)loadAnalysisEngines()};
  $('#analysis-attach-menu').onclick=e=>e.stopPropagation();$('#analysis-engine-menu').onclick=e=>e.stopPropagation();document.addEventListener('click',()=>{const attach=$('#analysis-attach-menu');if(attach&&!attach.hidden){attach.hidden=true;$('#analysis-attach-toggle').setAttribute('aria-expanded','false')}const engines=$('#analysis-engine-menu');if(engines&&!engines.hidden){engines.hidden=true;$('#analysis-engine-toggle').setAttribute('aria-expanded','false')}});
  $('#mission-diagnosis').onchange=e=>setFiles('diagnosis_results',e.target.files);$('#mission-logs').onchange=e=>setFiles('log_reports',e.target.files);$('#mission-references').onchange=e=>setFiles('references',e.target.files);
  for(const id of ['mission-task','mission-scope'])$('#'+id).addEventListener('input',()=>{S.uploadRevision++;clearTimeout(S.autoTimer);S.preflight=null;S.preflightId='';$('#mission-start').disabled=true;$('#preflight-results').textContent='';$('#upload-state').textContent='材料信息已更改，准备重新校验输入。';schedulePreflight()});
  $('#quick-demo').onclick=()=>{$('#demo-modal').showModal();loadDemos()};$('#quick-history').onclick=()=>useView('history');$('#quick-resume').onclick=async()=>{if(S.jobId){try{await openJob(S.jobId)}catch(e){toast('无法继续任务：'+e.message)}}else useView('history')};
  $('#open-demo').onclick=()=>{$('#demo-modal').showModal();loadDemos()};$('#mission-preflight').onclick=preflight;$('#mission-start').onclick=startJob;
  $$('[data-report-tab]').forEach(b=>b.onclick=()=>{S.reportTab=b.dataset.reportTab;$$('[data-report-tab]').forEach(x=>x.classList.toggle('active',x===b));renderReport()});$('#report-review-jump').onclick=()=>useView('review');$('#report-export-json').onclick=()=>exportReport('json');$('#report-export-md').onclick=()=>exportReport('md');
  $('#refresh-history').onclick=loadHistory;$('#graph-search').oninput=renderGraph;$('#graph-domain').onchange=renderGraph;$('#graph-evidence').onchange=renderGraph;
  $('#knowledge-review-refresh').onclick=loadKnowledgeReview;$$('[data-knowledge-tab]').forEach(b=>b.onclick=()=>{S.knowledgeTab=b.dataset.knowledgeTab;$$('[data-knowledge-tab]').forEach(x=>x.classList.toggle('active',x===b));loadKnowledgeReview()});$('#graph-knowledge-review-link').onclick=()=>useView('knowledge-review');$('#users-refresh').onclick=loadUsers;$('#users-create-form').onsubmit=createUser;$('#evidence-detail-close').onclick=()=>$('#evidence-detail').close();$('#service-status-open').onclick=openServiceStatus;$('#service-close').onclick=()=>$('#service-modal').close();$('#service-probe').onclick=probeServices;$('#expert-login').onclick=expertLogin;$('#expert-logout').onclick=expertLogout;$('#graph-build-run').onclick=runGraphBuild;$('#llm-persist').onchange=e=>$('#llm-profile-name-field').hidden=!e.target.checked;$('#llm-form').onsubmit=saveLLM;$('#llm-test').onclick=testLLM;$('#engine-form').onsubmit=saveEngine;$('#engine-probe').onclick=probeEngine;
  window.addEventListener('hashchange',()=>{const route=location.hash.slice(1);if($('#screen-'+route))useView(route)});
}
async function main(){bind();
  try{const cfg=await api('/v1/platform/auth/status');S.authMode=cfg.mode||'local';
    if(S.authMode==='accounts'){
      try{S.token=sessionStorage.getItem('astrograph-account-session')||''}catch{}
      S.accountInit=!!cfg.bootstrap_required;
      $('#account-modal-title').textContent=S.accountInit?'初始化系统管理员':'登录 AstroGraph';
      $('#account-submit').textContent=S.accountInit?'创建管理员并登录':'登录';
      $('#account-display-box').hidden=!S.accountInit;$('#account-setup-box').hidden=!S.accountInit;
      $('#identity-modal').querySelector('#identity-close').textContent='关闭';
    }
  }catch{}
  await refreshIdentity();try{if(['local','disabled'].includes(S.identity?.mode))S.jobId=localStorage.getItem('astrograph-mission-job-id')||'';else localStorage.removeItem('astrograph-mission-job-id')}catch{}
  const route=location.hash.slice(1);if($('#screen-'+route))useView(route);else useView('command');
  const incomingJob=new URLSearchParams(location.search).get('job');
  if(incomingJob&&S.identity){try{await loadCommand();let item=S.overview?.recent_jobs?.find(x=>x.job_id===incomingJob);if(!item){const job=await api(`/v1/platform/jobs/${qs(incomingJob)}`);item={...job,job_id:incomingJob};if(S.overview)S.overview.recent_jobs.push(item)}enterEvent(item)}catch(e){toast('事件读取失败：'+e.message)}}
  if(S.identity&&S.jobId){try{S.report=await api(`/v1/platform/jobs/${qs(S.jobId)}/report`);renderReport();renderWorkbench()}catch{S.report=null}}
}
main();
