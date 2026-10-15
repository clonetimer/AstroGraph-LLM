/** Expert-gated graph construction, independent review and release workbench.
 * No administrator bootstrap credential is persisted or embedded in JavaScript.
 * The server sets an HttpOnly cookie and exposes an ephemeral CSRF value.
 */
import { $, $$, esc, api, showToast } from './platform_core.js';

const BASE='/v1/platform/material-knowledge';
let session=null;
let pending={snapshot:0,graph_proposals:[],documents:[],cases:[],reasoning_gaps:[]};
let selectedKind='manual';
let manualQueue=[];
let manualDetail=null;
let focusedManualJobId=null;
let releaseSnapshot=null;
let releaseId=null;

function scopeText(raw){return String(raw||'').trim().toUpperCase()}
function htmlValue(v){return esc(String(v??''))}
function jsonRequest(value){return {method:'POST',headers:{'Content-Type':'application/json',...(session?.csrf?{'X-Material-CSRF':session.csrf}:{})},body:JSON.stringify(value),credentials:'same-origin',cache:'no-store'}}
async function expertApi(route,opts={}){
  if(!session?.authenticated)throw new Error('请先到专家知识审核页面登录');
  const headers={...(opts.headers||{}),'X-Material-CSRF':session.csrf};
  return api(BASE+route,{...opts,headers,credentials:'same-origin',cache:'no-store'});
}
function statsTile(name,value){return `<div class="graph-stat"><span>${esc(name)}</span><strong>${htmlValue(value)}</strong></div>`}
function graphRelationCard(item,options={}){
  const provenance=item.provenance||{};
  const filename=provenance.filename||'未知来源';
  const sourceLink=provenance.source_role==='references' && /^JOB-[0-9A-F]{24}$/.test(item.source_group_id||'')
    ? `<a class="graph-link" href="${BASE}/graph/sources/${encodeURIComponent(item.source_group_id)}/${encodeURIComponent(filename)}" target="_blank" rel="noopener noreferrer">查看原始文献</a>` : '';
  return `<div class="graph-claim-main">
    <div class="graph-link-line"><span class="graph-node"><small>${htmlValue(item.source?.entity_type)}</small>${htmlValue(item.source?.name)}</span><span class="graph-edge-arrow"><strong>${htmlValue(item.relation_type)}</strong> → </span><span class="graph-node"><small>${htmlValue(item.target?.entity_type)}</small>${htmlValue(item.target?.name)}</span></div>
    <div class="graph-tag-row"><span class="graph-tag ${item.polarity==='NEGATED'?'negative':''}">${item.polarity==='NEGATED'?'否定陈述':'正向陈述'}</span><span class="graph-tag">${htmlValue(item.claim_modality||'待确认')}</span><span class="graph-tag">${htmlValue(item.equipment_scope||'未审核构型')}</span></div>
    <blockquote class="graph-evidence-quote">“${htmlValue(item.evidence_quote||'未提供原文引句')}”</blockquote>
    <p class="graph-evidence-meta"><strong>出处：</strong>${htmlValue(filename)} · ${htmlValue((provenance.source_locators||[]).join(', ')||'无位置')} · SHA-256 ${htmlValue((provenance.source_sha256||'').slice(0,16))}… ${sourceLink}</p>
    ${options.approved?`<p class="graph-evidence-meta"><strong>审核：</strong>${htmlValue(item.reviewer)} · ${htmlValue(item.truth_source_ref)} · 快照 ${htmlValue(item.snapshot)}</p>`:''}
  </div>`;
}

function readiness(){
  const box=$('#graph-build-readiness');if(!box)return;
  if(!session){box.innerHTML='<div class="empty">正在读取…</div>';return}
  const flags=[
    ['专家身份',session.authenticated?`已登录 · ${session.reviewer}`:(session.configured?'尚未登录':'管理员尚未启用审核')],
    ['结构化抽取模型',session.graph_model_configured?'已配置':'未配置（禁止伪造关系）'],
    ['建图审批',session.authenticated?'可提交待审核关系':'需要专家会话'],
    ['知识发布权限',session.publisher?'具备管理员资格':'仅允许已授权发布管理员'],
    ['发布签名',session.release_signing_configured?'已配置':'未配置（不可写入 Neo4j）'],
    ['Neo4j 写入',session.neo4j_publication_enabled?'已显式启用':'默认关闭']
  ];
  box.innerHTML=flags.map(([label,val])=>`<div><span>${esc(label)}</span><strong>${esc(val)}</strong></div>`).join('');
  const button=$('#graph-build-submit');if(button)button.disabled=!session.authenticated;
  const exportButton=$('#graph-export');if(exportButton)exportButton.disabled=!(session.authenticated&&session.publisher&&releaseSnapshot!==null);
  const publishButton=$('#graph-publish');if(publishButton)publishButton.disabled=!(session.authenticated&&session.publisher&&session.release_signing_configured&&session.neo4j_publication_enabled&&releaseId&&$('#graph-publish-confirm')?.checked);
}
function authPanel(){
  if(!session)return;
  const form=$('#graph-login-form');if(!form)return;
  form.classList.toggle('hidden',!!session.authenticated);
  $('#graph-logout').classList.toggle('hidden',!session.authenticated);
  const info=$('#graph-auth-status');
  info.textContent=!session.configured?'服务端尚未配置审核身份，请联系管理员设置审核人员与安全凭据。':
    session.authenticated?`已认证：${session.reviewer} · ${Math.floor((session.session_ttl_seconds||1800)/60)} 分钟短时会话；${session.publisher?'具备发布管理员权限':'不具备发布管理员权限'}`:
    '待登录。凭据仅在本次登录请求中提交，浏览器不持久保存。非本机访问需使用 HTTPS。';
  readiness();
}
async function refreshSession(){
  try{session=await api(BASE+'/review-session',{cache:'no-store',credentials:'same-origin'});}catch(e){session={configured:false,authenticated:false};showToast('审核服务不可用：'+e.message)}
  authPanel();return session;
}
async function loadGraphBuild(){await refreshSession();}
function renderBuildResult(result){
  const box=$('#graph-build-result');if(!box)return;
  if(!result){box.innerHTML='<div class="empty">尚未运行建图。</div>';return}
  const relations=result.candidate_relations||[];
  box.innerHTML=`<div class="graph-review-summary">${statsTile('模型状态',result.status||'UNKNOWN')}${statsTile('提案关系',result.proposal_count??0)}${statsTile('解析片段',result.chunks_processed??0)}${statsTile('入库状态',result.staging_status||'—')}</div>
    <p class="graph-evidence-meta">来源组 ${htmlValue(result.source_group_id)} · 绑定构型 ${htmlValue(result.equipment_scope_requested)}</p>
    ${Object.keys(result.rejected_by_rule||{}).length?`<p class="graph-warning">规则拒绝：${Object.entries(result.rejected_by_rule).map(([k,v])=>`${esc(k)} ${htmlValue(v)}`).join('，')}</p>`:''}
    ${(result.extraction_errors||[]).length?`<p class="graph-warning">抽取错误：${(result.extraction_errors||[]).map(x=>esc(x.error_type)).join('，')}</p>`:''}
    ${relations.length?relations.slice(0,12).map(c=>`<div class="graph-claim-card">${graphRelationCard(c)}</div>`).join(''):'<div class="empty">未生成有效关系。请核查模型配置、文献内容及约束校验结果。</div>'}
    ${relations.length?'<p class="graph-disclaimer">以上仅为未审核的来源陈述，请到专家审核工作台逐条确认。</p>':''}`;
}
async function onBuildSubmit(e){
  e.preventDefault();
  if(!session?.authenticated){showToast('请先登录专家审核台');return}
  const files=Array.from($('#graph-files').files||[]);
  if(!files.length||files.length>8||files.some(x=>!['pdf','docx','txt','md'].includes(x.name.split('.').pop().toLowerCase())||x.size>16*1024*1024||!x.size)){
    showToast('请选择 1–8 份非空的 PDF/DOCX/TXT/MD 文件（单份≤16 MiB）');return;
  }
  if(files.reduce((n,x)=>n+x.size,0)>64*1024*1024){showToast('文献总大小不得超过 64 MiB');return}
  const scope=scopeText($('#graph-equipment-scope').value);
  if(!/^[A-Z0-9][A-Z0-9_.:-]{1,79}$/.test(scope)||['UNKNOWN','UNSPECIFIED'].includes(scope)){showToast('请填写明确的设备型号/构型标识');return}
  const chunks=Number($('#graph-max-chunks').value);
  if(!Number.isInteger(chunks)||chunks<1||chunks>12){showToast('最大抽取片段数应为 1–12');return}
  const fd=new FormData();files.forEach(f=>fd.append('reference_files',f,f.name));fd.append('equipment_scope',scope);fd.append('max_chunks',String(chunks));
  const button=$('#graph-build-submit');button.disabled=true;button.textContent='正在解析和构建候选关系…';
  $('#graph-build-result').innerHTML='<div class="empty">正在解析原始材料、调用结构化模型并校验引句；不会自动审批或修改正式图谱。</div>';
  try{
    const result=await expertApi('/graph/build',{method:'POST',body:fd});
    renderBuildResult(result);
    if(result.proposal_count)showToast(`新增 ${result.proposal_count} 条候选关系，需专家审核`);
    else showToast('构建结束：未生成有效候选关系，请查看详细状态');
  }catch(error){$('#graph-build-result').innerHTML=`<div class="graph-warning">构建失败：${esc(error.message)}</div>`;showToast('建图失败：'+error.message)}
  finally{button.textContent='开始构建候选图谱';button.disabled=!session?.authenticated}
}

function reviewCard(item,kind){
  const id=kind==='graph'?item.proposal_id:kind==='documents'?item.document_id:item.case_id;
  const title=kind==='graph'?'图谱关系':kind==='documents'?'参考文献':'历史案例';
  const src=kind==='graph'?graphRelationCard(item):
    `<h3>${htmlValue(item.filename||item.fault_id||id)}</h3><p class="graph-evidence-meta">来源组 ${htmlValue(item.source_group_id||'')} · 设备标识 ${htmlValue(item.equipment_scope||'待绑定')}</p>
     <p class="graph-evidence-quote">${htmlValue(item.fault_name||item.title||((item.sections||[]).map(x=>x.text||x.title).join(' ').slice(0,420))||'未提供简介')}</p>`;
  return `<article class="card card-pad graph-review-card" data-review-id="${esc(id)}" data-review-type="${esc(kind)}">
    <div class="cap-action"><strong>${esc(title)} · ${esc(id)}</strong><span class="badge review">待独立审核</span></div>
    ${src}
    <div class="graph-review-fields">
      <div class="field-group"><label>独立核验依据 / 单号</label><input class="field" data-role="truth" placeholder="手册版本、实测记录或复核编号" maxlength="500" required></div>
      <div class="field-group"><label>确认的设备构型</label><input class="field" data-role="scope" placeholder="SAT-A_ADCS_V2" maxlength="80" value="${esc(item.equipment_scope==='UNKNOWN'?'':item.equipment_scope||$('#graph-equipment-scope')?.value||'')}"></div>
      ${kind==='cases'?`<div class="field-group"><label>人工确诊故障编码</label><input class="field" data-role="fault" placeholder="已确认的故障编码" maxlength="160" value="${esc(item.fault_id||'')}"></div><div class="field-group"><label>审核确认的措施（每行一项，可空）</label><textarea class="field" data-role="actions" rows="2" maxlength="4000" placeholder="仅填写独立核验后的措施"></textarea></div>`:''}
    </div>
    <div class="graph-actions"><button class="btn primary" type="button" data-graph-decision="CONFIRM">确认知识</button><button class="btn" type="button" data-graph-decision="REJECT">驳回</button></div>
    <p class="graph-disclaimer">批准只表示已核验文献/案例的适用事实，不代表当前航天器已确诊，且不会直接执行处置指令。</p>
  </article>`;
}
function reasoningGapCard(item){
  return `<article class="card card-pad graph-review-card" data-gap-id="${esc(item.gap_id)}">
    <div class="cap-action"><strong>待补知识 · ${esc(item.fault_id)}</strong><span class="badge review">拒答/补证任务</span></div>
    <p class="graph-evidence-meta">设备构型：${esc(item.equipment_scope)} · 来源任务 ${esc(item.source_group_id)} · 快照 ${esc(item.graph_snapshot??'-')}</p>
    <div class="graph-tag-row">${(item.knowledge_gap_codes||[]).map(code=>`<span class="graph-tag">${esc(code)}</span>`).join('')}</div>
    <p class="graph-disclaimer">知识缺口不是已经证实的故障，也不是可发布的图谱关系。需独立补齐资料并经过专家审核。</p>
    <div class="field-group"><label>独立处理依据 / 文献补录编号（至少 8 字符）</label><input class="field" data-gap-resolution placeholder="经审核的知识构建编号或专家复核单" maxlength="500"></div>
    <div class="graph-actions"><button class="btn" type="button" data-gap-resolve="${esc(item.gap_id)}">记录已处理（不自动发布）</button></div>
  </article>`;
}
async function resolveReasoningGap(button){
  const card=button.closest('[data-gap-id]');if(!card)return;
  const id=card.dataset.gapId;
  const resolution_ref=card.querySelector('[data-gap-resolution]')?.value.trim()||'';
  if(resolution_ref.length<8){showToast('请提供独立审核或补录依据（至少 8 个字符）');return}
  if(!confirm(`确认已人工处理知识缺口 ${id}？不会自动写入知识图谱。`))return;
  button.disabled=true;
  try{await expertApi(`/reasoning/gaps/${encodeURIComponent(id)}/resolve`,jsonRequest({resolution_ref}));showToast('知识缺口已标记处理，不会自动发布知识');await loadGraphReview()}
  catch(e){showToast('处理失败：'+e.message);button.disabled=false}
}

function manualStatusText(status){return ({PENDING:'待复核',IN_PROGRESS:'部分复核',REVIEWED:'已记录所有复核意见',REVIEWED_WITH_GAPS:'已复核 · 仍有证据缺口'})[status]||'待复核'}
function manualReviewCard(item){
  return `<article class="report-review-queue-item"><div><strong>${esc(item.job_id)}</strong><span>${esc(item.completed_at||'')} · ${esc(manualStatusText(item.status))} · ${esc(item.reviewed_count)}/${esc(item.total_count)} 项</span></div><button type="button" class="btn small" data-report-review-open="${esc(item.job_id)}">${focusedManualJobId===item.job_id?'刷新详情':'查看并复核'}</button></article>`;
}
function manualReviewDetailHtml(detail){
  if(!detail)return '<div class="empty">请选择待复核的材料分析任务，或从原因与处置报告点击“进入人工复核”。</div>';
  const items=(detail.candidates||[]).map(item=>{
    const r=item.review;
    const previous=r?`<div class="report-review-existing"><strong>最近复核：${esc(({FOLLOW_UP_HYPOTHESIS:'保留为排查假设',NOT_SUPPORTED:'不支持此候选',EVIDENCE_INSUFFICIENT:'证据不足'})[r.decision]||r.decision)}</strong><span>${esc(r.actor)} · ${esc(r.recorded_at)} · ${esc(r.note)}</span></div>`:'';
    return `<article class="card card-pad report-review-editor" data-manual-candidate="${esc(item.fault_id)}">
      <div class="card-title">${esc(item.fault_name)}</div><div class="graph-evidence-meta">${esc(item.fault_id)}</div>${previous}
      <div class="field-group"><label>复核意见</label><select class="field" data-manual-decision><option value="EVIDENCE_INSUFFICIENT">证据不足 · 继续补充材料</option><option value="FOLLOW_UP_HYPOTHESIS">支持继续排查（不是已确诊）</option><option value="NOT_SUPPORTED">独立证据不支持该候选</option></select></div>
      <div class="field-group"><label>独立核验材料编号 / 来源</label><input class="field" data-manual-evidence maxlength="500" placeholder="例如手册版本号、复测记录或工单；证据不足可留空"></div>
      <div class="field-group"><label>复核记录与下一步检查（至少 8 字符）</label><textarea class="field" data-manual-note rows="3" maxlength="2000" placeholder="填写实际核对过的证据及处理建议，不能仅重复模型结论"></textarea></div>
      <div class="graph-actions"><button class="btn primary" type="button" data-manual-review-submit="${esc(item.fault_id)}">保存本项复核</button></div>
    </article>`;
  }).join('');
  const history=(detail.history||[]).slice(0,12).map(x=>`<div class="evidence-row"><span><strong>${esc(x.fault_id)} · ${esc(manualStatusText('REVIEWED'))} · ${esc(x.decision)}</strong><small>${esc(x.actor)} · ${esc(x.recorded_at)} · 来源 ${esc(x.evidence_ref||'未提供')}</small><small>${esc(x.note)}</small></span></div>`).join('');
  return `<div class="report-review-detail"><div class="cap-action"><div><h2>任务 ${esc(detail.job_id)}</h2><p>${esc(manualStatusText(detail.status))} · 已办理 ${detail.reviewed_count}/${detail.total_count}</p></div><button class="btn small" type="button" data-report-review-back>收起详情</button></div>
   <p class="review-scope-note">复核意见属于独立人工记录，不修改原始报告、已审知识图谱和设备控制权限。</p>
   <div class="report-review-editors">${items}</div>
   <details class="card card-pad"><summary>复核修订历史（追加记录，不覆盖）</summary><div class="evidence-list">${history||'<div class="empty">暂无复核修订。</div>'}</div></details></div>`;
}
function showManualReviewQueue(){
  const list=$('#graph-review-list');if(!list)return;
  const queue=manualQueue.filter(x=>x.status!=='REVIEWED'&&x.status!=='REVIEWED_WITH_GAPS');
  const rest=manualQueue.filter(x=>x.status==='REVIEWED'||x.status==='REVIEWED_WITH_GAPS');
  const cards=[...queue,...rest].map(manualReviewCard).join('');
  list.innerHTML=`<section class="card card-pad"><div class="card-title">材料分析报告复核</div><p class="card-sub">先检查报告中的候选原因与证据，再逐项登记独立复核意见。</p><div class="report-review-queue">${cards||'<div class="empty">近期没有已完成的材料分析任务。</div>'}</div></section>${manualReviewDetailHtml(manualDetail)}`;
}
async function openManualReview(jobId){
  focusedManualJobId=jobId;selectedKind='manual';
  if(!session?.authenticated){showPending();return}
  try{manualDetail=await expertApi(`/report-reviews/${encodeURIComponent(jobId)}`);showPending()}
  catch(e){manualDetail=null;showPending();showToast('读取报告复核失败：'+e.message)}
}
function focusManualReportReview(jobId){
  if(!/^JOB-[0-9A-F]{24}$/.test(String(jobId||'')))return;
  focusedManualJobId=jobId;selectedKind='manual';manualDetail=null;
}
async function submitManualReview(button){
  const card=button.closest('[data-manual-candidate]');if(!card||!manualDetail)return;
  const fault_id=card.dataset.manualCandidate;
  const decision=card.querySelector('[data-manual-decision]').value;
  const evidence_ref=card.querySelector('[data-manual-evidence]').value.trim();
  const note=card.querySelector('[data-manual-note]').value.trim();
  if(note.length<8){showToast('请填写至少 8 字符的独立复核意见');return}
  if(decision!=='EVIDENCE_INSUFFICIENT'&&evidence_ref.length<8){showToast('支持或否定候选必须提供独立证据编号');return}
  if(!confirm(`确认保存 ${fault_id} 的复核意见？记录将保留历史，不改变诊断或发布状态。`))return;
  button.disabled=true;
  try{
    manualDetail=await expertApi(`/report-reviews/${encodeURIComponent(manualDetail.job_id)}/decisions`,jsonRequest({report_sha256:manualDetail.report_sha256,
      fault_id,decision,evidence_ref,note,expected_revision:manualDetail.revision}));
    const queue=await expertApi('/report-reviews?limit=100');manualQueue=queue.items||[];
    showPending();showToast('已保存独立复核意见');
  }catch(e){showToast('保存失败：'+e.message);button.disabled=false}
}

function showPending(){
  const counts=$('#graph-review-counts'), list=$('#graph-review-list');if(!counts||!list)return;
  if(!session?.authenticated){counts.innerHTML='';list.innerHTML='<div class="empty">请先登录专家审核台；待审核知识及来源资料不对普通用户开放。</div>';return}
  counts.innerHTML=(selectedKind==='manual'?statsTile('待办材料复核',manualQueue.filter(x=>x.status==='PENDING'||x.status==='IN_PROGRESS').length)+statsTile('近期完成',manualQueue.filter(x=>x.status==='REVIEWED'||x.status==='REVIEWED_WITH_GAPS').length):statsTile('当前知识快照',pending.snapshot)+statsTile('待审图谱关系',(pending.graph_proposals||[]).length)+statsTile('待审文献',(pending.documents||[]).length)+statsTile('待审案例',(pending.cases||[]).length)+statsTile('待补知识',(pending.reasoning_gaps||[]).length));
  if(selectedKind==='manual')showManualReviewQueue();
  else {
    const items=pending[selectedKind==='graph'?'graph_proposals':selectedKind]||[];
    list.innerHTML=items.length?items.map(x=>selectedKind==='reasoning_gaps'?reasoningGapCard(x):reviewCard(x,selectedKind)).join(''):'<div class="card card-pad empty">该类别暂无待审核记录。</div>';
  }
  $$('[data-review-kind]').forEach(b=>{b.classList.toggle('active',b.dataset.reviewKind===selectedKind);b.setAttribute('aria-pressed',String(b.dataset.reviewKind===selectedKind))});
}
async function loadGraphReview(){
  await refreshSession();
  if(!session?.authenticated){showPending();return}
  try{const [data, queue]=await Promise.all([expertApi('/pending?limit=100'),expertApi('/report-reviews?limit=100')]);pending=data;manualQueue=queue.items||[];
    if(focusedManualJobId)manualDetail=await expertApi(`/report-reviews/${encodeURIComponent(focusedManualJobId)}`);showPending()}
  catch(e){$('#graph-review-list').innerHTML=`<div class="graph-warning">审核队列读取失败：${esc(e.message)}</div>`}
}
async function onReviewDecision(button){
  const card=button.closest('[data-review-id]');if(!card)return;
  const kind=card.dataset.reviewType,id=card.dataset.reviewId,decision=button.dataset.graphDecision;
  const truth=card.querySelector('[data-role="truth"]').value.trim();
  const scope=scopeText(card.querySelector('[data-role="scope"]').value);
  if(truth.length<3){showToast('请输入独立核验依据，至少 3 个字符');return}
  if(decision==='CONFIRM'&&(!/^[A-Z0-9][A-Z0-9_.:-]{1,79}$/.test(scope)||['UNKNOWN','UNSPECIFIED'].includes(scope))){showToast('确认知识必须绑定有效设备构型');return}
  const item=kind==='graph'?pending.graph_proposals.find(x=>x.proposal_id===id):kind==='cases'?pending.cases.find(x=>x.case_id===id):pending.documents.find(x=>x.document_id===id);
  if(!item)return;
  const payload={decision,truth_source_ref:truth,equipment_scope:decision==='CONFIRM'?scope:null,expected_snapshot:pending.snapshot};
  if(kind==='cases'){
    payload.verified_fault_id=card.querySelector('[data-role="fault"]').value.trim()||null;
    payload.approved_actions=card.querySelector('[data-role="actions"]').value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean);
    if(decision==='CONFIRM'&&!payload.verified_fault_id){showToast('案例确认需要人工确诊故障编码');return}
  }
  if(!confirm(`${decision==='CONFIRM'?'确认':'驳回'} ${kind==='graph'?'关系':kind==='cases'?'案例':'文献'} ${id}？该操作会记录审核人和核验依据。`))return;
  const part=kind==='graph'?`/graph/${encodeURIComponent(id)}/decision`:kind==='cases'?`/cases/${encodeURIComponent(id)}/decision`:`/documents/${encodeURIComponent(id)}/decision`;
  button.disabled=true;
  try{const out=await expertApi(part,jsonRequest(payload));showToast(`审核记录已提交：${out.status}`);await loadGraphReview()}
  catch(e){showToast('审核失败：'+e.message);button.disabled=false;await loadGraphReview()}
}

function releasedCard(item){
  return `<article class="graph-claim-card">${graphRelationCard(item,{approved:true})}<div class="graph-actions"><button class="btn small" type="button" data-graph-retract="${esc(item.proposal_id)}">撤销该关系（需独立证明）</button></div></article>`;
}
async function loadReviewed(){
  if(!session?.authenticated){await refreshSession()}
  if(!session?.authenticated){showToast('请先登录专家审核台');return}
  const scope=scopeText($('#graph-release-scope').value);
  if(!/^[A-Z0-9][A-Z0-9_.:-]{1,79}$/.test(scope)){showToast('请填写有效设备构型');return}
  const box=$('#graph-reviewed-list');box.innerHTML='<div class="empty">正在读取已审核图谱…</div>';
  try{
    const [graph,stats]=await Promise.all([
      expertApi('/graph/reviewed?equipment_scope='+encodeURIComponent(scope)+'&limit=120'),
      expertApi('/graph/status?equipment_scope='+encodeURIComponent(scope)),
    ]);
    releaseSnapshot=graph.snapshot;releaseId=`${scope}-SNAPSHOT-${String(releaseSnapshot).padStart(8,'0')}`;
    $('#graph-release-stats').innerHTML=statsTile('知识快照',graph.snapshot)+statsTile('当前有效关系',stats.active_approved_relations??graph.total)+statsTile('已撤销关系',stats.retracted_relations||0)+statsTile('争议关系组',stats.disputed_relation_groups||0);
    box.innerHTML=(graph.relations||[]).map(releasedCard).join('')||'<div class="empty">该型号暂时没有已审核的图谱关系。</div>';
    if(graph.truncated)box.innerHTML+='<p class="graph-warning">已到显示上限，列表并非全部关系；请通过导出的完整快照核验。</p>';
    if(graph.retractions?.length)box.innerHTML+=`<details class="card card-pad"><summary>历史撤销 ${graph.retractions.length} 条</summary>${graph.retractions.map(r=>`<div class="graph-evidence-meta">${esc(r.proposal_id)} · 版本 ${htmlValue(r.version)} · ${esc(r.reason)}</div>`).join('')}</details>`;
    $('#graph-publish-result').innerHTML=`<div class="graph-evidence-meta">已查询 ${esc(scope)} · 当前版本 ${graph.snapshot}。导出/发布前请复核实际的知识出处及争议关系。</div>`;
    $('#graph-publish-confirm').checked=false;
    readiness();
  }catch(e){box.innerHTML=`<div class="graph-warning">查询失败：${esc(e.message)}</div>`;releaseSnapshot=null;releaseId=null;readiness()}
}
async function retractRelation(button){
  const id=button.dataset.graphRetract;
  if(!id||releaseSnapshot===null)return;
  const reason=prompt(`撤销 ${id} 的原因（至少 8 个字符）：`);
  if(reason===null)return;
  const truth=prompt('请填写独立更正依据 / 复核单号：');
  if(truth===null)return;
  if(reason.trim().length<8||truth.trim().length<3){showToast('撤销必须有明确原因和独立核验依据');return}
  if(!confirm('撤销后新的知识快照将不再使用该关系，旧版本保持可审计。继续？'))return;
  button.disabled=true;
  try{await expertApi(`/graph/${encodeURIComponent(id)}/retract`,jsonRequest({reason:reason.trim(),truth_source_ref:truth.trim(),expected_snapshot:releaseSnapshot}));showToast('关系已撤销，新的知识快照已形成');await loadReviewed()}
  catch(e){showToast('撤销失败：'+e.message);button.disabled=false;await loadReviewed()}
}
async function exportRelease(){
  if(releaseSnapshot===null||!session?.publisher)return;
  const scope=scopeText($('#graph-release-scope').value);
  if(!confirm(`从已审知识导出 ${scope} 的快照 ${releaseSnapshot}？`))return;
  try{
    const receipt=await expertApi('/graph/export',jsonRequest({equipment_scope:scope,snapshot:releaseSnapshot}));
    $('#graph-publish-result').innerHTML=`<div class="graph-tag-row"><span class="graph-tag">${esc(receipt.release_id||'')}</span><span class="graph-tag">关系 ${htmlValue(receipt.edge_count||0)}</span><span class="graph-tag">节点 ${htmlValue(receipt.node_count||0)}</span><span class="graph-tag">${receipt.signature_hmac_sha256?'签名存在':'无签名，不可 Neo4j 发布'}</span></div><p class="graph-evidence-meta">导出完成，文件保存在服务器受治理的 graph_releases 目录，后续发布需核对签名。</p>`;
    showToast('知识图谱快照导出完成');
  }catch(e){showToast('导出失败：'+e.message)}
}
async function publishRelease(){
  if(!session?.publisher||!$('#graph-publish-confirm').checked||!releaseId)return;
  const input=prompt(`管理员二次确认：请输入完整发布编号\n${releaseId}`);
  if(input===null||input.trim()!==releaseId){showToast('发布编号不匹配，操作取消');return}
  if(!confirm(`确认将已签名快照 ${releaseId} 写入隔离 Neo4j？这是管理员明确的写操作。`))return;
  const button=$('#graph-publish');button.disabled=true;
  try{
    const result=await expertApi('/graph/publish',jsonRequest({equipment_scope:scopeText($('#graph-release-scope').value),snapshot:releaseSnapshot,confirm_release_id:releaseId,confirm_write:true}));
    $('#graph-publish-result').innerHTML=`<div class="graph-stat"><strong>${esc(result.status||'发布已提交')}</strong><span>Neo4j 写入：${result.neo4j_write_performed?'已执行':'未执行'}；版本：${htmlValue(releaseSnapshot)}</span></div>`;
    showToast('发布接口已返回，请按服务器结果确认');
  }catch(e){showToast('发布失败：'+e.message)}
  finally{$('#graph-publish-confirm').checked=false;readiness()}
}
function setupGraphWorkbench(){
  $('#graph-login-form').addEventListener('submit',async e=>{
    e.preventDefault();const input=$('#graph-review-credential'),credential=input.value;
    input.value='';
    try{await api(BASE+'/review-session',jsonRequest({credential}));await loadGraphReview();showToast('专家会话已建立')}
    catch(error){showToast('专家登录失败：'+error.message)}
  });
  $('#graph-logout').onclick=async()=>{try{await expertApi('/review-session',{method:'DELETE'});session=null;pending={snapshot:0,graph_proposals:[],documents:[],cases:[]};releaseSnapshot=null;releaseId=null;await refreshSession();showPending();showToast('已退出审核会话')}catch(e){showToast('退出失败：'+e.message)}};
  $('#graph-build-form').addEventListener('submit',onBuildSubmit);
  $('#graph-files').onchange=()=>{$('#graph-files-note').textContent=Array.from($('#graph-files').files||[]).map(f=>`${f.name} (${(f.size/1024).toFixed(1)} KB)`).join('，')||'尚未选择文件'};
  $('#graph-refresh-review').onclick=loadGraphReview;
  $('#graph-review-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-review-kind]');if(b){selectedKind=b.dataset.reviewKind;showPending()}});
  $('#graph-review-list').addEventListener('click',e=>{
    const open=e.target.closest('[data-report-review-open]');if(open){openManualReview(open.dataset.reportReviewOpen);return}
    const submit=e.target.closest('[data-manual-review-submit]');if(submit){submitManualReview(submit);return}
    if(e.target.closest('[data-report-review-back]')){manualDetail=null;focusedManualJobId=null;showPending();return}
    const g=e.target.closest('[data-gap-resolve]');if(g){resolveReasoningGap(g);return}const b=e.target.closest('[data-graph-decision]');if(b)onReviewDecision(b)
  });
  $('#graph-load-reviewed').onclick=loadReviewed;
  $('#graph-reviewed-list').addEventListener('click',e=>{const b=e.target.closest('[data-graph-retract]');if(b)retractRelation(b)});
  $('#graph-export').onclick=exportRelease;
  $('#graph-publish-confirm').onchange=readiness;
  $('#graph-publish').onclick=publishRelease;
  refreshSession();
}
async function loadGraphRelease(){await refreshSession();if(session?.authenticated&&$('#graph-release-scope')?.value.trim())await loadReviewed()}
export { setupGraphWorkbench, loadGraphBuild, loadGraphReview, loadGraphRelease, focusManualReportReview };
