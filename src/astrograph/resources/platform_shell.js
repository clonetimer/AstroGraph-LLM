import { $, $$, state, esc } from './platform_core.js';

let hooks={};
function setupShellHooks(value){hooks={...hooks,...value}}

function nav(view,updateHash=true){
  $$('.view').forEach(v=>v.classList.remove('active'));
  const target=$('#view-'+view);if(!target)return;
  target.classList.add('active');
  $$('.nav-btn').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  const titles={dashboard:'分析总览',new:'新建故障分析',progress:'分析中',result:'原因与处置报告',history:'任务记录',agent:'Agent 调查',knowledge:'内置知识图谱','graph-build':'文献图谱构建','graph-review':'人工复核与知识审核','graph-release':'知识版本与发布',bench:'Bench · 工程验证',system:'大模型配置',engine:'检索与多模态配置'};
  $('#top-title').textContent=titles[view]||'AstroGraph';
  const expert=$('#nav-knowledge-more'),advanced=$('#nav-advanced-more');
  if(expert&&['graph-review','graph-release'].includes(view))expert.open=true;
  if(advanced&&['agent','bench'].includes(view))advanced.open=true;
  if(updateHash&&view!=='progress')history.replaceState(null,'','#'+view);
  if(view==='dashboard')hooks.loadDashboard?.();
  if(view==='new')hooks.loadJobDiagnostics?.();
  if(view==='history')hooks.loadHistory?.();
  if(view==='agent')Promise.all([hooks.loadAgentCapabilities?.(),hooks.loadTools?.()]);
  if(view==='knowledge')hooks.loadKnowledgeWorkspace?.();
  if(view==='graph-build')hooks.loadGraphBuild?.();
  if(view==='graph-review')hooks.loadGraphReview?.();
  if(view==='graph-release')hooks.loadGraphRelease?.();
  if(view==='engine')hooks.loadEngineSettings?.();
  if(view==='system')Promise.all([hooks.loadLLMSettings?.(),hooks.loadLLMProfiles?.(),hooks.loadNeo4jSettings?.(),hooks.loadMCPProviders?.(),hooks.loadCapabilities?.()]);
  if(view==='result'&&state.report)hooks.renderReport?.(state.report);
}

function applyFontScale(v,persist=true){
  v=Math.max(90,Math.min(160,Math.round(Number(v)/5)*5));state.fontScale=v;
  document.documentElement.style.setProperty('--font-scale',(v/100).toFixed(2));
  document.documentElement.dataset.fontBand=v>=145?'xlarge':v>=125?'large':'normal';
  $('#font-range').value=String(v);$('#font-output').textContent=v+'%';
  if(persist)localStorage.setItem('astrograph-font-scale-v04',String(v));
}

function renderCommands(q=''){
  const box=$('#command-list'),raw=q.trim(),low=raw.toLowerCase();
  let defs=[
    {title:'分析总览',keywords:'首页 工作台 dashboard',run:()=>nav('dashboard')},{title:'新建故障分析',keywords:'新建 故障分析 上传',run:()=>nav('new')},{title:'原因与处置报告',keywords:'结果 报告 result',run:()=>nav('result')},{title:'任务记录',keywords:'历史 任务',run:()=>nav('history')},{title:'Agent 调查',keywords:'agent 智能体 工具 调查',run:()=>nav('agent')},{title:'内置知识图谱',keywords:'知识 图谱 公开 NASA 故障 资料库',run:()=>nav('knowledge')},{title:'文献图谱构建',keywords:'graph build 文献 抽取',run:()=>nav('graph-build')},{title:'人工复核与知识审核',keywords:'review 知识 审批',run:()=>nav('graph-review')},{title:'知识版本与发布',keywords:'release Neo4j 快照',run:()=>nav('graph-release')},{title:'Bench · 工程验证',keywords:'bench 工程 验证',run:()=>nav('bench')},{title:'大模型配置',keywords:'系统 模型 llm deepseek ollama LM Studio 大模型 配置',run:()=>nav('system')},{title:'检索与多模态配置',keywords:'GraphRAG Qdrant Embedding OCR Docling VLM 向量 配置',run:()=>nav('engine')},
    {title:'切换：标准诊断',keywords:'模式 normal 标准 正常',run:()=>hooks.setMode?.('normal')},{title:'切换：增强诊断',keywords:'模式 thinking 增强 深入 复核 思考',run:()=>hooks.setMode?.('thinking')},{title:'切换：扩展排查（高级）',keywords:'模式 creative 高级 扩展 排查 创造',run:()=>hooks.setMode?.('creative')},{title:'刷新驾驶舱',keywords:'刷新 reload',run:()=>hooks.loadDashboard?.(true)},
  ].filter(c=>!low||(`${c.title} ${c.keywords}`).toLowerCase().includes(low));
  const fm=raw.match(/(?:字体|font)\s*(\d{2,3})/i);if(fm){const v=Math.max(90,Math.min(160,Number(fm[1])));defs=[{title:`设置字体 ${v}%`,keywords:'',run:()=>applyFontScale(v)},...defs]}
  const jm=raw.match(/([A-Za-z0-9_-]{8,})/);if(jm)defs=[{title:`打开任务 ${jm[1]}`,keywords:'',run:()=>hooks.openHistory?.(jm[1])},...defs];
  box.innerHTML=defs.slice(0,12).map((c,i)=>`<button type="button" class="command-item" data-command-index="${i}"><span><strong>${esc(c.title)}</strong></span><span class="command-key">Enter</span></button>`).join('')||'<div class="empty">没有匹配命令。</div>';
  $$('[data-command-index]').forEach(b=>b.onclick=()=>{const c=defs[Number(b.dataset.commandIndex)];if(c){$('#command-dialog').close();c.run()}});box._defs=defs;
}
function openCommand(){renderCommands('');$('#command-input').value='';$('#command-dialog').showModal();setTimeout(()=>$('#command-input').focus(),0)}

function initShell(){
  $$('[data-view]').forEach(b=>b.onclick=()=>nav(b.dataset.view));$$('[data-view-jump]').forEach(b=>b.onclick=()=>nav(b.dataset.viewJump));
  window.addEventListener('hashchange',()=>{const v=location.hash.slice(1);if(v)nav(v,false)});
  $('#show-capabilities').onclick=async()=>{if(!state.capabilities)await hooks.loadCapabilities?.();$('#cap-dialog-body').innerHTML=hooks.capabilitiesHtml?.()||'';$('#cap-dialog').showModal()};
  $('#font-range').addEventListener('input',e=>applyFontScale(e.target.value));$('#font-dec').onclick=()=>applyFontScale(state.fontScale-5);$('#font-inc').onclick=()=>applyFontScale(state.fontScale+5);$('#font-reset').onclick=()=>applyFontScale(110);applyFontScale(state.fontScale,false);
  $('#command-open').onclick=openCommand;$('#command-input').addEventListener('input',e=>renderCommands(e.target.value));$('#command-input').addEventListener('keydown',e=>{if(e.key==='Enter'){const defs=$('#command-list')._defs||[];if(defs[0]){$('#command-dialog').close();defs[0].run()}}});window.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openCommand()}});
  $$('[data-dialog-close]').forEach(b=>b.onclick=()=>$('#'+b.dataset.dialogClose).close());$$('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d)d.close()}));
}

export { applyFontScale, initShell, nav, setupShellHooks };
