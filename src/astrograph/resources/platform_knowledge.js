import { $, state, esc, api, showToast } from './platform_core.js';
import { renderKnowledgeOperations } from './platform_operations.js';

// The public seed is a *separate*, immutable reference catalogue. It is never
// mixed into the equipment-scoped, expert-approved graph or used for release.
let publicSeed=null;
let selectedFaultCode='';
let seedLoadPromise=null;
const domainNames={ADCS:'姿态控制',EPS:'电源',THERMAL:'热控',PROP:'推进',COMM:'通信数传',CDH:'星载计算',PAYLOAD:'有效载荷'};
const kindNames={COMPONENT:'部件',FAULT_MODE:'故障模式',FAILURE_MECHANISM:'失效机理',SYMPTOM:'异常现象',VERIFICATION:'检查项目',ACTION:'建议措施',CONDITION:'适用条件'};
const edgeNames={ASSOCIATED_WITH:'关联机理',SUSCEPTIBLE_TO:'可能失效',MANIFESTS_AS:'可能表现',INDICATED_BY:'异常线索',CHECKED_BY:'检查',MITIGATED_BY:'建议措施',REQUIRES:'实施前提'};
const nodePositions={FAULT_MODE:[130,140],COMPONENT:[130,54],FAILURE_MECHANISM:[365,66],SYMPTOM:[600,65],VERIFICATION:[600,216],ACTION:[366,238],CONDITION:[831,238]};

function safeSourceUrl(value){
  try{const u=new URL(String(value||''));return ['https:','http:'].includes(u.protocol)?u.href:null}catch{return null}
}
function svgLineLabel(text,max=10){const value=String(text||'');return value.length>max?value.slice(0,max-1)+'…':value}
function localGraphSvg(entities,relations){
  const nodeById=new Map(entities.map(n=>[n.id,n]));
  const edges=relations.map(r=>{
    const a=nodeById.get(r.source),b=nodeById.get(r.target);if(!a||!b)return '';
    const p=nodePositions[a.entity_type],q=nodePositions[b.entity_type];if(!p||!q)return '';
    if(p[0]===q[0]&&p[1]===q[1])return '';
    return `<path d="M${p[0]},${p[1]} Q${Math.round((p[0]+q[0])/2)},${Math.round((p[1]+q[1])/2-15)} ${q[0]},${q[1]}" fill="none" stroke="#9bb3cc" stroke-width="1.7" marker-end="url(#seed-arrow)" />`;
  }).join('');
  const nodes=entities.map(n=>{
    const [x,y]=nodePositions[n.entity_type]||[730,130];
    return `<g><title>${esc(n.name)}</title><rect x="${x-88}" y="${y-29}" width="176" height="58" rx="12" class="seed-svg-node seed-svg-${esc(n.entity_type.toLowerCase())}"/><text x="${x}" y="${y-5}" text-anchor="middle" class="seed-svg-kind">${esc(kindNames[n.entity_type]||n.entity_type)}</text><text x="${x}" y="${y+15}" text-anchor="middle" class="seed-svg-name">${esc(svgLineLabel(n.name))}</text></g>`;
  }).join('');
  return `<svg viewBox="0 0 940 310" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(entities.find(x=>x.entity_type==='FAULT_MODE')?.name||'故障')}相关的实体和关系示意图"><defs><marker id="seed-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8z" fill="#7895b2"/></marker></defs>${edges}${nodes}</svg>`;
}
function renderPublicSeed(){
  if(!publicSeed)return;
  const domain=$('#seed-domain')?.value||'';
  const term=($('#seed-search')?.value||'').trim().toLowerCase();
  const entities=new Map(publicSeed.entities.map(n=>[n.id,n]));
  const relations=new Map(publicSeed.relations.map(e=>[e.id,e]));
  const sources=new Map(publicSeed.sources.map(s=>[s.id,s]));
  const clusters=publicSeed.clusters.filter(c=>{
    if(domain&&c.domain!==domain)return false;
    if(!term)return true;
    const names=[c.code,c.title,c.summary,...(c.edge_ids||[]).flatMap(id=>{
      const r=relations.get(id);return r?[entities.get(r.source)?.name,entities.get(r.target)?.name]:[];
    })];
    return names.some(s=>String(s||'').toLowerCase().includes(term));
  });
  const statHost=$('#seed-overview');
  if(statHost)statHost.innerHTML=`<div class="seed-kpis"><div><strong>${publicSeed.clusters.length}</strong><span>故障主题</span></div><div><strong>${publicSeed.entities.length}</strong><span>知识实体</span></div><div><strong>${publicSeed.relations.length}</strong><span>候选关系</span></div><div><strong>${publicSeed.sources.length}</strong><span>参考来源</span></div></div><p class="seed-notice">公开教材与技术资料形成的通用知识，支持浏览与学习；尚未经型号级审核，不参与自动确诊。</p>`;
  const list=$('#seed-fault-list');
  if(!clusters.some(c=>c.code===selectedFaultCode))selectedFaultCode=clusters[0]?.code||'';
  if(list){
    list.innerHTML=clusters.map(c=>`<button type="button" class="seed-fault-btn ${c.code===selectedFaultCode?'selected':''}" data-seed-fault="${esc(c.code)}" aria-pressed="${c.code===selectedFaultCode}"><strong>${esc(c.title)}</strong><span>${esc(domainNames[c.domain]||c.domain)} · ${esc(c.code)}</span></button>`).join('')||'<div class="empty">没有匹配的故障主题，试试其他关键词。</div>';
    list.querySelectorAll('[data-seed-fault]').forEach(b=>b.addEventListener('click',()=>{selectedFaultCode=b.dataset.seedFault;renderPublicSeed()}));
  }
  const host=$('#seed-topic-detail');if(!host)return;
  const cluster=clusters.find(c=>c.code===selectedFaultCode);
  if(!cluster){host.innerHTML='<div class="empty">请选择一个故障主题。</div>';return}
  const links=(cluster.edge_ids||[]).map(id=>relations.get(id)).filter(Boolean);
  const ids=new Set(links.flatMap(e=>[e.source,e.target]));
  const nodes=[...ids].map(id=>entities.get(id)).filter(Boolean);
  const sourceRefs=[...new Set(links.flatMap(e=>e.source_ids||[]))];
  const rows=links.map(edge=>{
    const source=entities.get(edge.source)?.name||edge.source;
    const target=entities.get(edge.target)?.name||edge.target;
    return `<details class="seed-relation"><summary><span class="seed-relation-kind">${esc(edgeNames[edge.relation_type]||edge.relation_type)}</span><strong>${esc(source)} → ${esc(target)}</strong></summary><div class="seed-relation-body"><p>${esc(edge.source_summary_zh||'本关系仅来自公开资料归纳。')}</p><div class="seed-source-line">${esc(edge.source_locator||'无页码')} · ${esc((edge.source_ids||[]).join(' / '))}</div></div></details>`;
  }).join('');
  host.innerHTML=`<div class="seed-topic-head"><div><div class="seed-topic-label">${esc(domainNames[cluster.domain]||cluster.domain)} / 故障主题</div><h2>${esc(cluster.title)}</h2><code>${esc(cluster.code)}</code></div><span class="badge review">未审核知识</span></div>
    <div class="seed-diagram">${localGraphSvg(nodes,links)}</div>
    <div class="seed-topic-grid"><div><h3>关系链</h3><div class="seed-relation-list">${rows||'<div class="empty">暂无关系。</div>'}</div></div>
    <div class="seed-source-panel"><h3>资料依据</h3><p>${esc(cluster.summary||'')}</p>${sourceRefs.map(id=>{
      const s=sources.get(id);if(!s)return '';
      const href=safeSourceUrl(s.url);
      return `<div class="seed-source-entry"><strong>${esc(s.org||id)}</strong><span>${esc(s.title||id)}</span>${href?`<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">查看公开来源 ↗</a>`:''}</div>`;
    }).join('')}</div></div>`;
}

async function loadPublicSeedGraph(force=false){
  if(publicSeed&&!force){renderPublicSeed();return}
  if(seedLoadPromise&&!force)return seedLoadPromise;
  seedLoadPromise=(async()=>{
    const data=await api('/v1/platform/knowledge/public-seed');
    if(data.graph_kind!=='UNREVIEWED_PUBLIC_REFERENCE_SEED'||data.is_flight_qualified!==false||!Array.isArray(data.clusters)||!Array.isArray(data.relations)||!Array.isArray(data.entities)||!Array.isArray(data.sources))throw new Error('图谱数据格式不符合公开种子知识合同');
    publicSeed=data;
    renderPublicSeed();
  })().catch(err=>{
    const node=$('#seed-overview');if(node)node.innerHTML=`<div class="empty">内置图谱暂不可用：${esc(err.message)}</div>`;
    showToast('知识图谱加载失败');
  }).finally(()=>{seedLoadPromise=null});
  return seedLoadPromise;
}
function renderKnowledgeLibrary(){
  renderKnowledgeOperations();
  const box=$('#knowledge-library');if(!box)return;
  const k=state.report?.analysis?.knowledge_enrichment;
  if(!k){box.innerHTML='<div class="empty">尚未选择包含参考文献的任务。</div>';return}
  const docs=k.document_ingestion||[];
  const hits=k.local_retrieval?.hits||[];
  if(!docs.length){box.innerHTML='<div class="empty">当前任务没有解析到参考文献。</div>';return}
  box.innerHTML=docs.map(d=>{const hitCount=hits.filter(h=>h.filename===d.filename).length;const url=state.report?`/v1/platform/jobs/${encodeURIComponent(state.report.job_id)}/documents/${encodeURIComponent(d.filename)}`:'#';return `<div class="file-row"><div class="file-type doc">DOC</div><div><div class="file-name">${esc(d.filename)}</div><div class="file-meta">${esc(d.parser||'')} · ${d.chunk_count??0} 片段 · 检索命中 ${hitCount}</div></div><a class="btn small" target="_blank" rel="noreferrer" href="${url}">查看</a></div>`}).join('');
}
async function loadKnowledgeWorkspace(force=false){
  const refresh=$('#seed-refresh');
  if(refresh&&!refresh.dataset.bound){refresh.dataset.bound='1';refresh.addEventListener('click',()=>loadPublicSeedGraph(true))}
  for(const id of ['seed-domain','seed-search']){const node=$('#'+id);if(node&&!node.dataset.bound){node.dataset.bound='1';node.addEventListener(id==='seed-search'?'input':'change',renderPublicSeed)}}
  await loadPublicSeedGraph(force);
  const jobId=state.currentJobId;
  if(!jobId){renderKnowledgeLibrary();return}
  const host=$('#knowledge-operations');
  try{
    if(force||state.job?.job_id!==jobId)state.job=await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}`);
    if((force||state.report?.job_id!==jobId)&&state.job?.status==='SUCCEEDED')state.report=await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}/report`);
    if(force||state.maintenance?.job_id!==jobId)state.maintenance=await api(`/v1/platform/jobs/${encodeURIComponent(jobId)}/maintenance`);
    renderKnowledgeLibrary();
  }catch(e){
    if(host)host.innerHTML=`<div class="empty">任务资料读取失败：${esc(e.message)}</div>`;
    renderKnowledgeLibrary();
  }
}
export { loadPublicSeedGraph, loadKnowledgeWorkspace, renderKnowledgeLibrary };
