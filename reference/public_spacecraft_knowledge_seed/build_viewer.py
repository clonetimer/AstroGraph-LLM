#!/usr/bin/env python3
"""Produce a one-file offline SVG browser for the unreviewed public KG."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent
j=json.dumps(json.loads((R/'graph.json').read_text(encoding='utf8')),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
html=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AstroGraph · 公开航天知识种子图谱</title>
<style>
:root{color-scheme:light;--paper:#f4f6fa;--card:#fff;--line:#dce4ef;--ink:#203047;--muted:#64748b;--blue:#2666a9;--sub:#f2f6fc}*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;background:var(--paper);color:var(--ink)}
header{background:#122841;color:#fff;padding:20px clamp(16px,3vw,42px);display:flex;justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap}
header strong{display:block;font-size:20px;letter-spacing:.4px}header span{display:block;color:#a7c3de;font-size:13px;margin-top:5px}
.warn{background:#fff5e5;border:1px solid #e4c289;padding:11px 16px;color:#6d5027;font-size:13px;margin:18px clamp(16px,3vw,42px);border-radius:10px}
main{display:grid;grid-template-columns:minmax(245px,300px) minmax(400px,1fr) minmax(270px,320px);gap:14px;margin:0 clamp(16px,3vw,42px) 30px;align-items:start}
section{border:1px solid var(--line);background:var(--card);border-radius:13px;box-shadow:0 2px 9px #112b4310;min-width:0;overflow:hidden}
section>.hd{padding:16px;border-bottom:1px solid var(--line)}.hd h2{font-size:16px;margin:0}.muted{font-size:12px;color:var(--muted);line-height:1.6}
input,select{font:inherit;color:var(--ink);width:100%;padding:10px 11px;border-radius:8px;border:1px solid #cbd5e1;margin:10px 0 0;background:#fff}
.list{max-height:calc(100vh - 275px);min-height:420px;overflow:auto;padding:9px}.fault{display:block;text-align:left;border:1px solid transparent;background:transparent;width:100%;border-radius:9px;padding:10px 11px;cursor:pointer;color:var(--ink);margin-bottom:5px}.fault:hover{background:#f2f6fb}.fault.sel{background:#eaf2fb;border-color:#b5cee9}.fault b{display:block;font-size:13px;line-height:1.6}.fault small{display:block;color:#75849b;margin-top:3px;font-size:11px}
.charthead{padding:17px 19px;border-bottom:1px solid var(--line)}.charthead h2{font-size:19px;margin:0 0 6px}.charthead .tags{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.tag{font-size:11px;padding:4px 7px;background:#eaf1fa;color:#355d88;border-radius:5px}.tag.warn2{background:#fff1d7;color:#a06900}
.svgwrap{padding:10px;overflow:auto}svg{width:100%;min-width:590px;height:auto;display:block}svg .connector{stroke:#9babc1;stroke-width:1.8;fill:none;marker-end:url(#arrow)}svg .connector.extra{stroke:#c78c5c;stroke-dasharray:5 5}.vnode{cursor:pointer}.vnode rect{fill:#f8fafc;stroke:#93abc8;stroke-width:1.2;rx:11}.vnode.current rect,.vnode:hover rect{fill:#e8f3ff;stroke:#1e639e;stroke-width:2}.vnode text{font-size:12px;fill:#25364e;text-anchor:middle}.vnode .ncap{font-size:11px;fill:#68819e}.edge-summary{margin:0 15px 17px;padding:13px;background:#f4f8fd;border-radius:9px;color:#3c5474;font-size:13px;line-height:1.8}
.right{padding:16px}.right h3{margin:0 0 10px;font-size:15px}.right .row{border-bottom:1px dashed #e0e6ee;padding:10px 0;font-size:13px;line-height:1.75}.right .row:last-child{border:0}.right .small{font-size:12px;color:var(--muted)}.right a{color:var(--blue);word-break:break-all;text-decoration:none}.right a:hover{text-decoration:underline}.alt{margin-top:5px;padding:7px 9px;border-left:3px solid #e6a566;background:#fff6ea;font-size:12px}.foot{font-size:12px;color:#66758d;line-height:1.6;padding:0 40px 24px}
@media(max-width:1200px){main{grid-template-columns:245px minmax(400px,1fr)}.right-panel{grid-column:1/-1}}@media(max-width:740px){main{display:block;margin:0 12px 24px}section{margin:0 0 13px}.list{max-height:220px;min-height:120px}header{padding:20px}.warn{margin:12px}.foot{padding:0 14px 22px}}
</style></head><body>
<header><div><strong>AstroGraph · 公开航天知识图谱（初始种子）</strong><span>按分系统筛选 · 30 组故障线索 · 带文献依据的多跳候选路径</span></div><div style="font-size:12px;color:#bdd7ee">离线可用 · 无外部脚本 · 非飞行级知识</div></header>
<div class="warn"><b>知识状态：待审核。</b> 文献摘要为中文转述，跨系统关系包含工程推导；所有“可能原因／处置”均不构成实际故障确诊或遥控授权。真正型号库需要独立专家审核与版本发布。</div>
<main><section><div class="hd"><h2>故障模式目录 <span id="shown" class="muted"></span></h2><input id="filter" placeholder="搜索：飞轮、欠压、地面站、SEE…" aria-label="检索知识"><select id="domain" aria-label="按系统筛选"><option value="">全部分系统</option><option>ADCS</option><option>EPS</option><option>THERMAL</option><option>PROP</option><option>COMM</option><option>CDH</option><option>PAYLOAD</option></select></div><div class="list" id="list"></div></section>
<section><div class="charthead"><h2 id="title"></h2><div id="code" class="muted"></div><div class="tags"><span class="tag">候选机理</span><span class="tag">原文可追溯</span><span class="tag warn2">未经专家确认</span></div></div><div class="svgwrap"><svg id="graph" viewBox="0 0 930 525" role="img" aria-label="故障原因与处置知识关系图"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="#8aa0be"></path></marker></defs><g id="links"></g><g id="nodes"></g></svg></div><div id="summary" class="edge-summary"></div></section>
<section class="right-panel"><div class="right"><h3>证据与研判详情</h3><div id="details"></div></div></section></main><div class="foot">资料来自 NASA / ECSS 官方公开页面和教材。此页面与知识种子属于辅助检索工具；外部链接需要联网访问，离线查询与图示不依赖互联网。来源列表参见 sources.json。</div>
<script>const G=__DATA__;const NM=Object.fromEntries(G.entities.map(n=>[n.id,n]));const EM=Object.fromEntries(G.relations.map(e=>[e.id,e]));const SM=Object.fromEntries(G.sources.map(s=>[s.id,s]));
const P={component:[130,110],mechanism:[455,110],symptom:[780,110],fault:[130,295],verification:[455,295],action:[780,295],condition:[780,475]};
const labels={component:'相关部件',mechanism:'可能机理',symptom:'异常征兆',fault:'故障候选',verification:'复核方法',action:'建议措施',condition:'执行条件'};
let active=G.clusters.find(x=>x.code===decodeURIComponent(location.hash.slice(1)))||G.clusters[0];let selected='fault';
const $=s=>document.querySelector(s);const svg=(tag,props={})=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(props).forEach(([k,v])=>e.setAttribute(k,v));return e};
function sourceHtml(s){return '<a href="'+s.url+'" target="_blank" rel="noopener noreferrer">'+esc(s.id)+' · '+esc(s.title)+'</a>'}
function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function draw(){if(!active)return;let c=active;$('#title').textContent=c.title;$('#code').textContent=c.domain+' / '+c.code;
 let links=$('#links'),nodes=$('#nodes');links.replaceChildren();nodes.replaceChildren();
 for(let id of c.edge_ids){let e=EM[id];if(e.interpretation==='ENGINEERING_INFERENCE_CROSS_SUBSYSTEM')continue;
 let keyA=Object.keys(c.source_ids).find(k=>c.source_ids[k]===e.source);let keyB=Object.keys(c.source_ids).find(k=>c.source_ids[k]===e.target);
 if(!P[keyA]||!P[keyB])continue;
 let [x1,y1]=P[keyA],[x2,y2]=P[keyB],dx=x2-x1,dy=y2-y1,l=Math.hypot(dx,dy);
 let trim=dx?((dx/l)*96):0;let tx=dx?((dx/l)*96):0;
 let path=svg('path',{d:'M'+(x1+trim)+' '+(y1+dy/l*27)+' L'+(x2-tx)+' '+(y2-dy/l*27),class:'connector'});
 path.setAttribute('title',e.relation_type);links.appendChild(path);
 }
 for(let [key,p] of Object.entries(P)){let name=NM[c.source_ids[key]].name;let [x,y]=p;
 let g=svg('g',{class:'vnode '+(key===selected?'current':''),tabindex:'0',role:'button','aria-label':labels[key]+'：'+name});
 let r=svg('rect',{x:x-106,y:y-36,width:212,height:75});g.appendChild(r);
 let cap=svg('text',{class:'ncap',x,y:y-15});cap.textContent=labels[key];g.appendChild(cap);
 const split=name.match(/.{1,13}/g)||[name];for(let i=0;i<Math.min(2,split.length);i++){
  let t=svg('text',{x,y:y+4+i*18});t.textContent=split[i]+(i===1&&split.length>2?'…':'');g.appendChild(t);
 }g.addEventListener('click',()=>{selected=key;drawDetails();draw()});g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selected=key;drawDetails();draw()}});nodes.appendChild(g)}
 $('#summary').textContent='参考文献摘要：'+c.summary;
 drawDetails();}
function drawDetails(){let c=active;let key=selected;let v=NM[c.source_ids[key]];
 let html='<div class="row"><b>'+esc(labels[key])+'</b><div>'+esc(v.name)+'</div><div class="small">实体类型：'+esc(v.entity_type)+'</div></div>';
 for(let k of ['component','mechanism','symptom','verification','action','condition'])if(k!==key&&key==='fault')html+='<div class="row"><b>'+esc(labels[k])+'：</b> '+esc(NM[c.source_ids[k]].name)+'</div>';
 let extra=c.edge_ids.map(id=>EM[id]).filter(e=>e.interpretation==='ENGINEERING_INFERENCE_CROSS_SUBSYSTEM');
 if(extra.length){html+='<div class="row"><b>跨系统替代假设（需要核实）</b>';for(let e of extra)html+='<div class="alt">'+esc(NM[e.target].name)+'<div class="small">'+esc(e.source_summary_zh)+'</div></div>';html+='</div>'}
 html+='<div class="row"><b>文献定位</b><div>'+esc(c.source_locator)+'</div><div class="small">摘要为转述，不代表文献逐字引文</div></div>';
 for(let id of c.sources)html+='<div class="row">'+sourceHtml(SM[id])+'</div>';
 html+='<div class="row" style="color:#8b5d24"><b>使用边界：</b>仅作故障假设枚举和专家复核，不可自动确诊或操作。</div>';
 $('#details').innerHTML=html;}
function list(){let q=$('#filter').value.toLowerCase().trim(),d=$('#domain').value;
 let matches=G.clusters.filter(c=>(!d||c.domain===d)&&( !q||[c.code,c.title,c.summary,...Object.values(c.source_ids).map(id=>NM[id].name)].join('|').toLowerCase().includes(q)));
 $('#shown').textContent='('+matches.length+'/'+G.clusters.length+')';const box=$('#list');box.replaceChildren();for(let c of matches){let b=document.createElement('button');b.className='fault '+(c===active?'sel':'');let t=document.createElement('b');t.textContent=c.title;b.appendChild(t);let sm=document.createElement('small');sm.textContent=c.domain+' / '+c.code;b.appendChild(sm);b.onclick=()=>{active=c;selected='fault';location.hash=encodeURIComponent(c.code);list();draw()};box.appendChild(b)} }
$('#filter').addEventListener('input',list);$('#domain').addEventListener('change',list);list();draw();</script></body></html>'''
(R/'graph_viewer.html').write_text(html.replace('__DATA__',j),encoding='utf8')
print('wrote offline graph viewer',round((R/'graph_viewer.html').stat().st_size/1024),'KiB')
