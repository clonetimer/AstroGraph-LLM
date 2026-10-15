import { $, state, modeZh, esc } from './platform_core.js';
import { agentExecutionHtml, safeUiSection } from './platform_report.js';
import { probeDisplayName } from './platform_operations.js';

function renderAgentExecution(){
  const agent=state.report?.agent_execution||state.report?.analysis?.agent_execution;
  const shell=$('#agent-live-panels'),box=$('#agent-execution'),inv=$('#agent-investigation');
  if(!shell||!box||!inv)return;
  if(!agent){shell.hidden=true;box.innerHTML='';inv.innerHTML='';return}
  shell.hidden=false;
  $('#agent-execution-summary').textContent=`${modeZh[agent.work_mode]||agent.work_mode}`;
  box.innerHTML=safeUiSection('Agent 执行回执',()=>agentExecutionHtml(agent));
  let h='';
  const investigation=agent.investigation;
  if(investigation){
    const findings=(investigation.findings||[]).map(f=>{const citations=f.citation_ids?.length?`<div class="mini-tags">${f.citation_ids.map(x=>`<span class="mini-tag mono">${esc(x)}</span>`).join('')}</div>`:'';const preview=String(f.text||'').replace(/\s+/g,' ').slice(0,72)+(String(f.text||'').length>72?'…':'');const technical=f.probe_id||f.kind;return `<details class="finding"><summary><span class="badge info">${esc(probeDisplayName(technical))}</span><span>${esc(preview)}</span></summary><div class="finding-body">${esc(f.text)}</div><div class="evidence-role mono">${esc(technical||'')}</div>${citations}</details>`}).join('');
    if(findings)h+=`<div class="investigation-list">${findings}</div>`;
    if(investigation.required_observations?.length)h+=`<div class="divider"></div><div class="mini-tags">${investigation.required_observations.map(x=>`<span class="mini-tag">待补充：${esc(x)}</span>`).join('')}</div>`;
  }
  const exp=agent.exploration;
  if(exp){const proposals=exp.hypotheses||[];if(proposals.length)h+=proposals.map(p=>`<div class="proposal" style="margin-top:8px"><strong>未验证假设</strong><p>${esc(p.hypothesis)}</p><div class="evidence-role">证伪测试：${esc(p.falsification_test)}</div><div class="mini-tags">${(p.citation_ids||[]).map(x=>`<span class="mini-tag mono">${esc(x)}</span>`).join('')}</div></div>`).join('')}
  const card=$('#agent-investigation-card');
  if(!h){if(card)card.hidden=true;inv.innerHTML=''}else{if(card)card.hidden=false;inv.innerHTML=h}
}

export { renderAgentExecution };
