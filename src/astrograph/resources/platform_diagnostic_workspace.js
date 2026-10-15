import { $, $$, esc, showToast } from './platform_core.js';
import { probeDisplayName } from './platform_operations.js';

const WORKSPACE_SCHEMA = 'astrograph-diagnostic-workspace-presentation.v1';
const COVERAGE_REPLAY_CONTRACT = 'BROWSER_REVIEW_COVERAGE_REPLAY_V1';
const REREVIEW_CONVERGENCE_CONTRACT = 'BROWSER_INCOMING_REREVIEW_CONVERGENCE_V1';
const PRODUCT_EXPERIENCE_STABLE_CANDIDATE_BOUNDARY = 'R35_PRODUCT_EXPERIENCE_STABLE_CANDIDATE_READ_ONLY';
const COVERAGE_REPLAY_BEGIN = 'BEGIN_REVIEW_COVERAGE_REPLAY_V1';
const COVERAGE_REPLAY_END = 'END_REVIEW_COVERAGE_REPLAY_V1';
const REPLAY_DELTA_STATUSES = new Set(['CHANGED','NOT_REPRODUCED','NOT_PROVIDED']);
const workspaceSelection = {candidateId:null, componentId:null, evidenceId:null, graphNodeId:null, observedTraceId:null, simulationTraceId:null, mainView:'overlay'};
const REVIEW_AREAS = [
  {id:'identity',label:'来源身份'}, {id:'candidate',label:'候选上下文'}, {id:'evidence',label:'Evidence'},
  {id:'graph',label:'机理图谱'}, {id:'traces',label:'遥测与仿真'}, {id:'agent',label:'Agent 回执'},
];
const workspaceReviewSession = {identity:false,candidate:false,evidence:false,graph:false,traces:false,agent:false};
const workspaceReviewCoverage = {identity:null,candidate:null,evidence:null,graph:null,traces:null,agent:null};
const workspaceIncomingCoverage = {manifest:null,raw:'',error:null};
const workspaceReplayDeltaQueue = {cursorArea:null,acknowledgements:{identity:null,candidate:null,evidence:null,graph:null,traces:null,agent:null}};
let workspaceData = {report:null, maintenance:null, job:null, visualization:null, diagnosticWorkspace:null};

function resetWorkspaceSelection(){
  Object.assign(workspaceSelection,{candidateId:null,componentId:null,evidenceId:null,graphNodeId:null,observedTraceId:null,simulationTraceId:null,mainView:'overlay'});
}
function workspaceSnapshot(){return {...workspaceSelection}}
function resetWorkspaceReviewSession(){REVIEW_AREAS.forEach(({id})=>{workspaceReviewSession[id]=false;workspaceReviewCoverage[id]=null})}
function reviewSessionSnapshot(){return {...workspaceReviewSession}}
function reviewCoverageSnapshot(){return Object.fromEntries(REVIEW_AREAS.map(({id})=>[id,workspaceReviewCoverage[id]?{...workspaceReviewCoverage[id],basis:[...workspaceReviewCoverage[id].basis]}:null]))}
function resetReplayDeltaQueue(){workspaceReplayDeltaQueue.cursorArea=null;REVIEW_AREAS.forEach(({id})=>{workspaceReplayDeltaQueue.acknowledgements[id]=null})}
function resetIncomingReviewCoverage(){workspaceIncomingCoverage.manifest=null;workspaceIncomingCoverage.raw='';workspaceIncomingCoverage.error=null;resetReplayDeltaQueue()}
function incomingReviewCoverageSnapshot(){return workspaceIncomingCoverage.manifest?JSON.parse(JSON.stringify(workspaceIncomingCoverage.manifest)):null}
function setReviewAreaReviewed(area,reviewed=true){
  if(Object.prototype.hasOwnProperty.call(workspaceReviewSession,area)){
    workspaceReviewSession[area]=Boolean(reviewed);
    workspaceReviewCoverage[area]=reviewed?reviewCoverageRecord(validWorkspaceProjection(workspaceData.diagnosticWorkspace)||{},area):null;
  }
  renderWorkspacePanels();return reviewSessionSnapshot();
}
function setWorkspaceSelection(patch={}){Object.assign(workspaceSelection,patch);renderWorkspacePanels();return workspaceSnapshot()}
function idOf(value,fallback=''){
  if(value&&typeof value==='object')return String(value.id||value.trace_id||value.link_id||value.evidence_id||value.candidate_id||value.fault_id||value.component_id||value.name||fallback);
  return String(value??fallback);
}
function validPoints(points){
  return (Array.isArray(points)?points:[]).map(p=>({t:Number(p.t??p.time_s??p.x),v:Number(p.v??p.value??p.y)})).filter(p=>Number.isFinite(p.t)&&Number.isFinite(p.v));
}
function validWorkspaceProjection(value){return value?.schema_version===WORKSPACE_SCHEMA?value:null}
function simulationTraces(source={}){
  const raw=source.simulation_traces||source.simulation_review?.numerical_traces||source.simulation_review?.traces||[];
  return (Array.isArray(raw)?raw:[]).map((trace,i)=>({
    id:idOf(trace,`simulation-${i}`), label:trace.label||trace.name||trace.candidate_id||trace.channel||`Simulation ${i+1}`,
    channel:trace.channel||trace.observed_channel||null, unit:trace.unit||'', semantics:trace.semantics||null,
    providerId:trace.provider_id||null, diagnosisId:trace.diagnosis_id||null, candidateId:trace.candidate_id||null,
    sourceTraceSha256:trace.source_trace_sha256||trace.trace_sha256||null,
    points:validPoints(trace.points||trace.samples),
  })).filter(x=>x.points.length>=2);
}
function observedTraces(source={}){
  const raw=source.observed_traces||source.channels||[];
  return (Array.isArray(raw)?raw:[]).map((trace,i)=>({
    id:idOf(trace,`observed-${i}`), label:trace.label||trace.name||trace.channel||`Observed ${i+1}`,
    channel:trace.channel||trace.name||null, unit:trace.unit||'', points:validPoints(trace.points||trace.samples),
  })).filter(x=>x.points.length>=2);
}
function workspaceAvailability(projection={}){
  const graph=projection.mechanism_graph||{},nodes=Array.isArray(graph.nodes)?graph.nodes:[];
  const observed=observedTraces(projection),simulated=simulationTraces(projection);
  const evidence=Array.isArray(projection.evidence_links)?projection.evidence_links:[];
  const agent=Array.isArray(projection.agent_links)?projection.agent_links:[];
  return {graphAvailable:graph.status==='AVAILABLE'&&nodes.length>0,graphNodes:nodes.length,observed:observed.length,simulated:simulated.length,evidence:evidence.length,agent:agent.length};
}
function overviewHtml(projection){
  if(!projection)return '<div class="dw-overview unavailable"><div><strong>诊断工作区投影不可用</strong><span>未展示图谱、轨迹、Evidence 或 Agent 材料；浏览器不会从其他页面内容推断补齐。</span></div></div>';
  const a=workspaceAvailability(projection);
  const cards=[
    ['graph','图谱',a.graphAvailable?`${a.graphNodes} 个节点`:'不可用',a.graphAvailable],
    ['overlay','遥测',a.observed?`${a.observed} 条观测`:'不可用',a.observed>0],
    ['overlay','数值仿真',a.simulated?`${a.simulated} 条轨迹`:'不可用',a.simulated>0],
    ['support','正式 Evidence',a.evidence?`${a.evidence} 项`:'不可用',a.evidence>0],
    ['support','Agent 回执',a.agent?`${a.agent} 项`:'不可用',a.agent>0],
  ];
  return `<div class="dw-overview" aria-label="工作区材料概览">${cards.map(([target,label,value,available])=>`<button type="button" class="dw-overview-item ${available?'available':'unavailable'}" data-dw-jump="${target}" ${available?'':'disabled'}><span>${esc(label)}</span><strong>${esc(value)}</strong></button>`).join('')}</div>`;
}
function labelForId(items,id,fallback){
  if(!id)return fallback;
  const item=(Array.isArray(items)?items:[]).find(x=>idOf(x)===id);
  return item?.label||item?.name||item?.evidence_id||item?.candidate_id||item?.fault_id||id;
}
function selectionSummaryHtml(projection={}){
  const context=projection.candidate_context||{},graph=projection.mechanism_graph||{},evidence=projection.evidence_links||[];
  const observed=observedTraces(projection),simulated=simulationTraces(projection);
  const candidate=workspaceSelection.candidateId||context.selected_fault_id||'未形成';
  const evidenceLabel=labelForId(evidence,workspaceSelection.evidenceId,'未选择');
  const nodeLabel=labelForId(graph.nodes,workspaceSelection.graphNodeId,'未选择');
  const observedId=workspaceSelection.observedTraceId&&observed.some(x=>x.id===workspaceSelection.observedTraceId)?workspaceSelection.observedTraceId:(observed[0]?.id||null);
  const observedLabel=labelForId(observed,observedId,'不可用');
  const simulationLabel=!simulated.length?'不可用':workspaceSelection.simulationTraceId==='__NONE__'?'未叠加':labelForId(simulated,workspaceSelection.simulationTraceId,'自动匹配');
  const alternativeSelected=Boolean(workspaceSelection.candidateId&&context.selected_fault_id&&workspaceSelection.candidateId!==context.selected_fault_id);
  const note=alternativeSelected?`<div class="dw-selection-note">当前只是浏览备选候选 <strong>${esc(workspaceSelection.candidateId)}</strong>；图谱、Evidence 与仿真材料仍来自并绑定冻结候选 <strong>${esc(context.selected_fault_id)}</strong>，不会被浏览选择重新解释。</div>`:'';
  return `<div class="dw-selection-bar" aria-live="polite"><div><span>当前候选</span><strong>${esc(candidate)}</strong></div><div><span>Evidence</span><strong>${esc(evidenceLabel)}</strong></div><div><span>图谱节点</span><strong>${esc(nodeLabel)}</strong></div><div><span>观测轨迹</span><strong>${esc(observedLabel)}</strong></div><div><span>仿真轨迹</span><strong>${esc(simulationLabel)}</strong></div><button type="button" class="btn small" id="dw-reset-context">恢复默认上下文</button></div>${note}`;
}

function identityValue(value,fallback='未提供'){return value===null||value===undefined||value===''?fallback:String(value)}
function effectiveTraceContext(projection={}){
  const observed=observedTraces(projection),simulated=simulationTraces(projection);
  const observedId=workspaceSelection.observedTraceId&&observed.some(x=>x.id===workspaceSelection.observedTraceId)?workspaceSelection.observedTraceId:(observed[0]?.id||null);
  const observedTrace=observed.find(x=>x.id===observedId)||null;
  const compatible=observedTrace?simulated.filter(x=>!x.channel||x.channel===observedTrace.channel):[];
  const simulationTrace=workspaceSelection.simulationTraceId==='__NONE__'?null:(compatible.find(x=>x.id===workspaceSelection.simulationTraceId)||compatible[0]||null);
  return {observedTrace,simulationTrace};
}
function reviewAttentionSignals(projection={}){
  const signals=[],evidence=Array.isArray(projection.evidence_links)?projection.evidence_links:[],agent=Array.isArray(projection.agent_links)?projection.agent_links:[];
  const graph=projection.mechanism_graph||{},context=projection.candidate_context||{},frozenCandidate=context.selected_fault_id||null;
  const contradicting=evidence.filter(x=>x.role==='CONTRADICTING').length,missing=evidence.filter(x=>x.role==='MISSING').length,pendingAgent=agent.filter(x=>x.status!=='COMPLETED').length;
  if(projection.source_bundle_status&&projection.source_bundle_status!=='VERIFIED')signals.push({code:'SOURCE_BUNDLE_NOT_VERIFIED',label:`来源包状态为 ${projection.source_bundle_status}`});
  if(contradicting)signals.push({code:'CONTRADICTING_EVIDENCE_PRESENT',label:`投影包含 ${contradicting} 项显式反证`});
  if(missing)signals.push({code:'MISSING_EVIDENCE_DECLARED',label:`投影声明 ${missing} 项缺失证据`});
  if(pendingAgent)signals.push({code:'AGENT_STEP_INCOMPLETE',label:`${pendingAgent} 个 Agent 回执尚未标记 COMPLETED`});
  if(graph.status!=='AVAILABLE'||!(Array.isArray(graph.nodes)&&graph.nodes.length))signals.push({code:'MECHANISM_GRAPH_UNAVAILABLE',label:'当前投影没有可交互机理图谱'});
  if(!observedTraces(projection).length)signals.push({code:'OBSERVED_TELEMETRY_UNAVAILABLE',label:'当前投影没有可叠加 observed telemetry'});
  if(!simulationTraces(projection).length)signals.push({code:'NUMERIC_SIMULATION_UNAVAILABLE',label:'当前投影没有数值仿真轨迹'});
  if(workspaceSelection.candidateId&&frozenCandidate&&workspaceSelection.candidateId!==frozenCandidate)signals.push({code:'TRANSIENT_CANDIDATE_BROWSE',label:`当前浏览 ${workspaceSelection.candidateId}，冻结候选仍为 ${frozenCandidate}`});
  const {simulationTrace}=effectiveTraceContext(projection);
  if(simulationTrace?.candidateId&&frozenCandidate&&simulationTrace.candidateId!==frozenCandidate)signals.push({code:'SIMULATION_CANDIDATE_IDENTITY_MISMATCH',label:`当前仿真轨迹 candidate=${simulationTrace.candidateId}，冻结候选=${frozenCandidate}`});
  const coverageDrift=reviewCoverageDrift(projection);
  if(coverageDrift.length)signals.push({code:'REVIEW_COVERAGE_CONTEXT_CHANGED',label:`${coverageDrift.length} 个已复核区域的当前材料范围与捕获记录不同：${coverageDrift.map(x=>x.label).join('、')}`});
  return signals;
}
function coverageValue(value,fallback='none'){return value===null||value===undefined||value===''?fallback:String(value).replaceAll('|','%7C').replaceAll('\n',' ')}
function sortedCoverage(values=[]){return values.map(x=>coverageValue(x)).sort()}
function reviewCoverageBasis(projection={},area){
  const authority=projection.authority_snapshot||{},context=projection.candidate_context||{},graph=projection.mechanism_graph||{};
  const evidence=Array.isArray(projection.evidence_links)?projection.evidence_links:[],agent=Array.isArray(projection.agent_links)?projection.agent_links:[];
  const observed=observedTraces(projection),simulated=simulationTraces(projection),{observedTrace,simulationTrace}=effectiveTraceContext(projection);
  const frozen=context.selected_fault_id||null,browsing=workspaceSelection.candidateId||frozen,candidateMode=workspaceSelection.candidateId&&frozen&&workspaceSelection.candidateId!==frozen?'TRANSIENT_ALTERNATIVE_BROWSE_ONLY':'FROZEN_REPORT_CANDIDATE';
  const common=[`presentation=${coverageValue(projection.presentation_sha256)}`,`source_bundle=${coverageValue(projection.source_bundle_sha256)}`];
  if(area==='identity')return [...common,`job=${coverageValue(projection.job_id)}`,`source_status=${coverageValue(projection.source_bundle_status)}`,`graph_provenance=${coverageValue(graph.provenance_sha256)}`,`evidence_bundle=${coverageValue(authority.evidence_bundle_sha256)}`];
  if(area==='candidate')return [...common,`frozen_candidate=${coverageValue(frozen)}`,`component=${coverageValue(context.component_id)}`,`candidate_view=${coverageValue(browsing)}`,`candidate_mode=${candidateMode}`];
  if(area==='evidence')return [...common,`evidence_bundle=${coverageValue(authority.evidence_bundle_sha256)}`,`evidence_items=${sortedCoverage(evidence.map(x=>`${coverageValue(x.role)}:${idOf(x)}`)).join(',')||'none'}`,`selected_evidence=${coverageValue(workspaceSelection.evidenceId)}`];
  if(area==='graph')return [...common,`graph_status=${coverageValue(graph.status)}`,`graph_provenance=${coverageValue(graph.provenance_sha256)}`,`graph_nodes=${sortedCoverage((graph.nodes||[]).map(x=>idOf(x))).join(',')||'none'}`,`graph_edges=${Array.isArray(graph.edges)?graph.edges.length:0}`,`selected_node=${coverageValue(workspaceSelection.graphNodeId)}`];
  if(area==='traces')return [...common,`observed_scope=${sortedCoverage(observed.map(x=>`${x.id}:${coverageValue(x.channel)}`)).join(',')||'none'}`,`simulation_scope=${sortedCoverage(simulated.map(x=>`${x.id}:${coverageValue(x.providerId)}:${coverageValue(x.candidateId)}:${coverageValue(x.sourceTraceSha256)}`)).join(',')||'none'}`,`observed_view=${coverageValue(observedTrace?.id)}`,`simulation_view=${coverageValue(simulationTrace?.id)}`];
  if(area==='agent')return [...common,`agent_scope=${sortedCoverage(agent.map(x=>`${coverageValue(x.link_id||idOf(x))}:${coverageValue(x.probe_id)}:${coverageValue(x.status)}`)).join(',')||'none'}`];
  return [...common];
}
function reviewCoverageRecord(projection={},area){
  const meta=REVIEW_AREAS.find(x=>x.id===area);if(!meta)return null;
  const basis=reviewCoverageBasis(projection,area);
  return {area,areaLabel:meta.label,presentationSha256:projection.presentation_sha256||null,sourceBundleSha256:projection.source_bundle_sha256||null,basis,coverageKey:basis.join('|')};
}
function reviewCoverageDrift(projection={}){
  return REVIEW_AREAS.filter(({id})=>workspaceReviewSession[id]&&workspaceReviewCoverage[id]).map(({id,label})=>{
    const captured=workspaceReviewCoverage[id],current=reviewCoverageRecord(projection,id);
    return current&&captured.coverageKey!==current.coverageKey?{area:id,label,captured,current}:null;
  }).filter(Boolean);
}
function reviewCoverageReplayManifest(projection={}){
  const valid=validWorkspaceProjection(projection)||{};
  return {contract:COVERAGE_REPLAY_CONTRACT,source:{job_id:valid.job_id||null,presentation_sha256:valid.presentation_sha256||null,source_bundle_sha256:valid.source_bundle_sha256||null},areas:Object.fromEntries(REVIEW_AREAS.map(({id,label})=>{
    const record=workspaceReviewCoverage[id];return [id,record?{area:id,label,coverage_key:record.coverageKey,basis:[...record.basis]}:null];
  }))};
}
function reviewCoverageReplayManifestText(projection={}){return JSON.stringify(reviewCoverageReplayManifest(projection),null,2)}
function reviewCoverageReplayManifestBlock(projection={}){return `${COVERAGE_REPLAY_BEGIN}\n${reviewCoverageReplayManifestText(projection)}\n${COVERAGE_REPLAY_END}`}
function replaySourceValue(lines,prefix){const line=lines.find(x=>x.startsWith(prefix));return line?line.slice(prefix.length).trim()||null:null}
function legacyReviewCoverageReplay(text){
  const lines=String(text||'').split(/\r?\n/),areas={};let found=0;
  REVIEW_AREAS.forEach(({id,label})=>{const prefix=`Coverage ${id} (`;const line=lines.find(x=>x.startsWith(prefix));if(!line){areas[id]=null;return}found+=1;const value=line.slice(line.indexOf(':')+1).trim();if(value==='NOT_CAPTURED'){areas[id]=null;return}const basis=value.split(' | ').filter(Boolean);areas[id]={area:id,label,coverage_key:basis.join('|'),basis};});
  if(!found)throw new Error('未识别到 coverage replay manifest 或 R35.5 handoff coverage 行');
  return {contract:COVERAGE_REPLAY_CONTRACT,import_format:'R35_5_HANDOFF_COVERAGE_LINES',source:{job_id:replaySourceValue(lines,'Job ID:'),presentation_sha256:replaySourceValue(lines,'Presentation SHA256:'),source_bundle_sha256:replaySourceValue(lines,'Source bundle SHA256:')},areas};
}
function normalizeReplayRecord(area,label,value){
  if(value===null||value===undefined)return null;
  if(!value||typeof value!=='object'||!Array.isArray(value.basis)||value.basis.some(x=>typeof x!=='string'))throw new Error(`coverage area ${area} 格式无效`);
  const basis=value.basis.map(x=>String(x));const coverageKey=basis.join('|');
  if(value.coverage_key&&String(value.coverage_key)!==coverageKey)throw new Error(`coverage area ${area} coverage_key 与 basis 不一致`);
  return {area,label,coverage_key:coverageKey,basis};
}
function parseReviewCoverageReplay(text=''){
  const raw=String(text??'').trim();if(!raw)throw new Error('请粘贴上一班 handoff 或 coverage manifest');if(raw.length>100000)throw new Error('coverage replay 输入超过 100000 字符限制');
  let candidate=raw;const begin=raw.indexOf(COVERAGE_REPLAY_BEGIN),end=raw.indexOf(COVERAGE_REPLAY_END);
  if(begin>=0&&end>begin){const start=raw.indexOf('\n',begin);candidate=raw.slice(start>=0?start+1:begin+COVERAGE_REPLAY_BEGIN.length,end).trim();}
  let parsed;try{parsed=JSON.parse(candidate)}catch(_){return legacyReviewCoverageReplay(raw)}
  if(!parsed||typeof parsed!=='object'||parsed.contract!==COVERAGE_REPLAY_CONTRACT||!parsed.areas||typeof parsed.areas!=='object')throw new Error(`仅接受 ${COVERAGE_REPLAY_CONTRACT} 或 R35.5 handoff coverage 行`);
  const source=parsed.source&&typeof parsed.source==='object'?parsed.source:{};
  return {contract:COVERAGE_REPLAY_CONTRACT,import_format:'REPLAY_MANIFEST_V1',source:{job_id:source.job_id==null?null:String(source.job_id),presentation_sha256:source.presentation_sha256==null?null:String(source.presentation_sha256),source_bundle_sha256:source.source_bundle_sha256==null?null:String(source.source_bundle_sha256)},areas:Object.fromEntries(REVIEW_AREAS.map(({id,label})=>[id,normalizeReplayRecord(id,label,parsed.areas[id])]))};
}
function loadIncomingReviewCoverage(text=''){
  resetReplayDeltaQueue();
  workspaceIncomingCoverage.raw=String(text??'');
  try{workspaceIncomingCoverage.manifest=parseReviewCoverageReplay(workspaceIncomingCoverage.raw);workspaceIncomingCoverage.error=null}catch(e){workspaceIncomingCoverage.manifest=null;workspaceIncomingCoverage.error=e?.message||String(e)}
  renderWorkspacePanels();return incomingReviewCoverageSnapshot();
}
function clearIncomingReviewCoverage(){resetIncomingReviewCoverage();renderWorkspacePanels()}
function replayBasisValue(record,key){const prefix=`${key}=`;const item=(record?.basis||[]).find(x=>x.startsWith(prefix));return item?item.slice(prefix.length):null}
function replayRecordReproducible(projection={},area,incoming,current){
  if(!validWorkspaceProjection(projection)||!current)return false;
  const missing=(key)=>replayBasisValue(incoming,key)!==null&&replayBasisValue(incoming,key)!=='none'&&(replayBasisValue(current,key)===null||replayBasisValue(current,key)==='none');
  if(area==='candidate')return !missing('frozen_candidate')&&!missing('component');
  if(area==='evidence')return !missing('evidence_items');
  if(area==='graph')return !missing('graph_nodes')&&!(replayBasisValue(incoming,'graph_status')==='AVAILABLE'&&replayBasisValue(current,'graph_status')!=='AVAILABLE');
  if(area==='traces')return !missing('observed_scope')&&!missing('simulation_scope');
  if(area==='agent')return !missing('agent_scope');
  return !missing('presentation')&&!missing('source_bundle');
}
function reviewCoverageReplayComparison(projection={}){
  const manifest=workspaceIncomingCoverage.manifest;if(!manifest)return [];
  return REVIEW_AREAS.map(({id,label})=>{
    const incoming=manifest.areas[id]||null,current=reviewCoverageRecord(projection,id);
    if(!incoming)return {area:id,label,status:'NOT_PROVIDED',incoming:null,current};
    if(!replayRecordReproducible(projection,id,incoming,current))return {area:id,label,status:'NOT_REPRODUCED',incoming,current};
    return {area:id,label,status:incoming.coverage_key===current.coverageKey?'MATCHED':'CHANGED',incoming,current};
  });
}
function replayBasisEntries(record){
  return (record?.basis||[]).map((item,index)=>{const text=String(item),split=text.indexOf('=');return {key:split>=0?text.slice(0,split):`basis_${index+1}`,value:split>=0?text.slice(split+1):text}});
}
function replayBasisDiff(item){
  if(!item||item.status!=='CHANGED'||!item.incoming||!item.current)return [];
  const incomingEntries=replayBasisEntries(item.incoming),currentEntries=replayBasisEntries(item.current);
  const incoming=new Map(incomingEntries.map(x=>[x.key,x.value])),current=new Map(currentEntries.map(x=>[x.key,x.value]));
  const keys=[...incomingEntries.map(x=>x.key),...currentEntries.map(x=>x.key).filter(key=>!incoming.has(key))];
  return keys.map(key=>{const hasIncoming=incoming.has(key),hasCurrent=current.has(key),incomingValue=hasIncoming?incoming.get(key):null,currentValue=hasCurrent?current.get(key):null;let status='SAME';if(!hasIncoming)status='CURRENT_ONLY';else if(!hasCurrent)status='INCOMING_ONLY';else if(incomingValue!==currentValue)status='VALUE_CHANGED';return {key,status,incoming:incomingValue,current:currentValue}});
}
function replayBasisDiffSnapshot(projection={}){
  return reviewCoverageReplayComparison(projection).filter(x=>x.status==='CHANGED').map(x=>{const fields=replayBasisDiff(x);return {area:x.area,label:x.label,status:x.status,changedFields:fields.filter(field=>field.status!=='SAME').map(field=>field.key),fields}});
}
function replayBasisDiffHtml(item){
  if(!item||item.status!=='CHANGED')return '';
  const fields=replayBasisDiff(item),changed=fields.filter(field=>field.status!=='SAME');
  const rows=fields.map(field=>`<div class="dw-replay-basis-row ${field.status.toLowerCase()}" data-dw-replay-basis-field="${esc(field.key)}"><code>${esc(field.key)}</code><span>${esc(field.status)}</span><pre>${esc(field.incoming===null?'∅':field.incoming)}</pre><pre>${esc(field.current===null?'∅':field.current)}</pre></div>`).join('');
  return `<details class="dw-replay-basis-diff" data-dw-replay-basis-area="${esc(item.area)}"><summary>字段级 basis diff · ${changed.length} 项变化</summary><div class="dw-replay-basis-head"><span>字段</span><span>状态</span><span>上一班 incoming</span><span>当前 current</span></div><div class="dw-replay-basis-list">${rows}</div><p>这里只对已导入与当前 coverage basis 的 exact key/value 做机械并列；不推断变化原因、不判断哪一侧正确，也不表示差异已处置。</p></details>`;
}
function replayDeltaFingerprint(item){
  if(!item)return null;
  const incoming=item.incoming?.coverage_key||'none',current=item.current?.coverageKey||'none';
  return `${item.area}|${item.status}|incoming=${incoming}|current=${current}`;
}
function reviewReplayDeltaQueue(projection={}){
  if(!workspaceIncomingCoverage.manifest)return [];
  return reviewCoverageReplayComparison(projection).filter(x=>REPLAY_DELTA_STATUSES.has(x.status)).map(x=>{
    const fingerprint=replayDeltaFingerprint(x),changedFields=x.status==='CHANGED'?replayBasisDiff(x).filter(field=>field.status!=='SAME').map(field=>field.key):[];
    return {...x,fingerprint,changedFields,revisited:workspaceReplayDeltaQueue.acknowledgements[x.area]===fingerprint};
  });
}
function reviewReplayDeltaQueueSnapshot(projection={}){
  return reviewReplayDeltaQueue(projection).map(x=>({area:x.area,label:x.label,status:x.status,fingerprint:x.fingerprint,changedFields:[...x.changedFields],revisited:x.revisited}));
}
function incomingReplayConvergenceState(projection={}){
  if(!workspaceIncomingCoverage.manifest)return {contract:REREVIEW_CONVERGENCE_CONTRACT,status:'NO_INCOMING_REPLAY',matchedCount:0,deltaCount:0,revisitedCount:0,pendingCount:0,matchedAreas:[],deltaAreas:[],revisitedAreas:[],pendingAreas:[]};
  const comparison=reviewCoverageReplayComparison(projection),queue=reviewReplayDeltaQueue(projection);
  const matched=comparison.filter(x=>x.status==='MATCHED'),revisited=queue.filter(x=>x.revisited),pending=queue.filter(x=>!x.revisited);
  const status=!queue.length?'ALL_MATCHED':pending.length?'REREVIEW_PENDING':'DELTAS_ALL_REVISITED';
  return {contract:REREVIEW_CONVERGENCE_CONTRACT,status,matchedCount:matched.length,deltaCount:queue.length,revisitedCount:revisited.length,pendingCount:pending.length,matchedAreas:matched.map(x=>x.area),deltaAreas:queue.map(x=>x.area),revisitedAreas:revisited.map(x=>x.area),pendingAreas:pending.map(x=>x.area)};
}
function incomingReplayConvergenceHtml(projection={}){
  if(!workspaceIncomingCoverage.manifest)return '';
  const state=incomingReplayConvergenceState(projection);
  const labels={ALL_MATCHED:'上一班范围全部一致',DELTAS_ALL_REVISITED:'存在差异范围 · 已全部重新查看',REREVIEW_PENDING:'仍有待重新查看范围'};
  const detail=state.status==='ALL_MATCHED'
    ?`上一班提供的 ${state.matchedCount} 个区域当前 deterministic coverage key 均一致；没有 replay delta 进入重新查看队列。`
    :state.status==='DELTAS_ALL_REVISITED'
      ?`当前仍有 ${state.deltaCount} 个 replay delta，但这些当前 comparison fingerprint 均已显式标记 REVISITED；原比较状态没有被改写。`
      :`当前有 ${state.pendingCount}/${state.deltaCount} 个 replay delta 仍为 PENDING：${state.pendingAreas.join('、')||'none'}。`;
  return `<div class="dw-rereview-convergence ${state.status.toLowerCase()}" data-dw-rereview-convergence="${state.status}" aria-live="polite"><div><span>接手重新查看状态</span><strong>${esc(labels[state.status])}</strong><code>${state.status}</code></div><div class="dw-rereview-convergence-counts"><span>MATCHED <strong>${state.matchedCount}</strong></span><span>DELTA <strong>${state.deltaCount}</strong></span><span>REVISITED <strong>${state.revisitedCount}</strong></span><span>PENDING <strong>${state.pendingCount}</strong></span></div><p>${esc(detail)}</p><small>${REREVIEW_CONVERGENCE_CONTRACT} · 只汇总当前 replay/re-review 会话状态；不是 discrepancy resolution、批准、签名或 release gate。</small></div>`;
}
function activeReplayDeltaArea(projection={}){
  const queue=reviewReplayDeltaQueue(projection);if(!queue.length){workspaceReplayDeltaQueue.cursorArea=null;return null}
  if(!queue.some(x=>x.area===workspaceReplayDeltaQueue.cursorArea))workspaceReplayDeltaQueue.cursorArea=(queue.find(x=>!x.revisited)||queue[0]).area;
  return workspaceReplayDeltaQueue.cursorArea;
}
function setReplayDeltaQueueRevisited(area,revisited=true,projection=validWorkspaceProjection(workspaceData.diagnosticWorkspace)||{}){
  const item=reviewReplayDeltaQueue(projection).find(x=>x.area===area);if(!item)return reviewReplayDeltaQueueSnapshot(projection);
  workspaceReplayDeltaQueue.cursorArea=area;workspaceReplayDeltaQueue.acknowledgements[area]=revisited?item.fingerprint:null;
  renderWorkspacePanels();return reviewReplayDeltaQueueSnapshot(projection);
}
function advanceReplayDeltaQueue(direction=1,projection=validWorkspaceProjection(workspaceData.diagnosticWorkspace)||{}){
  const queue=reviewReplayDeltaQueue(projection);if(!queue.length){workspaceReplayDeltaQueue.cursorArea=null;renderWorkspacePanels();return null}
  const current=activeReplayDeltaArea(projection),index=Math.max(0,queue.findIndex(x=>x.area===current));
  const step=Number(direction)<0?-1:1,next=queue[(index+step+queue.length)%queue.length];workspaceReplayDeltaQueue.cursorArea=next.area;
  renderWorkspacePanels();requestAnimationFrame(()=>{const target=$(`[data-dw-rereview-area="${next.area}"]`);target?.scrollIntoView?.({block:'center'});try{target?.focus?.({preventScroll:true})}catch(_){target?.focus?.()}});return next.area;
}
function navigateReplayDeltaQueueItem(projection={},area){
  const item=reviewReplayDeltaQueue(projection).find(x=>x.area===area);if(!item)return null;workspaceReplayDeltaQueue.cursorArea=area;return navigateReviewItem(projection,'area',area);
}
function reviewCoverageReplayHtml(projection={}){
  const manifest=workspaceIncomingCoverage.manifest,comparison=reviewCoverageReplayComparison(projection),error=workspaceIncomingCoverage.error;
  const statusLabel={MATCHED:'范围一致',CHANGED:'范围变化',NOT_REPRODUCED:'当前未能复现',NOT_PROVIDED:'上一班未提供'};
  const rows=manifest?comparison.map(x=>`<div class="dw-review-replay-row ${x.status.toLowerCase()}" data-dw-replay-area="${x.area}"><div class="dw-review-replay-row-main"><div><strong>${esc(x.label)}</strong><code>${x.status}</code><span>${esc(statusLabel[x.status])}</span></div><button type="button" data-dw-replay-nav-area="${x.area}">定位当前区域</button></div>${replayBasisDiffHtml(x)}</div>`).join(''):'';
  const counts=manifest?Object.fromEntries(['MATCHED','CHANGED','NOT_REPRODUCED','NOT_PROVIDED'].map(k=>[k,comparison.filter(x=>x.status===k).length])):null;
  const source=manifest?.source||{};
  return `<div class="dw-review-replay" aria-label="上一班复核范围复现对照"><div class="dw-review-replay-head"><div><strong>交接范围复现对照</strong><span>${COVERAGE_REPLAY_CONTRACT} · 仅当前页面</span></div>${manifest?`<span>${counts.MATCHED} 一致 / ${counts.CHANGED} 变化 / ${counts.NOT_REPRODUCED} 未复现 / ${counts.NOT_PROVIDED} 未提供</span>`:'<span>尚未加载上一班范围</span>'}</div><label class="dw-review-replay-input"><span>粘贴上一班交接摘要或 coverage manifest</span><textarea id="dw-incoming-coverage-manifest" rows="5" placeholder="可粘贴完整 handoff，或 BEGIN_REVIEW_COVERAGE_REPLAY_V1 ... END_REVIEW_COVERAGE_REPLAY_V1">${esc(workspaceIncomingCoverage.raw)}</textarea></label>${error?`<div class="dw-review-replay-error" role="alert">${esc(error)}</div>`:''}<div class="dw-review-replay-actions"><button type="button" class="btn small" id="dw-load-incoming-coverage">对照当前工作区</button><button type="button" class="btn small" id="dw-copy-coverage-manifest">复制当前 coverage manifest</button><button type="button" class="btn small" id="dw-clear-incoming-coverage" ${manifest||workspaceIncomingCoverage.raw?'':'disabled'}>清除对照</button></div>${manifest?`<div class="dw-review-replay-source"><span>输入来源</span><code>format=${esc(manifest.import_format||'REPLAY_MANIFEST_V1')} | job=${esc(identityValue(source.job_id))} | presentation=${esc(identityValue(source.presentation_sha256))} | source_bundle=${esc(identityValue(source.source_bundle_sha256))}</code></div><div class="dw-review-replay-grid">${rows}</div>`:'<div class="dw-review-replay-empty">加载后仅比较上一班捕获范围与当前可见投影/临时视图范围；不会自动恢复 selection、不会标记已复核，也不会改变 closure 或 attention signal。</div>'}<p>比较结果只用于人工复现交接范围：MATCHED 表示 deterministic coverage key 相同；CHANGED 表示两边均可比较但范围不同；NOT_REPRODUCED 表示上一班记录的显式材料在当前投影中缺少可比对象；NOT_PROVIDED 表示上一班没有该区域 coverage record。它们都不是审计结论、差异处置或批准。</p></div>`;
}
function reviewReplayDeltaQueueHtml(projection={}){
  const manifest=workspaceIncomingCoverage.manifest;if(!manifest)return '';
  const queue=reviewReplayDeltaQueue(projection),active=activeReplayDeltaArea(projection);
  if(!queue.length)return '<div class="dw-rereview-queue clear"><strong>接手重新查看队列</strong><span>上一班提供的区域当前全部为 MATCHED；没有 replay delta 需要排队重新查看。</span></div>';
  const pending=queue.filter(x=>!x.revisited),rows=queue.map(x=>`<div class="dw-rereview-row ${x.revisited?'revisited':'pending'} ${x.area===active?'active':''}" data-dw-rereview-area="${x.area}" tabindex="-1"><div><span>${esc(x.status)}</span><strong>${esc(x.label)}</strong><small>${x.revisited?'REVISITED · 当前 comparison fingerprint 已重新查看':'PENDING · 需要人工重新查看当前区域'}</small>${x.changedFields.length?`<small>变化字段：${esc(x.changedFields.join('、'))}</small>`:''}</div><div class="dw-rereview-actions"><button type="button" data-dw-rereview-nav-area="${x.area}">定位</button><button type="button" data-dw-rereview-toggle-area="${x.area}" aria-pressed="${x.revisited}">${x.revisited?'撤销重新查看':'标记已重新查看'}</button></div></div>`).join('');
  return `<div class="dw-rereview-queue" aria-label="接手重新查看队列"><div class="dw-rereview-head"><div><strong>接手重新查看队列</strong><span>仅收纳 CHANGED / NOT_REPRODUCED / NOT_PROVIDED；MATCHED 不进入队列。</span></div><span>${pending.length} 待查看 / ${queue.length} 项</span></div><div class="dw-rereview-guide"><button type="button" class="btn small" id="dw-rereview-prev">上一项</button><code>${esc(active||'none')}</code><button type="button" class="btn small" id="dw-rereview-next">下一项</button></div><div class="dw-rereview-list">${rows}</div><p>REVISITED 只表示接手人在当前 comparison fingerprint 下重新查看过该区域；它不会把 CHANGED/NOT_REPRODUCED/NOT_PROVIDED 改成 MATCHED，也不表示差异已处置、批准或 release。若当前材料范围再次变化，旧 fingerprint 的重新查看标记不会套用到新范围。</p></div>`;
}
function reviewCoverageHtml(projection={}){
  const records=REVIEW_AREAS.filter(({id})=>workspaceReviewSession[id]&&workspaceReviewCoverage[id]),drift=new Set(reviewCoverageDrift(projection).map(x=>x.area));
  if(!records.length)return '<div class="dw-review-coverage empty"><strong>复核覆盖记录</strong><span>尚未捕获覆盖记录；区域标记已复核时才会在当前页面内存中记录其材料身份范围。</span></div>';
  const rows=records.map(({id,label})=>{const record=workspaceReviewCoverage[id],changed=drift.has(id);return `<div class="dw-review-coverage-row ${changed?'changed':''}" data-dw-review-coverage-area="${id}" tabindex="-1"><div><strong>${esc(label)}</strong><span>${changed?'当前范围已变化':'与捕获范围一致'}</span></div><code>${esc(record.basis.join(' | '))}</code></div>`}).join('');
  return `<div class="dw-review-coverage" aria-label="复核覆盖来源记录"><div class="dw-review-coverage-head"><div><strong>复核覆盖记录</strong><span>BROWSER_SESSION_SCOPE_V1 · 仅当前页面</span></div><span>${records.length}/${REVIEW_AREAS.length} 区域已捕获</span></div>${rows}<p>记录只保存当时可见投影身份与临时查看范围，用于人工复现检查范围；不是审计签名、批准记录或材料充分性证明。</p></div>`;
}
function reviewAreaDetail(projection={},area){
  const a=workspaceAvailability(projection),context=projection.candidate_context||{},evidence=projection.evidence_links||[],agent=projection.agent_links||[];
  if(area==='identity')return `projection ${identityValue(projection.presentation_sha256,'不可用').slice(0,12)}… / source ${identityValue(projection.source_bundle_status)}`;
  if(area==='candidate')return `${identityValue(context.selected_fault_id,'未形成')} / ${(context.alternatives||[]).length} 个备选`;
  if(area==='evidence')return `${Array.isArray(evidence)?evidence.length:0} 项正式链接`;
  if(area==='graph')return a.graphAvailable?`${a.graphNodes} 个节点`:'不可用';
  if(area==='traces')return `${a.observed} 条观测 / ${a.simulated} 条仿真`;
  if(area==='agent')return `${Array.isArray(agent)?agent.length:0} 个回执`;
  return '';
}
function reviewClosureState(projection={}){
  const reviewedAreas=REVIEW_AREAS.filter(({id})=>workspaceReviewSession[id]),unreviewedAreas=REVIEW_AREAS.filter(({id})=>!workspaceReviewSession[id]);
  const attentionSignals=reviewAttentionSignals(projection);
  const status=unreviewedAreas.length?'REVIEW_INCOMPLETE':attentionSignals.length?'REVIEWED_WITH_ATTENTION':'REVIEWED_NO_EXPLICIT_ATTENTION';
  return {status,reviewedCount:reviewedAreas.length,totalAreas:REVIEW_AREAS.length,reviewedAreas:reviewedAreas.map(x=>x.label),unreviewedAreas:unreviewedAreas.map(x=>x.label),attentionSignals:attentionSignals.map(x=>({...x}))};
}
function reviewClosureHtml(projection={}){
  const state=reviewClosureState(projection);
  const labels={REVIEW_INCOMPLETE:'复核未完成',REVIEWED_WITH_ATTENTION:'已复核 · 仍有注意项',REVIEWED_NO_EXPLICIT_ATTENTION:'已复核 · 当前无显式注意项'};
  const detail=state.status==='REVIEW_INCOMPLETE'
    ?`仍有 ${state.unreviewedAreas.length} 个区域未标记已复核：${state.unreviewedAreas.join('、')}。`
    :state.status==='REVIEWED_WITH_ATTENTION'
      ?`${state.totalAreas} 个区域均已标记复核；仍保留 ${state.attentionSignals.length} 项显式注意信号，交接摘要会继续列出。`
      :`${state.totalAreas} 个区域均已标记复核；当前投影与页面上下文没有显式注意信号。`;
  return `<div class="dw-review-closure ${state.status.toLowerCase()}" data-dw-closure-status="${state.status}" aria-live="polite"><div><span>本页材料复核状态</span><strong>${esc(labels[state.status])}</strong><code>${state.status}</code></div><p>${esc(detail)}</p><small>仅描述当前页面会话的复核覆盖度与显式注意项；不是批准、签名、release gate，也不表示注意项已解决。</small></div>`;
}
function reviewStatusOverviewState(projection={}){
  const local=reviewClosureState(projection),signals=reviewAttentionSignals(projection),incoming=incomingReplayConvergenceState(projection);
  return {
    boundary:PRODUCT_EXPERIENCE_STABLE_CANDIDATE_BOUNDARY,
    localReviewStatus:local.status,
    reviewedCount:local.reviewedCount,
    totalAreas:local.totalAreas,
    attentionCount:signals.length,
    incomingReplayLoaded:Boolean(workspaceIncomingCoverage.manifest),
    incomingReReviewStatus:incoming.status,
    replayDeltaCount:incoming.deltaCount,
    pendingReReviewCount:incoming.pendingCount,
  };
}
function reviewStatusOverviewHtml(projection={}){
  const state=reviewStatusOverviewState(projection);
  const incomingLoaded=state.incomingReplayLoaded;
  const cards=[
    ['本页材料复核',state.localReviewStatus,`${state.reviewedCount}/${state.totalAreas} 区域已标记`],
    ['显式注意项',String(state.attentionCount),state.attentionCount?'来自当前投影/临时上下文':'当前无显式注意项'],
    ['接手范围对照',incomingLoaded?'已加载':'未加载',incomingLoaded?`${state.replayDeltaCount} 个 replay delta`:'尚未导入上一班 coverage'],
    ['接手重新查看',incomingLoaded?state.incomingReReviewStatus:'未加载',incomingLoaded?`${state.pendingReReviewCount} 个当前 fingerprint 待查看`:'不生成 convergence 结论'],
  ];
  return `<div class="dw-review-status-overview" aria-label="复核状态总览"><div class="dw-review-status-overview-head"><div><strong>复核状态总览</strong><span>四组状态来自既有只读契约，彼此独立，不构成统一审批 gate。</span></div><code>${PRODUCT_EXPERIENCE_STABLE_CANDIDATE_BOUNDARY}</code></div><div class="dw-review-status-overview-grid">${cards.map(([label,value,detail])=>`<div><span>${esc(label)}</span><strong>${esc(value)}</strong><small>${esc(detail)}</small></div>`).join('')}</div></div>`;
}
function evidenceSelectionPatch(item={}){
  const nodeRef=item.component_id?`component:${item.component_id}`:null;
  return {evidenceId:idOf(item),graphNodeId:nodeRef||workspaceSelection.graphNodeId,observedTraceId:item.channel||workspaceSelection.observedTraceId,mainView:nodeRef?'graph':'overlay'};
}
function agentSelectionPatch(item={}){
  const nodeRef=item.component_id?`component:${item.component_id}`:null;
  return {graphNodeId:nodeRef||workspaceSelection.graphNodeId,observedTraceId:item.channel||workspaceSelection.observedTraceId,mainView:nodeRef?'graph':'overlay'};
}
function reviewNavigationSpec(projection={},kind,key){
  const evidence=Array.isArray(projection.evidence_links)?projection.evidence_links:[],agent=Array.isArray(projection.agent_links)?projection.agent_links:[];
  const context=projection.candidate_context||{},area=REVIEW_AREAS.find(x=>x.id===key);
  if(kind==='area'){
    if(!area)return null;
    if(key==='identity')return {kind,key,label:area.label,target:'identity'};
    if(key==='candidate')return {kind,key,label:area.label,target:'candidate',targetId:workspaceSelection.candidateId||context.selected_fault_id||null};
    if(key==='evidence'){const item=evidence[0]||null;return {kind,key,label:area.label,target:item?'evidence':'support',targetId:item?idOf(item):null,selection:item?evidenceSelectionPatch(item):null}}
    if(key==='graph')return {kind,key,label:area.label,target:'graph',selection:{mainView:'graph'}};
    if(key==='traces')return {kind,key,label:area.label,target:'traces',selection:{mainView:'overlay'}};
    if(key==='agent'){const item=agent[0]||null;return {kind,key,label:area.label,target:item?'agent':'support',targetId:item?.link_id||null,selection:item?agentSelectionPatch(item):null}}
    return null;
  }
  if(kind!=='signal')return null;
  if(key==='SOURCE_BUNDLE_NOT_VERIFIED')return {kind,key,label:key,target:'identity'};
  if(key==='CONTRADICTING_EVIDENCE_PRESENT'||key==='MISSING_EVIDENCE_DECLARED'){
    const role=key==='CONTRADICTING_EVIDENCE_PRESENT'?'CONTRADICTING':'MISSING',item=evidence.find(x=>x.role===role)||null;
    return {kind,key,label:key,target:item?'evidence':'support',targetId:item?idOf(item):null,selection:item?evidenceSelectionPatch(item):null};
  }
  if(key==='AGENT_STEP_INCOMPLETE'){
    const item=agent.find(x=>x.status!=='COMPLETED')||null;
    return {kind,key,label:key,target:item?'agent':'support',targetId:item?.link_id||null,selection:item?agentSelectionPatch(item):null};
  }
  if(key==='MECHANISM_GRAPH_UNAVAILABLE')return {kind,key,label:key,target:'graph',selection:{mainView:'graph'}};
  if(key==='OBSERVED_TELEMETRY_UNAVAILABLE'||key==='NUMERIC_SIMULATION_UNAVAILABLE'||key==='SIMULATION_CANDIDATE_IDENTITY_MISMATCH')return {kind,key,label:key,target:'traces',selection:{mainView:'overlay'}};
  if(key==='TRANSIENT_CANDIDATE_BROWSE')return {kind,key,label:key,target:'candidate',targetId:workspaceSelection.candidateId||context.selected_fault_id||null};
  if(key==='REVIEW_COVERAGE_CONTEXT_CHANGED')return {kind,key,label:key,target:'coverage'};
  return null;
}
function resolveReviewNavigationTarget(spec){
  if(!spec)return null;
  if(spec.target==='identity')return $('#dw-review-identity summary')||$('#dw-review-identity');
  if(spec.target==='candidate')return $$('[data-dw-candidate]').find(x=>x.dataset.dwCandidate===spec.targetId)||$('#dw-context');
  if(spec.target==='evidence')return $$('[data-dw-evidence]').find(x=>x.dataset.dwEvidence===spec.targetId)||$('#dw-support');
  if(spec.target==='agent')return $$('[data-dw-agent-step]').find(x=>x.dataset.dwAgentStep===spec.targetId)||$('#dw-support');
  if(spec.target==='graph')return $$('[data-dw-main]').find(x=>x.dataset.dwMain==='graph')||$('#dw-main');
  if(spec.target==='traces')return $$('[data-dw-main]').find(x=>x.dataset.dwMain==='overlay')||$('#dw-main');
  if(spec.target==='support')return $('#dw-support');
  if(spec.target==='coverage')return $('.dw-review-coverage-row.changed')||$('.dw-review-coverage')||$('#dw-review-session');
  return $('#diagnostic-workspace');
}
function navigateReviewItem(projection={},kind,key){
  const spec=reviewNavigationSpec(projection,kind,key);if(!spec)return null;
  if(spec.selection)Object.assign(workspaceSelection,spec.selection);
  renderWorkspacePanels();
  requestAnimationFrame(()=>{const target=resolveReviewNavigationTarget(spec);if(!target)return;target.scrollIntoView?.({block:'center'});try{target.focus?.({preventScroll:true})}catch(_){target.focus?.()}});
  return spec;
}
function reviewNavigationHtml(projection={}){
  const unreviewed=REVIEW_AREAS.filter(({id})=>!workspaceReviewSession[id]),signals=reviewAttentionSignals(projection);
  if(!unreviewed.length&&!signals.length)return '<div class="dw-review-navigation clear"><strong>复核定位</strong><span>当前没有未复核区域或显式注意信号需要定位。</span></div>';
  const reviewButtons=unreviewed.map(({id,label})=>`<button type="button" data-dw-review-nav-area="${id}"><span>未复核</span><strong>${esc(label)}</strong><small>定位</small></button>`).join('');
  const signalButtons=signals.map(x=>`<button type="button" data-dw-review-nav-signal="${esc(x.code)}"><span>注意</span><strong>${esc(x.code)}</strong><small>定位</small></button>`).join('');
  return `<div class="dw-review-navigation" aria-label="未关闭复核项定位"><div class="dw-review-navigation-head"><strong>未关闭项定位</strong><span>${unreviewed.length} 个未复核区域 / ${signals.length} 个显式注意信号</span></div><div class="dw-review-navigation-items">${reviewButtons}${signalButtons}</div><p>定位只改变页面焦点、主视图或现有临时 selection；不会自动标记已复核，也不会清除 attention signal。</p></div>`;
}
function reviewSessionHtml(projection){
  if(!projection)return '<section class="dw-review-session unavailable"><strong>复核会话不可用</strong><span>生产工作区投影不可用时，不创建浏览器侧复核结论。</span></section>';
  const reviewed=REVIEW_AREAS.filter(({id})=>workspaceReviewSession[id]).length,signals=reviewAttentionSignals(projection);
  const checks=REVIEW_AREAS.map(({id,label})=>`<button type="button" class="dw-review-check ${workspaceReviewSession[id]?'reviewed':''}" aria-pressed="${workspaceReviewSession[id]}" data-dw-review-area="${id}"><span>${workspaceReviewSession[id]?'✓':'○'}</span><strong>${esc(label)}</strong><small>${esc(reviewAreaDetail(projection,id))}</small></button>`).join('');
  const signalHtml=signals.length?`<ul>${signals.map(x=>`<li><code>${esc(x.code)}</code><span>${esc(x.label)}</span></li>`).join('')}</ul>`:'<div class="dw-review-clear">当前投影与页面上下文未出现需要注意的显式材料信号。</div>';
  return `<section class="dw-review-session" aria-label="本页材料复核"><div class="dw-review-session-head"><div><strong>本页材料复核</strong><span>勾选只保存在当前页面内存；不是正式审批、签名或 release gate。</span></div><div class="dw-review-count"><strong>${reviewed}/${REVIEW_AREAS.length}</strong><span>已复核</span></div></div>${reviewStatusOverviewHtml(projection)}<div class="dw-review-checklist">${checks}</div><div class="dw-review-signals"><div class="dw-review-signals-head"><strong>需注意的显式材料信号</strong><span>${signals.length?`${signals.length} 项`:'0 项'}</span></div>${signalHtml}</div>${reviewClosureHtml(projection)}${reviewCoverageHtml(projection)}${reviewCoverageReplayHtml(projection)}${incomingReplayConvergenceHtml(projection)}${reviewReplayDeltaQueueHtml(projection)}${reviewNavigationHtml(projection)}<button type="button" class="btn small dw-review-reset" id="dw-reset-review-session" ${reviewed?'':'disabled'}>清除本页复核标记</button><p class="dw-boundary">信号只转述投影中的显式角色、状态、材料可用性或页面临时选择；不生成新的故障判断、EvidenceFact、概率或资格结论。</p></section>`;
}
function handoffSummaryText(projection={}){
  const authority=projection.authority_snapshot||{},context=projection.candidate_context||{},graph=projection.mechanism_graph||{};
  const {observedTrace,simulationTrace}=effectiveTraceContext(projection);
  const frozenCandidate=context.selected_fault_id||null,browsingCandidate=workspaceSelection.candidateId||frozenCandidate;
  const candidateMode=workspaceSelection.candidateId&&frozenCandidate&&workspaceSelection.candidateId!==frozenCandidate?'TRANSIENT_ALTERNATIVE_BROWSE_ONLY':'FROZEN_REPORT_CANDIDATE';
  const evidence=projection.evidence_links||[],selectedEvidence=workspaceSelection.evidenceId?evidence.find(x=>idOf(x)===workspaceSelection.evidenceId):null;
  const closure=reviewClosureState(projection);
  return [
    'AstroGraph Diagnostic Workspace Handoff',
    `Schema: ${identityValue(projection.schema_version)}`,
    `Job ID: ${identityValue(projection.job_id)}`,
    `Presentation SHA256: ${identityValue(projection.presentation_sha256)}`,
    `Source bundle status: ${identityValue(projection.source_bundle_status)}`,
    `Source bundle SHA256: ${identityValue(projection.source_bundle_sha256)}`,
    `Graph provenance SHA256: ${identityValue(graph.provenance_sha256)}`,
    `Diagnosis disposition: ${identityValue(authority.diagnosis_disposition)}`,
    `Release status: ${identityValue(authority.release_status)}`,
    `EvidenceBundle SHA256: ${identityValue(authority.evidence_bundle_sha256,'未提供 / 当前投影未绑定')}`,
    `Frozen candidate: ${identityValue(frozenCandidate)}`,
    `Component: ${identityValue(context.component_id)}`,
    '',
    'Transient UI context:',
    `Candidate view: ${identityValue(browsingCandidate)} [${candidateMode}]`,
    `Evidence selection: ${identityValue(selectedEvidence?.evidence_id||workspaceSelection.evidenceId,'未选择')}`,
    `Graph node: ${identityValue(workspaceSelection.graphNodeId,'未选择')}`,
    `Observed trace: ${identityValue(observedTrace?.id,'不可用')} | channel=${identityValue(observedTrace?.channel)} | unit=${identityValue(observedTrace?.unit)}`,
    `Simulation trace: ${identityValue(simulationTrace?.id,'未叠加 / 不可用')} | provider=${identityValue(simulationTrace?.providerId)} | diagnosis=${identityValue(simulationTrace?.diagnosisId)} | candidate=${identityValue(simulationTrace?.candidateId)} | source_trace_sha256=${identityValue(simulationTrace?.sourceTraceSha256)}`,
    '',
    'Review session (ephemeral):',
    `Reviewed areas: ${REVIEW_AREAS.filter(({id})=>workspaceReviewSession[id]).map(x=>x.label).join(', ')||'none'}`,
    `Unreviewed areas: ${REVIEW_AREAS.filter(({id})=>!workspaceReviewSession[id]).map(x=>x.label).join(', ')||'none'}`,
    `Attention signals: ${reviewAttentionSignals(projection).map(x=>`${x.code}: ${x.label}`).join(' | ')||'none'}`,
    `Local page review status: ${closure.status}`,
    `Review closure status: ${closure.status}`,
    `Incoming re-review status: ${incomingReplayConvergenceState(projection).status}`,
    `Review status boundary: ${PRODUCT_EXPERIENCE_STABLE_CANDIDATE_BOUNDARY}`,
    `Open review areas: ${closure.unreviewedAreas.join(', ')||'none'}`,
    `Open attention signal codes: ${closure.attentionSignals.map(x=>x.code).join(', ')||'none'}`,
    'Review closure meaning: SESSION_TRACE_ONLY_NOT_APPROVAL',
    '',
    'Review coverage provenance (ephemeral):',
    'Coverage contract: BROWSER_SESSION_SCOPE_V1',
    ...REVIEW_AREAS.map(({id,label})=>workspaceReviewCoverage[id]?`Coverage ${id} (${label}): ${workspaceReviewCoverage[id].basis.join(' | ')}`:`Coverage ${id} (${label}): NOT_CAPTURED`),
    `Coverage drift areas: ${reviewCoverageDrift(projection).map(x=>x.label).join(', ')||'none'}`,
    'Coverage meaning: MATERIAL_SCOPE_TRACE_ONLY_NOT_AUDIT_OR_APPROVAL',
    '',
    'Outgoing review coverage replay manifest:',
    reviewCoverageReplayManifestBlock(projection),
    '',
    `Incoming coverage replay: ${workspaceIncomingCoverage.manifest?workspaceIncomingCoverage.manifest.contract:'NOT_LOADED'}`,
    `Incoming scope comparison: ${workspaceIncomingCoverage.manifest?reviewCoverageReplayComparison(projection).map(x=>`${x.area}=${x.status}`).join(', '):'none'}`,
    `Incoming changed basis fields: ${workspaceIncomingCoverage.manifest?replayBasisDiffSnapshot(projection).map(x=>`${x.area}=[${x.changedFields.join(',')}]`).join(', ')||'none':'none'}`,
    `Incoming re-review queue: ${workspaceIncomingCoverage.manifest?reviewReplayDeltaQueue(projection).map(x=>`${x.area}=${x.status}/${x.revisited?'REVISITED':'PENDING'}`).join(', ')||'none':'none'}`,
    `Incoming re-review pending areas: ${workspaceIncomingCoverage.manifest?reviewReplayDeltaQueue(projection).filter(x=>!x.revisited).map(x=>x.label).join(', ')||'none':'none'}`,
    `Incoming re-review convergence: ${incomingReplayConvergenceState(projection).status}`,
    `Incoming re-review convergence counts: matched=${incomingReplayConvergenceState(projection).matchedCount}, delta=${incomingReplayConvergenceState(projection).deltaCount}, revisited=${incomingReplayConvergenceState(projection).revisitedCount}, pending=${incomingReplayConvergenceState(projection).pendingCount}`,
    'Incoming replay meaning: INCOMING_SCOPE_COMPARISON_ONLY_NOT_AUDIT_OR_APPROVAL',
    'Incoming basis diff meaning: INCOMING_REPLAY_BASIS_DIFF_DISPLAY_ONLY_NOT_CAUSAL_EXPLANATION_OR_RESOLUTION',
    'Incoming re-review meaning: INCOMING_REPLAY_REREVIEW_TRACE_ONLY_NOT_RESOLUTION_OR_APPROVAL',
    'Incoming re-review convergence meaning: INCOMING_REREVIEW_CONVERGENCE_SUMMARY_ONLY_NOT_DISCREPANCY_RESOLUTION_OR_APPROVAL',
    '',
    'Boundary: generated locally from the read-only presentation.v1 projection; review marks, coverage records, incoming replay comparison, re-review queue traces, closure status, and transient selections do not modify formal diagnosis, EvidenceFact, retained source bytes, or release authority.'
  ].join('\n');
}
function reviewIdentityHtml(projection){
  if(!projection)return '<div class="dw-review-identity unavailable"><strong>审查身份不可用</strong><span>生产工作区投影不可用时，不从 maintenance/report 页面拼装来源身份。</span></div>';
  const authority=projection.authority_snapshot||{},graph=projection.mechanism_graph||{},context=projection.candidate_context||{};
  const rows=[
    ['任务 ID',projection.job_id],['工作区投影 SHA256',projection.presentation_sha256],['来源包状态',projection.source_bundle_status],
    ['来源包 SHA256',projection.source_bundle_sha256],['图谱 provenance SHA256',graph.provenance_sha256],['EvidenceBundle SHA256',authority.evidence_bundle_sha256],
    ['冻结候选',context.selected_fault_id],['正式状态',`${identityValue(authority.diagnosis_disposition)} / ${identityValue(authority.release_status)}`],
  ];
  return `<details class="dw-review-identity" id="dw-review-identity" open><summary><span>审查身份与交接</span><small>projection/source/provenance identity</small></summary><div class="dw-identity-grid">${rows.map(([label,value])=>`<div><span>${esc(label)}</span><code>${esc(identityValue(value))}</code></div>`).join('')}</div><div class="dw-handoff-head"><div><strong>交接摘要</strong><span>由当前只读投影和页面内临时选择即时生成，不写回任务。</span></div><button type="button" class="btn small" id="dw-copy-handoff">复制交接摘要</button></div><pre class="dw-handoff-text" id="dw-handoff-text" tabindex="0">${esc(handoffSummaryText(projection))}</pre></details>`;
}
async function copyWorkspaceText(text){
  if(navigator.clipboard&&navigator.clipboard.writeText){await navigator.clipboard.writeText(text);return}
  const ta=document.createElement('textarea');ta.value=text;ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();
}

function contextHtml(context={}){
  const alternatives=context.alternatives||[],primary=context.selected_fault_id||null;
  const candidates=[...(primary?[{candidate_id:primary,label:primary,primary:true}]:[]),...alternatives].filter((x,i,a)=>a.findIndex(y=>idOf(y)===idOf(x))===i);
  return `<div class="dw-context-block"><span>Subsystem</span><strong>${esc(context.subsystem_id||'未提供')}</strong></div><div class="dw-context-block"><span>Component</span><strong>${esc(context.component_id||'未提供')}</strong></div><div class="dw-context-block"><span>Fault mode</span><strong>${esc(primary||'未形成')}</strong></div><div class="dw-context-list"><span>Candidates</span>${candidates.length?candidates.map(x=>`<button type="button" aria-pressed="${workspaceSelection.candidateId===idOf(x)}" class="dw-context-choice ${workspaceSelection.candidateId===idOf(x)?'selected':''}" data-dw-candidate="${esc(idOf(x))}"><strong>${esc(x.label||idOf(x))}</strong><small>${x.primary?'冻结报告候选':'备选假设'}</small></button>`).join(''):'<div class="dw-empty">当前报告没有候选集合。</div>'}</div><p class="dw-boundary">选择只改变当前工作区上下文，不改写正式 diagnosis。</p>`;
}
function graphHtml(graph={}){
  const nodes=Array.isArray(graph.nodes)?graph.nodes:[],edges=Array.isArray(graph.edges)?graph.edges:[];
  if(graph.status!=='AVAILABLE'||!nodes.length)return '<div class="dw-empty">当前报告没有可交互图谱数据。</div>';
  const width=760,height=Math.max(300,Math.ceil(nodes.length/4)*150),positions={};
  nodes.forEach((n,i)=>{positions[idOf(n,`node-${i}`)]={x:100+(i%4)*185,y:75+Math.floor(i/4)*145}});
  const edgeSvg=edges.map(e=>{const a=positions[idOf(e.from)],b=positions[idOf(e.to)];if(!a||!b)return '';const active=workspaceSelection.graphNodeId&&[idOf(e.from),idOf(e.to)].includes(workspaceSelection.graphNodeId);return `<g class="dw-edge ${active?'active':''}"><line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/><text x="${(a.x+b.x)/2}" y="${(a.y+b.y)/2-6}">${esc(e.label||e.relation||'关联')}</text></g>`}).join('');
  const nodeSvg=nodes.map((n,i)=>{const id=idOf(n,`node-${i}`),p=positions[id],active=workspaceSelection.graphNodeId===id||workspaceSelection.componentId===id||workspaceSelection.candidateId===id;return `<g class="dw-node ${active?'selected':''}" tabindex="0" role="button" aria-pressed="${active}" data-dw-node="${esc(id)}" transform="translate(${p.x},${p.y})"><circle r="31"/><text text-anchor="middle" y="4">${esc(String(n.label||id).slice(0,18))}</text><title>${esc(n.label||id)}</title></g>`}).join('');
  const selected=workspaceSelection.graphNodeId?nodes.find(x=>idOf(x)===workspaceSelection.graphNodeId):null;
  const selectedId=selected?idOf(selected):null,relatedEdges=selectedId?edges.filter(e=>idOf(e.from)===selectedId||idOf(e.to)===selectedId):[];
  const detail=selected?`<div class="dw-focus-detail"><span>当前图谱节点</span><strong>${esc(selected.label||selectedId)}</strong><small class="mono">${esc(selectedId)}</small><div>${relatedEdges.length} 条显式关联${Array.isArray(selected.related_channels)&&selected.related_channels.length?` · 关联遥测 ${selected.related_channels.map(esc).join(' / ')}`:''}</div></div>`:'<div class="dw-focus-detail muted">选择节点后显示该节点已有的显式关联和遥测引用。</div>';
  return `<svg class="dw-graph" viewBox="0 0 ${width} ${height}" role="img" aria-label="只读故障传播图">${edgeSvg}${nodeSvg}</svg>${detail}<p class="dw-boundary">只读结构化图谱；关联数量只来自当前投影内显式 edges，无任意 Cypher、编辑或自动补边。</p>`;
}
function linePath(points,x,y){return points.map((p,i)=>`${i?'L':'M'}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(' ')}
function overlayHtml(projection={}){
  const observed=observedTraces(projection),simulated=simulationTraces(projection);
  if(!observed.length)return '<div class="dw-empty">当前任务没有可叠加的 observed telemetry。</div>';
  const observedId=workspaceSelection.observedTraceId&&observed.some(x=>x.id===workspaceSelection.observedTraceId)?workspaceSelection.observedTraceId:observed[0].id;
  const obs=observed.find(x=>x.id===observedId),compatible=simulated.filter(x=>!x.channel||x.channel===obs.channel),simId=workspaceSelection.simulationTraceId;
  const chosen=simId==='__NONE__'?null:(compatible.find(x=>x.id===simId)||compatible[0]||null);
  const all=[...obs.points,...(chosen?.points||[])];let xmin=Math.min(...all.map(p=>p.t)),xmax=Math.max(...all.map(p=>p.t)),ymin=Math.min(...all.map(p=>p.v)),ymax=Math.max(...all.map(p=>p.v));if(xmax===xmin){xmax+=1;xmin-=1}if(ymax===ymin){ymax+=1;ymin-=1}const x=t=>58+(t-xmin)/(xmax-xmin)*642,y=v=>22+(ymax-v)/(ymax-ymin)*210;
  const unit=obs.unit||chosen?.unit||'',channel=obs.channel||obs.label;
  const chart=`<svg class="dw-overlay-chart" viewBox="0 0 740 290" role="img" aria-label="Observed versus simulated telemetry"><line x1="58" y1="232" x2="700" y2="232" class="axis"/><line x1="58" y1="22" x2="58" y2="232" class="axis"/><path d="${linePath(obs.points,x,y)}" class="observed"/>${chosen?`<path d="${linePath(chosen.points,x,y)}" class="simulated"/>`:''}<text x="58" y="255">${esc(xmin)} s</text><text x="700" y="255" text-anchor="end">${esc(xmax)} s</text><text x="50" y="30" text-anchor="end">${esc(ymax.toPrecision(4))}</text><text x="50" y="232" text-anchor="end">${esc(ymin.toPrecision(4))}</text><text x="379" y="278" text-anchor="middle">时间 (s)</text><text x="14" y="128" text-anchor="middle" transform="rotate(-90 14 128)">${esc(unit||'value')}</text></svg>`;
  const stateMessage=!simulated.length?'当前诊断没有可叠加的数值仿真轨迹':!compatible.length?'当前遥测通道没有匹配的仿真轨迹':'';
  return `<div class="dw-trace-toolbar"><label>观测轨迹<select id="dw-observed-select">${observed.map(t=>`<option value="${esc(t.id)}" ${t.id===obs.id?'selected':''}>${esc(t.label)}</option>`).join('')}</select></label>${compatible.length?`<label>仿真轨迹<select id="dw-simulation-select"><option value="__NONE__" ${chosen?'':'selected'}>不叠加</option>${compatible.map(t=>`<option value="${esc(t.id)}" ${chosen&&t.id===chosen.id?'selected':''}>${esc(t.label)}</option>`).join('')}</select></label>`:''}</div><div class="dw-trace-meta"><span>通道 <strong>${esc(channel||'未提供')}</strong></span><span>单位 <strong>${esc(unit||'未提供')}</strong></span><span>时间范围 <strong>${esc(xmin)}–${esc(xmax)} s</strong></span></div>${chart}<div class="dw-legend"><span><i class="observed"></i>Observed</span>${chosen?`<span><i class="simulated"></i>${esc(chosen.label)}</span>`:''}</div>${stateMessage?`<div class="dw-empty compact">${stateMessage}</div>`:''}<p class="dw-boundary">simulation fit / distance ≠ fault probability。曲线只显示已验证投影中的数值样本，浏览器不执行仿真。</p>`;
}
function evidenceHtml(links=[]){
  if(!Array.isArray(links)||!links.length)return '<div class="dw-empty">正式证据摘要不可用。</div>';
  const groups=[['支持证据','SUPPORTING'],['反证','CONTRADICTING'],['缺失证据','MISSING']];
  return groups.map(([label,role])=>{const items=links.filter(x=>x.role===role);return `<section class="dw-evidence-group"><h3>${label} <span>${items.length}</span></h3>${items.length?items.map((item,i)=>{const id=idOf(item,`${role}-${i}`);return `<button type="button" aria-pressed="${workspaceSelection.evidenceId===id}" class="dw-evidence ${workspaceSelection.evidenceId===id?'selected':''}" data-dw-evidence="${esc(id)}" data-dw-channel="${esc(item.channel||'')}" data-dw-node-ref="${esc(item.component_id?`component:${item.component_id}`:'')}"><strong>${esc(item.label||item.evidence_id||id)}</strong><small>${esc(item.evidence_id||role)}</small></button>`}).join(''):'<div class="dw-empty compact">无</div>'}</section>`}).join('');
}
function agentHtml(links=[]){
  if(!Array.isArray(links)||!links.length)return '<div class="dw-empty">当前报告没有 Agent 调查回执。</div>';
  return `<div class="dw-agent-steps">${links.map(x=>`<button type="button" data-dw-agent-step="${esc(x.link_id)}" data-dw-channel="${esc(x.channel||'')}" data-dw-node-ref="${esc(x.component_id?`component:${x.component_id}`:'')}"><span>${x.status==='COMPLETED'?'✓':'○'}</span><strong>${esc(probeDisplayName(x.probe_id))}</strong><small>${x.status==='COMPLETED'?'已完成':'待补充'}</small></button>`).join('')}</div><p class="dw-boundary">点击步骤只导航到已有材料，不会自动执行 Probe。</p>`;
}
function renderWorkspacePanels(){
  const host=$('#diagnostic-workspace');if(!host)return;
  const projection=validWorkspaceProjection(workspaceData.diagnosticWorkspace),safeProjection=projection||{},graph=safeProjection.mechanism_graph||{};
  $('#dw-review').innerHTML=reviewIdentityHtml(projection);
  $('#dw-review-session').innerHTML=reviewSessionHtml(projection);
  $('#dw-overview').innerHTML=overviewHtml(projection);
  $('#dw-selection').innerHTML=selectionSummaryHtml(safeProjection);
  $('#dw-context').innerHTML=contextHtml(safeProjection.candidate_context||{});
  $('#dw-main').innerHTML=`<div class="dw-main-tabs" role="tablist" aria-label="诊断工作区主视图"><button type="button" role="tab" aria-selected="${workspaceSelection.mainView==='overlay'}" data-dw-main="overlay" class="${workspaceSelection.mainView==='overlay'?'active':''}">遥测对比</button><button type="button" role="tab" aria-selected="${workspaceSelection.mainView==='graph'}" data-dw-main="graph" class="${workspaceSelection.mainView==='graph'?'active':''}">故障传播图</button></div><div class="dw-main-stage">${workspaceSelection.mainView==='graph'?graphHtml(graph):overlayHtml(safeProjection)}</div>`;
  $('#dw-support').innerHTML=`<div class="dw-support-tabs"><strong>Evidence</strong></div>${evidenceHtml(safeProjection.evidence_links)}<div class="dw-support-tabs"><strong>Agent steps</strong></div>${agentHtml(safeProjection.agent_links)}`;
  $$('[data-dw-candidate]').forEach(b=>b.onclick=()=>setWorkspaceSelection({candidateId:b.dataset.dwCandidate}));
  $$('[data-dw-main]').forEach(b=>b.onclick=()=>setWorkspaceSelection({mainView:b.dataset.dwMain}));
  $$('[data-dw-jump]').forEach(b=>b.onclick=()=>{if(b.dataset.dwJump==='graph'||b.dataset.dwJump==='overlay')setWorkspaceSelection({mainView:b.dataset.dwJump});else $('#dw-support')?.scrollIntoView({block:'nearest'})});
  $$('[data-dw-node]').forEach(b=>{const select=()=>{const n=(graph.nodes||[]).find(x=>idOf(x)===b.dataset.dwNode),channel=(n?.related_channels||[])[0]||null;setWorkspaceSelection({graphNodeId:b.dataset.dwNode,componentId:b.dataset.dwNode,observedTraceId:channel||workspaceSelection.observedTraceId})};b.onclick=select;b.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select()}}});
  $$('[data-dw-evidence]').forEach(b=>b.onclick=()=>setWorkspaceSelection({evidenceId:b.dataset.dwEvidence,graphNodeId:b.dataset.dwNodeRef||workspaceSelection.graphNodeId,observedTraceId:b.dataset.dwChannel||workspaceSelection.observedTraceId,mainView:b.dataset.dwNodeRef?'graph':'overlay'}));
  $$('[data-dw-agent-step]').forEach(b=>b.onclick=()=>setWorkspaceSelection({graphNodeId:b.dataset.dwNodeRef||workspaceSelection.graphNodeId,observedTraceId:b.dataset.dwChannel||workspaceSelection.observedTraceId,mainView:b.dataset.dwNodeRef?'graph':'overlay'}));
  const observed=$('#dw-observed-select');if(observed)observed.onchange=()=>setWorkspaceSelection({observedTraceId:observed.value,simulationTraceId:null});
  const simulation=$('#dw-simulation-select');if(simulation)simulation.onchange=()=>setWorkspaceSelection({simulationTraceId:simulation.value});
  const reset=$('#dw-reset-context');if(reset)reset.onclick=()=>{resetWorkspaceSelection();renderWorkspacePanels()};
  $$('[data-dw-review-area]').forEach(b=>b.onclick=()=>setReviewAreaReviewed(b.dataset.dwReviewArea,!workspaceReviewSession[b.dataset.dwReviewArea]));
  $$('[data-dw-review-nav-area]').forEach(b=>b.onclick=()=>navigateReviewItem(safeProjection,'area',b.dataset.dwReviewNavArea));
  $$('[data-dw-review-nav-signal]').forEach(b=>b.onclick=()=>navigateReviewItem(safeProjection,'signal',b.dataset.dwReviewNavSignal));
  $$('[data-dw-replay-nav-area]').forEach(b=>b.onclick=()=>navigateReviewItem(safeProjection,'area',b.dataset.dwReplayNavArea));
  $$('[data-dw-rereview-nav-area]').forEach(b=>b.onclick=()=>navigateReplayDeltaQueueItem(safeProjection,b.dataset.dwRereviewNavArea));
  $$('[data-dw-rereview-toggle-area]').forEach(b=>b.onclick=()=>{const item=reviewReplayDeltaQueue(safeProjection).find(x=>x.area===b.dataset.dwRereviewToggleArea);if(item)setReplayDeltaQueueRevisited(item.area,!item.revisited,safeProjection)});
  const prevRereview=$('#dw-rereview-prev');if(prevRereview)prevRereview.onclick=()=>advanceReplayDeltaQueue(-1,safeProjection);
  const nextRereview=$('#dw-rereview-next');if(nextRereview)nextRereview.onclick=()=>advanceReplayDeltaQueue(1,safeProjection);
  const loadReplay=$('#dw-load-incoming-coverage');if(loadReplay)loadReplay.onclick=()=>loadIncomingReviewCoverage($('#dw-incoming-coverage-manifest')?.value||'');
  const clearReplay=$('#dw-clear-incoming-coverage');if(clearReplay)clearReplay.onclick=()=>clearIncomingReviewCoverage();
  const copyReplay=$('#dw-copy-coverage-manifest');if(copyReplay)copyReplay.onclick=async()=>{try{await copyWorkspaceText(reviewCoverageReplayManifestBlock(safeProjection));showToast('已复制 coverage replay manifest')}catch(e){console.error('[AstroGraph UI] copy coverage replay manifest failed',e);showToast('复制失败，请从交接摘要手动复制 manifest')}};
  const resetReview=$('#dw-reset-review-session');if(resetReview)resetReview.onclick=()=>{resetWorkspaceReviewSession();renderWorkspacePanels()};
  const copyHandoff=$('#dw-copy-handoff');if(copyHandoff)copyHandoff.onclick=async()=>{try{await copyWorkspaceText(handoffSummaryText(safeProjection));showToast('已复制审查交接摘要')}catch(e){console.error('[AstroGraph UI] copy diagnostic handoff failed',e);showToast('复制失败，请手动选择交接摘要')}};
}
function renderDiagnosticWorkspace({report,maintenance,job,visualization,diagnosticWorkspace}={}){
  workspaceData={report:report||null,maintenance:maintenance||null,job:job||null,visualization:visualization||null,diagnosticWorkspace:diagnosticWorkspace||null};resetWorkspaceSelection();resetWorkspaceReviewSession();resetIncomingReviewCoverage();
  const container=$('#result-content');if(!container)return;container.querySelector('#diagnostic-workspace')?.remove();
  const section=document.createElement('section');section.id='diagnostic-workspace';section.className='diagnostic-workspace card';section.tabIndex=-1;section.innerHTML='<div class="dw-header"><div><span class="eyebrow">Diagnostic Workspace</span><h2>诊断分析工作区</h2><p>复核已有候选、正式 Evidence、机理图谱与数值轨迹；所有交互均为页面内只读上下文。</p></div><span class="badge audit">只读上下文联动</span></div><div id="dw-review"></div><div id="dw-review-session"></div><div id="dw-overview"></div><div id="dw-selection"></div><div class="dw-layout"><aside id="dw-context" class="dw-context" tabindex="-1"></aside><main id="dw-main" class="dw-main" tabindex="-1"></main><aside id="dw-support" class="dw-support" tabindex="-1"></aside></div><p class="dw-boundary">工作区 selection 为页面内临时状态；不写 report、EvidenceBundle、diagnosis、Agent trajectory 或 release state。按 Esc 可恢复默认上下文。</p>';
  section.onkeydown=e=>{if(e.key==='Escape'){resetWorkspaceSelection();renderWorkspacePanels()}};
  const nav=$('#ops-result-navigation');nav?.insertAdjacentElement('afterend',section);if(!nav)container.prepend(section);renderWorkspacePanels();
}

export { PRODUCT_EXPERIENCE_STABLE_CANDIDATE_BOUNDARY, advanceReplayDeltaQueue, clearIncomingReviewCoverage, handoffSummaryText, incomingReviewCoverageSnapshot, loadIncomingReviewCoverage, navigateReplayDeltaQueueItem, navigateReviewItem, parseReviewCoverageReplay, renderDiagnosticWorkspace, replayBasisDiff, replayBasisDiffSnapshot, resetWorkspaceReviewSession, resetWorkspaceSelection, reviewAttentionSignals, reviewClosureState, reviewCoverageBasis, reviewCoverageDrift, reviewCoverageReplayComparison, reviewCoverageReplayManifest, reviewCoverageReplayManifestBlock, reviewCoverageSnapshot, reviewNavigationSpec, reviewReplayDeltaQueue, reviewReplayDeltaQueueSnapshot, reviewSessionSnapshot, reviewStatusOverviewState, setReplayDeltaQueueRevisited, setReviewAreaReviewed, setWorkspaceSelection, simulationTraces, workspaceAvailability, workspaceIncomingCoverage, workspaceReplayDeltaQueue, workspaceReviewCoverage, workspaceReviewSession, workspaceSelection, workspaceSnapshot };
