import { $, esc, showToast } from './platform_core.js';

// Never store administrator tokens or service secrets in localStorage, URLs or HTML.
let adminToken='';
let currentRevision=null;

async function engineApi(path,options={}){
  const headers={...(options.headers||{})};
  if(adminToken)headers.Authorization=`Bearer ${adminToken}`;
  const response=await fetch('/v1/platform/settings/knowledge-engine'+path,{...options,headers,cache:'no-store'});
  if(!response.ok){
    let detail=response.statusText;
    try{const data=await response.json();detail=data.detail||data.error?.message||detail;}catch{}
    throw new Error(typeof detail==='string'?detail:JSON.stringify(detail));
  }
  return response.json();
}
function fillSettings(s){
  const fields={
    'engine-embedding-model':s.embedding_model, 'engine-vector-backend':s.vector_backend,
    'engine-qdrant-url':s.qdrant_url,'engine-qdrant-collection':s.qdrant_collection,
    'engine-rag-docs':s.rag_documents,'engine-rag-sections':s.rag_sections,
    'engine-rag-hits':s.rag_hits,'engine-graph-model':s.graph_llm_model,
    'engine-ocr':s.ocr_engine,'engine-ocr-langs':s.ocr_languages,
    'engine-vision-endpoint':s.vision_endpoint,'engine-vision-model':s.vision_model,
    'engine-vision-pages':s.vision_pages,
  };
  for(const [id,value] of Object.entries(fields))$('#'+id).value=String(value??'');
  $('#engine-community-llm').checked=!!s.community_llm_enabled;
  $('#engine-graph-build').checked=!!s.graph_build_enabled;
  $('#engine-vision-enabled').checked=!!s.vision_enabled;
  $('#engine-qdrant-key').value='';
}
function settingsPayload(){
  const s={
    embedding_model:$('#engine-embedding-model').value.trim(),
    vector_backend:$('#engine-vector-backend').value,
    qdrant_url:$('#engine-qdrant-url').value.trim(),
    qdrant_collection:$('#engine-qdrant-collection').value.trim(),
    community_llm_enabled:$('#engine-community-llm').checked,
    graph_build_enabled:$('#engine-graph-build').checked,
    graph_llm_model:$('#engine-graph-model').value.trim(),
    ocr_engine:$('#engine-ocr').value,
    ocr_languages:$('#engine-ocr-langs').value.trim(),
    vision_enabled:$('#engine-vision-enabled').checked,
    vision_endpoint:$('#engine-vision-endpoint').value.trim(),
    vision_model:$('#engine-vision-model').value.trim(),
    vision_pages:Number($('#engine-vision-pages').value),
    rag_documents:Number($('#engine-rag-docs').value),
    rag_sections:Number($('#engine-rag-sections').value),
    rag_hits:Number($('#engine-rag-hits').value),
  };
  if($('#engine-qdrant-key').value)s.qdrant_api_key=$('#engine-qdrant-key').value;
  return s;
}
async function loadEngineSettings(){
  try{
    const x=await engineApi('');
    currentRevision=x.revision??null;
    fillSettings(x.settings);
    $('#engine-key-indicator').textContent=x.qdrant_api_key_configured?'已有安全存储密钥（不回显）':'未配置持久密钥';
    $('#engine-revision-label').textContent=x.revision||'默认配置 / 未保存';
    const history=await engineApi('/revisions');
    const sel=$('#engine-revision-select');
    sel.textContent='';
    const initial=document.createElement('option'); initial.value='';initial.textContent='选择历史版本…';sel.appendChild(initial);
    for(const row of history.revisions||[]){
      const option=document.createElement('option');option.value=row.revision;
      option.textContent=`${row.updated_at||''} · ${row.revision===currentRevision?'当前 · ':''}${row.revision}`;
      sel.appendChild(option);
    }
    return x;
  }catch(e){showToast('知识引擎配置读取失败：'+e.message);return null;}
}
function setupEngineSettings(){
  $('#engine-admin-token').addEventListener('change',e=>{adminToken=e.target.value; e.target.value='';loadEngineSettings();});
  $('#engine-refresh').onclick=()=>loadEngineSettings();
  $('#engine-form').addEventListener('submit',async e=>{
    e.preventDefault();const button=$('#engine-save');button.disabled=true;
    try{
      const x=await engineApi('',{method:'PUT',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({expected_revision:currentRevision,settings:settingsPayload()})});
      $('#engine-qdrant-key').value='';showToast('新版本已保存并应用于后续任务');await loadEngineSettings();
    }catch(error){showToast('保存失败：'+error.message);}finally{button.disabled=false;}
  });
  for(const [id,network] of [['engine-probe',false],['engine-probe-network',true]]){
    $('#'+id).onclick=async()=>{
      const box=$('#engine-probe-result');box.classList.remove('hidden');box.textContent='检查中（不会执行模型生成或文档识别）…';
      try{
        const x=await engineApi('/probe?network='+network,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(settingsPayload())});
        box.className='llm-probe ready';box.textContent=JSON.stringify(x.checks,null,2);
      }catch(error){box.className='llm-probe failed';box.textContent='检查失败：'+error.message;}
    };
  }
  $('#engine-rollback').onclick=async()=>{
    const target=$('#engine-revision-select').value;
    if(!target||!currentRevision){showToast('请选择可恢复的历史版本');return;}
    if(target===currentRevision){showToast('已是当前版本');return;}
    if(!confirm('确认将所选历史版本复制为新生效版本？该操作影响后续新任务。'))return;
    try{
      await engineApi('/rollback',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({target_revision:target,expected_revision:currentRevision})});
      await loadEngineSettings();showToast('恢复完成，新版本已生效');
    }catch(error){showToast('恢复失败：'+error.message);}
  };
}
export { setupEngineSettings,loadEngineSettings };
