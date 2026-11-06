/* Explicit reviewed context; never automatically recalls another workspace. */
(function(global){'use strict';
 const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
 function mount(host,{api,baseUrl='/api/workspaces',onChange=()=>{}}){
  host.replaceChildren();host.classList.add('context-workbench');
  const summary=el('summary',null,'工作区与已审阅上下文（可选）');host.append(summary);
  const content=el('div','context-content');host.append(content);
  const note=el('p','muted','只使用本次明确勾选的已审阅条目；审阅表示允许用于分析，不代表事实已被证实。');content.append(note);
  const toolbar=el('div','context-toolbar'),select=el('select','context-workspace'),refresh=el('button','context-refresh','刷新');refresh.type='button';select.setAttribute('aria-label','选择工作区');toolbar.append(select,refresh);content.append(toolbar);
  const status=el('p','muted context-status');status.setAttribute('role','status');content.append(status);
  const list=el('div','context-entries');content.append(list);
  const create=el('details','context-new'),cs=el('summary',null,'新建工作区');create.append(cs);
  const name=el('input'),device=el('input'),event=el('input');name.placeholder='工作区名称';device.placeholder='设备／构型（可选）';event.placeholder='故障事件（可选）';[name,device,event].forEach((x,i)=>{x.maxLength=300;x.setAttribute('aria-label',x.placeholder);create.append(x);});
  const createButton=el('button',null,'创建工作区');createButton.type='button';create.append(createButton);content.append(create);
  const add=el('details','context-add'),as=el('summary',null,'新增补充说明或假设');add.append(as);
  const kind=el('select'),title=el('input'),body=el('textarea'),source=el('input');
  [['user_note','用户补充说明'],['model_hypothesis','模型假设（未证实）']].forEach(([v,t])=>{const o=el('option',null,t);o.value=v;kind.append(o);});kind.setAttribute('aria-label','条目类型');title.placeholder='条目标题';title.maxLength=240;body.placeholder='补充内容，不修改原始报告';body.maxLength=6000;source.placeholder='来源说明（可选）';source.maxLength=1000;[title,body,source].forEach(x=>x.setAttribute('aria-label',x.placeholder));
  const addButton=el('button',null,'保存为待审阅');addButton.type='button';add.append(kind,title,body,source,addButton);content.append(add);
  let epoch=0,selected=new Map(),disposed=false;
  function selection(){return select.value?{workspace_id:select.value,entries:Array.from(selected.values())}:null;}
  function changed(){onChange(selection());}
  async function request(path,method='GET',data){return api(baseUrl+path,{method,...(data===undefined?{}:{body:JSON.stringify(data)})});}
  function error(e){status.textContent=e.message||String(e);}
  async function entries(){const ticket=++epoch,wid=select.value;selected.clear();changed();list.replaceChildren();add.hidden=!wid;
    if(!wid){status.textContent='未选工作区；不会加入历史上下文。';return;}
    status.textContent='读取条目…';
    try{const data=await request('/'+wid+'/entries');if(ticket!==epoch||select.value!==wid||disposed)return;
      for(const item of data.entries||[]){const row=el('div','context-entry'),label=el('label','context-entry-heading'),cb=el('input','context-select-entry');cb.type='checkbox';cb.disabled=item.state!=='REVIEWED';label.append(cb,el('strong',null,item.title),el('span','muted','v'+item.revision+' · '+(item.state==='REVIEWED'?'已审阅':'待审阅')));row.append(label,el('p',null,item.body),el('small','muted',item.source_ref||'未提供来源说明'));
        cb.addEventListener('change',()=>{if(cb.checked&&selected.size>=8){cb.checked=false;status.textContent='最多选择 8 项。';return;}if(cb.checked)selected.set(item.entry_id,{entry_id:item.entry_id,revision:item.revision});else selected.delete(item.entry_id);changed();});
        const actions=el('div','context-toolbar'),review=el('button','context-review',item.state==='REVIEWED'?'撤回审阅':'审阅并允许使用'),history=el('button','context-history','修订记录');review.type=history.type='button';actions.append(review,history);row.append(actions);
        review.addEventListener('click',async()=>{review.disabled=true;try{await request('/'+wid+'/entries/'+item.entry_id+'/review','POST',{expected_revision:item.revision,reviewed:item.state!=='REVIEWED'});if(select.value===wid)await entries();}catch(e){error(e);}finally{review.disabled=false;}});
        const edit=el('details'),es=el('summary',null,'修订内容'),editor=el('textarea'),save=el('button',null,'保存新修订（重新待审阅）');editor.value=item.body;editor.maxLength=6000;editor.setAttribute('aria-label','修订 '+item.title);save.type='button';edit.append(es,editor,save);row.append(edit);
        save.addEventListener('click',async()=>{save.disabled=true;try{await request('/'+wid+'/entries/'+item.entry_id,'POST',{expected_revision:item.revision,title:item.title,body:editor.value,source_ref:item.source_ref||''});if(select.value===wid)await entries();}catch(e){error(e);}finally{save.disabled=false;}});
        history.addEventListener('click',async()=>{try{const h=await request('/'+wid+'/entries/'+item.entry_id+'/history');if(select.value!==wid)return;const pre=el('pre','context-history-data',JSON.stringify(h.revisions,null,2));row.querySelector('.context-history-data')?.remove();row.append(pre);}catch(e){error(e);}});
        list.append(row);
      }status.textContent=(data.entries||[]).length?'请勾选本次需要使用的条目。':'暂无条目，可先新增说明。';
    }catch(e){if(ticket===epoch)error(e);}
  }
  async function refreshAll(){const wid=select.value,ticket=++epoch;try{const data=await request('');if(disposed||ticket!==epoch)return;select.replaceChildren();const none=el('option',null,'不使用工作区上下文');none.value='';select.append(none);for(const w of data.workspaces||[]){const o=el('option',null,[w.name,w.device,w.event].filter(Boolean).join(' · '));o.value=w.workspace_id;select.append(o);}select.value=Array.from(select.options).some(x=>x.value===wid)?wid:'';await entries();}catch(e){error(e);}}
  createButton.addEventListener('click',async()=>{createButton.disabled=true;try{const w=await request('','POST',{name:name.value,device:device.value,event:event.value});await refreshAll();select.value=w.workspace_id;await entries();create.open=false;name.value=device.value=event.value='';}catch(e){error(e);}finally{createButton.disabled=false;}});
  addButton.addEventListener('click',async()=>{const wid=select.value;if(!wid){status.textContent='先选择工作区。';return;}addButton.disabled=true;try{await request('/'+wid+'/entries','POST',{kind:kind.value,title:title.value,body:body.value,source_ref:source.value});if(select.value===wid){await entries();title.value=body.value=source.value='';add.open=false;}}catch(e){error(e);}finally{addButton.disabled=false;}});
  refresh.addEventListener('click',refreshAll);select.addEventListener('change',entries);const ready=refreshAll();
  return {selection,refresh:refreshAll,ready,reset(){++epoch;selected.clear();select.value='';list.replaceChildren();add.hidden=true;changed();},dispose(){disposed=true;++epoch;selected.clear();host.replaceChildren();}};
 }
 function renderSnapshot(host,snapshot){host.replaceChildren();host.hidden=!snapshot;if(!snapshot)return;const d=el('details','context-workbench');d.append(el('summary',null,'本次冻结的工作区上下文 · '+snapshot.entries.length+' 项'));const c=el('div','context-content');d.append(c);c.append(el('p','muted',[snapshot.workspace.name,snapshot.workspace.device,snapshot.workspace.event].filter(Boolean).join(' · ')),el('small','muted','审阅不等于事实认证；后续修改不改变本次快照。'));
  for(const v of snapshot.entries)c.append(el('h4',null,v.title+' · v'+v.revision),el('p',null,v.body),el('small','muted',v.source_ref||'未提供来源说明'));c.append(el('code',null,snapshot.sha256));host.append(d);
 }
 global.AstroGraphContexts={mount,renderSnapshot};
})(window);
