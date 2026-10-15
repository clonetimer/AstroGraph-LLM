(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let requestId = 0;
  const text = (id, value) => { $(id).textContent = value; };
  const jobLink = job => '/platform/mission?job=' + encodeURIComponent(job.job_id) + '#workbench';
  function eventState(job) {
    if (job.status === 'FAILED') return {label:'执行失败',note:'任务未完成，进入事件中心查看失败原因。',rank:0};
    if (['RUNNING','QUEUED'].includes(job.status)) return {label:'研判进行中',note:'材料正在处理，进入工作台查看任务进度。',rank:1};
    if (job.manual_review_status === 'REVIEWED') return {label:'已记录复核意见',note:'复核记录已保存，可继续查看完整证据与决策包。',rank:4};
    if (job.manual_review_status === 'REVIEWED_WITH_GAPS') return {label:'复核后待补证',note:'复核仍有证据缺口，需补齐材料后再作判断。',rank:2};
    if (job.requires_human_review || /ABSTAIN|INSUFFICIENT/.test(job.diagnosis_disposition || '')) return {label:'待人工复核',note:'当前证据不足以确诊。请核对原始材料与候选原因。',rank:2};
    return {label:job.status === 'SUCCEEDED'?'报告已生成':'待启动',note:'打开事件，查看已保存的材料、报告与后续动作。',rank:3};
  }
  function reset(title, description, action, href) {
    text('focus-title', title);text('focus-description', description);text('focus-job','');
    text('event-state','');text('event-category','开始使用');
    $('primary-action').textContent = action + ' →';$('primary-action').href = href;
    $('show-events').hidden = true;$('event-list').replaceChildren();
    for (const id of ['count-attention','count-active','count-done']) text(id,'—');
  }
  async function refresh() {
    const id=++requestId;
    let token='';try{token=sessionStorage.getItem('astrograph-account-session')||'';}catch{}
    $('refresh-data').disabled=true;$('focus-title').parentElement.setAttribute('aria-busy','true');
    try {
      const response=await fetch('/v1/platform/overview?limit=100',{headers:token?{Authorization:'Bearer '+token}:{},cache:'no-store',signal:AbortSignal.timeout(10000)});
      if(id!==requestId)return;
      if(response.status===401||response.status===403){
        reset('从一项研判开始','登录后汇集异常线索、核验证据，并记录专家复核意见。','登录工作台','/platform/mission');
        text('connection','平台可连接');text('data-note','账户登录后显示真实任务');$('refresh-data').className='quiet status online';return;
      }
      if(!response.ok)throw new Error('HTTP '+response.status);
      const overview=await response.json();if(id!==requestId)return;
      const jobs=[...(overview.recent_jobs||[])].sort((a,b)=>eventState(a).rank-eventState(b).rank);
      if(!jobs.length){
        const identityResponse=await fetch('/v1/platform/mission/identity',{headers:token?{Authorization:'Bearer '+token}:{},cache:'no-store',signal:AbortSignal.timeout(10000)});
        if(!identityResponse.ok)throw new Error('Identity unavailable');
        const identity=await identityResponse.json();if(id!==requestId)return;
        if(identity.permissions?.analyze)reset('开始第一次故障分析','用自然语言描述现象并添加诊断、日志和参考资料，系统会自动检查材料。','新建分析','/platform/mission#upload');
        else reset('暂无可查看的事件','当前还没有可见任务。新报告生成后，可在这里进入证据与决策包。','进入事件中心','/platform/mission#command');
      }
      else {
        const job=jobs[0],state=eventState(job);
        text('event-category','优先关注');text('event-state',state.label);
        text('focus-title',job.fault_name||job.task||'研判任务');text('focus-description',state.note);text('focus-job',job.job_id);
        $('primary-action').textContent='继续研判 →';$('primary-action').href=jobLink(job);
        $('show-events').hidden=false;$('show-events').textContent='展开事件 · '+jobs.length;
        $('event-list').replaceChildren(...jobs.map(job=>{
          const a=document.createElement('a');a.className='drawer-event';a.href=jobLink(job);
          for(const [tag,value] of [['span',eventState(job).label],['strong',job.fault_name||job.task||job.job_id],['small',job.job_id]]){const el=document.createElement(tag);el.textContent=value;a.append(el);}return a;
        }));
      }
      text('count-attention',overview.review_or_attention_count||0);text('count-active',overview.active_job_count||0);text('count-done',overview.status_counts?.SUCCEEDED||0);
      text('connection','任务数据已同步');$('refresh-data').className='quiet status online';
      text('data-note','当前账号可见任务 · '+new Date().toLocaleTimeString('zh-CN',{hour12:false})+' 更新');
    } catch(error) {
      if(id!==requestId)return;
      reset('暂时无法读取任务','点击右上角重新连接，或进入事件中心检查服务状态。','进入事件中心','/platform/mission#command');
      text('connection','连接失败 · 重试');text('data-note','任务数据暂不可用');$('refresh-data').className='quiet status error';
    } finally {if(id===requestId){$('refresh-data').disabled=false;$('focus-title').parentElement.setAttribute('aria-busy','false');}}
  }
  $('show-events').onclick=()=>$('event-drawer').showModal();
  $('close-drawer').onclick=()=>$('event-drawer').close();
  $('event-drawer').addEventListener('click',e=>{if(e.target===$('event-drawer')&&e.clientX<$('event-drawer').getBoundingClientRect().left)$('event-drawer').close();});
  $('refresh-data').onclick=refresh;
  // A restored page must not reveal the previous account's cached task.
  window.addEventListener('pageshow',()=>{reset('连接事件中心','正在获取已授权的任务。','进入工作台','/platform/mission');refresh();});
  window.addEventListener('pagehide',()=>{++requestId;reset('连接事件中心','正在获取已授权的任务。','进入工作台','/platform/mission');});
})();
