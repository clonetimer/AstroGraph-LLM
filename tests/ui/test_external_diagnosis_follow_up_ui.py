"""Execute the production follow-up controller and report renderers in Node."""
from pathlib import Path
import shutil
import subprocess
import pytest

RESOURCES = Path(__file__).resolve().parents[2] / 'src/astrograph/resources'


def test_follow_up_binds_requests_preserves_failures_and_discards_late_results():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    script = r'''
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context={console,crypto:require('node:crypto').webcrypto};context.window=context;vm.createContext(context);
const file=process.argv[1]+'/external_follow_up.js';
assert.ok(fs.existsSync(file),'production external follow-up controller is missing');
vm.runInContext(fs.readFileSync(file,'utf8'),context);
const F=context.ExternalDiagnosisFollowUp;
const hash='a'.repeat(64),source={source_id:'source-1',filename:'报告<img>.txt',locator:'line:9',text:'全文 <script>bad()</script>',sha256:hash,role:'diagnosis_results'};
const good={request_id:'request-123',question:'问题',status:'SUCCEEDED',citation_reference_validation:'PASS',answer:[{text:'答复 <img>',citation_ids:['source-1']}],information_gaps:['仍缺 <script>']};
const base={schema_version:'astrograph-external-diagnosis-conversation.v1',job_id:'JOB-A',input_sha256:hash,report_sha256:hash,revision:1,turns:[good],sources:[source],limits:{}};
(async()=>{
 let active=true,mode='load',calls=[],seen=[],resolvePending;
 const api=async(path,options={})=>{calls.push([path,options]);if(mode==='throw')throw new Error('SECRET provider error');if(mode==='pending')return new Promise(resolve=>resolvePending=resolve);if(options.method==='POST')return {...base,revision:2,turns:[good,{request_id:JSON.parse(options.body).request_id,question:'保留问题',status:'FAILED',citation_reference_validation:'FAILED',reason_code:'MODEL_GENERATION_FAILED',answer:[{text:'BAD FAILED TEXT',citation_ids:['source-1']}]}]};return base};
 const options={api,jobId:'JOB-A',inputHash:hash,identity:'alice',canAsk:true,current:()=>active,onChange:s=>seen.push(s)};
 const c=F.create(options);await c.load();
 let html=F.turnsHtml(c.state);assert.ok(html.includes('答复 &lt;img&gt;'));assert.ok(html.includes('报告&lt;img&gt;.txt'));assert.ok(html.includes('line:9'));assert.ok(html.includes('全文 &lt;script&gt;'));assert.ok(!html.includes('<img>'));
 c.setDraft('保留问题');mode='throw';await c.submit();assert.equal(c.draft,'保留问题');assert.ok(!c.message.includes('SECRET'));
 const attempt=JSON.parse(calls.at(-1)[1].body);mode='load';await c.submit();assert.equal(JSON.parse(calls.at(-1)[1].body).request_id,attempt.request_id,'uncertain retry must reuse id');assert.equal(c.draft,'保留问题');assert.ok(!F.turnsHtml(c.state).includes('BAD FAILED TEXT'));
 assert.equal(attempt.input_sha256,hash);assert.equal(attempt.question,'保留问题');assert.deepEqual(Object.keys(attempt).sort(),['input_sha256','question','request_id']);
 mode='pending';const pending=c.submit();await Promise.resolve();const before=calls.length;await c.submit();assert.equal(calls.length,before,'duplicate pending submit must not call API');assert.equal(c.busy,true);
 const observed=seen.length;active=false;resolvePending({...base,turns:[]});await pending;assert.equal(seen.length,observed,'late response must not render after task/account switch');assert.equal(c.state.turns.length,2);
 c.dispose();const restored=F.create({...options,current:()=>true,onChange(){}});assert.equal(restored.draft,'保留问题');restored.dispose();
 const other=F.create({...options,identity:'bob',current:()=>true,onChange(){}});assert.equal(other.draft,'','drafts must not cross identities');other.dispose();
 const reader=F.create({...options,canAsk:false,current:()=>true,onChange(){}});reader.setDraft('read only');await reader.submit();assert.equal(calls.length,before);reader.dispose();
 const bad={...good,status:'SUCCEEDED',citation_reference_validation:'FAILED',answer:[{text:'HIDDEN UNVALIDATED',citation_ids:['source-1']}]};assert.ok(!F.turnsHtml({...base,turns:[bad]}).includes('HIDDEN UNVALIDATED'));
 let limitedCalls=0;
 const limited=F.create({...options,identity:'limits',current:()=>true,onChange(){},api:async()=>{limitedCalls++;return {...base,limits:{max_turns:1,max_question_characters:5,max_history_characters:50,max_source_characters:60}}}});
 await limited.load();limited.setDraft('short');await limited.submit();assert.equal(limitedCalls,1,'server turn limit must prevent another POST');assert.ok(limited.limitsText.includes('50'),'capacity guidance must use server limits');limited.dispose();
 let fallbackId=null;context.crypto={getRandomValues:require('node:crypto').webcrypto.getRandomValues.bind(require('node:crypto').webcrypto)};
 const fallback=F.create({...options,identity:'fallback',current:()=>true,onChange(){},api:async(path,init)=>{fallbackId=JSON.parse(init.body).request_id;return {...base,turns:[{...good,request_id:fallbackId}]}}});
 fallback.setDraft('问题');await fallback.submit();assert.match(fallbackId,/^[A-Za-z0-9_-]{8,80}$/,'HTTP contexts without randomUUID still submit valid unique request ids');fallback.dispose();
 console.log('follow-up controller behavior passed');
})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run([node, '-e', script, str(RESOURCES)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('surface', ['mission', 'legacy'])
def test_production_surfaces_mount_history_and_submit_real_follow_up(surface):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    script = r'''
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const [root,surface]=process.argv.slice(1),nodes=new Map();
function element(){const children=new Map();return {value:'',textContent:'',innerHTML:'',disabled:false,hidden:false,isConnected:true,style:{},dataset:{},classList:{add(){},remove(){},toggle(){},contains(){return false}},replaceChildren(){this.innerHTML=''},setAttribute(){},addEventListener(){},querySelectorAll:()=>[],querySelector(s){if(!children.has(s))children.set(s,element());return children.get(s)},append(n){this.lastChild=n},remove(){this.isConnected=false},scrollIntoView(){}}}
const $=s=>{if(!nodes.has(s))nodes.set(s,element());return nodes.get(s)};
const hash='a'.repeat(64),binding={job_id:'JOB-A',input_sha256:hash,report_sha256:hash,revision:0,turns:[],sources:[{source_id:'s1',filename:'报告.txt',locator:'line:1',text:'原文',sha256:hash,role:'diagnosis_results'}],limits:{}};
const fixture={job_id:'JOB-A',artifact_kind:'RESULT_LOG_REVIEW',analysis:{analysis_mode:'EXTERNAL_DIAGNOSIS_ANALYSIS',knowledge_enrichment:{generation:{status:'SUCCEEDED'},model_explanation:{status:'AVAILABLE_REQUIRES_EXPERT_REVIEW',citation_reference_validation:'PASS',summary:'摘要',sources:binding.sources,input_sha256:hash}}}};
const calls=[];let revision=0,lateMode=false,resolveLate;
const request=async(path,init={})=>{calls.push([path,init]);if(path.endsWith('/identity'))return {subject:'token-user',roles:['operator'],mode:'bearer'};if(path.endsWith('/follow-up')){if(init.method==='POST'){const q=JSON.parse(init.body);if(lateMode)return new Promise(resolve=>resolveLate=()=>resolve({...binding,turns:[{request_id:q.request_id,question:q.question,status:'SUCCEEDED',citation_reference_validation:'PASS',answer:[{text:'旧凭据迟到回答',citation_ids:['s1']}],information_gaps:[]}]}));return {...binding,revision:++revision,turns:[{request_id:q.request_id,question:q.question,status:'SUCCEEDED',citation_reference_validation:'PASS',answer:[{text:'真实模型回答',citation_ids:['s1']}],information_gaps:[]}]}}return binding}return {}};
const ctx={console,history:{replaceState(){}},scrollTo(){},crypto:require('node:crypto').webcrypto,document:{querySelector:$,querySelectorAll:()=>[],createElement:element},clearTimeout(){},setTimeout(){},localStorage:{},state:{},$,$$:()=>[],api:request,artifactKindZh:{},modeZh:{},focusManualReportReview(){},nav(){},enhanceOperationsResult(){}};ctx.window=ctx;vm.createContext(ctx);
vm.runInContext(fs.readFileSync(root+'/external_follow_up.js','utf8'),ctx);
let source=fs.readFileSync(root+'/'+(surface==='mission'?'mission_console.js':'platform_report.js'),'utf8');
if(surface==='mission'){source=source.slice(0,source.lastIndexOf('main();'));vm.runInContext(source,ctx);vm.runInContext('api=__request',Object.assign(ctx,{__request:request}));vm.runInContext('S.identity={subject:"alice",roles:["operator"]};S.report=__fixture;S.jobId="JOB-A";renderReport()',Object.assign(ctx,{__fixture:fixture}));}
else{vm.runInContext(fs.readFileSync(root+'/platform_core.js','utf8').split('\n').find(s=>s.startsWith('function esc(')),ctx);source=source.split('\n').filter(s=>!s.startsWith('import ')).join('\n').split('\nexport {')[0];vm.runInContext(source,ctx);ctx.fixture=fixture;vm.runInContext('renderReport(fixture)',ctx);}
(async()=>{
 for(let n=0;n<12;n++)await Promise.resolve();
 assert.ok(calls.some(c=>c[0].endsWith('/follow-up')&&!c[1].method),'surface must recover persisted server history');
 if(surface==='mission'){$('#chat-text').value='比较现有证据';await vm.runInContext('sendChat()',ctx);assert.ok($('#conversation').innerHTML.includes('真实模型回答'));}
 else{const host=$('#result-content').lastChild;assert.ok(host,'legacy must mount a composer');const input=host.querySelector('[data-follow-up-input]');input.value='比较现有证据';input.oninput();await host.querySelector('[data-follow-up-form]').onsubmit({preventDefault(){}});for(let n=0;n<12;n++)await Promise.resolve();assert.ok(host.querySelector('[data-follow-up-turns]').innerHTML.includes('真实模型回答'));}
 if(surface==='mission'){
  ctx.nextFixture=JSON.parse(JSON.stringify(fixture));vm.runInContext('S.report=nextFixture;renderReport()',ctx);
  for(let n=0;n<12;n++)await Promise.resolve();
  $('#chat-text').value='重新打开后的问题';await vm.runInContext('sendChat()',ctx);
  assert.equal(calls.filter(c=>c[1].method==='POST').length,2,'reopening the same task must bind a live controller to the new report instance');
  // List-form bearer credentials deliberately share subject=token-user.
  vm.runInContext('S.token="token-A";S.identity={subject:"token-user",roles:["operator"],mode:"bearer"};S.report=nextFixture;renderReport()',ctx);
  for(let n=0;n<12;n++)await Promise.resolve();
  vm.runInContext('externalConversation.setDraft("凭据 A 的未发送草稿")',ctx);
  const otherReport=JSON.parse(JSON.stringify(fixture));otherReport.job_id='JOB-B';ctx.otherReport=otherReport;
  vm.runInContext('S.report=otherReport;renderReport();S.report=nextFixture;renderReport()',ctx);
  for(let n=0;n<12;n++)await Promise.resolve();
  assert.equal($('#chat-text').value,'凭据 A 的未发送草稿','same authenticated session must retain drafts across task navigation');
  lateMode=true;$('#chat-text').value='凭据 A 的未发送草稿';const oldCall=vm.runInContext('sendChat()',ctx);
  for(let n=0;n<12;n++)await Promise.resolve();
  const oldId=JSON.parse(calls.filter(c=>c[1].method==='POST').at(-1)[1].body).request_id;
  $('#identity-token').value='token-B';await vm.runInContext('loginToken()',ctx);
  vm.runInContext('S.report=nextFixture;S.jobId="JOB-A";renderReport()',ctx);
  for(let n=0;n<12;n++)await Promise.resolve();
  assert.equal(vm.runInContext('S.identity.subject',ctx),'token-user');
  assert.equal($('#chat-text').value,'','same-subject replacement credential must not inherit unsent drafts');
  lateMode=false;$('#chat-text').value='凭据 A 的未发送草稿';await vm.runInContext('sendChat()',ctx);
  const newId=JSON.parse(calls.filter(c=>c[1].method==='POST').at(-1)[1].body).request_id;
  assert.notEqual(newId,oldId,'replacement credential must not inherit a previous submission attempt');
  vm.runInContext('externalConversation.setDraft("凭据 B 的草稿")',ctx);resolveLate();await oldCall;
  vm.runInContext('resetConversation();ensureExternalConversation()',ctx);
  for(let n=0;n<12;n++)await Promise.resolve();
  assert.equal($('#chat-text').value,'凭据 B 的草稿','late old response must not repopulate cleared credential cache');
  assert.ok(!$('#conversation').innerHTML.includes('旧凭据迟到回答'));
  $('#chat-send').disabled=true;$('#chat-status').textContent='正在处理旧任务';vm.runInContext('resetConversation()',ctx);assert.equal($('#chat-send').disabled,false);assert.equal($('#chat-status').textContent,'');
 }
 const post=calls.find(c=>c[1].method==='POST');assert.ok(post,'surface must make a real POST');assert.equal(post[0],'/v1/platform/jobs/JOB-A/follow-up');assert.equal(JSON.parse(post[1].body).question,'比较现有证据');
})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run([node, '-e', script, str(RESOURCES), surface], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
