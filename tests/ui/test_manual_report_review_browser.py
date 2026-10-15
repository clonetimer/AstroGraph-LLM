"""Offline Chromium: result-page review CTA, expert login, single-candidate decision."""
from __future__ import annotations
import os
from pathlib import Path
import pytest

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]/'src/astrograph/resources'
HTML=(ROOT/'platform_ui.html').read_text('utf-8')
OFFLINE=HTML.replace('<link rel="stylesheet" href="/platform/assets/v0.21.1/platform_ui.css">','').replace('<script type="module" src="/platform/assets/v0.21.1/platform_ui.js"></script>','')
CSS=(ROOT/'platform_ui.css').read_text('utf-8')

def script(name):
    src=(ROOT/name).read_text('utf-8')
    return '\n'.join(line for line in src.splitlines() if not line.startswith('import ')).rsplit('\nexport {',1)[0]

SCRIPT='\n'.join([script(x) for x in ['platform_core.js','platform_graph_workbench.js','platform_report.js','platform_shell.js']])
STUB=r"""
const j='JOB-1234567890ABCDEF12345678',f='ADCS_RW_FRICTION_INCREASE';
window.__rv={auth:false,rev:0,decision:null};
const summary=()=>({job_id:j,report_sha256:'f'.repeat(64),status:window.__rv.rev?'REVIEWED':'PENDING',
  reviewed_count:window.__rv.rev?1:0,total_count:1,revision:window.__rv.rev,
  current_root_cause_confirmed:false,automatic_control_authorized:false});
window.fetch=async(url,opts={})=>{
  let data={};const uri=String(url);let status=200;
  if(uri.includes('/manual-review/status'))data=summary();
  else if(uri.endsWith('/review-session')){
    if(opts.method==='POST'){window.__rv.auth=true;data={authenticated:true,reviewer:'expert',csrf:'csrf-123'};}
    else data={configured:true,authenticated:window.__rv.auth,reviewer:window.__rv.auth?'expert':null,csrf:window.__rv.auth?'csrf-123':null};
  }else if(uri.endsWith('/pending?limit=100'))data={snapshot:0,cases:[],documents:[],graph_proposals:[],reasoning_gaps:[]};
  else if(uri.includes('/report-reviews?'))data={items:[{...summary(),completed_at:'2026-10-09'}]};
  else if(uri.includes('/report-reviews/'+j)){
    if(opts.method==='POST'){
      const payload=JSON.parse(opts.body);window.__rv.decision=payload;window.__rv.rev++;
    }
    data={...summary(),candidates:[{fault_id:f,fault_name:'飞轮摩擦增大',review:window.__rv.rev?{actor:'expert',
      decision:window.__rv.decision.decision,note:window.__rv.decision.note,evidence_ref:window.__rv.decision.evidence_ref,
      recorded_at:'2026-10-09'}:null}],history:[]};
  }
  return {ok:status===200,status,statusText:'error',headers:{get:()=> 'application/json'},json:async()=>data};
};
"""

def test_report_review_button_opens_own_record_and_saves():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,**({'executable_path': os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']} if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE') else {}),args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1280,'height':900})
        page.set_default_timeout(10000)
        page.set_content(OFFLINE,wait_until='domcontentloaded')
        page.evaluate("Object.defineProperty(window,'localStorage',{configurable:true,value:{getItem(){return null},setItem(){}}})")
        page.add_style_tag(content=CSS)
        page.add_script_tag(content=STUB)
        page.add_script_tag(content=SCRIPT+'''\nsetupGraphWorkbench();setupShellHooks({loadGraphReview,renderReport});''')
        report={'job_id':'JOB-1234567890ABCDEF12345678','artifact_kind':'RESULT_LOG_REVIEW','analysis':{
         'result_log_review':{'conclusion':'日志提示飞轮摩擦异常，但仍需专家核验','candidates':[
            {'rank':1,'fault_id':'ADCS_RW_FRICTION_INCREASE','fault_name':'飞轮摩擦增大',
             'cause_hypotheses':[],'handling_measures':[],'verification_checks':[],'status':'PROVISIONAL_LOG_CUE'}],
             'evidence_gaps':[]},'material_reasoning':{'candidates':[]}}}
        page.evaluate('nav("result")')
        page.evaluate('(r)=>renderReport(r)',report)
        assert page.locator('#report-manual-review-open').is_visible()
        page.locator('#report-manual-review-open').click()
        assert page.locator('#view-graph-review.active').is_visible()
        assert page.locator('[data-review-kind="manual"]').is_visible()
        page.locator('#graph-review-credential').fill('test-token-never-persisted')
        page.locator('#graph-login-form button[type="submit"]').click()
        page.locator('[data-manual-candidate="ADCS_RW_FRICTION_INCREASE"]').wait_for()
        page.locator('[data-manual-decision]').select_option('FOLLOW_UP_HYPOTHESIS')
        page.locator('[data-manual-evidence]').fill('LAB-REFERENCE-0134')
        page.locator('[data-manual-note]').fill('经独立审核轮速与电流曲线，建议继续排查')
        page.once('dialog',lambda d:d.accept())
        page.locator('[data-manual-review-submit]').click()
        page.wait_for_function('window.__rv.rev===1')
        assert page.evaluate('window.__rv.decision.decision')=='FOLLOW_UP_HYPOTHESIS'
        assert page.locator('.report-review-existing').is_visible()
        page.evaluate("nav('result')")
        page.wait_for_function("document.querySelector('#report-manual-review-label')?.textContent.includes('已记录全部复核意见')")
        browser.close()
