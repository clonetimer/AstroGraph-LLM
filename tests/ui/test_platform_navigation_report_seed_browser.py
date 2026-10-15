"""Operator-first navigation, overview report and embedded reference graph (offline Chromium)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from astrograph.api import create_app

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright

RES=Path(__file__).resolve().parents[2]/'src/astrograph/resources'
SEED=json.loads((RES/'public_spacecraft_seed.v1.json').read_text('utf-8'))
HTML=(RES/'platform_ui.html').read_text('utf-8')
OFFLINE=HTML.replace('<link rel="stylesheet" href="/platform/assets/v0.21.1/platform_ui.css">','').replace('<script type="module" src="/platform/assets/v0.21.1/platform_ui.js"></script>','')
CSS=(RES/'platform_ui.css').read_text('utf-8')

def script(name):
    source=(RES/name).read_text('utf-8')
    return '\n'.join(line for line in source.splitlines() if not line.startswith('import ')).rsplit('\nexport {',1)[0]

SCRIPT='\n'.join([script('platform_core.js'),'function renderKnowledgeOperations(){}',script('platform_knowledge.js'),script('platform_report.js'),script('platform_shell.js')])

@pytest.fixture
def browser_page():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True, **({'executable_path': os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']} if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE') else {}),args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1366,'height':900})
        page.set_default_timeout(8000)
        page.set_content(OFFLINE,wait_until='domcontentloaded')
        page.evaluate("""Object.defineProperty(window,'localStorage',{configurable:true,value:{getItem(){return null},setItem(){},removeItem(){}}});""")
        page.add_style_tag(content=CSS)
        page.add_script_tag(content='window.fetch=async()=>({ok:true,status:200,headers:{get:()=>"application/json"},json:async()=>('+json.dumps(SEED,ensure_ascii=False)+')});')
        page.add_script_tag(content=SCRIPT+'\nsetupShellHooks({loadKnowledgeWorkspace});initShell();')
        yield page
        browser.close()


def test_public_reference_seed_api_is_unreviewed_and_bundled(monkeypatch,tmp_path):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'jobs'))
    response=TestClient(create_app()).get('/v1/platform/knowledge/public-seed')
    assert response.status_code==200
    payload=response.json()
    assert len(payload['entities'])==206 and len(payload['relations'])==225
    assert len(payload['clusters'])==30
    assert payload['is_flight_qualified'] is False
    assert payload['graph_kind']=='UNREVIEWED_PUBLIC_REFERENCE_SEED'
    assert all(r['review_status']=='PUBLIC_REFERENCE_SEED_UNREVIEWED' for r in payload['relations'])


def test_navigation_exposes_llm_and_collapses_advanced(browser_page):
    page=browser_page
    assert page.locator('#view-system').count()==1
    assert page.get_by_role('button',name='大模型配置',exact=True).count()==1
    assert not page.locator('#nav-advanced-more').evaluate('(el)=>el.open')
    assert not page.locator('#nav-knowledge-more').evaluate('(el)=>el.open')
    page.get_by_role('button',name='大模型配置',exact=True).click()
    assert page.locator('#view-system.active #llm-model').is_visible()
    assert not page.locator('#view-system #neo4j-form').is_visible()
    page.evaluate('nav("graph-review")')
    assert page.locator('#nav-knowledge-more').evaluate('(el)=>el.open')
    page.evaluate('nav("agent")')
    assert page.locator('#nav-advanced-more').evaluate('(el)=>el.open')


def test_embedded_seed_browsing_and_source(browser_page):
    page=browser_page
    page.evaluate('nav("knowledge")')
    page.wait_for_function('document.querySelectorAll("#seed-fault-list [data-seed-fault]").length === 30')
    assert page.locator('#seed-overview').inner_text().find('206') >= 0
    assert page.locator('#seed-overview').inner_text().find('225') >= 0
    assert page.locator('#seed-topic-detail svg').count()==1
    page.locator('#seed-domain').select_option('ADCS')
    assert 1<=page.locator('[data-seed-fault]').count()<30
    page.locator('#seed-search').fill('飞轮')
    assert page.locator('[data-seed-fault]').count()>=1
    page.locator('[data-seed-fault]').first.click()
    assert page.locator('#seed-topic-detail').get_by_text('资料依据').count()==1
    assert page.locator('#seed-topic-detail a[target="_blank"]').count()>=1


def test_report_tabs_hide_technical_information_by_default(browser_page):
    page=browser_page
    report={
       'analysis':{'result_log_review':{
         'conclusion':'反作用轮机械阻力可能增大，尚需人工核验。',
         'candidates':[{'rank':1,'fault_id':'ADCS_RW_FRICTION_INCREASE','fault_name':'反作用轮摩擦增大',
           'status':'PROVISIONAL_LOG_CUE','cause_hypotheses':[{'text':'摩擦或润滑变化'}],
           'handling_measures':[{'action':'对齐轮速、电流与温度'}],
           'verification_checks':['检查温度趋势'], 'diagnosis_source_ids':[],'log_evidence_ids':[],'reference_citation_ids':[]}],
         'evidence_gaps':['缺少完整轮速曲线']},
         'material_reasoning':{'candidates':[{'fault_id':'ADCS_RW_FRICTION_INCREASE','fault_name':'反作用轮摩擦增大',
           'multi_hop':{'decision':'ABSTAIN'},'log_evidence_matrix':{'events':[]},
           'reference_retrieval':{'hits':[]},'evidence_assessment':{}}]}}
    }
    page.evaluate('nav("result")')
    page.evaluate('(report)=>renderResultLogReview(report)',report)
    assert page.locator('[data-report-panel="summary"]').is_visible()
    assert page.locator('[data-report-panel="graph"]').is_hidden()
    assert page.locator('.report-case').count()==1
    assert '对齐轮速' in page.locator('[data-report-panel="summary"]').inner_text()
    page.locator('[data-report-tab="graph"]').click()
    assert page.locator('[data-report-panel="summary"]').is_hidden()
    assert page.locator('[data-report-panel="graph"]').is_visible()
    assert 'ABSTAIN' in page.locator('[data-report-panel="graph"]').inner_text()
    page.locator('[data-report-tab="governance"]').click()
    assert page.locator('[data-report-panel="governance"]').is_visible()


def test_small_viewport_keeps_model_configuration_accessible(browser_page):
    page=browser_page
    page.set_viewport_size({'width':390,'height':844})
    page.get_by_role('button',name='大模型配置',exact=True).click()
    assert page.locator('#view-system.active #llm-form').is_visible()
    # No document-wide horizontal overflow; tabs and graph visualizations may scroll locally.
    assert page.evaluate('document.documentElement.scrollWidth <= document.documentElement.clientWidth + 3')
