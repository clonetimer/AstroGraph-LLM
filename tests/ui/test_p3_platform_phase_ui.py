"""Real Chromium component regression for the production Mission phase-model selector."""
from pathlib import Path
import shutil
import pytest

ROOT = Path(__file__).resolve().parents[2] / 'src/astrograph/resources'


def test_followup_selector_is_optional_and_invalidates_preflight():
    chromium = shutil.which('chromium') or shutil.which('chromium-browser')
    if not chromium:
        pytest.skip('Chromium is unavailable')
    from playwright.sync_api import sync_playwright
    source = (ROOT / 'mission_console.js').read_text(encoding='utf-8')
    component = source[source.index('function renderFollowupEngines(){'):source.index('function getRoleFiles(')]
    handler = source[source.index("  $('#mission-followup-engine').onchange="):source.index("  $('#analysis-engine-toggle').onclick=")]
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, headless=True, args=['--no-sandbox'])
        page = browser.new_page()
        try:
            page.set_content('''<div id="mission-followup-options" hidden>
             <select id="mission-followup-engine"></select></div>
             <div id="analysis-engine-options"></div><div id="analysis-engine-label"></div>
             <div id="analysis-engine-menu" hidden></div><button id="analysis-engine-toggle"></button>
             <div id="preflight-results"></div><button id="mission-start"></button>
             <div id="upload-state"></div>''')
            page.evaluate('''() => {
                window.S={analysisEngines:[
                  {engine_id:'rules',kind:'rules',available:true,name:'规则'},
                  {engine_id:'llm:MAIN',kind:'llm',available:true,name:'Main'},
                  {engine_id:'llm:QA',kind:'llm',available:true,name:'QA'}],
                  analysisEngineId:'llm:MAIN',followupEngineId:'',analysisEngineLoaded:true,
                  uploadRevision:0,preflight:{can_start:true},preflightId:'JOB-ID'};
                window.$=selector=>document.querySelector(selector);
                window.$$=selector=>[...document.querySelectorAll(selector)];
                window.html=value=>String(value);
                window.schedulePreflight=()=>{window.__preflightSchedules=(window.__preflightSchedules||0)+1};
                window.setUploadMessage=()=>{};
                window.renderAnalysisEngines=()=>{};
              }''')
            page.add_script_tag(content=component + '\n'+handler)
            page.evaluate('renderFollowupEngines()')
            assert page.locator('#mission-followup-options').is_visible()
            assert page.locator('#mission-followup-engine option').count() == 3
            page.locator('#mission-followup-engine').select_option('llm:QA')
            assert page.evaluate('S.followupEngineId') == 'llm:QA'
            assert page.evaluate('S.preflight') is None
            assert page.locator('#mission-start').is_disabled()
            page.evaluate("S.analysisEngineId='rules';renderFollowupEngines()")
            assert page.evaluate('S.followupEngineId') == ''
            assert page.locator('#mission-followup-options').is_hidden()
            page.evaluate("S.analysisEngineId='llm:MAIN';renderFollowupEngines()")
            page.locator('#mission-followup-engine').select_option('llm:QA')
            page.evaluate("S.analysisEngines=S.analysisEngines.filter(e=>e.engine_id!=='llm:QA');renderFollowupEngines()")
            assert page.evaluate('S.followupEngineId') == ''
            assert page.evaluate('window.__preflightSchedules') >= 3
        finally:
            browser.close()
