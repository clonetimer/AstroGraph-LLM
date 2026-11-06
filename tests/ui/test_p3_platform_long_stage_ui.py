"""Mission must allow explicit long-report stage model selection without auto routing."""
from pathlib import Path
import shutil
import pytest

RESOURCES = Path(__file__).resolve().parents[2] / 'src/astrograph/resources'


def test_mission_long_stage_selectors_and_backend_payload():
    html = (RESOURCES / 'mission_console.html').read_text(encoding='utf-8')
    js = (RESOURCES / 'mission_console.js').read_text(encoding='utf-8')
    assert 'id="mission-chunk-engine"' in html
    assert 'id="mission-synthesis-engine"' in html
    assert 'chunkEngineId' in js and 'synthesisEngineId' in js
    assert 'phase_engine_ids:phaseEngineSelections()' in js


def test_mission_long_stage_model_selections_are_optional_and_invalidate_preflight():
    binary = shutil.which('chromium') or shutil.which('chromium-browser')
    if not binary:
        pytest.skip('Chromium not available')
    from playwright.sync_api import sync_playwright
    source = (RESOURCES / 'mission_console.js').read_text(encoding='utf-8')
    component = source[source.index('function renderFollowupEngines(){'):source.index('function getRoleFiles(')]
    handler = source[source.index("  $('#mission-followup-engine').onchange="):source.index("  $('#analysis-engine-toggle').onclick=")]
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=binary, headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page()
            page.set_content('''<section id="mission-followup-options" hidden><select id="mission-followup-engine"></select></section>
              <section id="mission-chunk-options" hidden><select id="mission-chunk-engine"></select></section>
              <section id="mission-synthesis-options" hidden><select id="mission-synthesis-engine"></select></section>
              <div id="analysis-engine-options"></div><div id="analysis-engine-label"></div>
              <div id="analysis-engine-menu" hidden></div><button id="analysis-engine-toggle"></button>
              <div id="preflight-results"></div><button id="mission-start"></button>
              <div id="upload-state"></div>''')
            page.evaluate('''() => {
              window.S={analysisEngines:[
                {engine_id:'rules',kind:'rules',available:true,name:'规则'},
                {engine_id:'llm:MAIN',kind:'llm',available:true,name:'主模型'},
                {engine_id:'llm:CHUNK',kind:'llm',available:true,name:'片段模型'},
                {engine_id:'llm:SYNTH',kind:'llm',available:true,name:'综合模型'}],
                analysisEngineId:'llm:MAIN',followupEngineId:'',chunkEngineId:'',synthesisEngineId:'',
                analysisEngineLoaded:true,uploadRevision:0,preflight:{can_start:true},preflightId:'JOB-ID'};
              window.$=selector=>document.querySelector(selector);
              window.$$=selector=>[...document.querySelectorAll(selector)];
              window.html=x=>String(x);
              window.schedulePreflight=()=>{window.__checks=(window.__checks||0)+1};
              window.setUploadMessage=()=>{};window.renderAnalysisEngines=()=>{};
            }''')
            page.add_script_tag(content=component + '\n' + handler)
            page.evaluate('renderFollowupEngines()')
            for key,model in [('chunk', 'llm:CHUNK'), ('synthesis', 'llm:SYNTH')]:
                assert page.locator('#mission-'+key+'-options').is_visible()
                page.locator('#mission-'+key+'-engine').select_option(model)
                assert page.evaluate('S.'+key+'EngineId') == model
                assert page.locator('#mission-start').is_disabled()
            assert page.evaluate('phaseEngineSelections()') == {'chunk': 'llm:CHUNK', 'synthesis': 'llm:SYNTH'}
            page.evaluate("S.analysisEngineId='rules';renderFollowupEngines()")
            assert page.evaluate('phaseEngineSelections()') == {}
            assert page.locator('#mission-chunk-options').is_hidden()
            assert page.locator('#mission-synthesis-options').is_hidden()
            assert page.evaluate('window.__checks') >= 2
        finally:
            browser.close()
