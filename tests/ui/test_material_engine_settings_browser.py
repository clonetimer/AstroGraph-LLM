"""Chromium click-through with browser-local API stub (network blocked in CI)."""
from __future__ import annotations
from pathlib import Path
import os
import pytest
pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright

RESOURCES=Path(__file__).resolve().parents[2]/'src/astrograph/resources'
HTML=(RESOURCES/'platform_ui.html').read_text(encoding='utf-8')
OFFLINE=HTML.replace('<link rel="stylesheet" href="/platform/assets/v0.21.1/platform_ui.css">','').replace('<script type="module" src="/platform/assets/v0.21.1/platform_ui.js"></script>','')
CSS=(RESOURCES/'platform_ui.css').read_text(encoding='utf-8')
NAMES=('platform_core.js','platform_engine_settings.js','platform_shell.js')
SCRIPTS='\n'.join('\n'.join(line for line in (RESOURCES/name).read_text(encoding='utf-8').splitlines() if not line.startswith('import ')).split('\nexport {')[0] for name in NAMES)

MOCK_FETCH=r"""
window.__engineState={revision:null,settings:{embedding_model:'',vector_backend:'sqlite',qdrant_url:'',qdrant_collection:'astrograph_material_embeddings',
 community_llm_enabled:false,graph_build_enabled:false,graph_llm_model:'',ocr_engine:'disabled',ocr_languages:'chi_sim+eng',
 vision_enabled:false,vision_endpoint:'',vision_model:'',vision_pages:0,rag_documents:8,rag_sections:16,rag_hits:5},history:[]};
window.fetch=async(path,options={})=>{
 let status=200,body={};const data=window.__engineState;
 if(path.includes('/knowledge-engine')){
   if(path.endsWith('/revisions'))body={revisions:data.history.map(x=>({revision:x.revision,updated_at:x.updated_at}))};
   else if(path.includes('/probe'))body={checks:{ocr:{engine:data.settings.ocr_engine},vision:{status:'DISABLED'}},production_qualified:false};
   else if(path.endsWith('/rollback')){
     const request=JSON.parse(options.body);const target=data.history.find(x=>x.revision===request.target_revision);
     data.revision='revision-rollback-'+(++window.__mockCounter);data.settings={...target.settings};
     data.history.unshift({revision:data.revision,updated_at:'2026-10-09',settings:{...data.settings}});
     body={revision:data.revision,settings:data.settings};
   }else if(options.method==='PUT'){
     const request=JSON.parse(options.body);if(request.expected_revision!==data.revision){status=409;body={detail:'revision conflict'};}
     else{data.settings=request.settings;data.revision='revision-save-'+(++window.__mockCounter);
       data.history.unshift({revision:data.revision,updated_at:'2026-10-09',settings:{...data.settings}});
       body={revision:data.revision,settings:data.settings};}
   }else body={revision:data.revision,settings:data.settings,qdrant_api_key_configured:false};
 }
 return {ok:status===200,status,statusText:'error',headers:{get:()=> 'application/json'},json:async()=>body};
};
window.__mockCounter=0;
"""


def test_engine_config_click_through_and_version_restore():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,**({'executable_path': os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']} if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE') else {}),args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1280,'height':850})
        page.set_content(OFFLINE,wait_until='domcontentloaded')
        page.evaluate("""Object.defineProperty(window,'localStorage',{configurable:true,value:{
          _data:{},getItem(k){return this._data[k]??null},setItem(k,v){this._data[k]=String(v)},
          removeItem(k){delete this._data[k]},clear(){this._data={}}}});""")
        page.add_style_tag(content=CSS)
        page.add_script_tag(content=MOCK_FETCH)
        page.add_script_tag(content=SCRIPTS+'\nsetupEngineSettings();setupShellHooks({loadEngineSettings});nav("engine");loadEngineSettings();')
        page.locator('#view-engine.active').wait_for()
        page.wait_for_function("document.querySelector('#engine-rag-docs').value === '8'")
        page.locator('#engine-rag-docs').fill('12')
        page.locator('#engine-ocr').select_option('tesseract')
        page.locator('#engine-community-llm').check()
        page.locator('#engine-save').click()
        page.wait_for_function("document.querySelector('#engine-revision-label').textContent === 'revision-save-1'")
        first=page.locator('#engine-revision-label').inner_text()
        page.locator('#engine-probe').click()
        page.wait_for_function("document.querySelector('#engine-probe-result').textContent.includes('ocr')")
        page.locator('#engine-rag-docs').fill('6')
        page.locator('#engine-save').click()
        page.wait_for_function("document.querySelector('#engine-revision-label').textContent === 'revision-save-2'")
        page.locator('#engine-revision-select').select_option(first)
        page.once('dialog',lambda dialog:dialog.accept())
        page.locator('#engine-rollback').click()
        page.wait_for_function("document.querySelector('#engine-revision-label').textContent.startsWith('revision-rollback')")
        assert page.locator('#engine-rag-docs').input_value()=='12'
        assert page.locator('#engine-community-llm').is_checked()
        assert page.locator('#engine-ocr').input_value()=='tesseract'
        assert page.evaluate("localStorage.getItem('adminToken')") is None
        browser.close()
