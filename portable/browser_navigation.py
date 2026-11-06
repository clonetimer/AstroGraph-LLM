"""Production HTML/JS/CSS in Chromium; fixture API, never a production model.

Development only; reports each behavioral scenario and writes screenshots.
"""
from pathlib import Path
import argparse
import json
import re
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
JOB = 'a' * 32
SID = 'source:test-original'
SOURCE = {'source_id': SID, 'filename': '诊断报告.pdf', 'locator': 'pdf#page:1:block:1',
          'text': '原文：温度偏高。<img src=x onerror=alert(1)>', 'sha256': 'a'*64, 'role': 'diagnosis_results'}
JOBROW = {'job_id': JOB, 'task': '反作用轮温度异常核查', 'status': 'SUCCEEDED',
          'created_at': '2026-11-04T01:00:00Z', 'mode': 'local', 'model': 'test-model'}
MODEL = {'status':'AVAILABLE_REQUIRES_EXPERT_REVIEW','citation_reference_validation':'PASS','input_sha256':'b'*64,
         'sources':[SOURCE], 'summary':'回归测试示例（非真实诊断结论）',
         'findings':[{'text':'材料中记载温度变化。','citation_ids':[SID]}], 'hypotheses':[],
         'information_gaps':['需要核对观测时段'], 'recommended_review_actions':['核对原始时间戳'],
         'common_findings':[{'text':'两份材料均记录温度偏高。','citation_ids':[SID]}],
         'report_disagreements':[{'text':'时段尚未对齐，不能认定为同一次异常。','citation_ids':[SID]}],
         'integrated_conclusion':'请先对齐时段，再核查原因。'}
REPORT = {'analysis':{'knowledge_enrichment':{'generation':{'status':'SUCCEEDED'},'model_explanation':MODEL}}}
ROUTES = ['start', 'result', 'history', 'chat', 'settings', 'help']
LABELS = ['开始分析', '分析结果', '历史任务', '证据问答', '模型设置', '使用帮助']

def exercise(chromium: str, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    passed, failed = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, headless=True, args=['--no-sandbox'])
        def new_page(width=1440, fixture=None):
            page = browser.new_page(viewport={'width':width,'height':1000})
            page.set_default_timeout(5000)
            html=(ROOT/'assets/index.html').read_text(encoding='utf-8')
            scripts=re.findall(r'<script[^>]*src="/([^"]+)"[^>]*></script>',html)
            html=re.sub(r'<script[^>]*src="/[^"]+"[^>]*></script>|<link[^>]*href="/app.css"[^>]*>','',html)
            page.set_content(html)
            page.add_style_tag(content=(ROOT/'assets/app.css').read_text(encoding='utf-8'))
            shared_css=ROOT/'assets/execution_view.css'
            if not shared_css.is_file():shared_css=ROOT.parent/'src/astrograph/resources/execution_view.css'
            page.add_style_tag(content=shared_css.read_text(encoding='utf-8'))
            page.evaluate('''fixture => {
              window.__requests=[]; window.__fixture=fixture; window.__postCount=0;
              window.fetch=async (url, options={})=>{
                const body=options.body && typeof options.body==='string' ? JSON.parse(options.body):null;
                window.__requests.push({url,body,method:options.method||'GET'});
                const f=window.__fixture; let data;
                if(url==='/api/session') data={token:'local-test-token',version:'fixture-source-version', sample:'虚构示例：设备温度升高。'};
                else if(url==='/api/jobs' && body) {window.__postCount++;data={job:f.job};}
                else if(url==='/api/jobs') data={jobs:f.jobs};
                else if(url==='/api/probe') data={models:[{id:'test-model'}]};
                else if(url.endsWith('/conversation')) data={job_id:f.job.job_id,input_sha256:'b'.repeat(64),sources:f.sources,turns:f.turns||[]};
                else if(url.endsWith('/follow-up')) {await new Promise(r=>setTimeout(r,100));data={job_id:f.job.job_id,input_sha256:'b'.repeat(64),sources:f.sources,turns:[{request_id:body.request_id,question:body.question,status:'SUCCEEDED',citation_reference_validation:'PASS',answer:[{text:'核对材料中的观测时段。',citation_ids:['source:test-original']}],information_gaps:[]}]};}
                else data={job:f.job,report:f.report};
                return {ok:true,json:async()=>data};
              };
            }''',fixture or {'jobs':[JOBROW], 'job':JOBROW, 'sources':[SOURCE], 'report':REPORT})
            for script in scripts:
                page.add_script_tag(content=((ROOT/'assets'/script) if (ROOT/'assets'/script).is_file() else ROOT.parent/'src/astrograph/resources'/script).read_text(encoding='utf-8'))
            page.wait_for_function("document.querySelector('#service-status').textContent==='本机工作台已就绪'")
            return page
        def nav(page, route):
            if page.viewport_size['width']<900 and not page.locator('body').evaluate("e=>e.classList.contains('nav-open')"):
                page.locator('#mobile-nav').click()
            page.locator(f'#main-nav [data-route="{route}"]').click()
            page.wait_for_function('(r)=>!document.querySelector(`[data-page="${r}"]`).hidden', arg=route)
            if page.viewport_size['width']<900:
                page.wait_for_function("document.querySelector('#sidebar').getBoundingClientRect().right <= 1")
        def scenario(name, fn):
            try:
                fn();passed.append(name)
            except Exception as e:
                failed.append({'scenario':name,'error':str(e)})
        def navigation_preserves():
            page=new_page()
            try:
                assert page.locator('.version-label').inner_text()=='fixture-source-version'
                assert 'Win10 x64' not in page.locator('.header-status').inner_text()
                assert page.locator('#main-nav [data-route]').all_text_contents()==LABELS
                page.locator('#paste').fill('保留中的诊断文本')
                page.locator('#files').set_input_files({'name':'原始诊断.txt','mimeType':'text/plain','buffer':'温度异常'.encode()})
                nav(page,'settings');page.locator('#model').fill('keep-model');page.locator('#api-key').fill('memory-only-key')
                for route in ROUTES:
                    nav(page,route)
                    assert page.locator('[data-page]:not([hidden])').count()==1
                    assert page.locator(f'#main-nav [data-route="{route}"]').get_attribute('aria-current')=='page'
                nav(page,'start');assert page.locator('#paste').input_value()=='保留中的诊断文本'
                assert page.locator('#file-list > li').count()==1
                page.screenshot(path=str(output/'desktop-start-analysis.png'),full_page=True)
                nav(page,'settings');assert page.locator('#model').input_value()=='keep-model'
                assert page.locator('#api-key').input_value()=='memory-only-key'
                assert page.evaluate('window.__requests.filter(r=>r.method==="POST").length')==0
                page.screenshot(path=str(output/'desktop-model-settings.png'),full_page=True)
            finally:page.close()
        def history_report_chat():
            page=new_page()
            try:
                nav(page,'history');page.locator('#history-search').fill('不匹配')
                assert page.locator('.history-item').count()==0
                page.locator('#history-search').fill('反作用轮');assert page.locator('.history-item').count()==1
                page.locator('.history-open').click();page.wait_for_selector('#analysis-content h3')
                assert not page.locator('[data-page="result"]').evaluate('e=>e.hidden')
                assert page.locator('#source-list img').count()==0
                page.screenshot(path=str(output/'desktop-analysis-results.png'),full_page=True)
                nav(page,'chat');page.locator('#question').fill('哪些依据需要补充？')
                nav(page,'settings');page.locator('#model').fill('test-model')
                nav(page,'chat');assert page.locator('#question').input_value()=='哪些依据需要补充？'
                page.locator('#send-question').click();page.wait_for_selector('#turns .citation-button')
                page.locator('#turns .citation-button').first.click()
                assert not page.locator('[data-page="result"]').evaluate('e=>e.hidden')
                assert page.locator('#source-0').evaluate('e=>e.open')
                assert page.locator('#source-0').is_visible()
                assert page.evaluate('window.__requests.filter(r=>r.url.endsWith("/follow-up")).length')==1
            finally:page.close()
        def file_submission():
            page=new_page()
            try:
                page.locator('#files').set_input_files([
                    {'name':'诊断报告.pdf','mimeType':'application/pdf','buffer':b'%PDF-fixture'},
                    {'name':'参考.docx','mimeType':'application/vnd.openxmlformats-officedocument.wordprocessingml.document','buffer':b'PK-fixture'}])
                page.locator('#file-list > li').nth(1).locator('select').first.select_option('references')
                nav(page,'settings');page.locator('#probe').click();page.wait_for_function("document.querySelector('#model').value==='test-model'")
                nav(page,'start');page.locator('#analyze').click();page.wait_for_selector('#analysis-content h3')
                sent=page.evaluate("window.__requests.find(r=>r.url==='/api/jobs' && r.body).body")
                assert sent['files'][1]['role']=='references' and sent['files'][0]['filename']=='诊断报告.pdf'
                assert page.evaluate('window.__postCount')==1
            finally:page.close()
        def failed_resume():
            row={**JOBROW,'resumable':True}
            report={'analysis':{'knowledge_enrichment':{'generation':{'status':'FAILED','reason_code':'MODEL_GENERATION_FAILED'},'model_explanation':{'sources':[SOURCE]}}}}
            page=new_page(fixture={'jobs':[row],'job':row,'sources':[SOURCE],'report':report})
            try:
                nav(page,'history');page.locator('.history-open').click();page.wait_for_selector('#resume-job')
                nav(page,'settings');page.locator('#model').fill('test-model');nav(page,'result');page.locator('#resume-job').click()
                page.wait_for_function("window.__requests.some(r=>r.url.endsWith('/resume'))")
            finally:page.close()
        def narrow():
            page=new_page(width=390)
            try:
                for route in ROUTES:
                    nav(page,route)
                    assert page.locator('#mobile-nav').get_attribute('aria-expanded')=='false'
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                nav(page,'settings');page.screenshot(path=str(output/'mobile-model-settings.png'),full_page=True)
                page.locator('#mobile-nav').click();page.keyboard.press('Escape')
                assert page.locator('#mobile-nav').get_attribute('aria-expanded')=='false'
            finally:page.close()
        def empty_help():
            page=new_page(fixture={'jobs':[],'job':JOBROW,'sources':[],'report':None})
            try:
                nav(page,'history');assert page.locator('#history-empty').is_visible()
                nav(page,'chat');assert page.locator('#chat-empty').is_visible()
                nav(page,'help');assert 'CHECK.bat' in page.locator('[data-page="help"]').inner_text()
                page.screenshot(path=str(output/'desktop-help.png'),full_page=True)
                page.evaluate("location.hash='unknown-route'")
                page.wait_for_function("!document.querySelector('[data-page=start]').hidden")
            finally:page.close()
        def back_forward_keyboard():
            page=new_page()
            try:
                nav(page,'settings');nav(page,'help')
                page.evaluate('history.back()')
                page.wait_for_function("!document.querySelector('[data-page=settings]').hidden")
                page.evaluate('history.forward()')
                page.wait_for_function("!document.querySelector('[data-page=help]').hidden")
                page.locator('[data-route=history]').focus();page.keyboard.press('Enter')
                page.wait_for_function("!document.querySelector('[data-page=history]').hidden")
                assert page.locator('#title-history').evaluate('e=>document.activeElement===e')
            finally:page.close()
        def running_does_not_hijack_view():
            running={**JOBROW,'status':'RUNNING','progress':{'stage':'CHUNK_ANALYSIS','completed':2,'total':6}}
            page=new_page(fixture={'jobs':[running],'job':running,'sources':[SOURCE],'report':None})
            try:
                nav(page,'history');page.locator('.history-open').click();page.wait_for_selector('#job-progress')
                nav(page,'help');assert page.locator('#active-operation').is_visible()
                page.evaluate('(f)=>{window.__fixture.job=f.job;window.__fixture.report=f.report;window.__fixture.jobs=[f.job]}',{'job':JOBROW,'report':REPORT})
                page.wait_for_function("document.querySelector('#job-badge').textContent==='模型分析已生成'")
                assert not page.locator('[data-page=help]').evaluate('e=>e.hidden')
                assert not page.locator('#active-operation').is_visible()
                nav(page,'result');assert page.locator('#analysis-content h3').count()>0
                assert page.evaluate('window.__postCount')==0
            finally:page.close()
        def desktop_zoom_layout():
            # 900/1024 CSS pixels represent narrow laptop windows and common zoomed layouts.
            for width in [900,1024,1280]:
                page=new_page(width=width)
                try:
                    for route in ROUTES:
                        nav(page,route)
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(width,route)
                finally:page.close()
        scenario('Six routes, one visible panel, draft/files/model/key preservation, no automatic model call',navigation_preserves)
        scenario('History filter/open, report rendering, question preservation, real submission and source jump',history_report_chat)
        scenario('PDF/DOCX role retention, probe and one analysis submission',file_submission)
        scenario('Failed report resumes with current model settings',failed_resume)
        scenario('390px navigation drawer, Escape and no horizontal overflow',narrow)
        scenario('Empty history/chat, built-in help and invalid route fallback',empty_help)
        scenario('Browser back/forward and keyboard focus follow the active route',back_forward_keyboard)
        scenario('Background polling preserves the chosen page and never resubmits analysis',running_does_not_hijack_view)
        scenario('900/1024/1280px desktop workspaces have no horizontal overflow',desktop_zoom_layout)
        browser.close()
    result={'environment':'Linux Chromium; production DOM/JS; scripted API responses',
            'native_windows':False,'real_model':False,'passed':passed,'failures':failed}
    (output/'navigation-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));return 1 if failed else 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chromium',default='/usr/bin/chromium');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(exercise(args.chromium,args.output))
