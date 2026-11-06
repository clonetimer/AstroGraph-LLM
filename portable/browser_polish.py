"""V2.7.1 layout/behavior regression on production DOM/CSS/JS, fixture APIs.

No production model or external network request. Screenshots show fictional data.
Run on a development machine with Playwright and Chromium, not on the user PC.
"""
from pathlib import Path
import argparse
import copy
import json
import re
from playwright.sync_api import sync_playwright
from browser_navigation import JOB, JOBROW, MODEL, REPORT, SOURCE, ROUTES

ROOT = Path(__file__).resolve().parent


def exercise(chromium: str, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    passed, failed, measurements = [], [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=chromium, headless=True, args=['--no-sandbox'])

        def page_for(width=1366, height=768, report=None, defer_probe=False):
            page = browser.new_page(viewport={'width': width, 'height': height})
            page.set_default_timeout(5000)
            html = (ROOT / 'assets/index.html').read_text(encoding='utf-8')
            scripts = re.findall(r'<script[^>]*src="/([^"]+)"[^>]*></script>', html)
            html = re.sub(r'<script[^>]*src="/[^"]+"[^>]*></script>|<link[^>]*href="/app.css"[^>]*>', '', html)
            page.set_content(html)
            page.add_style_tag(content=(ROOT / 'assets/app.css').read_text(encoding='utf-8'))
            shared_css=ROOT/'assets/execution_view.css'
            if not shared_css.is_file():shared_css=ROOT.parent/'src/astrograph/resources/execution_view.css'
            page.add_style_tag(content=shared_css.read_text(encoding='utf-8'))
            page.evaluate('''fixture => {
              window.__requests=[]; window.__fixture=fixture;
              window.fetch=async(url,options={})=>{
                const body=options.body && typeof options.body==='string' ? JSON.parse(options.body) : null;
                window.__requests.push({url,body,method:options.method||'GET'});
                const f=window.__fixture; let data;
                if(url==='/api/session') data={token:'fixture-only-token',sample:'虚构诊断：温度变化。'};
                else if(url==='/api/jobs' && body) data={job:f.job};
                else if(url==='/api/jobs') data={jobs:[f.job]};
                else if(url==='/api/probe') {if(f.defer_probe) await new Promise(resolve=>window.__resolveProbe=resolve);data={models:[{id:'test-model'}]};}
                else if(url.endsWith('/conversation')) data={job_id:f.job.job_id,input_sha256:'b'.repeat(64),sources:f.sources,turns:[]};
                else data={job:f.job,report:f.report};
                return {ok:true,json:async()=>data};
              };
            }''', {'job': JOBROW, 'sources': [SOURCE], 'report': report or REPORT, 'defer_probe': defer_probe})
            for name in scripts:
                page.add_script_tag(content=((ROOT / 'assets' / name) if (ROOT / 'assets' / name).is_file() else ROOT.parent / 'src/astrograph/resources' / name).read_text(encoding='utf-8'))
            page.wait_for_function("document.querySelector('#service-status').textContent==='本机工作台已就绪'")
            return page

        def nav(page, route):
            if page.viewport_size['width'] < 900:
                page.locator('#mobile-nav').click()
            page.locator(f'#main-nav [data-route="{route}"]').click()
            page.wait_for_function('(r)=>!document.querySelector(`[data-page="${r}"]`).hidden', arg=route)
            if page.viewport_size['width'] < 900:
                page.wait_for_function('document.querySelector("#sidebar").getBoundingClientRect().right<=1')

        def visible_button(page, selector):
            result = page.locator(selector).evaluate('''e=>{
              const r=e.getBoundingClientRect(), c=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
              return {top:r.top,bottom:r.bottom,left:r.left,right:r.right,width:innerWidth,height:innerHeight,
                      hit:!!c&&(c===e||e.contains(c)),visible:!!(r.width&&r.height)};
            }''')
            assert result['visible'] and 0 <= result['top'] < result['bottom'] <= result['height'], (selector, result)
            assert 0 <= result['left'] < result['right'] <= result['width'], (selector, result)
            assert result['hit'], ('button covered', selector, result)
            measurements.append({'selector': selector, **result})

        def scenario(title, test):
            try:
                test()
                passed.append(title)
            except Exception as exc:
                failed.append({'scenario': title, 'error': str(exc)})

        def action_visibility():
            page = page_for()
            try:
                visible_button(page, '#analyze')
                page.locator('#paste').fill('首次试用：诊断结果（虚构）。')
                page.evaluate('scrollTo(0,document.documentElement.scrollHeight/2)')
                visible_button(page, '#analyze')
                page.locator('#task').focus()
                page.locator('#task').fill('保留自定义分析目标')
                page.locator('#task').scroll_into_view_if_needed()
                box=page.locator('#task').bounding_box();bar=page.locator('#analyze').bounding_box()
                assert box['y']+box['height'] <= bar['y'], 'sticky action covers task text'
                page.evaluate('scrollTo(0,0)')
                page.screenshot(path=str(output / 'start-1366.png'))
            finally:
                page.close()

        def settings_actions():
            page=page_for()
            try:
                nav(page,'settings');visible_button(page,'#probe')
                page.locator('#api-key').fill('never-display-this-secret')
                page.locator('#probe').click()
                page.wait_for_function("document.querySelector('#connection-badge').textContent==='已连通'")
                assert page.locator('#model').input_value()=='test-model'
                assert page.evaluate('window.__requests.filter(r=>r.url==="/api/probe").length')==1
                page.locator('.connection-card .advanced:not(#phase-settings) summary').click()
                visible_button(page,'#probe')
                page.locator('.connection-card .advanced:not(#phase-settings) summary').click()
                page.locator('#api-key').fill('')
                page.evaluate('scrollTo(0,0)');page.screenshot(path=str(output/'settings-1366.png'))
            finally:page.close()

        def essential_settings_visible():
            page=page_for()
            try:
                nav(page,'settings')
                field=page.locator('#api-key').bounding_box();button=page.locator('#probe').bounding_box()
                assert field['y']+field['height'] <= button['y'], 'essential API Key field below/behind action bar'
                page.locator('#model').focus();page.keyboard.press('Tab')
                assert page.locator('#api-key').evaluate('e=>document.activeElement===e')
                field=page.locator('#api-key').bounding_box();button=page.locator('#probe').bounding_box()
                assert field['y']>=76 and field['y']+field['height'] <= button['y'], 'keyboard focus covered by sticky toolbar'
            finally:page.close()

        def model_strip():
            page=page_for(1024)
            try:
                text=page.locator('#analysis-model-summary')
                assert text.bounding_box()['y'] < page.locator('#paste').bounding_box()['y'], 'model state still after form'
                assert text.bounding_box()['y'] < 400, 'model state not in first screen'
                assert '尚未填写模型 ID' in text.inner_text()
                nav(page,'settings');page.locator('#model').fill('test-model')
                page.locator('#api-key').fill('private-key-do-not-show')
                nav(page,'start')
                assert 'test-model' in text.inner_text()
                assert page.locator('#analysis-model-status').count()==1, 'missing explicit model state'
                assert '未测试' in page.locator('#analysis-model-status').inner_text()
                assert 'private-key-do-not-show' not in page.locator('[data-page=start]').inner_text()
                nav(page,'settings');page.locator('#probe').click()
                page.wait_for_function("document.querySelector('#connection-badge').textContent==='已连通'")
                nav(page,'start');assert '列表接口已连通' in page.locator('#analysis-model-status').inner_text()
                nav(page,'settings');page.locator('#model').fill('another-model')
                nav(page,'start');assert '未测试' in page.locator('#analysis-model-status').inner_text()
                page.screenshot(path=str(output/'start-1024.png'))
            finally:page.close()

        def stale_connection_response():
            page=page_for(defer_probe=True)
            try:
                nav(page,'settings');page.locator('#model').fill('old-model')
                page.locator('#probe').click()
                page.wait_for_function('typeof window.__resolveProbe==="function"')
                page.locator('#base-url').fill('http://127.0.0.1:11434/v1')
                page.locator('#model').fill('new-model')
                page.evaluate('window.__resolveProbe()')
                page.wait_for_function('!document.querySelector("#probe").disabled')
                assert page.locator('#connection-badge').inner_text()=='未测试', 'late old probe marked new connection tested'
                assert '配置已改变' in page.locator('#probe-status').inner_text()
                nav(page,'start');assert '未测试' in page.locator('#analysis-model-status').inner_text()
                assert 'new-model' in page.locator('#analysis-model-summary').inner_text()
            finally:page.close()

        def advanced_encoding():
            page=page_for()
            try:
                assert page.locator('#input-options').count()==1, 'capacity/encoding notes are not collapsible'
                assert not page.locator('#input-options').evaluate('e=>e.open')
                raw='外部诊断：转速降低'.encode('gb18030')
                page.locator('#files').set_input_files({'name':'旧诊断.txt','mimeType':'text/plain','buffer':raw})
                role=page.get_by_label('旧诊断.txt 材料角色',exact=True)
                assert role.is_visible()
                encoding=page.get_by_label('旧诊断.txt 文件编码',exact=True)
                assert not encoding.is_visible(), 'encoding should default to collapsed'
                page.locator('.file-advanced summary').click()
                encoding.select_option('gb18030')
                page.locator('.file-advanced summary').click()
                assert 'GB18030' in page.locator('.file-advanced summary').inner_text()
                nav(page,'settings');page.locator('#model').fill('test-model')
                nav(page,'start');assert encoding.input_value()=='gb18030'
                visible_button(page,'#analyze');page.locator('#analyze').click()
                page.wait_for_function('window.__requests.some(r=>r.url==="/api/jobs"&&r.body)')
                sent=page.evaluate('window.__requests.find(r=>r.url==="/api/jobs"&&r.body).body.files')
                assert len(sent)==1 and sent[0]['encoding']=='gb18030' and sent[0]['filename']=='旧诊断.txt'
                import base64
                assert base64.b64decode(sent[0]['content_base64'])==raw
                assert page.evaluate('window.__requests.filter(r=>r.url==="/api/jobs"&&r.body).length')==1
            finally:page.close()

        def conclusion_first():
            page=page_for()
            try:
                nav(page,'history');page.locator('.history-open').click();page.wait_for_selector('#analysis-content h3')
                headings=page.locator('#analysis-content > section > h3').all_text_contents()
                assert headings[:3]==['多报告综合结论','建议核查','分析摘要'], headings
                assert headings.count('多报告综合结论')==1 and headings.count('建议核查')==1
                assert MODEL['integrated_conclusion'] in page.locator('#analysis-content').inner_text()
                assert MODEL['recommended_review_actions'][0] in page.locator('#analysis-content').inner_text()
                assert '对诊断材料的解读' in headings and '仍缺少的信息' in headings
                page.locator('.comparison-grid .citation-button').first.click()
                assert page.locator('#source-0').evaluate('e=>e.open')
                assert page.locator('#source-list img').count()==0
                page.evaluate('scrollTo(0,0)');page.screenshot(path=str(output/'results-1366.png'))
            finally:page.close()

        def single_report():
            report=copy.deepcopy(REPORT);report['analysis']['knowledge_enrichment']['model_explanation'].pop('integrated_conclusion')
            page=page_for(report=report)
            try:
                nav(page,'history');page.locator('.history-open').click();page.wait_for_selector('#analysis-content h3')
                headings=page.locator('#analysis-content > section > h3').all_text_contents()
                assert headings[:2]==['分析摘要','建议核查'],headings
                assert '多报告综合结论' not in headings
            finally:page.close()

        def readable_type():
            page=page_for()
            try:
                for selector in ['#paste','#task','#paste-role']:
                    font=page.locator(selector).evaluate('e=>parseFloat(getComputedStyle(e).fontSize)')
                    assert font>=14,(selector,font)
                nav(page,'settings')
                for selector in ['#model','#base-url','#api-key','#probe']:
                    assert page.locator(selector).evaluate('e=>parseFloat(getComputedStyle(e).fontSize)')>=14,selector
                nav(page,'history');page.locator('.history-open').click();page.wait_for_selector('#analysis-content h3')
                for selector in ['#analysis-content .body-text','#analysis-content .plain-list']:
                    assert page.locator(selector).first.evaluate('e=>parseFloat(getComputedStyle(e).fontSize)')>=14,selector
            finally:page.close()

        def responsive():
            for width in [390,768,900,1024,1093,1280,1366,1920]:
                page=page_for(width,640 if width<600 else 768)
                try:
                    for route in ROUTES:
                        nav(page,route)
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(width,route,'overflow')
                        if route=='start':visible_button(page,'#analyze')
                        if route=='settings':visible_button(page,'#probe')
                    if width==390:
                        nav(page,'start');page.screenshot(path=str(output/'start-390.png'))
                finally:page.close()

        for name, test in [
            ('Analysis action visible in first screen and after scroll; task field not covered',action_visibility),
            ('Connection test action visible, works once and remains accessible with advanced settings',settings_actions),
            ('Essential connection fields visible above action bar; keyboard focus unobscured',essential_settings_visible),
            ('Model status before input at 1024px; distinguishes model configured from tested; no Key display',model_strip),
            ('Changed connection ignores the previous in-flight probe result',stale_connection_response),
            ('Collapsed guidance/encoding; material role visible; selected encoding and bytes preserved in submission',advanced_encoding),
            ('Integrated conclusion and review actions first; evidence sections retained and citations work',conclusion_first),
            ('Single-report summary and review actions first without invented integrated conclusion',single_report),
            ('Primary form and report text at least 14 CSS pixels',readable_type),
            ('Six routes across eight viewport widths; no horizontal overflow; actions visible',responsive),
        ]:
            scenario(name,test)
        browser.close()
    result={'environment':'Linux Chromium, real production DOM/CSS/JS, scripted API responses',
            'native_windows':False,'real_model':False,'passed':passed,'failures':failed,'button_measurements':measurements}
    (output/'polish-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='button_measurements'},ensure_ascii=False,indent=2))
    return int(bool(failed))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chromium',default='/usr/bin/chromium');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(exercise(args.chromium,args.output))
