"""Offline Chromium DOM regression. Uses real production JS; API is a test fixture.

Run with a development Python that has Playwright, plus installed Chromium:
python portable/browser_smoke.py --chromium /usr/bin/chromium --output artifacts/ui
This is not a live-server or Windows end-to-end acceptance test.
"""
from pathlib import Path
import argparse
import json
import re
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
JOB = 'a'*32
SID = 'source:test-original'
SOURCE = {'source_id': SID, 'filename': '诊断报告.pdf', 'locator': 'pdf#page:1:block:1',
          'text': '原始观测：温度偏高。<img src=x onerror=alert(1)>', 'sha256': 'a'*64,
          'role': 'diagnosis_results'}


def exercise(chromium: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, headless=True, args=['--no-sandbox'])
        def page_for(fixture):
            page = browser.new_page(viewport={'width': 1440, 'height': 1100})
            document = (ROOT/'assets/index.html').read_text(encoding='utf-8')
            scripts = re.findall(r'<script[^>]*src="/([^"]+)"[^>]*></script>', document)
            html = re.sub(r'<script[^>]*src="/[^"]+"[^>]*></script>|<link[^>]*href="/app.css"[^>]*>', '', document)
            page.set_content(html)
            page.add_style_tag(content=(ROOT/'assets/app.css').read_text(encoding='utf-8'))
            shared_css=ROOT/'assets/execution_view.css'
            if not shared_css.is_file():shared_css=ROOT.parent/'src/astrograph/resources/execution_view.css'
            page.add_style_tag(content=shared_css.read_text(encoding='utf-8'))
            page.evaluate('''fixture => {
              window.__requests=[]; window.__fixture=fixture;
              window.fetch=async (url, options={}) => {
                const body=options.body ? JSON.parse(options.body) : null;
                window.__requests.push({url,body});
                let data;
                if (url==='/api/session') data={token:'offline-test-token'};
                else if(url==='/api/jobs' && body) data={job:fixture.job};
                else if(url==='/api/jobs') data={jobs:fixture.history ? [fixture.job] : []};
                else if(url.endsWith('/conversation')) data={job_id:fixture.job.job_id, input_sha256:'b'.repeat(64), turns:[],sources:fixture.sources||[]};
                else data={job:fixture.job, report:fixture.report||null};
                return {ok:true,json:async()=>data};
              };
            }''', fixture)
            for script in scripts:
                page.add_script_tag(content=((ROOT/'assets'/script) if (ROOT/'assets'/script).is_file() else ROOT.parent/'src/astrograph/resources'/script).read_text(encoding='utf-8'))
            page.wait_for_function("document.querySelector('#service-status').textContent==='本机工作台已就绪'")
            return page

        failures=[]
        try:
            page=page_for({'job':{'job_id':JOB,'status':'FAILED','task':'测试'}})
            page.locator('#files').set_input_files([
                {'name':'诊断报告.pdf','mimeType':'application/pdf','buffer':b'%PDF-fixture'},
                {'name':'参考资料.docx','mimeType':'application/vnd.openxmlformats-officedocument.wordprocessingml.document','buffer':b'PK-fixture'}])
            assert page.locator('#file-list > li').count()==2, 'PDF/DOCX rejected by real frontend validation'
            role_select=page.locator('#file-list > li').nth(1).locator('select').first
            assert role_select.locator('option[value="references"]').count()==1, 'DOCX cannot be reference material'
            role_select.select_option('references')
            page.locator('[data-route=settings]').click();page.locator('#model').fill('test-model')
            page.locator('[data-route=start]').click();page.locator('#analyze').click()
            page.wait_for_function("window.__requests.some(r=>r.url==='/api/jobs' && r.body)")
            sent=page.evaluate("window.__requests.find(r=>r.url==='/api/jobs' && r.body).body.files")
            assert [s['filename'] for s in sent]==['诊断报告.pdf','参考资料.docx']
            assert sent[1]['role']=='references'
            cases.append('PDF/DOCX file input, reference role and serialized upload: PASS')
            page.close()
        except Exception as e:
            failures.append(str(e))

        try:
            model={'status':'AVAILABLE_REQUIRES_EXPERT_REVIEW','citation_reference_validation':'PASS',
                   'input_sha256':'b'*64,'sources':[SOURCE],'summary':'界面回归示例（非真实模型结论）',
                   'findings':[{'text':'材料中记录了温度变化。','citation_ids':[SID]}],
                   'hypotheses':[], 'information_gaps':['需补充观测时段'], 'recommended_review_actions':['核对时间戳'],
                   'common_findings':[{'text':'两份材料均记录温度偏高。','citation_ids':[SID]}],
                   'report_disagreements':[{'text':'采样时段不同，暂不合并事件。','citation_ids':[SID]}],
                   'complementary_evidence':[], 'event_alignment':[], 'integrated_conclusion':'需先核对事件与时间。'}
            page=page_for({'history':True,'job':{'job_id':JOB,'status':'SUCCEEDED','task':'对照诊断报告','model':'test-model'},
                          'sources':[SOURCE],'report':{'analysis':{'knowledge_enrichment':{'generation':{'status':'SUCCEEDED'},'model_explanation':model}}}})
            page.locator('[data-route=history]').click();page.locator('#history').select_option(JOB)
            page.wait_for_selector('#analysis-content h3')
            headings=page.locator('#analysis-content h3').all_text_contents()
            assert headings.count('跨报告共识')==1 and headings.count('报告间分歧')==1, 'comparison sections rendered twice'
            page.locator('.comparison-grid .citation-button').first.click()
            assert page.locator('#source-0').evaluate('(e)=>e.open'), 'citation did not open original source'
            assert page.locator('#source-list img').count()==0, 'original source interpreted as HTML'
            page.screenshot(path=str(output/'report-comparison-demo.png'),full_page=True)
            cases.append('Single comparison workspace, citation navigation and text escaping: PASS')
            page.close()
        except Exception as e:
            failures.append(str(e))

        try:
            page=page_for({'history':True,'job':{'job_id':JOB,'status':'SUCCEEDED','task':'失败重试','resumable':True},
                          'sources':[SOURCE],'report':{'analysis':{'knowledge_enrichment':{'generation':{'status':'FAILED','reason_code':'MODEL_GENERATION_FAILED'},'model_explanation':{'sources':[SOURCE]}}}}})
            page.locator('[data-route=history]').click();page.locator('#history').select_option(JOB)
            page.wait_for_selector('#resume-job',state='visible')
            assert '重试' in page.locator('#resume-job').inner_text()
            page.locator('[data-route=settings]').click();page.locator('#model').fill('test-model')
            page.locator('[data-route=result]').click();page.locator('#resume-job').click()
            page.wait_for_function("window.__requests.some(r=>r.url.endsWith('/resume'))")
            cases.append('Failed-model report offers real resume action: PASS')
            page.close()
        except Exception as e:
            failures.append(str(e))
        browser.close()
    result={'environment':'Linux system Chromium; production DOM/JS; fixture API responses',
            'native_windows':False,'live_server_navigation':False,'passed':cases,'failures':failures}
    (output/'browser-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if failures else 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--chromium',required=True);parser.add_argument('--output',type=Path,default=Path('artifacts/ui'))
    args=parser.parse_args();raise SystemExit(exercise(args.chromium,args.output))
