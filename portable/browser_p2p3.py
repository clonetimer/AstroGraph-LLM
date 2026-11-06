"""P2/P3 production DOM/JS tests; scripted API responses, not model quality tests."""
import argparse
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright
from browser_navigation import JOBROW, REPORT, SOURCE
ROOT = Path(__file__).resolve().parent
WID = 'c' * 32
EID = 'd' * 32

def exercise(chromium, output):
    output.mkdir(parents=True, exist_ok=True)
    passed, failed, errors = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, headless=True, args=['--no-sandbox'])
        def page_new():
            page = browser.new_page(viewport={'width':1366,'height':768})
            page.set_default_timeout(4000)
            page.on('pageerror', lambda e: errors.append(str(e)))
            html = (ROOT/'assets/index.html').read_text()
            scripts = re.findall(r'<script[^>]*src="/([^"]+)"[^>]*></script>', html)
            html = re.sub(r'<script[^>]*src="/[^"]+"[^>]*></script>|<link[^>]*href="/[^"]+"[^>]*>', '', html)
            page.set_content(html)
            for name in ('app.css','execution_view.css','workspace_view.css'):
                path = ROOT/'assets'/name
                if not path.exists(): path=ROOT.parent/'src/astrograph/resources'/name
                page.add_style_tag(content=path.read_text())
            page.evaluate('''f=>{
              window.__f=f;window.__requests=[];window.__running=false;window.__revision=1;window.__reviewed=false;
              window.fetch=async(url,options={})=>{
                const body=typeof options.body==='string'?JSON.parse(options.body):null;
                window.__requests.push({url,body,method:options.method||'GET'});
                let data={};const wid='c'.repeat(32),eid='d'.repeat(32),job={...f.job,status:window.__running?'RUNNING':'SUCCEEDED'};
                if(url==='/api/session')data={token:'fixture-token',version:'P2P3-fixture'};
                else if(url==='/api/workspaces')data={workspaces:[{workspace_id:wid,name:'卫星 A',device:'ADCS',event:'E1'}]};
                else if(url.endsWith('/review')){window.__reviewed=!window.__reviewed;window.__revision++;}
                else if(url.startsWith('/api/workspaces/')&&url.endsWith('/entries'))data={entries:[{entry_id:eid,revision:window.__revision,title:'单位 <img src=x>',body:'人工补充：核对单位',state:window.__reviewed?'REVIEWED':'DRAFT'}]};
                else if(url==='/api/jobs')data=body?{job}:{jobs:[job]};
                else if(url.endsWith('/cancel')){window.__running=false;data={job:{...job,status:'CANCEL_REQUESTED'}};}
                else if(url.endsWith('/conversation'))data={job_id:job.job_id,input_sha256:'b'.repeat(64),sources:[f.source],turns:[]};
                else if(url==='/api/probe')data={models:[{id:'test-model'}]};
                else data={job,report:window.__running?null:f.report};
                return {ok:true,json:async()=>data};
              };
            }''',{'job':JOBROW,'report':REPORT,'source':SOURCE})
            for name in scripts:
                path=ROOT/'assets'/name
                if not path.exists(): path=ROOT.parent/'src/astrograph/resources'/name
                page.add_script_tag(content=path.read_text())
            page.wait_for_function("document.querySelector('#service-status').textContent==='本机工作台已就绪'")
            return page
        def nav(page, route):
            page.evaluate('(r)=>AstroGraphShell.navigate(r)',route)
        def run(name, fn):
            page=page_new()
            try: fn(page);passed.append(name)
            except Exception as e: failed.append({'name':name,'error':str(e)})
            finally:page.close()
        def select(page):
            page.locator('#context-picker').evaluate('e=>e.open=true')
            page.locator('.context-workspace').select_option(WID)
            page.wait_for_selector('.context-select-entry')
            assert page.locator('.context-select-entry').is_disabled()
            page.locator('.context-review').click()
            page.wait_for_function("!document.querySelector('.context-select-entry').disabled")
            page.locator('.context-select-entry').check()
            assert page.locator('.context-entry img').count()==0
            assert page.locator('.context-select-entry').bounding_box()['width']<=24, 'context checkbox must not stretch to a full row'
        def context_phases(page):
            select(page);nav(page,'settings');page.locator('#model').fill('main')
            page.locator('#phase-settings').evaluate('e=>e.open=true')
            page.locator('#phase-chunk').fill('small');page.locator('#phase-synthesis').fill('large');page.locator('#phase-follow-up').fill('qa')
            nav(page,'start');page.locator('#paste').fill('诊断结果：轮速异常');page.locator('#queue-analysis').check()
            page.locator('#analyze').click();page.wait_for_function("__requests.some(x=>x.url==='/api/jobs'&&x.body)")
            data=page.evaluate("__requests.find(x=>x.url==='/api/jobs'&&x.body).body")
            assert data['context']=={'workspace_id':WID,'entries':[{'entry_id':EID,'revision':2}]}
            assert data['phases']=={'chunk':{'model':'small'},'synthesis':{'model':'large'},'follow_up':{'model':'qa'}} and data['queue'] is True
        def reset_scope(page):
            select(page);page.locator('.context-workspace').select_option('')
            nav(page,'settings');page.locator('#model').fill('main');nav(page,'start');page.locator('#paste').fill('诊断结果')
            page.locator('#analyze').click();page.wait_for_function("__requests.some(x=>x.url==='/api/jobs'&&x.body)")
            data=page.evaluate("__requests.find(x=>x.url==='/api/jobs'&&x.body).body")
            assert not data.get('context') and not data.get('phases') and not data.get('queue')
        def queue_cancel(page):
            nav(page,'settings');page.locator('#model').fill('main');nav(page,'start');page.locator('#paste').fill('诊断结果')
            page.evaluate('__running=true');page.locator('#analyze').click();page.wait_for_selector('#cancel-job')
            nav(page,'start');assert page.locator('#analyze').is_disabled()
            page.locator('#queue-analysis').check();assert page.locator('#analyze').is_enabled()
            page.locator('#analyze').click();page.wait_for_function("__requests.filter(x=>x.url==='/api/jobs'&&x.body).length===2")
            page.locator('#cancel-job').click();page.wait_for_function("__requests.some(x=>x.url.endsWith('/cancel'))")
            assert len(page.evaluate("__requests.filter(x=>x.url.endsWith('/cancel'))"))==1
        def responsive(page):
            select(page)
            for width in (390,768,1024,1366):
                page.set_viewport_size({'width':width,'height':768})
                for route in ('start','settings','result'):
                    nav(page,route)
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'),(width,route)
            nav(page,'start');page.screenshot(path=str(output/'workspace-1366.png'),full_page=True)
        run('explicit-reviewed-selection-and-stage-models',context_phases)
        run('scope-reset-no-implicit-context',reset_scope)
        run('opt-in-queue-and-explicit-cancel',queue_cancel)
        run('responsive-context-stage-forms',responsive)
        browser.close()
    data={'passed':passed,'failed':failed,'page_errors':errors,'model_responses':'scripted API fixtures'}
    (output/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print(json.dumps(data,ensure_ascii=False,indent=2))
    return bool(failed or errors)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--chromium',default='/usr/bin/chromium');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    raise SystemExit(exercise(a.chromium,a.output))
