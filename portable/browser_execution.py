"""Render the shared production receipt component in Chromium, using fixture data."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent

def script_path():
    paths=[ROOT/'assets/execution_view.js',ROOT.parent/'src/astrograph/resources/execution_view.js']
    return next((p for p in paths if p.is_file()), paths[-1])

def exercise(chromium, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    script=script_path()
    assert script.is_file(), 'Shared execution view is missing'
    passed=[]
    fixture={'task_id':'a'*32,'legacy':False,'unavailable_runs':0,
        'usage':{'http_requests':2,'failed_requests':1,'input_tokens':None,'known_input_tokens':17,
                 'output_tokens':None,'known_output_tokens':4,'input_reported_requests':1,'output_reported_requests':1},
        'runs':[{'run_id':'b'*32,'kind':'analysis','status':'FAILED','started_at':'2026-11-04T01:02:03Z',
                 'ended_at':'2026-11-04T01:02:04Z','model':'fixture-model','reason_code':'MODEL_OUTPUT_INVALID',
                 'materials':[{'filename':'<img src=x onerror=alert(1)>.txt','role':'diagnosis_results',
                   'characters':42,'segments':2,'first_locator':'line:1','last_locator':'line:2','original_sha256':'c'*64}],
                 'events':[{'sequence':1,'code':'INPUT_VALIDATED','at':'now','counts':{}},
                           {'sequence':2,'code':'CHUNK_ANALYSIS','at':'now','counts':{'index':2,'total':4}}],
                 'requests':[{'sequence':1,'status':'RESPONSE_RECEIVED','stage':'MODEL_ANALYSIS','http_status':200,
                      'latency_ms':11,'usage':{'input_tokens':17,'output_tokens':4}},
                     {'sequence':2,'status':'FAILED','stage':'SYNTHESIS','http_status':401,'latency_ms':20,
                      'usage':{'input_tokens':None,'output_tokens':None}}],
                 'artifacts':[{'name':'report.json','sha256':'d'*64,'bytes':20}]}]}
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=chromium,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1366,'height':768})
        page.set_content('<main id="target"></main>')
        page.add_script_tag(content=script.read_text(encoding='utf-8'))
        page.add_style_tag(content=script.with_suffix('.css').read_text(encoding='utf-8'))
        def render(data):
            page.evaluate('d=>AstroGraphExecution.render(document.getElementById("target"),d,{baseUrl:"/api/jobs/"+d.task_id})',data)
        render(fixture)
        assert '未提供' in page.locator('#target').inner_text()
        assert '已知部分 17' in page.locator('#target').inner_text()
        assert page.locator('#target img').count()==0
        assert page.locator('#target a').get_attribute('href')=='/api/jobs/'+'a'*32+'/runs/'+'b'*32+'/report.json'
        assert 'SYNTHESIS' not in page.locator('#target').inner_text()  # translated stage label
        passed.append('nullable usage, per-request status, escaping and version links')
        page.locator('#target details').first.evaluate('(e)=>e.open=true')
        page.screenshot(path=str(output/'execution-receipts.png'),full_page=True)
        render({'task_id':'a'*32,'legacy':True,'runs':[]})
        assert '旧任务' in page.locator('#target').inner_text()
        assert '0 tokens' not in page.locator('#target').inner_text()
        passed.append('legacy task does not invent zero-cost historical usage')
        render(fixture)
        for width in [390,768,1024,1366]:
            page.set_viewport_size({'width':width,'height':768})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'),width
        passed.append('four-width renderer does not overflow')
        # Late response for an old selection must never replace a new task.
        page.evaluate('''() => {window.pending=null;window.active=true;
          AstroGraphExecution.mount(document.getElementById('target'),{
            baseUrl:'/api/jobs/'+'a'.repeat(32),
            api:()=>new Promise(resolve=>window.pending=resolve),current:()=>window.active});}''')
        page.evaluate('''d=>{window.active=false;document.getElementById('target').textContent='NEW TASK';window.pending(d);}''',fixture)
        page.wait_for_timeout(50)
        assert page.locator('#target').inner_text()=='NEW TASK'
        passed.append('stale execution response cannot overwrite another task')
        browser.close()
    result={'environment':'Linux Chromium, production renderer, fixture API','passed':passed,'failed':[]}
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chromium',default='/usr/bin/chromium');parser.add_argument('--output',required=True)
    args=parser.parse_args();exercise(args.chromium,args.output)
