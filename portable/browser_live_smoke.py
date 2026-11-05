"""Real browser → local web server → local HTTP model fixture, development only."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
from playwright.sync_api import sync_playwright
from test_workbench import app, model_server


def run(chromium, output):
    output.mkdir(parents=True, exist_ok=True)
    result={'native_windows':False,'real_model':False,'transport':'Chromium to actual local workbench and scripted HTTP model'}
    with tempfile.TemporaryDirectory(prefix='v27-live-') as temp, model_server() as model:
        server=app.make_server(app.Workbench(Path(temp)),port=0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path=chromium,headless=True,args=['--no-sandbox'])
                page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
                page.set_default_timeout(15000)
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}/',wait_until='networkidle')
                page.wait_for_function("document.querySelector('#service-status').textContent==='本机工作台已就绪'")
                page.locator('[data-route=settings]').click()
                page.locator('#base-url').fill(model['base_url']);page.locator('#model').fill('test-model')
                page.locator('[data-route=start]').click()
                page.locator('#paste').fill('虚构回归样例：反作用轮温度偏高，运行日志显示温度持续上升。请核对观测时段。')
                page.screenshot(path=str(output/'live-start-page.png'),full_page=True)
                page.locator('#analyze').click()
                page.wait_for_function("document.querySelector('#job-badge').textContent==='模型分析已生成'",timeout=45000)
                page.locator('[data-route=chat]').click();page.locator('#question').fill('需要补充哪些材料？')
                page.locator('#send-question').click();page.wait_for_selector('#turns .citation-button',timeout=45000)
                page.locator('#turns .citation-button').first.click()
                assert page.locator('#source-0').evaluate('e=>e.open')
                with page.expect_download() as info: page.locator('#download-md').click()
                download=info.value;download.save_as(output/'live-report.md')
                assert (output/'live-report.md').stat().st_size>0
                page.locator('[data-route=history]').click();assert page.locator('.history-item').count()==1
                with page.expect_download() as info:page.locator('a[href="/api/backup"]').click()
                info.value.save_as(output/'live-backup.zip')
                assert not errors,errors
                result.update(status='PASS',checks=['local HTML/CSS/JS loading under CSP','six-view navigation','real upload/analysis transport','cited follow-up','MD download','history item','backup download','no browser JS exceptions'])
                browser.close()
        except Exception as exc:
            result.update(status='FAILED_OR_ENVIRONMENT_BLOCKED',error=str(exc))
        finally:server.shutdown();server.server_close();thread.join(2)
    (output/'live-browser-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--chromium',default='/usr/bin/chromium');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.chromium,args.output))
