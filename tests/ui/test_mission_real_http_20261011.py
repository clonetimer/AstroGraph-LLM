"""Actual local HTTP and Playwright chromium: 3-file upload automatically enables analysis."""
from __future__ import annotations
import json, os, socket, threading, time
from fastapi.testclient import TestClient
import pytest
from pathlib import Path
pytest.importorskip('playwright.sync_api')
uvicorn=pytest.importorskip('uvicorn')
from playwright.sync_api import sync_playwright
from astrograph.api import create_app


def _freeport():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

@pytest.fixture
def server(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'jobs'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','local')
    app=create_app();port=_freeport()
    cfg=uvicorn.Config(app,host='127.0.0.1',port=port,access_log=False,log_level='error')
    srv=uvicorn.Server(cfg);srv.install_signal_handlers=lambda:None
    thread=threading.Thread(target=srv.run,daemon=True);thread.start()
    limit=time.monotonic()+20
    while not srv.started and time.monotonic()<limit and thread.is_alive():time.sleep(.04)
    if not srv.started:pytest.fail('local server failed')
    yield f'http://127.0.0.1:{port}'
    srv.should_exit=True;thread.join(timeout=8)


def configure_browser_network(page, origin):
    """Optional browser-sandbox bridge: actual backend HTTP via Python, not simulated responses.

    This is ONLY for environments where managed Chromium blocks loopback; default
    tests use the browser's normal networking on a developer machine / Windows.
    """
    if os.environ.get('ASTROGRAPH_TEST_NETWORK_BRIDGE') != '1':
        return
    import httpx
    from urllib.parse import urlsplit
    expected = urlsplit(origin).netloc
    def request_real_server(route):
        req=route.request
        if urlsplit(req.url).netloc != expected:
            return route.abort()
        headers={k:v for k,v in req.headers.items() if k.lower() not in ('host','content-length','connection')}
        try:
            with httpx.Client(trust_env=False,follow_redirects=False,timeout=50) as client:
                res=client.request(req.method,req.url,headers=headers,content=req.post_data_buffer)
            safe={k:v for k,v in res.headers.items() if k.lower() not in ('transfer-encoding','connection','content-length','content-encoding')}
            return route.fulfill(status=res.status_code,headers=safe,body=res.content)
        except Exception as exc:
            print('backend_bridge_failed:',exc)
            return route.abort()
    page.route('**/*',request_real_server)


def launch(p):
    kwargs={'headless':True,'args':['--no-sandbox']}
    executable=os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE')
    if executable:kwargs['executable_path']=executable
    return p.chromium.launch(**kwargs)


def open_workbench(page, server):
    """Use native browser networking on Windows; only bridge in restricted CI.

    The offline bridge still requests the real HTTP backend, and must not be
    reported as an unrestricted native-browser HTTP test.
    """
    if os.environ.get('ASTROGRAPH_TEST_NETWORK_BRIDGE')=='1':
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_mission_offline_bridge_20261011 import install_html_and_api_bridge
        install_html_and_api_bridge(page,server)
    else:
        page.goto(server+'/platform',wait_until='networkidle')
        page.wait_for_url('**/platform/mission*')


def test_default_platform_real_http_auto_preflight_and_abstain(server):
    with sync_playwright() as p:
        b=launch(p);page=b.new_page(viewport={'width':1440,'height':900},locale='zh-CN')
        if os.environ.get('ASTROGRAPH_TEST_NETWORK_BRIDGE')!='1': configure_browser_network(page,server)
        errors=[];page.on('pageerror',lambda err:errors.append(str(err)))
        open_workbench(page,server)
        assert page.locator('#screen-command.active').is_visible()
        page.locator('[data-route="upload"]').click()
        assert not page.locator('#mission-preflight').is_disabled()
        with TestClient(create_app()) as c:
            demo=c.get('/platform/mission/assets/mission_demo_cases.json').json()['cases'][0]
        for role,target in [('diagnosis_results','mission-diagnosis'),('log_reports','mission-logs'),('references','mission-references')]:
            file=demo['files'][role][0]
            page.locator('#'+target).set_input_files({'name':file['name'],'mimeType':'text/plain', 'buffer':file['content'].encode('utf8')})
        page.wait_for_function("!document.querySelector('#mission-start').disabled",timeout=40000)
        assert '通过' in page.locator('#preflight-results').inner_text()
        assert page.locator('[data-upload-step="3"]').get_attribute('class').find('active')>=0
        page.locator('#mission-start').click()
        page.wait_for_selector('#screen-report.active',timeout=90000)
        page.locator('#action-evidence').wait_for(state='visible')
        assert 'ABSTAIN' in page.locator('#action-evidence').inner_text()
        assert page.locator('#action-evidence').get_attribute('data-decision-code').startswith('ABSTAIN')
        assert page.locator('#report-export-json').is_visible()
        assert not errors,errors
        b.close()


def test_ui_role_nav_graph_and_service_panel(server):
    with sync_playwright() as p:
        b=launch(p);page=b.new_page(viewport={'width':390,'height':850},locale='zh-CN')
        if os.environ.get('ASTROGRAPH_TEST_NETWORK_BRIDGE')!='1': configure_browser_network(page,server)
        open_workbench(page,server)
        page.locator('#mobile-nav').click()
        page.locator('[data-route="graph"]').click()
        page.locator('#graph-topics .graph-topic').first.wait_for(timeout=18000)
        assert page.locator('[data-route="knowledge-review"]').is_visible()
        page.locator('#service-status-open').click()
        assert page.locator('#service-details').is_visible()
        page.locator('#service-close').click()
        page.locator('#theme-select').select_option('light');page.locator('#font-select').select_option('large')
        assert page.locator('body').get_attribute('data-theme')=='light'
        assert page.locator('body').get_attribute('data-size')=='large'
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+1')
        b.close()
