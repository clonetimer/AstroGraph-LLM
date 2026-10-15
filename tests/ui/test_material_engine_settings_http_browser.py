"""Real HTTP + Chromium scenario for the knowledge-engine settings page."""
from __future__ import annotations
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def live_platform(tmp_path):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    env={**os.environ,'ASTROGRAPH_PLATFORM_ROOT':str(tmp_path/'platform'),
         'ASTROGRAPH_AUTH_MODE':'local','PYTHONPATH':str(ROOT/'src')}
    cmd=[sys.executable,'-m','uvicorn','astrograph.api:create_app','--factory',
         '--host','127.0.0.1','--port',str(port),'--log-level','error']
    server=subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    url=f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            if server.poll() is not None:
                pytest.fail('API server exited unexpectedly before becoming ready')
            try:
                with urlopen(url+'/live',timeout=0.3) as response:
                    if response.status==200:break
            except Exception:
                time.sleep(0.08)
        else:
            pytest.fail('API server failed to become ready')
        yield url
    finally:
        server.terminate()
        try:server.wait(timeout=5)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)


@pytest.mark.skipif(os.environ.get('ASTROGRAPH_BROWSER_HTTP_E2E') != '1',
    reason='Actual Chromium HTTP navigation is blocked in this CI; opt in on deployment host')
def test_engine_settings_navigation_save_probe_and_rollback(live_platform):
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,**({'executable_path': os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']} if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE') else {}),args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1280,'height':850})
        page.goto(live_platform+'/platform#engine',wait_until='domcontentloaded')
        page.locator('#view-engine.active').wait_for(timeout=15000)
        page.wait_for_function("document.querySelector('#engine-revision-label').textContent.includes('默认配置')",timeout=15000)
        page.locator('#engine-rag-docs').fill('12')
        page.locator('#engine-ocr').select_option('tesseract')
        page.locator('#engine-community-llm').check()
        page.locator('#engine-save').click()
        page.wait_for_function("document.querySelector('#engine-revision-label').textContent.startsWith('revision-')")
        first=page.locator('#engine-revision-label').inner_text()
        assert page.evaluate("document.querySelector('#engine-qdrant-key').value") == ''
        page.locator('#engine-probe').click()
        page.locator('#engine-probe-result').get_by_text('tesseract').wait_for(timeout=10000)
        page.locator('#engine-rag-docs').fill('6')
        page.locator('#engine-save').click()
        page.wait_for_function("document.querySelector('#engine-rag-docs').value === '6' && document.querySelectorAll('#engine-revision-select option').length >= 3")
        page.locator('#engine-revision-select').select_option(first)
        page.once('dialog', lambda dialog: dialog.accept())
        page.locator('#engine-rollback').click()
        page.wait_for_function("document.querySelector('#engine-rag-docs').value === '12'")
        assert page.locator('#engine-ocr').input_value()=='tesseract'
        assert page.locator('#engine-community-llm').is_checked()
        browser.close()
