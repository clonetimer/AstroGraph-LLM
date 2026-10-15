"""Real-material report + event-oriented UI smoke test via true HTTP backend bridge."""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pytest
pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_mission_offline_bridge_20261011 import local_server,install_html_and_api_bridge


def test_event_inbox_evidence_filters_and_report(local_server):
    with sync_playwright() as p:
        options={'headless':True,'args':['--no-sandbox']}
        if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE'):
            options['executable_path']=os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']
        browser=p.chromium.launch(**options)
        page=browser.new_page(viewport={'width':1520,'height':950},locale='zh-CN')
        errors=[]
        page.on('pageerror',lambda e: errors.append(str(e)))
        install_html_and_api_bridge(page,local_server)
        shots=os.environ.get('ASTROGRAPH_UI_SCREENSHOT_DIR')
        if shots:
            Path(shots).mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(Path(shots)/'01_首次研判引导.png'),full_page=True)
        page.locator('#command-first-create').click()
        page.locator('#demo-cards [data-demo-inline]').first.wait_for(timeout=15000)
        page.locator('#demo-cards [data-demo-inline]').first.click()
        page.wait_for_function("document.querySelector('#mission-start').disabled===false",timeout=45000)
        page.locator('#mission-start').click()
        page.locator('#screen-report.active').wait_for(timeout=90000)
        page.locator('[data-route="command"]').click()
        page.locator('#command-items [data-command-job]').first.wait_for(timeout=10000)
        assert page.locator('.command-preview').is_visible()
        assert page.locator('#nav-secondary').get_attribute('open') is None
        # Event inbox no longer repeats evidence and next action in every event row.
        assert page.locator('#command-items .event-meta').count()==0
        if shots:
            page.screenshot(path=str(Path(shots)/'02_事件中心.png'),full_page=True)
        assert page.locator('#command-next-content').is_visible()
        assert page.locator('.command-table-head').count()==0
        page.locator('#command-items [data-command-job]').first.click()
        page.locator('#screen-workbench.active').wait_for(timeout=10000)
        page.locator('#workbench-evidence-list .evidence-tile').first.wait_for(timeout=15000)
        if shots:
            page.screenshot(path=str(Path(shots)/'03_证据研判工作台.png'),full_page=True)
        assert page.locator('#workbench-evidence-list .source-locator').first.is_visible()
        assert page.locator('#workbench-evidence-list [data-evidence-index]').count() >= 3
        assert page.locator('#workbench-evidence-list .evidence-spot').count()>0
        assert not any(v.startswith('T08') for v in page.locator('#workbench-evidence-list .evidence-spot').all_inner_texts())
        page.locator('[data-evidence-index]').first.click()
        assert page.locator('#evidence-detail[open]').is_visible()
        assert page.locator('#evidence-source-meta').inner_text().find('原文定位') != -1
        assert page.locator('#evidence-source-meta').inner_text().find('SHA-256') != -1
        assert page.locator('#evidence-source-quote').inner_text().strip()
        page.locator('#evidence-detail-close').click()
        assert not page.locator('#evidence-detail').is_visible()
        filters=page.locator('[data-evidence-filter]')
        assert filters.count()==4
        page.locator('[data-evidence-filter="log"]').click()
        assert page.locator('[data-evidence-filter="log"]').get_attribute('aria-pressed')=='true'
        assert page.locator('#workbench-evidence-list [data-evidence-type="log"]:visible').count()>=1
        if shots:
            page.screenshot(path=str(Path(shots)/'04_日志筛选.png'),full_page=True)
        assert not page.locator('#workbench-evidence-list [data-evidence-type="diagnosis"]').first.is_visible()
        page.locator('[data-evidence-filter="all"]').click()
        assert page.locator('#workbench-evidence-list .evidence-tile:visible').count()>=3
        page.locator('#workbench-report-open').click()
        assert page.locator('#screen-report.active').is_visible()
        assert 'ABSTAIN' in page.locator('#action-evidence').inner_text()
        page.locator('[data-route="command"]').click()
        page.set_viewport_size({'width':390,'height':844})
        page.locator('body').evaluate("node=>node.dataset.size='large'")
        page.locator('#command-items [data-command-job]').first.wait_for(timeout=12000)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+2'), '390px large-font overflow'
        if shots:
            page.screenshot(path=str(Path(shots)/'05_窄屏事件中心.png'),full_page=True)
        assert not errors,errors
        browser.close()
