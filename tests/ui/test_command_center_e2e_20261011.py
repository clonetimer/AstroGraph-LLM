"""Task-first command center: actual backend HTTP bridged into sandbox Chromium.

The network bridge forwards real requests; no fabricated success responses.
On Windows normal Playwright Chromium uses its own browser transport.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pytest
pytest.importorskip('playwright.sync_api')
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mission_offline_bridge_20261011 import local_server, account_server, install_html_and_api_bridge


def browser(p):
    opts={'headless':True,'args':['--no-sandbox']}
    if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE'):
        opts['executable_path']=os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']
    return p.chromium.launch(**opts)


def test_task_center_demo_preflight_workbench_and_report(local_server):
    with sync_playwright() as p:
        b=browser(p)
        page=b.new_page(viewport={'width':1440,'height':900},locale='zh-CN')
        errors=[];page.on('pageerror', lambda err:errors.append(str(err)))
        install_html_and_api_bridge(page,local_server)
        page.locator('#screen-command.active').wait_for(timeout=10000)
        # First-use command center shows one guided workflow, not empty KPIs and an empty table.
        page.locator('#command-first-run:visible').wait_for(timeout=15000)
        assert not page.locator('#command-items').is_visible()
        assert not page.locator('#command-metrics').is_visible()
        assert page.locator('#command-first-run .first-run-steps li').count()==4
        assert not page.locator('#command-create').is_visible()
        assert page.locator('#command-first-create').is_visible()
        page.locator('#command-first-create').click()
        page.locator('#screen-upload.active').wait_for()
        page.locator('#demo-cards [data-demo-inline]').first.wait_for(timeout=15000)
        page.locator('#demo-cards [data-demo-inline]').first.click()
        assert page.locator('#mission-diagnosis').get_attribute('type')=='file'
        page.wait_for_function("document.querySelector('#mission-start').disabled === false",timeout=45000)
        assert '通过' in page.locator('#preflight-results').inner_text()
        page.locator('#mission-start').click()
        page.locator('#screen-report.active').wait_for(timeout=90000)
        assert page.locator('#action-approval').inner_text().startswith('未授权')
        assert 'ABSTAIN' in page.locator('#action-evidence').inner_text()
        page.locator('[data-route="workbench"]').click()
        assert page.locator('#workbench-timeline .case-timeline li').count()==4
        assert page.locator('#workbench-evidence').is_visible()
        assert page.locator('#workbench-actions').is_visible()
        assert '未评定' in page.locator('#workbench-context').inner_text()
        assert '未指派' in page.locator('#workbench-context').inner_text()
        page.locator('#workbench-evidence-open').click()
        assert page.locator('#screen-report.active').is_visible()
        assert page.locator('[data-report-tab="evidence"]').get_attribute('class').find('active')!=-1
        page.locator('[data-route="command"]').click()
        page.locator('#command-items [data-command-job]').first.wait_for(timeout=12000)
        page.locator('#command-filter').select_option('all')
        assert page.locator('#metric-done').inner_text()=='1'
        page.locator('#command-items [data-command-job]').first.click()
        # Navigation occurs immediately, independently of the report fetch completing.
        assert page.locator('#screen-workbench.active').is_visible()
        page.wait_for_function("document.querySelector('#workbench-evidence-open')!==null",timeout=15000)
        assert page.locator('#workbench-job').inner_text()==page.locator('#report-job-label').inner_text()
        assert not errors,errors
        b.close()


def test_mobile_service_action_access_and_zoom():
    root=Path(__file__).resolve().parents[2]/'src'/'astrograph'/'resources'
    with sync_playwright() as p:
        b=browser(p)
        page=b.new_page(viewport={'width':390,'height':844},locale='zh-CN')
        page.set_content((root/'mission_console.html').read_text(encoding='utf8'))
        page.add_style_tag(content=(root/'mission_console.css').read_text(encoding='utf8'))
        page.locator('body').evaluate("node=>node.dataset.size='large'")
        for w in (390,312,260):
            page.set_viewport_size({'width':w,'height':844})
            btn=page.locator('#service-status-open')
            assert btn.is_visible()
            rect=btn.bounding_box()
            assert rect and rect['x'] >= -1 and rect['x']+rect['width']<=w+2, (w,rect)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+2'),w
        b.close()


def test_admin_access_visible_via_identity_and_create_three_users(account_server):
    with sync_playwright() as p:
        b=browser(p)
        page=b.new_page(viewport={'width':1300,'height':850},locale='zh-CN')
        errors=[];page.on('pageerror',lambda err:errors.append(str(err)))
        install_html_and_api_bridge(page,account_server)
        page.locator('#account-modal[open]').wait_for(timeout=15000)
        page.locator('#account-name').fill('chief01')
        page.locator('#account-display').fill('测试管理员')
        page.locator('#account-pass').fill('StrongPassword123!')
        page.locator('#account-setup-key').fill('2026_AstroGraph_DeployChosenSecret_39chars')
        page.locator('#account-submit').click()
        page.wait_for_function("document.querySelector('#identity-label').textContent==='平台管理员'",timeout=20000)
        # Successful login must refresh the command center, replacing the blank KPI placeholders.
        page.locator('#command-first-run:visible').wait_for(timeout=15000)
        assert not page.locator('#command-metrics').is_visible()
        page.locator('#identity-trigger').click()
        page.locator('#identity-open-users').click()
        page.locator('#screen-users.active').wait_for(timeout=10000)
        for username,label,role in [
            ('analyst01','分析员','operator'),
            ('expert01','领域专家','reviewer'),
            ('guest01','只读用户','viewer'),
        ]:
            page.locator('#new-user-name').fill(username)
            page.locator('#new-user-display').fill(label)
            page.locator('#new-user-role').select_option(role)
            page.locator('#new-user-password').fill('SomeLongPassword2026!')
            page.locator('#users-create-form button[type=submit]').click()
            page.wait_for_function("name=>document.querySelector('#users-list').textContent.includes(name)",arg=username,timeout=15000)
        assert page.locator('#users-list [data-user-row]').count()==4
        assert not errors,errors
        b.close()


def test_native_file_input_real_browser_upload_and_automatic_preflight(local_server):
    """P0: native file pickers upload bytes; auto-preflight enables actual analysis."""
    import json
    case=json.loads((Path(__file__).resolve().parents[2]/'src'/'astrograph'/'resources'/'mission_demo_cases.json').read_text(encoding='utf8'))['cases'][0]
    with sync_playwright() as p:
        b=browser(p)
        page=b.new_page(viewport={'width':1280,'height':800},locale='zh-CN')
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        install_html_and_api_bridge(page,local_server)
        page.locator('#screen-command.active').wait_for(timeout=10000)
        page.locator('[data-route="upload"]').click()
        for role,id in [('diagnosis_results','mission-diagnosis'),('log_reports','mission-logs'),('references','mission-references')]:
            source=case['files'][role][0]
            target=page.locator('#'+id)
            assert target.get_attribute('type')=='file'
            target.set_input_files({'name':source['name'],'mimeType':'text/plain','buffer':source['content'].encode('utf8')})
            assert page.locator('#'+{'diagnosis_results':'diagnosis-summary','log_reports':'logs-summary','references':'references-summary'}[role]).inner_text().find(source['name'])>=0
        page.wait_for_function("document.querySelector('#mission-start').disabled===false",timeout=45000)
        assert '通过' in page.locator('#preflight-results').inner_text()
        assert not errors,errors
        b.close()


def test_account_role_navigation_and_previous_account_data_cleared(account_server):
    with sync_playwright() as p:
        b=browser(p)
        page=b.new_page(viewport={'width':1240,'height':870},locale='zh-CN')
        install_html_and_api_bridge(page,account_server)
        page.locator('#account-modal[open]').wait_for(timeout=15000)
        page.locator('#account-name').fill('chief01')
        page.locator('#account-display').fill('管理员')
        page.locator('#account-pass').fill('StrongPassword123!')
        page.locator('#account-setup-key').fill('2026_AstroGraph_DeployChosenSecret_39chars')
        page.locator('#account-submit').click()
        page.wait_for_function("document.querySelector('#identity-label').textContent==='平台管理员'",timeout=18000)
        page.locator('[data-route="users"]').click()
        for username,role in [('analyst01','operator'),('expert01','reviewer'),('guest01','viewer')]:
            page.locator('#new-user-name').fill(username)
            page.locator('#new-user-display').fill(username)
            page.locator('#new-user-role').select_option(role)
            page.locator('#new-user-password').fill('SomeLongPassword2026!')
            page.locator('#users-create-form button[type=submit]').click()
            page.wait_for_function("name=>document.querySelector('#users-list').textContent.includes(name)",arg=username,timeout=13000)
        for username,label,expected in [('analyst01','分析人员',{'upload':True,'review':False,'users':False}),('expert01','领域审核专家',{'upload':False,'review':True,'users':False}),('guest01','只读用户',{'upload':False,'review':False,'users':False})]:
            page.locator('#identity-trigger').click()
            page.locator('#identity-logout').click()
            page.locator('#account-modal[open]').wait_for(timeout=10000)
            assert not page.locator('#users-list').inner_text().strip()
            page.locator('#account-name').fill(username)
            page.locator('#account-pass').fill('SomeLongPassword2026!')
            page.locator('#account-submit').click()
            page.wait_for_function("name=>document.querySelector('#identity-label').textContent===name",arg=label,timeout=15000)
            for target,visible in expected.items():
                assert page.locator(f'[data-route="{target}"]').is_visible()==visible,(username,target)
            assert page.locator('[data-route="command"]').is_visible()
        b.close()


def test_queued_event_also_opens_its_workbench_without_stale_report(local_server):
    """The event row is never a dead end, even before a report is available."""
    import httpx
    with httpx.Client(base_url=local_server, trust_env=False) as client:
        created=client.post('/v1/platform/jobs',json={'task':'待检查的新任务','work_mode':'normal'})
        assert created.status_code==200,created.text
        job_id=created.json()['job_id']
    with sync_playwright() as p:
        b=browser(p)
        page=b.new_page(viewport={'width':1440,'height':900},locale='zh-CN')
        install_html_and_api_bridge(page,local_server)
        row=page.locator(f'#command-items [data-command-job="{job_id}"]')
        row.wait_for(timeout=15000)
        row.click()
        assert page.locator('#screen-workbench.active').is_visible()
        assert page.locator('#workbench-job').inner_text()==job_id
        context=page.locator('#workbench-context').inner_text()
        assert '未评定' in context and '未指派' in context
        assert page.locator('#workbench-back').is_visible()
        page.locator('#workbench-back').click()
        assert page.locator('#screen-command.active').is_visible()
        b.close()
