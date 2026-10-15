"""Offline Chromium UI + real backend HTTP bridge for managed browser sandboxes.

In normal environments prefer test_mission_real_http_20261011.py, whose network
requests originate in Chromium. The bridge below calls the actual uvicorn HTTP
server but feeds responses back through Playwright to avoid sandbox loopback bans.
"""
from __future__ import annotations
import base64
import os
import re
import socket
import threading
import time
from pathlib import Path

import pytest
from astrograph.api import create_app
from playwright.sync_api import sync_playwright
import httpx
import uvicorn

ROOT=Path(__file__).resolve().parents[2]
UI=ROOT/'src'/'astrograph'/'resources'

@pytest.fixture
def local_server(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'jobs'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','local')
    with socket.socket() as sk:sk.bind(('127.0.0.1',0));port=sk.getsockname()[1]
    s=uvicorn.Server(uvicorn.Config(create_app(),host='127.0.0.1',port=port,access_log=False,log_level='error'))
    s.install_signal_handlers=lambda:None
    t=threading.Thread(target=s.run,daemon=True);t.start()
    expiry=time.monotonic()+15
    while not s.started and time.monotonic()<expiry:time.sleep(.04)
    assert s.started
    yield f'http://127.0.0.1:{port}'
    s.should_exit=True;t.join(timeout=7)



@pytest.fixture
def account_server(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'accounts'))
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','accounts')
    monkeypatch.setenv('ASTROGRAPH_ACCOUNT_SETUP_SECRET','2026_AstroGraph_DeployChosenSecret_39chars')
    with socket.socket() as sk:sk.bind(('127.0.0.1',0));port=sk.getsockname()[1]
    s=uvicorn.Server(uvicorn.Config(create_app(),host='127.0.0.1',port=port,access_log=False,log_level='error'))
    s.install_signal_handlers=lambda:None
    t=threading.Thread(target=s.run,daemon=True);t.start()
    expiry=time.monotonic()+15
    while not s.started and time.monotonic()<expiry:time.sleep(.04)
    assert s.started
    yield f'http://127.0.0.1:{port}'
    s.should_exit=True;t.join(timeout=7)


def install_html_and_api_bridge(page, origin):
    def relay(data):
        headers=data.get('headers') or {}
        content=base64.b64decode(data.get('body') or '')
        with httpx.Client(trust_env=False,timeout=90) as client:
            res=client.request(data['method'],origin+data['url'],headers=headers,content=content)
        return {'status':res.status_code,'headers':{'content-type':res.headers.get('content-type','text/plain')},'body':base64.b64encode(res.content).decode()}
    page.expose_function('astrographTestRequest',relay)
    fetch_override='''window.fetch=async function(url, options={}) {
       let bytes=new Uint8Array(0);let ct=null;
       if (options.body instanceof Blob) {bytes=new Uint8Array(await options.body.arrayBuffer());ct=options.body.type||'application/octet-stream';}
       else if(typeof options.body==='string') {bytes=new TextEncoder().encode(options.body);}
       else if(options.body!=null) {throw Error('unsupported bridge body');}
       const headers=Object.fromEntries(new Headers(options.headers||{}).entries());
       if(ct&&!headers['content-type'])headers['content-type']=ct;
       let binary=''; for(let i=0;i<bytes.length;i++)binary+=String.fromCharCode(bytes[i]);
       const reply=await astrographTestRequest({url:String(url),method:options.method||'GET',headers,body:btoa(binary)});
       const data=Uint8Array.from(atob(reply.body),c=>c.charCodeAt(0));
       return new Response(data,{status:reply.status,headers:reply.headers});
    };'''
    markup=(UI/'mission_console.html').read_text(encoding='utf8')
    markup=re.sub(r'<link rel="stylesheet"[^>]*>', '',markup)
    markup=re.sub(r'<script type="module" src="[^"]+"></script>', '',markup)
    page.set_content(markup,wait_until='domcontentloaded')
    page.add_style_tag(content=(UI/'mission_console.css').read_text(encoding='utf8'))
    page.evaluate(fetch_override+'\nvoid 0;')
    page.add_script_tag(content=(UI/'mission_console.js').read_text(encoding='utf8'))


def test_real_http_preflight_via_browser_isolated_transport(local_server):
    """End-to-end front-end events backed by actual HTTP API; no mocked returns."""
    with sync_playwright() as p:
        kwargs={'headless':True,'args':['--no-sandbox']}
        if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE'):kwargs['executable_path']=os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']
        b=p.chromium.launch(**kwargs)
        page=b.new_page(viewport={'width':1366,'height':860},locale='zh-CN')
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        install_html_and_api_bridge(page,local_server)
        page.wait_for_function("document.querySelector('#identity-label').textContent.includes('本机管理员')",timeout=10000)
        if os.environ.get('ASTROGRAPH_UI_SCREENSHOTS'):
            dest=Path(os.environ['ASTROGRAPH_UI_SCREENSHOTS']);dest.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(dest/'01_智能助理.png'))
        page.locator('[data-route="upload"]').click()
        assert not page.locator('#mission-preflight').is_disabled()
        # Load actual demo payload from server; selecting via UI uses the real File payloads.
        page.locator('#open-demo').click()
        page.locator('[data-demo-case]').first.wait_for(timeout=8000)
        page.locator('[data-demo-case]').first.click()
        page.wait_for_function("document.querySelector('#mission-start').disabled === false",timeout=50000)
        assert '通过' in page.locator('#preflight-results').inner_text()
        if os.environ.get('ASTROGRAPH_UI_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['ASTROGRAPH_UI_SCREENSHOTS'])/'02_三步上传.png'))
        page.locator('#mission-start').click()
        page.locator('#screen-report.active').wait_for(timeout=90000)
        assert page.locator('#action-evidence').get_attribute('data-decision-code').startswith('ABSTAIN')
        if os.environ.get('ASTROGRAPH_UI_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['ASTROGRAPH_UI_SCREENSHOTS'])/'03_报告行动摘要.png'))
        page.locator('[data-route="graph"]').click()
        page.locator('#graph-topics .graph-topic').first.wait_for(timeout=25000)
        page.locator('#graph-detail [data-focus-edge]').first.click()
        assert page.locator('#graph-detail .graph-relation.selected').count()==1
        if os.environ.get('ASTROGRAPH_UI_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['ASTROGRAPH_UI_SCREENSHOTS'])/'04_图谱高亮.png'))
        page.locator('#service-status-open').click()
        assert page.locator('#service-details').is_visible()
        page.locator('#service-close').click()
        assert not errors, errors
        b.close()


def test_real_account_ui_login_admin_users_reviewer_rights(account_server):
    """Browser controls must map to backend accounts; role-switching is not cosmetic."""
    with sync_playwright() as p:
        kwargs={'headless':True,'args':['--no-sandbox']}
        if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE'):kwargs['executable_path']=os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']
        b=p.chromium.launch(**kwargs)
        page=b.new_page(viewport={'width':1440,'height':900},locale='zh-CN')
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        install_html_and_api_bridge(page,account_server)
        page.locator('#account-modal[open]').wait_for(timeout=15000)
        assert page.locator('#account-modal-title').inner_text()=='初始化系统管理员'
        page.locator('#account-name').fill('chief01')
        page.locator('#account-display').fill('管理员')
        page.locator('#account-pass').fill('StrongPassword123!')
        page.locator('#account-setup-key').fill('2026_AstroGraph_DeployChosenSecret_39chars')
        page.locator('#account-submit').click()
        page.wait_for_function("document.querySelector('#identity-label').textContent==='平台管理员'",timeout=20000)
        page.locator('[data-route="users"]').click()
        page.locator('#new-user-name').fill('domain01')
        page.locator('#new-user-display').fill('领域专家')
        page.locator('#new-user-role').select_option('reviewer')
        page.locator('#new-user-password').fill('ExpertPassword123!')
        page.locator('#users-create-form button[type=submit]').click()
        page.wait_for_function("document.querySelector('#users-list').textContent.includes('domain01')",timeout=13000)
        page.locator('#identity-trigger').click()
        page.locator('#identity-logout').click()
        page.locator('#account-modal[open]').wait_for(timeout=10000)
        page.locator('#account-name').fill('domain01')
        page.locator('#account-pass').fill('ExpertPassword123!')
        page.locator('#account-submit').click()
        try:
            page.wait_for_function("document.querySelector('#identity-label').textContent==='领域审核专家'",timeout=9000)
        except Exception:
            print('DEBUG_ACCOUNT_UI',page.locator('#account-message').inner_text(),page.locator('#identity-label').inner_text(),'modal',page.locator('#account-modal').get_attribute('open'),'errors',errors)
            raise
        assert page.locator('[data-route="review"]').is_visible()
        assert page.locator('[data-route="knowledge-review"]').is_visible()
        assert not page.locator('[data-route="users"]').is_visible()
        assert not page.locator('[data-route="settings"]').is_visible()
        assert not page.locator('[data-route="upload"]').is_visible()
        page.locator('[data-route="knowledge-review"]').click()
        assert page.locator('#screen-knowledge-review.active').is_visible()
        assert not errors, errors
        b.close()


def test_reviewer_can_record_real_report_decision_and_nonreviewer_cannot(account_server):
    """Full account mode: real task -> review queue -> expert decision, not cosmetic UI."""
    with httpx.Client(base_url=account_server, trust_env=False, timeout=60) as client:
        init=client.post('/v1/platform/auth/bootstrap',json={
            'username':'admin01','password':'Administrator2026!','display_name':'平台管理员',
            'setup_secret':'2026_AstroGraph_DeployChosenSecret_39chars'})
        assert init.status_code==200,init.text
        admin_token=client.post('/v1/platform/auth/login',json={
            'username':'admin01','password':'Administrator2026!'}).json()['token']
        h={'Authorization':'Bearer '+admin_token}
        assert client.post('/v1/platform/accounts',headers=h,json={
            'username':'reviewer01','password':'ReviewerPassword2026!','display_name':'专家',
            'role':'reviewer'}).status_code==200
        assert client.post('/v1/platform/accounts',headers=h,json={
            'username':'operator01','password':'OperatorPassword2026!','display_name':'分析员',
            'role':'operator'}).status_code==200
        cases=client.get('/platform/mission/assets/mission_demo_cases.json').json()['cases']
        case=next(c for c in cases if c['id']=='TC01')
        create=client.post('/v1/platform/jobs',headers=h,json={'task':'复核演示 TC01','work_mode':'normal'})
        assert create.status_code==200,create.text
        job_id=create.json()['job_id']
        manifest={'schema_version':'astrograph-result-log-review.v1','equipment_scope':case['equipment_scope']}
        for role, files in case['files'].items():
            manifest[role]=[f['name'] for f in files]
            for f in files:
                r=client.put(f"/v1/platform/jobs/{job_id}/files/{f['name']}",headers=h,
                             content=f['content'].encode('utf8'))
                assert r.status_code==200,r.text
        assert client.put(f'/v1/platform/jobs/{job_id}/files/analysis_input.json',headers=h,
                          content=__import__('json').dumps(manifest).encode()).status_code==200
        check=client.post(f'/v1/platform/jobs/{job_id}/preflight',headers=h)
        assert check.status_code==200 and check.json()['can_start'],check.text
        result=client.post(f'/v1/platform/jobs/{job_id}/run',headers=h)
        assert result.status_code==200 and result.json()['status']=='SUCCEEDED',result.text
        operator_token=client.post('/v1/platform/auth/login',json={
            'username':'operator01','password':'OperatorPassword2026!'}).json()['token']
        assert client.get('/v1/platform/material-knowledge/report-reviews',headers={
            'Authorization':'Bearer '+operator_token}).status_code==403

    with sync_playwright() as p:
        kwargs={'headless':True,'args':['--no-sandbox']}
        if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE'):
            kwargs['executable_path']=os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']
        browser=p.chromium.launch(**kwargs)
        page=browser.new_page(viewport={'width':1330,'height':950},locale='zh-CN')
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        install_html_and_api_bridge(page,account_server)
        page.locator('#account-modal[open]').wait_for(timeout=12000)
        page.locator('#account-name').fill('reviewer01')
        page.locator('#account-pass').fill('ReviewerPassword2026!')
        page.locator('#account-submit').click()
        page.wait_for_function("document.querySelector('#identity-label').textContent==='领域审核专家'",timeout=15000)
        page.locator('[data-route="review"]').click()
        page.locator(f'[data-review-job="{job_id}"]').wait_for(timeout=20000)
        page.locator(f'[data-review-job="{job_id}"]').click()
        candidate=page.locator('[data-review-fault]').first
        candidate.wait_for(timeout=15000)
        candidate.locator('select[name=decision]').select_option('EVIDENCE_INSUFFICIENT')
        candidate.locator('input[name=evidence_ref]').fill('资料缺失待核验')
        candidate.locator('textarea[name=note]').fill('同型号的独立日志与正式规程尚未确认，暂不能下根因结论。')
        candidate.locator('button[type=submit]').click()
        page.wait_for_function("document.querySelector('#toast').textContent.includes('已记录复核意见')",timeout=15000)
        assert not errors,errors
        browser.close()

    with httpx.Client(base_url=account_server, trust_env=False, timeout=30) as client:
        reviewer_token=client.post('/v1/platform/auth/login',json={
            'username':'reviewer01','password':'ReviewerPassword2026!'}).json()['token']
        detail=client.get(f'/v1/platform/material-knowledge/report-reviews/{job_id}',headers={
            'Authorization':'Bearer '+reviewer_token})
        assert detail.status_code==200,detail.text
        assert detail.json()['reviewed_count']>=1


def test_chinese_file_picker_keyboard_and_mobile_zoom_layout():
    """No network needed to verify actual bundled markup/CSS and native file input."""
    with sync_playwright() as p:
        kwargs={'headless':True,'args':['--no-sandbox']}
        if os.environ.get('ASTROGRAPH_CHROMIUM_EXECUTABLE'):
            kwargs['executable_path']=os.environ['ASTROGRAPH_CHROMIUM_EXECUTABLE']
        browser=p.chromium.launch(**kwargs)
        try:
            page=browser.new_page(viewport={'width':390,'height':844},locale='zh-CN')
            page.set_content((UI/'mission_console.html').read_text(encoding='utf8'))
            page.add_style_tag(content=(UI/'mission_console.css').read_text(encoding='utf8'))
            page.evaluate("document.querySelector('#screen-assistant').classList.remove('active');document.querySelector('#screen-upload').classList.add('active')")
            assert page.locator('.file-picker').count()==3
            assert page.locator('.file-picker').first.inner_text().strip()=='＋ 选择文件'
            file_input=page.locator('#mission-diagnosis')
            file_input.focus()
            assert page.evaluate('document.activeElement.id')=='mission-diagnosis'
            file_input.set_input_files({'name':'诊断结果.json','mimeType':'application/json',
                                        'buffer':'{"fault_id":"ADCS_RW_FRICTION_INCREASE"}'.encode('utf8')})
            assert page.evaluate("document.getElementById('mission-diagnosis').files.length") == 1
            for size in (390,312,260): # 260 CSS px approximates 390 physical px / 150% zoom
                page.set_viewport_size({'width':size,'height':844})
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+2'), f'overflow at {size}px'
        finally:
            browser.close()


def test_upload_revision_guard_covers_files_task_and_stale_preflight():
    code=(UI/'mission_console.js').read_text(encoding='utf8')
    assert 'uploadRevision:0' in code
    assert 'S.uploadRevision!==revision' in code
    assert 'S.uploadRevision++;S.files[role]' in code
    assert "['mission-task','mission-scope']" in code
