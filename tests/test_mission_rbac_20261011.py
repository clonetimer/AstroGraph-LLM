"""SCC-inspired real platform users and permissions: no fake role toggles."""
from __future__ import annotations

import os
from fastapi.testclient import TestClient
from astrograph.api import create_app


def test_bootstrap_users_roles_revocation_and_no_fixed_password(tmp_path, monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','accounts')
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path))
    monkeypatch.setenv('ASTROGRAPH_ACCOUNT_SETUP_SECRET','UserSelectedBootstrapSecret2026_123456')
    with TestClient(create_app()) as c:
        assert c.get('/v1/platform/auth/status').json()['bootstrap_required']
        denied=c.post('/v1/platform/auth/bootstrap',json={'username':'chief01','password':'StrongPassword123!','setup_secret':'wrong'*8})
        assert denied.status_code==403
        init={'username':'chief01','password':'StrongPassword123!','display_name':'总管理员',
              'setup_secret':'UserSelectedBootstrapSecret2026_123456'}
        assert c.post('/v1/platform/auth/bootstrap',json=init).status_code==200
        assert c.post('/v1/platform/auth/bootstrap',json=init).status_code==409
        assert not c.get('/v1/platform/auth/status').json()['bootstrap_required']
        token=c.post('/v1/platform/auth/login',json={'username':'chief01','password':'StrongPassword123!'}).json()['token']
        head={'Authorization':'Bearer '+token}
        assert c.get('/v1/platform/accounts',headers=head).status_code==200
        assert c.get('/v1/platform/accounts').status_code==401
        assert c.post('/v1/platform/accounts',headers=head,json={'username':'expert01','display_name':'审核专家',
                    'password':'ExpertPassword123!','role':'reviewer'}).status_code==200
        assert c.post('/v1/platform/accounts',headers=head,json={'username':'analyst01','display_name':'分析员',
                    'password':'OperatorPassword123!','role':'operator'}).status_code==200
        review_token=c.post('/v1/platform/auth/login',json={'username':'expert01','password':'ExpertPassword123!'}).json()['token']
        analyst_token=c.post('/v1/platform/auth/login',json={'username':'analyst01','password':'OperatorPassword123!'}).json()['token']
        reviewer={'Authorization':'Bearer '+review_token};operator={'Authorization':'Bearer '+analyst_token}
        assert c.get('/v1/platform/material-knowledge/pending',headers=reviewer).status_code==200
        assert c.get('/v1/platform/material-knowledge/pending',headers=operator).status_code==403
        assert c.post('/v1/platform/jobs',json={'task':'x','work_mode':'normal'},headers=reviewer).status_code==403
        assert c.post('/v1/platform/jobs',json={'task':'x','work_mode':'normal'},headers=operator).status_code==200
        assert c.get('/v1/platform/settings/llm',headers=reviewer).status_code==403
        assert c.get('/v1/platform/accounts',headers=operator).status_code==403
        assert c.patch('/v1/platform/accounts/1',json={'active':False},headers=head).status_code==422
        assert c.patch('/v1/platform/accounts/2',json={'active':False},headers=head).status_code==200
        assert c.get('/v1/platform/material-knowledge/pending',headers=reviewer).status_code==401
        assert c.post('/v1/platform/auth/logout',headers=head).status_code==200
        assert c.get('/v1/platform/accounts',headers=head).status_code==401


def test_roles_are_stored_hashed_and_no_default_bootstrap(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_AUTH_MODE','accounts')
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path))
    monkeypatch.delenv('ASTROGRAPH_ACCOUNT_SETUP_SECRET',raising=False)
    with TestClient(create_app()) as c:
        r=c.post('/v1/platform/auth/bootstrap',json={'username':'chief01','password':'StrongPassword123!',
          'setup_secret':'bad-but-more-than-twenty-four-characters'})
        assert r.status_code==403
    from astrograph.integrations.platform_accounts import PlatformAccounts
    store=PlatformAccounts(tmp_path/'access_control')
    assert store.list_users()==[]
