"""ASCII Windows launch compatibility + real event-center markup contract.

Native PowerShell syntax is additionally checked by verify_powershell_scripts.ps1
on Windows; this test intentionally does not pretend to run PowerShell on Linux.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESOURCE=ROOT/'src'/'astrograph'/'resources'


def test_windows_launch_scripts_ascii_and_setup():
    names=('run_platform_accounts_demo.ps1','run_platform_demo.ps1',
           'run_p0_acceptance_windows.ps1','run_acceptance_cases.ps1',
           'verify_powershell_scripts.ps1')
    for name in names:
        text=(ROOT/name).read_bytes()
        assert text and text.isascii(),name
        assert b'$ErrorActionPreference' in text,name
    accounts=(ROOT/names[0]).read_text('ascii')
    assert "ASTROGRAPH_AUTH_MODE = 'accounts'" in accounts
    assert 'ASTROGRAPH_ACCOUNT_SETUP_SECRET' in accounts
    assert 'Read-Host' in accounts and '-AsSecureString' in accounts
    assert '24 characters' in accounts
    assert 'run_platform_demo.ps1' in accounts
    native=(ROOT/'verify_powershell_scripts.ps1').read_text('ascii')
    assert '[System.Management.Automation.Language.Parser]::ParseFile' in native


def test_event_center_keeps_primary_backend_selectors_and_accessibility():
    html=(RESOURCE/'mission_console.html').read_text('utf8')
    js=(RESOURCE/'mission_console.js').read_text('utf8')
    css=(RESOURCE/'mission_console.css').read_text('utf8')
    for key in ('command-items','command-action-content','command-next-content',
                'workbench-evidence','workbench-actions','workbench-context',
                'metric-done','command-filter','account-form'):
        assert f'id="{key}"' in html,key
    assert '事件收件箱' in html and '事件详情' in html
    assert 'id="evidence-detail"' in html and 'id="evidence-source-meta"' in html
    assert 'id="nav-secondary" open' not in html
    assert '证据链' in js and 'data-evidence-filter' in js
    assert 'aria-pressed' in js and 'data-evidence-type' in js
    assert 'enterEvent(item)' in js and 'renderSelectedEvent(item)' in js
    assert 'prefers-reduced-motion:reduce' in css
