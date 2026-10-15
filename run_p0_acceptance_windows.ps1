#Requires -Version 5.1
param([switch]$RealBrowserHTTP)
# Run with Windows PowerShell 5.1 or PowerShell 7; no hardcoded Chromium path.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
Write-Host 'AstroGraph P0 regression: accounts, upload, preflight, reports, events.'
python -m pytest -q tests/test_mission_rbac_20261011.py `
  tests/test_mission_console_integration.py `
  tests/test_command_center_overview_20261011.py `
  tests/test_manual_report_review_workflow.py `
  tests/ui/test_mission_offline_bridge_20261011.py `
  tests/ui/test_command_center_e2e_20261011.py
if ($LASTEXITCODE -ne 0) { throw 'P0 regression failed. Preserve the pytest log.' }
if ($RealBrowserHTTP) {
  Write-Host 'Running native Windows Chromium HTTP tests.'
  python -m pytest -q tests/ui/test_mission_real_http_20261011.py
  if ($LASTEXITCODE -ne 0) { throw 'Native Chromium HTTP tests failed.' }
}
