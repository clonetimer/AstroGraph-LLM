#Requires -Version 5.1
# Local Windows launcher. Honors explicit environment settings.
# Kept ASCII-only for Windows PowerShell 5.1 without UTF-8 BOM.
$ErrorActionPreference = 'Stop'
if (-not $env:ASTROGRAPH_PLATFORM_ROOT) {
  $env:ASTROGRAPH_PLATFORM_ROOT = Join-Path $PSScriptRoot 'runs\platform'
}
if (-not $env:ASTROGRAPH_PORT) {
  $env:ASTROGRAPH_PORT = '8081'
}
New-Item -ItemType Directory -Force -Path $env:ASTROGRAPH_PLATFORM_ROOT | Out-Null
Write-Host ('AstroGraph task data: {0}' -f $env:ASTROGRAPH_PLATFORM_ROOT)
Write-Host ('Main command center: http://127.0.0.1:{0}/platform' -f $env:ASTROGRAPH_PORT)
Write-Host ('Legacy console: http://127.0.0.1:{0}/platform/legacy' -f $env:ASTROGRAPH_PORT)
& (Join-Path $PSScriptRoot 'run_platform.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
