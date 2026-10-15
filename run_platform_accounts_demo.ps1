#Requires -Version 5.1
# Windows PowerShell 5.1 compatible account-mode launcher.
# ASCII-only by design: Windows PowerShell may misdecode UTF-8 without BOM.
# Local evaluation only; put the deployed server behind HTTPS.
$ErrorActionPreference = 'Stop'

if (-not $env:ASTROGRAPH_PLATFORM_ROOT) {
  $env:ASTROGRAPH_PLATFORM_ROOT = Join-Path $PSScriptRoot 'runs\platform'
}
if (-not $env:ASTROGRAPH_PORT) {
  $env:ASTROGRAPH_PORT = '8081'
}
$env:ASTROGRAPH_AUTH_MODE = 'accounts'
New-Item -ItemType Directory -Force -Path $env:ASTROGRAPH_PLATFORM_ROOT | Out-Null
$accountDatabase = Join-Path $env:ASTROGRAPH_PLATFORM_ROOT 'access_control\accounts.sqlite3'

if (-not (Test-Path -LiteralPath $accountDatabase)) {
  if (-not $env:ASTROGRAPH_ACCOUNT_SETUP_SECRET) {
    Write-Host 'First account-mode run: configure a unique setup secret (24+ characters).'
    $secure = Read-Host 'Administrator setup secret (hidden)' -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
      $env:ASTROGRAPH_ACCOUNT_SETUP_SECRET = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
      [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
  }
  if (-not $env:ASTROGRAPH_ACCOUNT_SETUP_SECRET -or $env:ASTROGRAPH_ACCOUNT_SETUP_SECRET.Length -lt 24) {
    throw 'Administrator setup secret must contain at least 24 characters.'
  }
}

Write-Host ('AstroGraph account mode: http://127.0.0.1:{0}/platform' -f $env:ASTROGRAPH_PORT)
Write-Host ('Task data directory: {0}' -f $env:ASTROGRAPH_PLATFORM_ROOT)
& (Join-Path $PSScriptRoot 'run_platform_demo.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
