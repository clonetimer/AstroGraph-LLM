#Requires -Version 5.1
# Native Windows parser check. ASCII-only to work on Windows PowerShell 5.1.
$ErrorActionPreference = 'Stop'
$targets = @('run_platform_accounts_demo.ps1','run_platform_demo.ps1',
             'run_p0_acceptance_windows.ps1','run_acceptance_cases.ps1',
             'run_platform.ps1')
$errorsFound = 0
foreach ($name in $targets) {
  $file = Join-Path $PSScriptRoot $name
  $tokens = $null
  $parseErrors = $null
  [System.Management.Automation.Language.Parser]::ParseFile($file,[ref]$tokens,[ref]$parseErrors) | Out-Null
  if ($parseErrors.Count -gt 0) {
    $errorsFound += $parseErrors.Count
    foreach ($item in $parseErrors) { Write-Error ('{0}: {1}' -f $name,$item.Message) }
  } else { Write-Host ('PASS {0}' -f $name) }
}
if ($errorsFound -gt 0) { throw ('PowerShell parser failures: {0}' -f $errorsFound) }
