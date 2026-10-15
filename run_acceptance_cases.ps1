#Requires -Version 5.1
# Synthetic material-review cases; isolated under project-local storage.
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$CaseScript = Join-Path $ProjectRoot 'examples\platform_acceptance_cases_v1\validate_cases.py'
$OutputDir = Join-Path $ProjectRoot 'runs\acceptance_cases'
if (-not (Test-Path -LiteralPath $CaseScript)) {
  throw ('Acceptance case script not found: {0}' -f $CaseScript)
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
python -X utf8 $CaseScript --project-root $ProjectRoot --output-dir $OutputDir
exit $LASTEXITCODE
