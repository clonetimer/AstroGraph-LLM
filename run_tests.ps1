$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Executable = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $Executable scripts/verify_release.py @args
exit $LASTEXITCODE
