$ErrorActionPreference = "Stop"
if (-not $env:ASTROGRAPH_AUTH_MODE) { $env:ASTROGRAPH_AUTH_MODE = "local" }
if (-not $env:ASTROGRAPH_PLATFORM_ROOT) {
  $env:ASTROGRAPH_PLATFORM_ROOT = Join-Path $HOME ".astrograph\platform"
}
if (-not $env:ASTROGRAPH_HOST) { $env:ASTROGRAPH_HOST = "127.0.0.1" }
if (-not $env:ASTROGRAPH_PORT) { $env:ASTROGRAPH_PORT = "8080" }
python -m uvicorn astrograph.api:app --host $env:ASTROGRAPH_HOST --port $env:ASTROGRAPH_PORT
