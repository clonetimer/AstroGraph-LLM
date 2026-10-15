#!/usr/bin/env bash
set -euo pipefail
export ASTROGRAPH_AUTH_MODE="${ASTROGRAPH_AUTH_MODE:-local}"
export ASTROGRAPH_PLATFORM_ROOT="${ASTROGRAPH_PLATFORM_ROOT:-$HOME/.astrograph/platform}"
exec python -m uvicorn astrograph.api:app --host "${ASTROGRAPH_HOST:-127.0.0.1}" --port "${ASTROGRAPH_PORT:-8080}"
