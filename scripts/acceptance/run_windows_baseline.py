#!/usr/bin/env python3
"""Emit a reproducible Windows runtime baseline for AstroGraph acceptance."""
from __future__ import annotations

import json
import platform
import sys


def collect() -> dict:
    return {
        "schema_version": "astrograph-windows-baseline.v1", "platform": platform.system(),
        "python": platform.python_version(), "executable": sys.executable,
        "windows": platform.system() == "Windows", "status": "PASS" if platform.system() == "Windows" else "NOT_WINDOWS",
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
