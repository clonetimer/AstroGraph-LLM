"""Small cross-platform filesystem publication contracts."""
from __future__ import annotations

import os
import time
from pathlib import Path


def publish_staged_directory(staging: str | Path, output: str | Path) -> None:
    """Atomically publish a completed directory with bounded Windows retries.

    POSIX errors and non-permission failures are surfaced immediately. Windows
    search indexers and antivirus scanners can transiently retain handles after
    the last staged file is closed, so only ``PermissionError`` is retried.
    The function never falls back to a recursive copy because that would expose
    a partially published transaction.
    """
    source = Path(staging)
    target = Path(output)
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))
