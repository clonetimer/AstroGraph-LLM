from __future__ import annotations

from pathlib import Path
from typing import Any


def build_legacy_whole_spacecraft_evidence_builder(protocol_root: str | Path) -> Any:
    """Late-load the pre-convergence simulation-specific evidence builder.

    The import intentionally occurs only after an explicit legacy builder selection so
    Stable configured evidence paths do not statically or eagerly depend on it.
    """
    from astrograph.simulation.runtime_adapter import WholeSpacecraftRuntimeEvidenceBuilder

    return WholeSpacecraftRuntimeEvidenceBuilder(protocol_root)


__all__ = ["build_legacy_whole_spacecraft_evidence_builder"]
