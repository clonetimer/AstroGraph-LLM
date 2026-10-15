"""Compatibility perimeter for historical adapters.

Importing a non-HTTP adapter must not require the optional FastAPI extra. The
router is loaded only when explicitly requested; HTTP tests still require api.
"""
from __future__ import annotations

__all__ = ["build_compat_router"]


def __getattr__(name: str):
    if name == "build_compat_router":
        from .api import build_compat_router
        return build_compat_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
