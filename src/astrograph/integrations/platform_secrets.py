from __future__ import annotations

import os
from typing import Any


class PlatformSecretStore:
    """OS-keyring backed secret storage with fail-closed persistence.

    Secrets are never written to SQLite. If keyring is unavailable, non-secret
    profile fields may still be saved but credentials must be entered again.
    """

    service_name = "AstroGraph"

    @staticmethod
    def available() -> bool:
        try:
            import keyring  # noqa: F401
            return True
        except Exception:
            return False

    def set(self, ref: str, value: str) -> bool:
        if not value:
            return False
        try:
            import keyring
            keyring.set_password(self.service_name, ref, value)
            return True
        except Exception:
            return False

    def get(self, ref: str | None) -> str | None:
        if not ref:
            return None
        try:
            import keyring
            value = keyring.get_password(self.service_name, ref)
            return value or None
        except Exception:
            return None

    def delete(self, ref: str | None) -> None:
        if not ref:
            return
        try:
            import keyring
            keyring.delete_password(self.service_name, ref)
        except Exception:
            pass

    @staticmethod
    def make_ref(kind: str, identifier: str) -> str:
        return f"astrograph:{kind}:{identifier}"
