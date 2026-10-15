from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from typing import Any

from fastapi import Request


_ROLE_LEVEL = {"viewer": 1, "operator": 2, "admin": 3, "reviewer": 2}
_PUBLIC_PATHS = {"/live", "/ready", "/health", "/docs", "/openapi.json", "/redoc"}
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


@dataclass(frozen=True)
class AuthIdentity:
    subject: str
    roles: frozenset[str]
    mode: str

    def permits(self, required_role: str) -> bool:
        rights = {"viewer": {"viewer"}, "operator": {"viewer", "operator"},
                  "reviewer": {"viewer", "reviewer"},
                  "admin": {"viewer", "operator", "reviewer", "admin"}}
        return any(required_role in rights.get(role, set()) for role in self.roles)


@dataclass(frozen=True)
class ApiAuthConfig:
    mode: str
    tokens: dict[str, AuthIdentity]
    allow_forwarded_local: bool = False

    @classmethod
    def from_environment(cls) -> "ApiAuthConfig":
        mode = os.environ.get("ASTROGRAPH_AUTH_MODE", "local").strip().lower()
        if mode not in {"local", "bearer", "disabled", "accounts"}:
            raise RuntimeError(f"unsupported ASTROGRAPH_AUTH_MODE: {mode}")
        tokens: dict[str, AuthIdentity] = {}
        raw = os.environ.get("ASTROGRAPH_API_TOKENS_JSON", "").strip()
        if raw:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("ASTROGRAPH_API_TOKENS_JSON must be a JSON object")
            for token, value in payload.items():
                if not isinstance(token, str) or not token:
                    raise RuntimeError("API token keys must be nonempty strings")
                if isinstance(value, list):
                    subject = "token-user"
                    roles = value
                elif isinstance(value, dict):
                    subject = str(value.get("subject", "token-user"))
                    roles = value.get("roles", [])
                else:
                    raise RuntimeError("API token entries must be role lists or identity objects")
                normalized = frozenset(str(role).lower() for role in roles)
                if not normalized or any(role not in _ROLE_LEVEL for role in normalized):
                    raise RuntimeError("API token roles must use viewer, operator, or admin")
                tokens[token] = AuthIdentity(subject=subject, roles=normalized, mode="bearer")
        if mode == "bearer" and not tokens:
            raise RuntimeError("bearer auth mode requires ASTROGRAPH_API_TOKENS_JSON")
        deployment_mode = os.environ.get("ASTROGRAPH_DEPLOYMENT_MODE", "development").strip().lower()
        if deployment_mode in {"production", "prod"} and mode in {"local", "disabled"}:
            raise RuntimeError("production deployment requires ASTROGRAPH_AUTH_MODE=bearer")
        allow_forwarded_local = os.environ.get("ASTROGRAPH_ALLOW_FORWARDED_LOCAL_AUTH", "").strip().lower() in {
            "1", "true", "yes"
        }
        return cls(mode=mode, tokens=tokens, allow_forwarded_local=allow_forwarded_local)


def required_role(method: str, path: str) -> str | None:
    if path in _PUBLIC_PATHS or path.startswith("/v1/platform/auth/") or not path.startswith("/v1/"):
        return None
    if path == "/v1/platform/accounts" or path.startswith("/v1/platform/accounts/"):
        return "admin"
    if path.startswith("/v1/platform/material-knowledge/") and path != "/v1/platform/material-knowledge/review-session":
        if path.startswith("/v1/platform/material-knowledge/graph/releases/") and method.upper() != "GET":
            return "admin"
        return "reviewer"
    if path.startswith("/v1/admin/") or path.startswith("/v1/platform/settings/"):
        return "admin"
    if path.startswith("/v1/platform/tool-plugins") or path.startswith("/v1/platform/tool-providers"):
        return "admin"
    if method.upper() == "DELETE" and path.startswith("/v1/cases/"):
        return "admin"
    if method.upper() == "GET":
        return "viewer"
    if path.startswith("/v1/knowledge/") or path == "/v1/experiments":
        return "admin"
    return "operator"


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def authenticate_request(request: Request, config: ApiAuthConfig) -> AuthIdentity | None:
    if config.mode == "accounts":
        return None  # resolved by application account manager
    if config.mode == "disabled":
        return AuthIdentity(subject="unauthenticated", roles=frozenset({"admin"}), mode="disabled")
    if config.mode == "local":
        host = request.client.host if request.client else ""
        forwarded = request.headers.get("Forwarded") or request.headers.get("X-Forwarded-For")
        if forwarded and not config.allow_forwarded_local:
            return None
        if host in _LOCAL_HOSTS:
            return AuthIdentity(subject=f"local:{host}", roles=frozenset({"admin"}), mode="local")
        return None
    presented = _bearer_token(request)
    if presented is None:
        return None
    # Constant-time comparison avoids leaking which configured token matched.
    for configured, identity in config.tokens.items():
        if hmac.compare_digest(presented, configured):
            return identity
    return None


def auth_error_details(config: ApiAuthConfig, required: str) -> dict[str, Any]:
    return {"auth_mode": config.mode, "required_role": required, "forwarded_local_allowed": config.allow_forwarded_local}
