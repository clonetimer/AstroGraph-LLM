"""Short-lived, process-local, CSRF-protected sessions for the material knowledge UI.

The bootstrap review credential is submitted to the server *only* at login, then
replaced with an opaque HttpOnly cookie. It must never be embedded in JS, a URL,
or localStorage. For multi-worker deployment use an external identity/session
provider; this local-store implementation deliberately does not claim that.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

COOKIE_NAME = 'astrograph_material_review'
TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class ReviewerSession:
    actor: str
    csrf: str
    expires: float
    credential_hash: str


class MaterialReviewerSessions:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, ReviewerSession] = {}
        self._failures: dict[str, list[float]] = {}

    @staticmethod
    def config() -> tuple[str, str, tuple[str, ...]]:
        secret = os.environ.get('ASTROGRAPH_MATERIAL_REVIEW_TOKEN', '')
        reviewer = os.environ.get('ASTROGRAPH_MATERIAL_REVIEWER', '').strip()
        reviewers = tuple(x.strip() for x in os.environ.get('ASTROGRAPH_MATERIAL_REVIEWERS', '').split(',') if x.strip())
        if len(secret) < 24 or not reviewer or reviewer not in reviewers:
            raise HTTPException(status_code=503, detail='Expert review is disabled until administrator credentials are configured')
        return secret, reviewer, reviewers

    @staticmethod
    def secure_transport(request: Request) -> bool:
        # TLS termination at an explicitly trusted ingress is an operator decision;
        # never trust client-supplied X-Forwarded-Proto unconditionally.
        return (request.url.scheme == 'https' or
                os.environ.get('ASTROGRAPH_MATERIAL_REVIEW_TRUSTED_HTTPS_PROXY') == '1')

    @staticmethod
    def permit_login_transport(request: Request) -> bool:
        return (MaterialReviewerSessions.secure_transport(request) or
                request.url.hostname in {'localhost', '127.0.0.1', '::1'})

    def login(self, request: Request, supplied: str) -> tuple[str, ReviewerSession]:
        secret, actor, _ = self.config()
        if not self.permit_login_transport(request):
            raise HTTPException(status_code=403, detail='Remote expert login requires HTTPS')
        ip = str(request.client.host if request.client else 'unknown')
        now = time.monotonic()
        with self._lock:
            recent = [when for when in self._failures.get(ip, []) if now - when < 300]
            if len(recent) >= 5:
                raise HTTPException(status_code=429, detail='Too many failed login attempts')
            if not hmac.compare_digest(supplied, secret):
                self._failures[ip] = (recent + [now])[-5:]
                raise HTTPException(status_code=403, detail='Invalid expert credential')
            self._failures.pop(ip, None)
            self._sessions = {key: value for key, value in self._sessions.items() if value.expires > now}
            if len(self._sessions) >= 200:
                raise HTTPException(status_code=503, detail='Expert session capacity reached')
            token = secrets.token_urlsafe(40)
            session = ReviewerSession(actor=actor, csrf=secrets.token_urlsafe(32),
                expires=now + TTL_SECONDS, credential_hash=hashlib.sha256(secret.encode()).hexdigest())
            self._sessions[token] = session
            return token, session

    def authenticate(self, request: Request, *, require_csrf: bool = False) -> ReviewerSession | None:
        token = request.cookies.get(COOKIE_NAME, '')
        with self._lock:
            entry = self._sessions.get(token) if token else None
            if entry is None:
                return None
            secret, actor, _ = self.config()
            if (entry.expires <= time.monotonic() or actor != entry.actor or
                    entry.credential_hash != hashlib.sha256(secret.encode()).hexdigest()):
                self._sessions.pop(token, None)
                return None
        if require_csrf and not hmac.compare_digest(request.headers.get('x-material-csrf', ''), entry.csrf):
            raise HTTPException(status_code=403, detail='Expert session CSRF validation failed')
        return entry

    def logout(self, request: Request):
        token = request.cookies.get(COOKIE_NAME, '')
        with self._lock:
            self._sessions.pop(token, None)
