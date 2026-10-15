"""Persistent platform accounts inspired by SCC governance patterns.

Opt-in ``ASTROGRAPH_AUTH_MODE=accounts``; bootstrap requires a strong password
explicitly supplied at deployment. No builtin credentials. Sessions are hashed,
revocable and expired; no plaintext password or bearer credential in SQLite.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROLES = {'admin', 'operator', 'reviewer', 'viewer'}
ROLE_LABELS = {'admin': '平台管理员', 'operator': '分析人员', 'reviewer': '领域审核专家', 'viewer': '只读用户'}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(value: str) -> str:
    if len(value) < 12 or len(value.encode('utf-8')) > 1024:
        raise ValueError('密码至少12位且不可超过1024字节')
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac('sha256', value.encode('utf-8'), salt, 310_000)
    return 'pbkdf2_sha256$310000$' + salt.hex() + '$' + digest.hex()


def verify_password(value: str, encoded: str) -> bool:
    try:
        algo, rounds, salt, digest = encoded.split('$')
        if algo != 'pbkdf2_sha256' or not 100_000 <= int(rounds) <= 600_000:
            return False
        actual = hashlib.pbkdf2_hmac('sha256', value.encode('utf-8'), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(actual, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


class PlatformAccounts:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'accounts.sqlite3'
        self._lock = threading.RLock()
        with self._connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
                  display_name TEXT NOT NULL, role TEXT NOT NULL, password_hash TEXT NOT NULL,
                  active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
                  expires_at TEXT NOT NULL, revoked_at TEXT, FOREIGN KEY(user_id) REFERENCES users(id));
              CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, actor TEXT, operation TEXT NOT NULL,
                  target TEXT, created_at TEXT NOT NULL);
            ''')

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=15000')
        return db

    @staticmethod
    def _name(value: str) -> str:
        if not re.fullmatch(r'[a-zA-Z0-9_.-]{3,60}', value or ''):
            raise ValueError('账号须为3至60位字母、数字、点、下划线或横线')
        return value

    @staticmethod
    def _safe(row) -> dict[str, Any]:
        return {'id':row['id'], 'username':row['username'], 'display_name':row['display_name'],
                'role':row['role'], 'role_label':ROLE_LABELS[row['role']], 'active':bool(row['active'])}

    def initialized(self) -> bool:
        with self._connect() as db:
            return bool(db.execute('SELECT COUNT(*) FROM users').fetchone()[0])

    def bootstrap(self, username: str, password: str, display_name: str, *, init_secret: str) -> dict:
        required = os.environ.get('ASTROGRAPH_ACCOUNT_SETUP_SECRET', '')
        if len(required) < 24 or not hmac.compare_digest(required, init_secret):
            raise PermissionError('需要服务端配置的初始化密钥（至少24位）')
        digest = hash_password(password)
        uname = self._name(username)
        with self._lock, self._connect() as db:
            try:
                db.execute('BEGIN IMMEDIATE')
                if db.execute('SELECT COUNT(*) FROM users').fetchone()[0]:
                    raise ValueError('账户系统已初始化')
                db.execute('INSERT INTO users(username,display_name,role,password_hash,created_at) VALUES(?,?,?,?,?)',
                           (uname, display_name.strip()[:80] or uname, 'admin', digest, now()))
                db.execute('INSERT INTO audit(actor,operation,target,created_at) VALUES(?,?,?,?)',
                           (uname, 'account.bootstrap', uname, now()))
                db.commit()
            except Exception:
                db.rollback()
                raise
        return self.list_users()[0]

    def authenticate(self, username: str, password: str) -> tuple[str, dict]:
        with self._lock, self._connect() as db:
            row = db.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
            if row is None or not bool(row['active']) or not verify_password(password, row['password_hash']):
                # indistinguishable auth failures; no password in audit
                raise PermissionError('账号或密码无效')
            token = secrets.token_urlsafe(40)
            expiration = (datetime.now(timezone.utc)+timedelta(hours=8)).isoformat()
            db.execute('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)',
                       (hashlib.sha256(token.encode()).hexdigest(), row['id'], expiration))
            db.execute('INSERT INTO audit(actor,operation,target,created_at) VALUES(?,?,?,?)',
                       (username,'auth.login',username,now()))
        return token, self._safe(row)

    def get_session(self, token: str) -> dict | None:
        if not token or len(token) > 256:
            return None
        with self._connect() as db:
            row = db.execute('''SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
                WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at>? AND u.active=1''',
                (hashlib.sha256(token.encode()).hexdigest(), now())).fetchone()
            return self._safe(row) if row else None

    def logout(self, token: str) -> None:
        with self._lock, self._connect() as db:
            db.execute('UPDATE sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL',
                       (now(), hashlib.sha256(token.encode()).hexdigest()))

    def list_users(self) -> list[dict]:
        with self._connect() as db:
            return [self._safe(x) for x in db.execute('SELECT * FROM users ORDER BY id')]

    def add_user(self, *, actor: str, username: str, password: str, display_name: str, role: str) -> dict:
        if role not in ROLES:
            raise ValueError('不支持的角色')
        uname=self._name(username)
        digest=hash_password(password)
        try:
            with self._lock, self._connect() as db:
                cur=db.execute('INSERT INTO users(username,display_name,role,password_hash,created_at) VALUES(?,?,?,?,?)',
                               (uname,display_name.strip()[:80] or uname,role,digest,now()))
                db.execute('INSERT INTO audit(actor,operation,target,created_at) VALUES(?,?,?,?)',
                           (actor,'user.create',uname,now()))
                row=db.execute('SELECT * FROM users WHERE id=?', (cur.lastrowid,)).fetchone()
                return self._safe(row)
        except sqlite3.IntegrityError as exc:
            raise ValueError('账号已存在') from exc

    def modify_user(self, user_id: int, *, actor: str, role: str | None = None,
                    active: bool | None = None, password: str | None = None) -> dict:
        if role is not None and role not in ROLES:
            raise ValueError('不支持的角色')
        digest = hash_password(password) if password else None
        with self._lock, self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                row = db.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
                if row is None: raise ValueError('账号不存在')
                next_role = role or row['role']; next_active = int(active) if active is not None else row['active']
                if row['username'] == actor and (not next_active or next_role != 'admin'):
                    raise ValueError('不能撤销自身管理员权限')
                if row['role'] == 'admin' and (not next_active or next_role != 'admin'):
                    count=db.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]
                    if count <= 1:raise ValueError('至少保留一名有效管理员')
                db.execute('UPDATE users SET role=?, active=?,password_hash=? WHERE id=?',
                           (next_role,next_active,digest or row['password_hash'],user_id))
                db.execute('UPDATE sessions SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL',(now(),user_id))
                db.execute('INSERT INTO audit(actor,operation,target,created_at) VALUES(?,?,?,?)',
                           (actor,'user.update',str(user_id),now()))
                db.commit()
                new=db.execute('SELECT * FROM users WHERE id=?',(user_id,)).fetchone()
                return self._safe(new)
            except Exception:
                db.rollback();raise
