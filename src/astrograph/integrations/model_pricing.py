"""Versioned, append-only model API price book.

Prices are explicit administrator-supplied records. The platform never guesses
provider prices, treats missing usage as zero, or retroactively rewrites receipts.
Only token-based provider charges are estimated; hosting, taxes, discounts and
unmetered requests remain outside this ledger.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any, Iterator


_PRICE_TYPES = {"OFFICIAL_PUBLISHED", "CONTRACT", "ADMIN_ESTIMATE"}
_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_MAX_RATE = Decimal("1000000")
_COST_QUANTUM = Decimal("0.000000000001")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc(value: str) -> str:
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError("INVALID_PRICE_EFFECTIVE_FROM")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("INVALID_PRICE_EFFECTIVE_FROM") from exc
    if parsed.tzinfo is None:
        raise ValueError("PRICE_EFFECTIVE_FROM_REQUIRES_TIMEZONE")
    return parsed.astimezone(timezone.utc).isoformat()


def _rate(value: Any) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("INVALID_MODEL_PRICE_RATE") from exc
    if not number.is_finite() or number < 0 or number > _MAX_RATE:
        raise ValueError("MODEL_PRICE_RATE_OUT_OF_RANGE")
    normalized = format(number.normalize(), "f")
    if len(normalized) > 40:
        raise ValueError("MODEL_PRICE_RATE_OUT_OF_RANGE")
    return normalized


def calculate_cost(pricing: dict[str, Any] | None, usage: dict[str, Any]) -> dict[str, Any]:
    if not pricing or pricing.get("status") != "PRICE_PINNED":
        return {"status": "PRICING_NOT_CONFIGURED", "currency": None, "amount": None}
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    if type(input_tokens) is not int or input_tokens < 0 or type(output_tokens) is not int or output_tokens < 0:
        return {"status": "USAGE_MISSING", "currency": pricing["currency"], "amount": None}
    input_rate = Decimal(pricing["input_per_million_tokens"])
    output_rate = Decimal(pricing["output_per_million_tokens"])
    amount = (
        Decimal(input_tokens) * input_rate + Decimal(output_tokens) * output_rate
    ) / Decimal(1_000_000)
    amount = amount.quantize(_COST_QUANTUM, rounding=ROUND_HALF_EVEN)
    return {
        "status": "CALCULATED",
        "currency": pricing["currency"],
        "amount": format(amount, "f"),
    }


class ModelPriceBook:
    def __init__(self, platform_root: str | Path):
        root = Path(platform_root).expanduser().resolve()
        self.root = root / "execution"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("MODEL_PRICEBOOK_PATH_INVALID")
        self.path = self.root / "model_pricing.sqlite3"
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=20000")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _migrate(self) -> None:
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS model_price_versions (
                    price_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    input_per_million_tokens TEXT NOT NULL,
                    output_per_million_tokens TEXT NOT NULL,
                    effective_from TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(provider, model, effective_from)
                )
            """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_price_resolve
                ON model_price_versions(provider, model, effective_from, revision)
            """)
            db.execute("""
                CREATE TRIGGER IF NOT EXISTS model_price_versions_no_update
                BEFORE UPDATE ON model_price_versions BEGIN
                    SELECT RAISE(ABORT, 'MODEL_PRICEBOOK_APPEND_ONLY');
                END
            """)
            db.execute("""
                CREATE TRIGGER IF NOT EXISTS model_price_versions_no_delete
                BEFORE DELETE ON model_price_versions BEGIN
                    SELECT RAISE(ABORT, 'MODEL_PRICEBOOK_APPEND_ONLY');
                END
            """)

    @staticmethod
    def _validate(
        *, provider: str, model: str, currency: str,
        input_per_million_tokens: Any, output_per_million_tokens: Any,
        effective_from: str, source_type: str, source_ref: str, created_by: str,
    ) -> dict[str, str]:
        provider = str(provider).strip().casefold()
        model = str(model).strip()
        currency = str(currency).strip().upper()
        source_type = str(source_type).strip().upper()
        source_ref = str(source_ref).strip()
        created_by = str(created_by).strip()
        if not _PROVIDER.fullmatch(provider):
            raise ValueError("INVALID_PRICE_PROVIDER")
        if not model or len(model) > 200 or any(ord(ch) < 32 for ch in model):
            raise ValueError("INVALID_PRICE_MODEL")
        if not _CURRENCY.fullmatch(currency):
            raise ValueError("INVALID_PRICE_CURRENCY")
        if source_type not in _PRICE_TYPES:
            raise ValueError("INVALID_PRICE_SOURCE_TYPE")
        if not source_ref or len(source_ref) > 500 or any(ord(ch) < 32 for ch in source_ref):
            raise ValueError("INVALID_PRICE_SOURCE_REFERENCE")
        if not created_by or len(created_by) > 160:
            raise ValueError("INVALID_PRICE_ACTOR")
        return {
            "provider": provider,
            "model": model,
            "currency": currency,
            "input_per_million_tokens": _rate(input_per_million_tokens),
            "output_per_million_tokens": _rate(output_per_million_tokens),
            "effective_from": _utc(effective_from),
            "source_type": source_type,
            "source_ref": source_ref,
            "created_by": created_by,
        }

    def add(
        self, *, provider: str, model: str, currency: str = "USD",
        input_per_million_tokens: Any, output_per_million_tokens: Any,
        effective_from: str, source_type: str, source_ref: str, created_by: str,
    ) -> dict[str, Any]:
        value = self._validate(
            provider=provider, model=model, currency=currency,
            input_per_million_tokens=input_per_million_tokens,
            output_per_million_tokens=output_per_million_tokens,
            effective_from=effective_from, source_type=source_type,
            source_ref=source_ref, created_by=created_by,
        )
        now, price_id = _now(), secrets.token_hex(16)
        with self._connect() as db:
            revision = int(db.execute(
                "SELECT COALESCE(MAX(revision),0)+1 AS n FROM model_price_versions WHERE provider=? AND model=?",
                (value["provider"], value["model"]),
            ).fetchone()["n"])
            try:
                db.execute("""
                    INSERT INTO model_price_versions(
                        price_id,revision,provider,model,currency,input_per_million_tokens,
                        output_per_million_tokens,effective_from,source_type,source_ref,created_by,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    price_id, revision, value["provider"], value["model"], value["currency"],
                    value["input_per_million_tokens"], value["output_per_million_tokens"],
                    value["effective_from"], value["source_type"], value["source_ref"],
                    value["created_by"], now,
                ))
            except sqlite3.IntegrityError as exc:
                raise ValueError("PRICE_VERSION_ALREADY_EXISTS_FOR_EFFECTIVE_TIME") from exc
            row = db.execute("SELECT * FROM model_price_versions WHERE price_id=?", (price_id,)).fetchone()
        return dict(row)

    def list(self, *, provider: str | None = None, model: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            query = "SELECT * FROM model_price_versions"
            filters, values = [], []
            if provider:
                filters.append("provider=?")
                values.append(str(provider).strip().casefold())
            if model:
                filters.append("model=?")
                values.append(str(model).strip())
            if filters:
                query += " WHERE " + " AND ".join(filters)
            query += " ORDER BY provider,model,effective_from DESC,revision DESC LIMIT 1000"
            rows = db.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def resolve(self, *, provider: str, model: str, at: str) -> dict[str, Any] | None:
        provider, model, at = str(provider or "").strip().casefold(), str(model or "").strip(), _utc(at)
        if not provider or not model:
            return None
        with self._connect() as db:
            row = db.execute("""
                SELECT * FROM model_price_versions
                WHERE provider=? AND model=? AND effective_from<=?
                ORDER BY effective_from DESC,revision DESC LIMIT 1
            """, (provider, model, at)).fetchone()
        if row is None:
            return None
        value = dict(row)
        return {
            "status": "PRICE_PINNED",
            "price_id": value["price_id"],
            "revision": value["revision"],
            "provider": value["provider"],
            "model": value["model"],
            "currency": value["currency"],
            "input_per_million_tokens": value["input_per_million_tokens"],
            "output_per_million_tokens": value["output_per_million_tokens"],
            "effective_from": value["effective_from"],
            "source_type": value["source_type"],
            "source_ref": value["source_ref"],
            "created_by": value["created_by"],
            "created_at": value["created_at"],
            "resolved_at": at,
        }
