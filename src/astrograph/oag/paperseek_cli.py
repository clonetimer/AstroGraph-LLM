"""Managed CLI adapter for a locally supplied PaperSeek source tree.

This adapter exists so the upload platform can use the user-supplied PaperSeek
project without requiring a separately started HTTP service.  It does not bundle
credentials, silently install dependencies, or fabricate results when PaperSeek
is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.integrations.process_control import run_managed_subprocess
from astrograph.oag.contracts import LiteratureDocument


_MAX_OUTPUT_BYTES = 4_000_000


class PaperSeekCLIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    source_root: str | None = None
    data_source: str = "openalex"
    timeout_s: float = Field(default=180.0, gt=0, le=3600)
    top_k: int = Field(default=5, ge=1, le=50)
    max_iterations: int = Field(default=2, ge=1, le=10)
    max_output_bytes: int = Field(default=_MAX_OUTPUT_BYTES, ge=1024, le=32_000_000)

    @field_validator("source_root")
    @classmethod
    def validate_source_root(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("PaperSeek source_root must be an existing directory")
        return str(root)


class PaperSeekCLIClient:
    """Run PaperSeek in a managed subprocess and normalize its JSON response."""

    def __init__(self, config: PaperSeekCLIConfig):
        self.config = config

    def _source_module_available(self) -> bool:
        if not self.config.source_root:
            return False
        root = Path(self.config.source_root)
        return (root / "paperseek" / "cli.py").is_file()

    def available(self) -> bool:
        # An explicitly configured source tree is a reproducibility pin.  Do not
        # silently shadow it with an unrelated globally installed PaperSeek.
        if self.config.source_root is not None:
            return self._source_module_available()
        return shutil.which("paperseek") is not None

    def backend_mode(self) -> str:
        if self.config.source_root is not None:
            return "SOURCE_TREE_MODULE" if self._source_module_available() else "UNAVAILABLE"
        return "INSTALLED_CLI" if shutil.which("paperseek") is not None else "UNAVAILABLE"


    def probe(self) -> dict[str, Any]:
        """Check CLI import/configuration without performing a literature search."""
        env = os.environ.copy()
        installed = shutil.which("paperseek")
        if self.config.source_root is not None:
            if not self._source_module_available():
                return {"status": "UNAVAILABLE", "backend_mode": "UNAVAILABLE", "checks": []}
            root = Path(str(self.config.source_root))
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(root) + (os.pathsep + existing if existing else "")
            command = [sys.executable, "-m", "paperseek.cli", "doctor", "--source", self.config.data_source, "--json"]
            mode = "SOURCE_TREE_MODULE"
        elif installed:
            command = [installed, "doctor", "--source", self.config.data_source, "--json"]
            mode = "INSTALLED_CLI"
        else:
            return {"status": "UNAVAILABLE", "backend_mode": "UNAVAILABLE", "checks": []}
        completed = run_managed_subprocess(
            command, capture_output=True, text=True, timeout=min(self.config.timeout_s, 30.0),
            check=False, env=env,
        )
        stdout = completed.stdout or ""
        try:
            payload = json.loads(stdout) if stdout.strip() else {}
        except json.JSONDecodeError:
            payload = {}
        return {
            "status": "READY" if completed.returncode == 0 else "CONFIGURATION_INCOMPLETE",
            "backend_mode": mode,
            "exit_code": completed.returncode,
            "checks": payload.get("checks", []) if isinstance(payload, dict) else [],
            "stderr_tail": (completed.stderr or "")[-4000:],
            "claim_boundary": "Doctor checks configuration/imports only; it does not prove live retrieval quality.",
        }

    def _command_and_env(self, query: str) -> tuple[list[str], dict[str, str]]:
        arguments = [
            "search",
            query,
            "--source",
            self.config.data_source,
            "--output",
            "json",
            "--min",
            "0",
            "--max",
            str(self.config.top_k),
            "--iterations",
            str(self.config.max_iterations),
            "--fetch-abstracts",
        ]
        env = os.environ.copy()
        installed = shutil.which("paperseek")
        if self.config.source_root is not None:
            if not self._source_module_available():
                raise RuntimeError("configured PaperSeek source tree does not expose paperseek.cli")
            root = Path(str(self.config.source_root))
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(root) + (os.pathsep + existing if existing else "")
            return [sys.executable, "-m", "paperseek.cli", *arguments], env
        if installed:
            return [installed, *arguments], env
        raise RuntimeError("PaperSeek CLI/source tree is not configured")

    def search(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("PaperSeek query cannot be blank")
        command, env = self._command_and_env(query)
        started = datetime.now(timezone.utc)
        completed = run_managed_subprocess(
            command,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_s,
            check=False,
            env=env,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if len(stdout.encode("utf-8", errors="replace")) > self.config.max_output_bytes:
            raise ValueError("PaperSeek CLI response exceeds size limit")
        if completed.returncode != 0:
            tail = stderr[-4000:].strip()
            raise RuntimeError(
                "PaperSeek CLI failed with exit code "
                f"{completed.returncode}" + (f": {tail}" if tail else "")
            )
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("PaperSeek CLI did not return JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("ranked"), list):
            raise ValueError("invalid PaperSeek CLI response: expected ranked list")

        now = datetime.now(timezone.utc).isoformat()
        documents: list[LiteratureDocument] = []
        rejected: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in payload["ranked"][: self.config.top_k]:
            if not isinstance(item, dict):
                raise ValueError("invalid PaperSeek ranked record")
            title = str(item.get("title") or "").strip()
            text = str(item.get("abstract") or "").strip()
            url = str(item.get("url") or "").strip()
            doi = str(item.get("doi") or "").strip()
            if not url and doi:
                url = "https://doi.org/" + doi
            if not title or not text or not url:
                rejected.append({
                    "id": str(item.get("id") or ""),
                    "reason": "MISSING_TITLE_ABSTRACT_OR_SOURCE",
                })
                continue
            content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if content_sha in seen or url in seen:
                continue
            seen.update({content_sha, url})
            documents.append(
                LiteratureDocument(
                    document_id="DOC-" + content_sha[:20],
                    title=title,
                    text=text,
                    source_url=url,
                    retrieved_at=now,
                    source_content_sha256=content_sha,
                )
            )

        duration = (datetime.now(timezone.utc) - started).total_seconds()
        return {
            "status": "SUCCEEDED",
            "backend": "PAPERSEEK_CLI",
            "backend_mode": self.backend_mode(),
            "query": query,
            "response_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "retrieved_at": now,
            "duration_s": round(duration, 6),
            "documents": [item.model_dump(mode="json") for item in documents],
            "rejected": rejected,
            "claim_boundary": "Live literature context, not observations of this spacecraft.",
        }


__all__ = ["PaperSeekCLIConfig", "PaperSeekCLIClient"]
