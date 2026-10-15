"""HTTP adapter for the supplied PaperSeek POST /api/search contract.

The service performs search; AstroGraph does not invent a replacement article when
it is unavailable. Credentials are configured on the PaperSeek server itself.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from urllib import request
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .contracts import LiteratureDocument


class PaperSeekConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    base_url: str = Field(min_length=1)
    data_source: str = "openalex"
    timeout_s: float = Field(default=120, gt=0, le=3600)
    top_k: int = Field(default=5, ge=1, le=50)
    max_iterations: int = Field(default=2, ge=1, le=10)
    max_response_bytes: int = Field(default=4_000_000, ge=1024, le=32_000_000)

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value):
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("PaperSeek endpoint must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("do not embed credentials, query, or fragment in the service URL")
        return value.rstrip("/")


class PaperSeekHTTPClient:
    def __init__(self, config: PaperSeekConfig):
        self.config = config

    def search(self, query: str) -> dict:
        if not query.strip():
            raise ValueError("PaperSeek query cannot be blank")
        endpoint = self.config.base_url
        if not endpoint.endswith("/api/search"):
            endpoint += "/api/search"
        payload = {"question": query, "data_source": self.config.data_source,
                   "target_min": 0, "target_max": self.config.top_k,
                   "max_iterations": self.config.max_iterations, "fetch_abstracts": True}
        from astrograph.integrations.bounded_http import request_json
        result=request_json(endpoint,payload=payload,timeout_s=self.config.timeout_s,
                            max_response_bytes=self.config.max_response_bytes)
        raw=json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
        if not isinstance(result, dict) or not isinstance(result.get("ranked"), list):
            raise ValueError("invalid PaperSeek response: expected ranked list")
        now = datetime.now(timezone.utc).isoformat()
        documents, rejected, seen = [], [], set()
        for item in result["ranked"][:self.config.top_k]:
            if not isinstance(item, dict):
                raise ValueError("invalid PaperSeek ranked record")
            title = str(item.get("title") or "").strip()
            text = str(item.get("abstract") or "").strip()
            url = str(item.get("url") or "").strip()
            doi = str(item.get("doi") or "").strip()
            if not url and doi:
                url = "https://doi.org/" + doi
            if not title or not text or not url:
                rejected.append({"id": str(item.get("id") or ""),
                                 "reason": "MISSING_TITLE_ABSTRACT_OR_SOURCE"})
                continue
            identity = content = hashlib.sha256(text.encode()).hexdigest()
            # Identical content or source is not additional independent knowledge.
            if content in seen or url in seen:
                continue
            seen.update([content, url])
            documents.append(LiteratureDocument(
                document_id="DOC-" + identity[:20], title=title, text=text,
                source_url=url, retrieved_at=now, source_content_sha256=content,
            ))
        return {"status": "SUCCEEDED", "backend": "PAPERSEEK_HTTP",
                "endpoint": endpoint, "query": query,
                "response_sha256": hashlib.sha256(raw).hexdigest(), "response_hash_semantics":"CANONICAL_PARSED_JSON", "retrieved_at": now,
                "documents": [d.model_dump(mode="json") for d in documents],
                "rejected": rejected,
                "claim_boundary": "Live literature context, not observations of this spacecraft."}
