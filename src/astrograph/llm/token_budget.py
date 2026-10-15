from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Protocol


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Any: ...


class TokenizerLoadError(RuntimeError):
    pass


def rough_token_estimate(value: Any, *, chars_per_token: float = 3.5) -> int:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, math.ceil(len(text) / chars_per_token))


def _token_count(encoded: Any) -> int:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, tuple):
        encoded = list(encoded)
    if isinstance(encoded, list):
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise TokenizerLoadError("batched tokenizer result is not supported")
            encoded = encoded[0]
        return len(encoded)
    raise TokenizerLoadError(f"unsupported tokenizer output type: {type(encoded).__name__}")


def count_chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return max(1, _token_count(encoded))
    raise TokenizerLoadError("tokenizer does not provide apply_chat_template")


def resolve_tokenizer_path(tokenizer_path: str | None, model: str) -> str | None:
    raw = tokenizer_path or (model if Path(os.path.expandvars(os.path.expanduser(model))).exists() else None)
    if not raw:
        return None
    return os.path.expandvars(os.path.expanduser(raw))


def load_huggingface_tokenizer(
    tokenizer_path: str,
    *,
    local_files_only: bool,
    trust_remote_code: bool,
) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TokenizerLoadError(
            "exact tokenizer counting requires the optional 'transformers' dependency"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise TokenizerLoadError(f"cannot load tokenizer from {tokenizer_path}: {exc}") from exc


def relative_error(estimate: int | None, actual: int | None) -> float | None:
    if estimate is None or actual in {None, 0}:
        return None
    return round((estimate - actual) / actual, 6)
