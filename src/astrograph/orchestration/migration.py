from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LEGACY_CONTRACT_KEYS = {
    "contract",
    "contract_id",
    "contract_version",
    "contract_sha256",
    "dataset_contract_version",
    "fault_contract_version",
}


def detect_legacy_contract_keys(payload: Mapping[str, Any], *, prefix: str = "") -> list[str]:
    """Return dotted paths that still use historical contract_* keys.

    This function is intentionally non-mutating. It supports migration warnings
    without rewriting archived payloads or invalidating hashes.
    """

    found: list[str] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key in LEGACY_CONTRACT_KEYS or str(key).endswith("_contract"):
            found.append(path)
        if isinstance(value, Mapping):
            found.extend(detect_legacy_contract_keys(value, prefix=path))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    found.extend(detect_legacy_contract_keys(item, prefix=f"{path}[{index}]"))
    return found


def legacy_protocol_warnings(payload: Mapping[str, Any]) -> list[str]:
    paths = detect_legacy_contract_keys(payload)
    if not paths:
        return []
    return [
        "Historical contract_* fields were accepted as read-only compatibility aliases; "
        "new V0.4.4 payloads should use protocol_* fields. Paths: " + ", ".join(paths)
    ]
