#!/usr/bin/env python3
"""Build a governed local model/tokenizer runtime spec for R16 confirmatory attestation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "astrograph-confirmatory-model-runtime-spec.v1"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--producer-id", required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--tokenizer-revision", required=True)
    p.add_argument("--tokenizer-root", type=Path, required=True)
    p.add_argument("--server-version")
    p.add_argument("--server-build-id")
    p.add_argument("--server-artifact-root", type=Path)
    p.add_argument("--container-image-digest")
    p.add_argument("--server-launch-config", type=Path, required=True, help="JSON with exact vLLM argv and generation_config_policy.")
    p.add_argument("--deployment-kind", choices=["BARE_METAL", "CONTAINER", "VM"], default="BARE_METAL")
    p.add_argument("--require-readonly-artifacts", action="store_true")
    a = p.parse_args()
    launch_path = a.server_launch_config.resolve()
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if not isinstance(launch, dict):
        p.error("--server-launch-config must contain a JSON object")
    if launch.get("schema_version") != "astrograph-vllm-server-launch-config.v1":
        p.error("--server-launch-config has unsupported schema_version")
    if str(launch.get("provider") or "").lower() != "vllm":
        p.error("--server-launch-config provider must be vllm")
    if str(launch.get("model_id") or "") != a.model_id:
        p.error("--server-launch-config model_id must equal --model-id")
    if str(launch.get("generation_config_policy") or "") != "vllm":
        p.error("--server-launch-config generation_config_policy must be vllm")
    argv = launch.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        p.error("--server-launch-config argv must be a non-empty string list")
    try:
        generation_index = argv.index("--generation-config")
        generation_value = argv[generation_index + 1]
    except (ValueError, IndexError):
        generation_value = None
    if generation_value != "vllm":
        p.error("--server-launch-config argv must explicitly include --generation-config vllm")
    server = {
        "implementation": "vllm",
        "version": a.server_version,
        "build_id": a.server_build_id,
        "artifact_root": str(a.server_artifact_root.resolve()) if a.server_artifact_root else None,
        "container_image_digest": a.container_image_digest,
        "launch_config": launch,
    }
    payload = {
        "schema_version": SCHEMA,
        "producer_id": a.producer_id,
        "model_id": a.model_id,
        "model_revision": a.model_revision,
        "model_root": str(a.model_root.resolve()),
        "tokenizer_revision": a.tokenizer_revision,
        "tokenizer_root": str(a.tokenizer_root.resolve()),
        "deployment_kind": a.deployment_kind,
        "require_readonly_artifacts": bool(a.require_readonly_artifacts),
        "server": server,
    }
    path = a.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
