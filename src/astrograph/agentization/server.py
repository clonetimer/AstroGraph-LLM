from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path
from typing import TextIO

from .capabilities import CapabilityEnvironment, build_registry
from .protocol import (
    CapabilityProtocolError,
    CapabilityRequest,
    CapabilityResponse,
    PROJECT_STATUS,
    PROTOCOL_VERSION,
    sha256_json,
)


def _receipt(request: CapabilityRequest, result: dict, instance_id: str) -> dict:
    return {
        "schema_version": "astrograph-capability-receipt.v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "server_instance_id": instance_id,
        "project_status": PROJECT_STATUS,
        "request_sha256": request.digest(),
        "result_sha256": sha256_json(result),
    }


def serve(
    project_root: Path,
    allowed_roots: list[Path],
    artifact_dir: Path,
    external_roots: list[Path],
    inp: TextIO = sys.stdin,
    out: TextIO = sys.stdout,
) -> None:
    env = CapabilityEnvironment(
        project_root.resolve(),
        tuple(root.resolve() for root in allowed_roots),
        artifact_dir.resolve(),
        (project_root / "configs/agentization/workflows").resolve(),
        tuple(root.resolve() for root in external_roots),
    )
    env.artifact_dir.mkdir(parents=True, exist_ok=True)
    registry = build_registry(env)
    instance_id = str(uuid.uuid4())

    for line in inp:
        if not line.strip():
            continue
        request: CapabilityRequest | None = None
        try:
            request = CapabilityRequest.from_mapping(json.loads(line))
            if request.method == "health":
                result = {
                    "status": "SERVING",
                    "protocol_version": PROTOCOL_VERSION,
                    "project_status": PROJECT_STATUS,
                }
            elif request.method == "capabilities.list":
                result = registry.manifest()
            elif request.method == "capabilities.invoke":
                name = str(request.params.get("capability", ""))
                arguments = request.params.get("arguments", {})
                if not isinstance(arguments, dict):
                    raise CapabilityProtocolError("INVALID_ARGUMENTS", "arguments must be an object")
                result = registry.invoke(name, request.profile, arguments)
            elif request.method == "jobs.status":
                result = {
                    "status": "NO_ASYNC_JOB_BACKEND",
                    "job_id": request.params.get("job_id"),
                }
            elif request.method == "shutdown":
                result = {"status": "SHUTTING_DOWN"}
            else:  # defensive; CapabilityRequest rejects unsupported methods
                raise CapabilityProtocolError("INVALID_METHOD", request.method)
            response = CapabilityResponse(
                request.request_id,
                "OK",
                result=result,
                receipt=_receipt(request, result, instance_id),
            )
        except CapabilityProtocolError as exc:
            request_id = request.request_id if request else "invalid-request"
            response = CapabilityResponse(
                request_id,
                "ERROR",
                error={"code": exc.code, "message": exc.message, "details": exc.details},
            )
        except Exception as exc:  # server boundary: convert unexpected errors to structured response
            request_id = request.request_id if request else "invalid-request"
            response = CapabilityResponse(
                request_id,
                "ERROR",
                error={
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                    "details": {"type": type(exc).__name__},
                },
            )
        out.write(json.dumps(response.to_mapping(), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        out.flush()
        if request and request.method == "shutdown":
            break


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--allowed-root", action="append", type=Path, default=[])
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--external-audit-root", action="append", type=Path, default=[])
    arguments = parser.parse_args(argv)
    roots = arguments.allowed_root or [arguments.project_root]
    artifact = arguments.artifact_dir or arguments.project_root / "research/agentization/runtime_artifacts"
    external = arguments.external_audit_root or [arguments.project_root]
    serve(arguments.project_root, roots, artifact, external)


if __name__ == "__main__":
    main()
