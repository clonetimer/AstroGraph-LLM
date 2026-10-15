from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .contracts import build_pipeline_runtime_recompute_request, provider_capabilities, validate_upstream_request
from .manual_provider import build_manual_request_bundle, build_manual_response_bundle, sha256_bytes, verify_manual_response_bundle


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AstroGraph optional upstream recompute protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("providers", help="Show upstream provider availability without importing optional providers")

    build = sub.add_parser("build-request", help="Build a zero-dependency manual upstream recompute request bundle")
    build.add_argument("--source-job-id", required=True)
    build.add_argument("--runtime", type=Path, required=True)
    build.add_argument("--mapping", type=Path, required=True)
    build.add_argument("--contract-id", action="append", default=[])
    build.add_argument("--out", type=Path, required=True)

    package = sub.add_parser("package-response", help="Package complete runtime/mapping outputs into a request-bound response ZIP")
    package.add_argument("--request-json", type=Path, required=True)
    package.add_argument("--runtime", type=Path, required=True)
    package.add_argument("--mapping", type=Path, required=True)
    package.add_argument("--source-raw-input-receipt-sha256")
    package.add_argument("--out", type=Path, required=True)

    verify = sub.add_parser("verify-response", help="Verify a manual upstream response ZIP and materialize admitted artifacts")
    verify.add_argument("--request-json", type=Path, required=True)
    verify.add_argument("--response-zip", type=Path, required=True)
    verify.add_argument("--out-dir", type=Path, required=True)
    verify.add_argument("--actor", default="operator")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "providers":
        print(json.dumps(provider_capabilities(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "build-request":
        runtime = args.runtime.read_bytes()
        mapping = args.mapping.read_bytes()
        request = build_pipeline_runtime_recompute_request(
            source_job_id=args.source_job_id,
            source_runtime_sha256=sha256_bytes(runtime),
            source_mapping_bundle_sha256=sha256_bytes(mapping),
            maintenance_input_contract_ids=args.contract_id,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(build_manual_request_bundle(request))
        print(json.dumps({"status": "READY", "request_id": request.request_id, "request_sha256": request.request_sha256, "bundle": str(args.out)}, ensure_ascii=False))
        return 0
    if args.command == "package-response":
        request = validate_upstream_request(json.loads(args.request_json.read_text(encoding="utf-8")))
        data = build_manual_response_bundle(
            request=request,
            runtime_bytes=args.runtime.read_bytes(),
            mapping_bytes=args.mapping.read_bytes(),
            source_raw_input_receipt_sha256=args.source_raw_input_receipt_sha256,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(data)
        print(json.dumps({"status": "PACKAGED", "request_id": request.request_id, "bundle": str(args.out), "bundle_sha256": sha256_bytes(data)}, ensure_ascii=False))
        return 0
    if args.command == "verify-response":
        request = validate_upstream_request(json.loads(args.request_json.read_text(encoding="utf-8")))
        admitted = verify_manual_response_bundle(
            request=request,
            bundle_bytes=args.response_zip.read_bytes(),
            admitted_at=_now(),
            actor=args.actor,
        )
        out = args.out_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "runtime.json").write_bytes(admitted.runtime_bytes)
        (out / "mapping.json").write_bytes(admitted.mapping_bytes)
        _write_json(out / "admission_receipt.json", admitted.admission_receipt)
        print(json.dumps({"status": "ADMITTED", "out_dir": str(out), "receipt_sha256": admitted.admission_receipt["receipt_sha256"]}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
