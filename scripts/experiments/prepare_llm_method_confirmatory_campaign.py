#!/usr/bin/env python3
"""Prepare the frozen LLM-method confirmatory campaign without mutating source by default.

The command can mount explicit ADCS/PHM dataset roots, freeze the 38-case manifest into an
external experiment directory, probe the exact vLLM model identity, emit readiness + launch
receipts, and optionally invoke the formal runner only when readiness is READY.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from astrograph.evaluation.confirmatory_runtime_identity import (
    acquire_model_runtime_attestation,
    build_confirmatory_runtime_identity_admission,
    load_confirmatory_python_environment_lock,
)
from astrograph.evaluation.llm_method_campaign import (
    build_confirmatory_dataset_snapshot_receipt,
    confirmatory_readiness,
    freeze_confirmatory_case_manifest,
    load_confirmatory_design,
    materialize_confirmatory_dataset_snapshot,
)


def _probe_vllm(url: str) -> tuple[bool, str, list[str]]:
    endpoint = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(endpoint, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = sorted(str(item.get("id")) for item in payload.get("data", []) if item.get("id"))
        return True, "models=" + ",".join(models), models
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}", []


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen3.5-9B")
    parser.add_argument("--adcs-root", type=Path)
    parser.add_argument("--phm-root", type=Path)
    parser.add_argument("--output-dir", type=Path, help="External campaign working directory; recommended for immutable source deliveries.")
    parser.add_argument("--case-manifest", type=Path, help="Use an already frozen manifest instead of the default output location.")
    parser.add_argument("--freeze-case-manifest", action="store_true")
    parser.add_argument(
        "--environment-lock", type=Path,
        help="Frozen AstroGraph orchestrator Python environment lock. R16 formal readiness is BLOCKED when omitted.",
    )
    identity_group = parser.add_mutually_exclusive_group()
    identity_group.add_argument(
        "--model-runtime-spec", type=Path,
        help="Local model/tokenizer/server runtime spec used to recompute content identity.",
    )
    identity_group.add_argument(
        "--model-runtime-attestation-url",
        help="Fresh model-host runtime attestation URL. R16 validates content identity but does not authenticate the remote producer.",
    )
    parser.add_argument(
        "--materialize-snapshot", action="store_true",
        help="Copy the frozen denominator into a self-contained content-addressed snapshot under the external work directory.",
    )
    parser.add_argument(
        "--materialized-snapshot-root", type=Path,
        help="Override the default <output-dir>/materialized_snapshot destination.",
    )
    parser.add_argument("--output-readiness", type=Path, help="Compatibility alias for writing readiness JSON.")
    parser.add_argument("--execute-if-ready", action="store_true", help="Invoke the formal confirmatory runner after a READY preparation gate.")
    parser.add_argument(
        "--require-readonly-datasets", action="store_true",
        help="Fail admission unless both dataset roots are observed on read-only filesystem mounts.",
    )
    args = parser.parse_args()
    if args.execute_if_ready and not args.output_dir:
        parser.error("--execute-if-ready requires --output-dir so the snapshot and execution receipts remain outside the source tree")
    if args.materialize_snapshot and not args.output_dir:
        parser.error("--materialize-snapshot requires --output-dir")

    design = load_confirmatory_design(ROOT)
    dataset_roots: dict[str, Path] = {}
    if args.adcs_root:
        dataset_roots["adcs"] = args.adcs_root.resolve()
    if args.phm_root:
        dataset_roots["phm"] = args.phm_root.resolve()

    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.case_manifest:
        case_path = args.case_manifest.resolve()
    elif output_dir:
        case_path = output_dir / "case_manifest.json"
    else:
        # Backward-compatible path. New governed runs should pass --output-dir.
        case_path = ROOT / str(design["case_manifest_path"])

    freeze_status = "NOT_REQUESTED"
    if args.freeze_case_manifest:
        try:
            freeze_confirmatory_case_manifest(
                ROOT,
                output_path=case_path,
                dataset_roots=dataset_roots or None,
            )
            freeze_status = "FROZEN"
        except Exception as exc:
            freeze_status = f"FAILED:{type(exc).__name__}:{exc}"

    endpoint_ok, model_detail, observed_models = _probe_vllm(args.vllm_url)
    environment_lock = None
    environment_lock_error = None
    if args.environment_lock:
        try:
            environment_lock = load_confirmatory_python_environment_lock(args.environment_lock.resolve())
        except Exception as exc:
            environment_lock_error = f"{type(exc).__name__}:{exc}"
    model_runtime_attestation = None
    model_runtime_source = {"mode": "UNCONFIGURED", "refreshable": False, "source": None}
    model_runtime_error = None
    if args.model_runtime_spec or args.model_runtime_attestation_url:
        try:
            model_runtime_attestation, model_runtime_source = acquire_model_runtime_attestation(
                spec_path=(args.model_runtime_spec.resolve() if args.model_runtime_spec else None),
                attestation_url=args.model_runtime_attestation_url,
            )
        except Exception as exc:
            model_runtime_error = f"{type(exc).__name__}:{exc}"
    runtime_identity_admission = build_confirmatory_runtime_identity_admission(
        ROOT,
        environment_lock=environment_lock,
        model_runtime_attestation=model_runtime_attestation,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
    )
    if environment_lock_error:
        runtime_identity_admission.setdefault("blockers", []).append(f"CLIENT_ENVIRONMENT_LOCK_LOAD_ERROR:{environment_lock_error}")
        runtime_identity_admission["status"] = "BLOCKED"
    if model_runtime_error:
        runtime_identity_admission.setdefault("blockers", []).append(f"MODEL_RUNTIME_ATTESTATION_ACQUISITION_ERROR:{model_runtime_error}")
        runtime_identity_admission["status"] = "BLOCKED"
    runtime_identity_admission["blockers"] = sorted(set(runtime_identity_admission.get("blockers") or []))
    readiness = confirmatory_readiness(
        ROOT,
        dataset_roots=dataset_roots or None,
        live_model_available=endpoint_ok,
        live_model_detail=model_detail,
        requested_model_id=args.vllm_model,
        observed_model_ids=observed_models,
        case_manifest_path=case_path,
        require_readonly_datasets=args.require_readonly_datasets,
        runtime_identity_admission=runtime_identity_admission,
        enforce_runtime_identity=True,
    )
    readiness["runtime_identity_attestation_source"] = model_runtime_source
    readiness["preparation"] = {
        "case_manifest_freeze_status": freeze_status,
        "source_tree_mutated": bool(case_path.is_relative_to(ROOT)),
        "recommended_external_workdir": str(output_dir) if output_dir else None,
    }

    readiness_path = args.output_readiness.resolve() if args.output_readiness else (output_dir / "readiness.json" if output_dir else None)
    if readiness_path:
        _write(readiness_path, readiness)
    if output_dir:
        _write(output_dir / "runtime_identity_admission.json", runtime_identity_admission)
        if model_runtime_attestation is not None:
            _write(output_dir / "model_runtime_attestation.json", model_runtime_attestation)

    snapshot_receipt = build_confirmatory_dataset_snapshot_receipt(
        readiness=readiness, require_readonly_datasets=args.require_readonly_datasets
    )
    snapshot_receipt_path = output_dir / "dataset_snapshot_receipt.json" if output_dir else None
    if snapshot_receipt_path:
        _write(snapshot_receipt_path, snapshot_receipt)

    materialized_snapshot_root = (
        args.materialized_snapshot_root.resolve() if args.materialized_snapshot_root
        else ((output_dir / "materialized_snapshot") if output_dir else None)
    )
    materialization_receipt = None
    if args.materialize_snapshot:
        if snapshot_receipt.get("status") != "VALID" or not case_path.is_file():
            materialization_receipt = {
                "schema_version": "astrograph-llm-method-confirmatory-snapshot-materialization.v1",
                "status": "BLOCKED",
                "blockers": ["FROZEN_DATASET_SNAPSHOT_NOT_VALID"],
                "materialization_root": str(materialized_snapshot_root) if materialized_snapshot_root else None,
            }
        else:
            materialization_receipt = materialize_confirmatory_dataset_snapshot(
                ROOT, snapshot_receipt=snapshot_receipt, case_manifest_path=case_path,
                output_root=materialized_snapshot_root, dataset_roots=(dataset_roots or None),
                harden_readonly_permissions=True,
            )

    mount_receipt = {
        "schema_version": "astrograph-llm-method-confirmatory-dataset-mount-receipt.v1",
        "campaign_id": design["campaign_id"],
        "status": ("VALID" if all(item.get("ready") for item in (readiness.get("datasets") or {}).values()) else "INVALID"),
        "case_manifest_path": str(case_path),
        "case_manifest_sha256": (readiness.get("case_manifest") or {}).get("sha256"),
        "datasets": {
            key: {
                "root": item.get("root"),
                "sample_index_sha256": item.get("sample_index_sha256"),
                "mount_audit_status": item.get("mount_audit_status"),
                "selected_fault_row_count": item.get("selected_fault_row_count"),
                "selected_nominal_sample_count": item.get("selected_nominal_sample_count"),
                "payload_file_count": item.get("payload_file_count"),
                "payload_identity_sha256": item.get("payload_identity_sha256"),
                "mount_access": item.get("mount_access"),
                "errors": item.get("errors") or [],
            }
            for key, item in sorted((readiness.get("datasets") or {}).items())
        },
        "claim_boundary": (
            "Dataset mount/admission identity only. VALID means selected confirmatory sample identities and referenced payload files were present and hashable; "
            "it is not a diagnostic-performance or qualification result."
        ),
    }
    mount_receipt_path = output_dir / "dataset_mount_receipt.json" if output_dir else None
    if mount_receipt_path:
        _write(mount_receipt_path, mount_receipt)

    runner_identity_args: list[str] = []
    runner_identity_blockers: list[str] = []
    if args.environment_lock:
        runner_identity_args += ["--environment-lock", str(args.environment_lock.resolve())]
    else:
        runner_identity_blockers.append("CLIENT_ENVIRONMENT_LOCK_MISSING")
    if args.model_runtime_spec:
        runner_identity_args += ["--model-runtime-spec", str(args.model_runtime_spec.resolve())]
    elif args.model_runtime_attestation_url:
        runner_identity_args += ["--model-runtime-attestation-url", args.model_runtime_attestation_url]
    else:
        runner_identity_blockers.append("MODEL_RUNTIME_ATTESTATION_SOURCE_MISSING")
    runner_command = None
    if not runner_identity_blockers:
        runner_command = [
            sys.executable,
            str(ROOT / "scripts/experiments/run_llm_method_confirmatory_campaign.py"),
            "--output-dir", str((output_dir / "execution") if output_dir else (ROOT / "governance/evidence/research/llm_method_confirmatory_v4/execution")),
            "--vllm-url", args.vllm_url,
            "--vllm-model", args.vllm_model,
            "--case-manifest", str(case_path),
            "--dataset-snapshot-receipt", str(snapshot_receipt_path) if snapshot_receipt_path else "",
            *runner_identity_args,
        ]
        if args.require_readonly_datasets:
            runner_command += ["--require-readonly-datasets"]
        if dataset_roots.get("adcs"):
            runner_command += ["--adcs-root", str(dataset_roots["adcs"])]
        if dataset_roots.get("phm"):
            runner_command += ["--phm-root", str(dataset_roots["phm"])]
    materialization_command = [
        sys.executable,
        str(ROOT / "scripts/experiments/materialize_llm_method_confirmatory_snapshot.py"),
        "--snapshot-receipt", str(snapshot_receipt_path) if snapshot_receipt_path else "",
        "--case-manifest", str(case_path),
        "--output-root", str(materialized_snapshot_root) if materialized_snapshot_root else "",
    ]
    if dataset_roots.get("adcs"):
        materialization_command += ["--adcs-root", str(dataset_roots["adcs"])]
    if dataset_roots.get("phm"):
        materialization_command += ["--phm-root", str(dataset_roots["phm"])]
    snapshot_export_command = [
        sys.executable,
        str(ROOT / "scripts/experiments/export_llm_method_confirmatory_snapshot.py"),
        "--snapshot-root", str(materialized_snapshot_root) if materialized_snapshot_root else "",
    ]
    snapshot_verify_command = [
        sys.executable,
        str(ROOT / "scripts/experiments/verify_llm_method_confirmatory_snapshot_export.py"),
        "--snapshot-root", str(materialized_snapshot_root) if materialized_snapshot_root else "",
    ]
    materialized_runner_command = None
    if (
        materialization_receipt and materialization_receipt.get("status") == "VALID"
        and materialized_snapshot_root and not runner_identity_blockers
    ):
        materialized_runner_command = [
            sys.executable,
            str(ROOT / "scripts/experiments/run_llm_method_confirmatory_campaign.py"),
            "--output-dir", str(output_dir / "execution-materialized"),
            "--vllm-url", args.vllm_url,
            "--vllm-model", args.vllm_model,
            "--case-manifest", str(materialized_snapshot_root / "case_manifest.json"),
            "--dataset-snapshot-receipt", str(materialized_snapshot_root / "snapshot_metadata/materialized_dataset_snapshot_receipt.json"),
            "--adcs-root", str(materialized_snapshot_root / "datasets/adcs"),
            "--phm-root", str(materialized_snapshot_root / "datasets/phm"),
            *runner_identity_args,
        ]
        if args.require_readonly_datasets:
            materialized_runner_command += ["--require-readonly-datasets"]
    launch_plan = {
        "schema_version": "astrograph-llm-method-confirmatory-launch-plan.v6",
        "campaign_id": design["campaign_id"],
        "readiness_status": readiness["status"],
        "case_manifest_path": str(case_path),
        "dataset_roots": {key: str(value) for key, value in sorted(dataset_roots.items())},
        "vllm_url": args.vllm_url,
        "vllm_model": args.vllm_model,
        "runner_argv": runner_command,
        "runner_identity_blockers": runner_identity_blockers,
        "runtime_identity_status": runtime_identity_admission.get("status"),
        "runtime_identity_sha256": runtime_identity_admission.get("runtime_identity_sha256"),
        "model_runtime_attestation_source": model_runtime_source,
        "environment_lock_path": str(args.environment_lock.resolve()) if args.environment_lock else None,
        "inference_profile_path": str(ROOT / str(design["inference_profile_path"])),
        "dataset_mount_receipt_path": (str(mount_receipt_path) if mount_receipt_path else None),
        "dataset_snapshot_receipt_path": (str(snapshot_receipt_path) if snapshot_receipt_path else None),
        "dataset_snapshot_sha256": snapshot_receipt.get("snapshot_sha256"),
        "dataset_snapshot_status": snapshot_receipt.get("status"),
        "require_readonly_datasets": bool(args.require_readonly_datasets),
        "dataset_payload_identities": {
            key: item.get("payload_identity_sha256")
            for key, item in sorted((readiness.get("datasets") or {}).items())
        },
        "snapshot_materialization_argv": materialization_command,
        "snapshot_export_argv": snapshot_export_command,
        "snapshot_verify_argv": snapshot_verify_command,
        "materialization_requested": bool(args.materialize_snapshot),
        "materialization_receipt": materialization_receipt,
        "materialized_runner_argv": materialized_runner_command,
        "materialized_snapshot_policy_note": (
            "chmod-based write-bit removal is only accidental-mutation hardening. When --require-readonly-datasets is used, "
            "the materialized dataset roots must still be exposed through a true read-only mount before formal execution."
        ),
        "claim_boundary": "Preparation/launch plan only. R16 formal execution remains gated by independent dataset snapshot, source/environment/model-runtime admission and fresh post-execution identity re-observation. Runtime content identity does not authenticate a remote producer.",
    }
    if output_dir:
        _write(output_dir / "launch_plan.json", launch_plan)

    print(json.dumps({
        "readiness": readiness,
        "dataset_mount_receipt": mount_receipt,
        "dataset_snapshot_receipt": snapshot_receipt,
        "snapshot_materialization": materialization_receipt,
        "launch_plan": launch_plan,
    }, ensure_ascii=False, indent=2))
    if readiness["status"] != "READY":
        return 2
    if args.execute_if_ready:
        if runner_command is None:
            return 2
        return subprocess.run(runner_command, cwd=ROOT, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
