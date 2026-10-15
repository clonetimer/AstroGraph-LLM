from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrograph.evaluation import confirmatory_runtime_identity as identity

ROOT = Path(__file__).resolve().parents[1]


def _valid_launch_config(model_id: str = "Qwen3.5-9B") -> dict:
    return {
        "schema_version": "astrograph-vllm-server-launch-config.v1",
        "provider": "vllm",
        "model_id": model_id,
        "generation_config_policy": "vllm",
        "argv": ["vllm", "serve", "/models/qwen", "--served-model-name", model_id, "--generation-config", "vllm"],
        "environment": {},
    }


def _runtime_spec(tmp_path: Path, *, model_id: str = "Qwen3.5-9B") -> dict:
    model_root = tmp_path / "model"
    tokenizer_root = tmp_path / "tokenizer"
    server_root = tmp_path / "server"
    model_root.mkdir()
    tokenizer_root.mkdir()
    server_root.mkdir()
    (model_root / "weights.safetensors").write_bytes(b"model-v1")
    (tokenizer_root / "tokenizer.json").write_bytes(b"tokenizer-v1")
    (server_root / "vllm-runtime.bin").write_bytes(b"server-v1")
    return {
        "schema_version": identity.MODEL_RUNTIME_SPEC_SCHEMA,
        "producer_id": "unit-test-model-host",
        "model_id": model_id,
        "model_revision": "rev-model-001",
        "model_root": str(model_root),
        "tokenizer_revision": "rev-tokenizer-001",
        "tokenizer_root": str(tokenizer_root),
        "deployment_kind": "BARE_METAL",
        "require_readonly_artifacts": False,
        "server": {
            "implementation": "vllm",
            "version": "test",
            "build_id": "build-001",
            "artifact_root": str(server_root),
            "container_image_digest": None,
            "launch_config": _valid_launch_config(model_id),
        },
    }


def _fake_accelerator() -> dict:
    return {
        "status": "NVIDIA_GPU_OBSERVED",
        "nvidia_smi": "/usr/bin/nvidia-smi",
        "gpus": [{
            "index": "0",
            "uuid": "GPU-test",
            "name": "Test GPU",
            "driver_version": "999.1",
            "memory_total_mib": "81920",
        }],
        "torch_version": "2.test",
        "torch_cuda_version": "12.test",
    }


def test_inference_profile_is_content_bound_and_changes_identity(tmp_path: Path):
    profile = json.loads((ROOT / identity.INFERENCE_PROFILE_RELATIVE_PATH).read_text(encoding="utf-8"))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    first = identity.build_confirmatory_inference_runtime_snapshot(
        vllm_url="http://127.0.0.1:8000/v1", vllm_model="Qwen3.5-9B", profile_path=path
    )
    second = identity.build_confirmatory_inference_runtime_snapshot(
        vllm_url="http://127.0.0.1:8000/v1", vllm_model="Qwen3.5-9B", profile_path=path
    )
    assert first["inference_config_sha256"] == second["inference_config_sha256"]
    assert first["inference_profile_file_sha256"] == second["inference_profile_file_sha256"]
    profile["max_tokens"] = int(profile["max_tokens"]) + 1
    path.write_text(json.dumps(profile), encoding="utf-8")
    changed = identity.build_confirmatory_inference_runtime_snapshot(
        vllm_url="http://127.0.0.1:8000/v1", vllm_model="Qwen3.5-9B", profile_path=path
    )
    assert changed["inference_config_sha256"] != first["inference_config_sha256"]
    assert changed["inference_profile_file_sha256"] != first["inference_profile_file_sha256"]


def test_python_environment_lock_and_cyclonedx_sbom_are_self_verifying():
    lock = identity.build_confirmatory_python_environment_lock()
    check = identity.verify_confirmatory_python_environment_lock(lock)
    assert check["status"] == "STABLE"
    assert lock["cyclonedx_sbom"]["bomFormat"] == "CycloneDX"
    assert lock["cyclonedx_sbom"]["specVersion"] == "1.7"
    assert lock["package_count"] == len(lock["packages"])
    assert len(lock["lock_sha256"]) == 64


def test_model_runtime_attestation_binds_model_tokenizer_server_launch_cuda_and_detects_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(identity, "_nvidia_identity", _fake_accelerator)
    spec = _runtime_spec(tmp_path)
    first = identity.build_model_runtime_attestation(spec)
    check = identity.verify_model_runtime_attestation(first, expected_model_id="Qwen3.5-9B")
    assert first["status"] == "VALID"
    assert check["status"] == "STABLE"
    runtime = first["runtime_identity"]
    assert runtime["model_identity_sha256"]
    assert runtime["tokenizer_identity_sha256"]
    assert runtime["server"]["launch_config"]["generation_config_policy"] == "vllm"
    assert runtime["server"]["launch_config_sha256"]
    assert runtime["accelerator"]["gpus"][0]["driver_version"] == "999.1"
    assert runtime["accelerator"]["torch_cuda_version"] == "12.test"
    assert runtime["cyclonedx_sbom_sha256"]

    Path(spec["model_root"]).joinpath("weights.safetensors").write_bytes(b"model-v2")
    second = identity.build_model_runtime_attestation(spec)
    assert second["runtime_identity_sha256"] != first["runtime_identity_sha256"]
    assert second["runtime_identity"]["model_identity_sha256"] != runtime["model_identity_sha256"]


def test_model_runtime_attestation_internal_relation_tamper_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(identity, "_nvidia_identity", _fake_accelerator)
    attestation = identity.build_model_runtime_attestation(_runtime_spec(tmp_path))
    tampered = json.loads(json.dumps(attestation))
    tampered["runtime_identity"]["server"]["launch_config"]["argv"].append("--max-model-len=1")
    # Re-hash the outer envelope to prove the verifier also checks internal identity relations.
    core = dict(tampered)
    core.pop("attestation_sha256", None)
    tampered["attestation_sha256"] = identity._canonical_sha256(core)
    tampered["runtime_identity_sha256"] = identity._canonical_sha256(tampered["runtime_identity"])
    core = dict(tampered)
    core.pop("attestation_sha256", None)
    tampered["attestation_sha256"] = identity._canonical_sha256(core)
    check = identity.verify_model_runtime_attestation(tampered, expected_model_id="Qwen3.5-9B")
    assert check["status"] == "DRIFTED"
    assert "SERVER_LAUNCH_CONFIG_IDENTITY_INVALID" in check["drift_codes"]


def test_model_runtime_spec_requires_explicit_vllm_generation_config_policy(tmp_path: Path):
    spec = _runtime_spec(tmp_path)
    spec["server"]["launch_config"]["generation_config_policy"] = "model-repository-default"
    path = tmp_path / "runtime_spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="generation_config_policy must be vllm"):
        identity.load_model_runtime_spec(path)


def test_model_artifact_symlink_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(identity, "_nvidia_identity", _fake_accelerator)
    spec = _runtime_spec(tmp_path)
    external = tmp_path / "external.bin"
    external.write_bytes(b"external")
    link = Path(spec["model_root"]) / "linked.bin"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="symlink is not allowed"):
        identity.build_model_runtime_attestation(spec)


def test_runtime_identity_drift_uses_content_identity_not_timestamped_attestation_hash():
    base = {
        "status": "VALID",
        "runtime_identity": {
            "client_identity_sha256": "a" * 64,
            "model_runtime_identity_sha256": "b" * 64,
            "inference_config_sha256": "c" * 64,
        },
        "runtime_identity_sha256": "d" * 64,
        "model_runtime_attestation_sha256": "1" * 64,
        "blockers": [],
    }
    post = json.loads(json.dumps(base))
    post["model_runtime_attestation_sha256"] = "2" * 64  # generated_at may legitimately change
    stable = identity.verify_confirmatory_runtime_identity_drift(pre_admission=base, post_admission=post)
    assert stable["status"] == "STABLE"
    post["runtime_identity"]["model_runtime_identity_sha256"] = "e" * 64
    drifted = identity.verify_confirmatory_runtime_identity_drift(pre_admission=base, post_admission=post)
    assert drifted["status"] == "DRIFTED"
    assert "MODEL_RUNTIME_IDENTITY_DRIFT" in drifted["drift_codes"]


def test_ablation_row_preserves_frozen_manifest_group_id(monkeypatch: pytest.MonkeyPatch):
    from scripts.experiments import run_llm_method_ablation as runner

    monkeypatch.setattr(runner, "_decision", lambda _result: {"accepted_fault_id": "FAULT"})
    row = runner._row_from_result(
        case={
            "case_id": "case-1",
            "group_id": "trajectory-pair-42",
            "dataset": "adcs",
            "true_fault_id": "FAULT",
        },
        arm_id="A2_CONTROLLED_FULL",
        result=SimpleNamespace(output_payload={}),
        elapsed=0.01,
        backend=None,
    )
    assert row.case_id == "case-1"
    assert row.group_id == "trajectory-pair-42"
