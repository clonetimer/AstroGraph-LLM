from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend

ENV_LOCK_SCHEMA = "astrograph-confirmatory-python-environment-lock.v1"
MODEL_RUNTIME_SPEC_SCHEMA = "astrograph-confirmatory-model-runtime-spec.v1"
MODEL_RUNTIME_ATTESTATION_SCHEMA = "astrograph-confirmatory-model-runtime-attestation.v1"
RUNTIME_ADMISSION_SCHEMA = "astrograph-confirmatory-runtime-identity-admission.v1"
INFERENCE_PROFILE_SCHEMA = "astrograph-llm-confirmatory-inference-profile.v1"
INFERENCE_PROFILE_RELATIVE_PATH = Path("configs/evaluation/llm_method_confirmatory_inference.v1.json")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_inference_profile(root: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    root_path = Path(root).resolve() if root is not None else _project_root()
    path = root_path / INFERENCE_PROFILE_RELATIVE_PATH
    payload = _read_json(path)
    if payload.get("schema_version") != INFERENCE_PROFILE_SCHEMA:
        raise ValueError("unsupported confirmatory inference profile schema")
    if payload.get("provider") != "vllm":
        raise ValueError("confirmatory inference profile provider must be vllm")
    return path, payload


def build_confirmatory_openai_config(
    *,
    vllm_url: str,
    vllm_model: str,
    root: str | Path | None = None,
) -> OpenAICompatibleConfig:
    _path, profile = _load_inference_profile(root)
    config_fields = set(OpenAICompatibleConfig.model_fields)
    kwargs = {key: value for key, value in profile.items() if key in config_fields}
    kwargs.update({"base_url": vllm_url, "model": vllm_model})
    return OpenAICompatibleConfig(**kwargs)


def build_confirmatory_inference_runtime_snapshot(
    *,
    vllm_url: str,
    vllm_model: str,
    profile_path: str | Path | None = None,
) -> dict[str, Any]:
    if profile_path is not None:
        path = Path(profile_path).resolve()
        profile = _read_json(path)
        if profile.get("schema_version") != "astrograph-llm-confirmatory-inference-profile.v1":
            raise ValueError("unsupported confirmatory inference profile schema")
        config_payload = {key: value for key, value in profile.items() if key not in {"schema_version", "claim_boundary"}}
        config = OpenAICompatibleConfig(base_url=vllm_url, model=vllm_model, **config_payload)
        profile_file_sha256 = _sha256_file(path)
        profile_canonical_sha256 = _canonical_sha256(profile)
        profile_schema = profile.get("schema_version")
    else:
        config = OpenAICompatibleConfig(
            base_url=vllm_url, provider="vllm", model=vllm_model, max_tokens=900, timeout_s=180,
            prompt_profile="evidence_protocol_v059_compact", candidate_top_k=2, tokenizer_mode="disabled", context_length=16384,
        )
        profile_file_sha256 = None; profile_canonical_sha256 = None; profile_schema = None
    snapshot = config.public_snapshot()
    snapshot.update({
        "backend_model_version": OpenAICompatibleStructuredBackend.model_version,
        "prompt_builder_sources": ["src/astrograph/llm/openai_compatible.py", "src/astrograph/llm/task_prompts.py"],
        "inference_profile_schema": profile_schema,
        "inference_profile_file_sha256": profile_file_sha256,
        "inference_profile_canonical_sha256": profile_canonical_sha256,
    })
    snapshot["inference_config_sha256"] = _canonical_sha256(snapshot)
    return snapshot


def _distribution_manifest() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for dist in importlib.metadata.distributions():
        name = str(dist.metadata.get("Name") or "").strip()
        version = str(dist.version or "").strip()
        if not name:
            continue
        key = (name.casefold(), version)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"name": name, "version": version})
    rows.sort(key=lambda item: (item["name"].casefold(), item["version"]))
    return rows


def _cyclonedx_python_sbom(packages: list[dict[str, str]], *, component_name: str) -> dict[str, Any]:
    package_hash = _canonical_sha256(packages)
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"astrograph:{component_name}:{package_hash}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {"component": {"type": "application", "name": component_name, "version": "1"}},
        "components": [
            {
                "type": "library",
                "bom-ref": f"python:{item['name']}=={item['version']}",
                "name": item["name"],
                "version": item["version"],
            }
            for item in packages
        ],
    }


def build_confirmatory_python_environment_lock() -> dict[str, Any]:
    packages = _distribution_manifest()
    sbom = _cyclonedx_python_sbom(packages, component_name="AstroGraph confirmatory Python runtime")
    executable = Path(sys.executable)
    core = {
        "schema_version": ENV_LOCK_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "abi": getattr(sys.implementation, "cache_tag", None),
            "executable": str(executable.resolve()),
            "executable_sha256": _sha256_file(executable) if executable.is_file() else None,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "package_count": len(packages),
        "packages_sha256": _canonical_sha256(packages),
        "cyclonedx_sbom": sbom,
        "cyclonedx_sbom_sha256": _canonical_sha256(sbom),
        "claim_boundary": (
            "Exact observed Python distribution manifest and CycloneDX SBOM for confirmatory reproducibility. "
            "It binds the observed environment but is not a dependency-resolver proof, package signature, or supply-chain attestation."
        ),
    }
    payload = dict(core)
    payload["lock_sha256"] = _canonical_sha256(core)
    return payload


def build_confirmatory_cyclonedx_sbom(environment_lock: dict[str, Any], *, component_name: str) -> dict[str, Any]:
    packages = environment_lock.get("packages") or []
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"astrograph:{component_name}:{environment_lock.get('packages_sha256')}")
    return {
        "bomFormat": "CycloneDX", "specVersion": "1.7", "serialNumber": f"urn:uuid:{serial}", "version": 1,
        "metadata": {"component": {"type": "application", "name": component_name, "version": "1"}},
        "components": [
            {"type": "library", "bom-ref": f"python:{str(item.get('name') or '').lower()}=={item.get('version')}", "name": item.get("name"), "version": item.get("version")}
            for item in packages
        ],
    }


def load_confirmatory_python_environment_lock(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path).resolve())
    if payload.get("schema_version") != ENV_LOCK_SCHEMA:
        raise ValueError("unsupported confirmatory Python environment lock schema")
    return payload


def verify_confirmatory_python_environment_lock(lock: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if lock.get("schema_version") != ENV_LOCK_SCHEMA:
        errors.append("ENVIRONMENT_LOCK_SCHEMA_MISMATCH")
    stored = str(lock.get("lock_sha256") or "")
    core = dict(lock)
    core.pop("lock_sha256", None)
    if not stored or stored != _canonical_sha256(core):
        errors.append("ENVIRONMENT_LOCK_SHA256_INVALID")
    packages = lock.get("packages") or []
    if lock.get("packages_sha256") != _canonical_sha256(packages):
        errors.append("ENVIRONMENT_PACKAGES_SHA256_INVALID")
    sbom = lock.get("cyclonedx_sbom") or {}
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.7":
        errors.append("ENVIRONMENT_SBOM_INVALID")
    if lock.get("cyclonedx_sbom_sha256") != _canonical_sha256(sbom):
        errors.append("ENVIRONMENT_SBOM_SHA256_INVALID")
    current = build_confirmatory_python_environment_lock()
    if lock.get("packages_sha256") != current.get("packages_sha256"):
        errors.append("PYTHON_DISTRIBUTION_MANIFEST_DRIFT")
    frozen_python = lock.get("python") or {}
    current_python = current.get("python") or {}
    for key in ("implementation", "version", "abi", "executable_sha256", "platform", "machine"):
        if frozen_python.get(key) != current_python.get(key):
            errors.append(f"PYTHON_RUNTIME_DRIFT:{key}")
    return {
        "schema_version": "astrograph-confirmatory-python-environment-check.v1",
        "status": "STABLE" if not errors else "DRIFTED",
        "drift_codes": sorted(set(errors)),
        "expected_lock_sha256": stored or None,
        "observed_lock_sha256": current.get("lock_sha256"),
        "observed_packages_sha256": current.get("packages_sha256"),
        "observed_cyclonedx_sbom_sha256": current.get("cyclonedx_sbom_sha256"),
    }


def _tree_file_entries(root: Path) -> tuple[str, list[dict[str, Any]]]:
    root = root.resolve()
    if not root.exists():
        raise ValueError(f"artifact root does not exist: {root}")
    entries: list[dict[str, Any]] = []
    if root.is_file() or root.is_symlink():
        resolved = root.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"artifact root is not a regular file: {root}")
        entries.append({
            "relative_path": root.name,
            "size_bytes": resolved.stat().st_size,
            "sha256": _sha256_file(resolved),
            "source_kind": "SYMLINK_FILE" if root.is_symlink() else "REGULAR_FILE",
        })
        return "FILE", entries
    if not root.is_dir():
        raise ValueError(f"artifact root is neither file nor directory: {root}")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for dirname in list(dirnames):
            candidate = base / dirname
            if candidate.is_symlink():
                raise ValueError(f"artifact directory symlink is not allowed: {candidate}")
        for filename in filenames:
            candidate = base / filename
            if candidate.is_symlink():
                raise ValueError(f"artifact file symlink is not allowed: {candidate}")
            resolved = candidate.resolve(strict=True)
            if root not in resolved.parents and resolved != root:
                raise ValueError(f"artifact member escapes root: {candidate}")
            if not resolved.is_file():
                raise ValueError(f"artifact member is not a regular file: {candidate}")
            entries.append({
                "relative_path": candidate.relative_to(root).as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
                "source_kind": "REGULAR_FILE",
            })
    entries.sort(key=lambda item: item["relative_path"])
    if not entries:
        raise ValueError(f"artifact root contains no files: {root}")
    return "DIRECTORY", entries


def _artifact_identity(root: Path, *, label: str, revision: str) -> dict[str, Any]:
    kind, entries = _tree_file_entries(root)
    identity = {
        "label": label,
        "revision": revision,
        "root_kind": kind,
        "file_count": len(entries),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in entries),
        "files": entries,
        "content_manifest_sha256": _canonical_sha256(entries),
    }
    return {"root": str(root.resolve()), "identity": identity, "identity_sha256": _canonical_sha256(identity)}


def _nvidia_identity() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    torch_version = None; torch_cuda_version = None
    try:
        torch_version = importlib.metadata.version("torch")
        import torch  # type: ignore
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        pass
    if not command:
        return {"status": "CPU_OR_NON_NVIDIA_OBSERVED", "nvidia_smi": None, "gpus": [], "torch_version": torch_version, "torch_cuda_version": torch_cuda_version}
    query = [command, "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return {"status": "ACCELERATOR_QUERY_FAILED", "nvidia_smi": command, "detail": f"{type(exc).__name__}:{exc}", "gpus": [], "torch_version": torch_version, "torch_cuda_version": torch_cuda_version}
    if completed.returncode != 0:
        return {"status": "ACCELERATOR_QUERY_FAILED", "nvidia_smi": command, "detail": completed.stderr.strip()[:1000], "gpus": [], "torch_version": torch_version, "torch_cuda_version": torch_cuda_version}
    gpus = []
    for line in completed.stdout.splitlines():
        parts = [item.strip() for item in line.split(",", 4)]
        if len(parts) == 5:
            gpus.append({"index": parts[0], "uuid": parts[1], "name": parts[2], "driver_version": parts[3], "memory_total_mib": parts[4]})
    return {"status": "NVIDIA_GPU_OBSERVED" if gpus else "CPU_OR_NON_NVIDIA_OBSERVED", "nvidia_smi": command, "gpus": gpus, "torch_version": torch_version, "torch_cuda_version": torch_cuda_version}


def _installed_distribution_artifact(distribution_name: str) -> dict[str, Any] | None:
    try:
        dist = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    entries: list[dict[str, Any]] = []
    for item in sorted(dist.files or [], key=lambda value: str(value)):
        path = dist.locate_file(item)
        try:
            if path.is_file():
                entries.append({
                    "relative_path": str(item).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })
        except OSError:
            continue
    identity = {
        "distribution": distribution_name,
        "version": dist.version,
        "files": entries,
        "file_count": len(entries),
        "content_manifest_sha256": _canonical_sha256(entries),
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _mount_access_status(path: Path) -> dict[str, Any]:
    path = path.resolve(); mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file(): return {"status": "UNKNOWN", "mount_point": None, "options": [], "detail": "mountinfo unavailable"}
    best = None
    try:
        for line in mountinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            fields=line.split()
            if len(fields)<7 or "-" not in fields: continue
            mp_raw=fields[4].replace("\\040"," "); options=fields[5].split(",")
            try:
                mp=Path(mp_raw).resolve()
                if path != mp and mp not in path.parents: continue
                score=len(str(mp))
            except OSError: continue
            if best is None or score>best[0]: best=(score,str(mp),options)
    except OSError as exc: return {"status":"UNKNOWN","mount_point":None,"options":[],"detail":f"{type(exc).__name__}:{exc}"}
    if best is None: return {"status":"UNKNOWN","mount_point":None,"options":[],"detail":"no covering mount found"}
    _,mount_point,options=best; status="READ_ONLY" if "ro" in options else ("READ_WRITE" if "rw" in options else "UNKNOWN")
    return {"status":status,"mount_point":mount_point,"options":options,"detail":None}


def load_model_runtime_spec(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(); payload = _read_json(path)
    if payload.get("schema_version") != MODEL_RUNTIME_SPEC_SCHEMA: raise ValueError("unsupported model runtime spec schema")
    required = ["producer_id", "model_id", "model_revision", "model_root", "tokenizer_revision", "tokenizer_root"]
    missing=[key for key in required if not str(payload.get(key) or "").strip()]
    if missing: raise ValueError("model runtime spec missing required fields: "+",".join(missing))
    server=payload.get("server") or {}
    if str(server.get("implementation") or "").lower() != "vllm": raise ValueError("confirmatory model runtime spec requires server.implementation=vllm")
    launch=server.get("launch_config")
    if not isinstance(launch,dict): raise ValueError("confirmatory model runtime spec requires server.launch_config")
    argv=launch.get("argv")
    if not isinstance(argv,list) or not argv or not all(isinstance(item,str) and item for item in argv): raise ValueError("server.launch_config.argv must be a non-empty string list")
    policy = str(launch.get("generation_config_policy") or "").strip()
    if policy != "vllm":
        raise ValueError("server.launch_config.generation_config_policy must be vllm for frozen confirmatory execution")
    launch_schema = str(launch.get("schema_version") or "")
    if launch_schema != "astrograph-vllm-server-launch-config.v1":
        raise ValueError("unsupported server.launch_config schema")
    if str(launch.get("provider") or "").lower() != "vllm":
        raise ValueError("server.launch_config.provider must be vllm")
    if str(launch.get("model_id") or "") != str(payload.get("model_id") or ""):
        raise ValueError("server.launch_config.model_id must match model runtime spec model_id")
    try:
        generation_index = argv.index("--generation-config")
        generation_value = argv[generation_index + 1]
    except (ValueError, IndexError):
        generation_value = None
    if generation_value != "vllm":
        raise ValueError("server.launch_config.argv must explicitly include --generation-config vllm")
    return payload


def build_model_runtime_attestation(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != MODEL_RUNTIME_SPEC_SCHEMA: raise ValueError("unsupported model runtime spec schema")
    model_root=Path(spec["model_root"]); tokenizer_root=Path(spec["tokenizer_root"])
    model=_artifact_identity(model_root,label=str(spec["model_id"]),revision=str(spec["model_revision"])); tokenizer=_artifact_identity(tokenizer_root,label="tokenizer",revision=str(spec["tokenizer_revision"]))
    server=dict(spec.get("server") or {}); launch_config=dict(server.get("launch_config") or {}); server_artifact=None
    if server.get("artifact_root"): server_artifact=_artifact_identity(Path(server["artifact_root"]),label="vllm-server-artifact",revision=str(server.get("build_id") or "UNSPECIFIED"))
    vllm_distribution=_installed_distribution_artifact("vllm")
    strong_server_identity=any([str(server.get("container_image_digest") or "").strip(),server_artifact is not None,vllm_distribution is not None and int(vllm_distribution.get("file_count") or 0)>0])
    environment=build_confirmatory_python_environment_lock(); sbom=build_confirmatory_cyclonedx_sbom(environment,component_name="AstroGraph confirmatory vLLM runtime"); accelerator=_nvidia_identity()
    readonly_required=bool(spec.get("require_readonly_artifacts")); model_mount=_mount_access_status(model_root); tokenizer_mount=_mount_access_status(tokenizer_root)
    blockers=[]
    if not strong_server_identity: blockers.append("SERVER_BINARY_IDENTITY_NOT_CONTENT_BOUND")
    if accelerator.get("status") != "NVIDIA_GPU_OBSERVED": blockers.append("NVIDIA_GPU_RUNTIME_IDENTITY_NOT_OBSERVED")
    if readonly_required and model_mount.get("status") != "READ_ONLY": blockers.append("MODEL_ARTIFACT_NOT_READ_ONLY")
    if readonly_required and tokenizer_mount.get("status") != "READ_ONLY": blockers.append("TOKENIZER_ARTIFACT_NOT_READ_ONLY")
    if str(spec.get("deployment_kind") or "BARE_METAL") == "CONTAINER" and not str(server.get("container_image_digest") or "").strip(): blockers.append("CONTAINER_IMAGE_DIGEST_MISSING")
    identity={
        "producer_id":spec["producer_id"],"model":model["identity"],"model_identity_sha256":model["identity_sha256"],"tokenizer":tokenizer["identity"],"tokenizer_identity_sha256":tokenizer["identity_sha256"],
        "server":{"implementation":"vllm","declared_version":server.get("version"),"build_id":server.get("build_id"),"container_image_digest":server.get("container_image_digest"),"artifact_identity_sha256":(server_artifact or {}).get("identity_sha256"),"vllm_distribution_identity_sha256":(vllm_distribution or {}).get("identity_sha256"),"vllm_distribution_version":(vllm_distribution or {}).get("version"),"strong_binary_identity_present":strong_server_identity,"launch_config":launch_config,"launch_config_sha256":_canonical_sha256(launch_config)},
        "python_environment_sha256":environment["lock_sha256"],"cyclonedx_sbom_sha256":_canonical_sha256(sbom),"accelerator":accelerator,"deployment_kind":str(spec.get("deployment_kind") or "BARE_METAL"),
        "readonly_artifacts_required":readonly_required,"model_mount_access":model_mount,"tokenizer_mount_access":tokenizer_mount,
    }
    core={"schema_version":MODEL_RUNTIME_ATTESTATION_SCHEMA,"status":"VALID" if not blockers else "INVALID","runtime_identity":identity,"runtime_identity_sha256":_canonical_sha256(identity),
          "audit":{"generated_at":datetime.now(timezone.utc).isoformat(),"model_root":model["root"],"tokenizer_root":tokenizer["root"],"server_artifact_root":(server_artifact or {}).get("root"),"python_environment_lock":environment,"cyclonedx_sbom":sbom,"server_artifact":server_artifact,"vllm_distribution":vllm_distribution},
          "origin_authentication":{"status":"UNVERIFIED","method":"NONE","claim_boundary":"Content identity only. R16 does not yet prove producer authenticity or hardware-rooted remote origin."},
          "blockers":blockers,"claim_boundary":"Content-bound model/tokenizer/server/runtime attestation. VALID binds model/tokenizer bytes, vLLM binary/container and launch configuration, exact Python environment/SBOM, CUDA/GPU/driver identity and optional read-only artifact policy. It does not authenticate a remote producer and does not qualify model performance."}
    payload=dict(core); payload["attestation_sha256"]=_canonical_sha256(core); return payload


def load_model_runtime_attestation(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path).resolve())
    if payload.get("schema_version") != MODEL_RUNTIME_ATTESTATION_SCHEMA:
        raise ValueError("unsupported model runtime attestation schema")
    return payload


def fetch_model_runtime_attestation(
    url: str,
    *,
    timeout_s: float = 10.0,
    max_bytes: int = 32_000_000,
) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("MODEL_RUNTIME_ATTESTATION_RESPONSE_TOO_LARGE")
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("MODEL_RUNTIME_ATTESTATION_RESPONSE_TOO_LARGE")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MODEL_RUNTIME_ATTESTATION_RESPONSE_NOT_OBJECT")
    if payload.get("schema_version") != MODEL_RUNTIME_ATTESTATION_SCHEMA:
        raise ValueError("unsupported model runtime attestation schema")
    return payload


def acquire_model_runtime_attestation(
    *,
    spec_path: str | Path | None = None,
    attestation_url: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if spec_path and attestation_url:
        raise ValueError("choose either local model runtime spec or attestation URL, not both")
    if spec_path:
        spec = load_model_runtime_spec(spec_path)
        return build_model_runtime_attestation(spec), {
            "mode": "LOCAL_RECOMPUTED",
            "refreshable": True,
            "source": str(Path(spec_path).resolve()),
        }
    if attestation_url:
        return fetch_model_runtime_attestation(attestation_url), {
            "mode": "HTTP_REOBSERVED",
            "refreshable": True,
            "source": attestation_url,
        }
    return None, {"mode": "UNCONFIGURED", "refreshable": False, "source": None}


def verify_model_runtime_attestation(attestation: dict[str, Any], *, expected_model_id: str) -> dict[str, Any]:
    errors=[]
    if attestation.get("schema_version") != MODEL_RUNTIME_ATTESTATION_SCHEMA: errors.append("MODEL_RUNTIME_ATTESTATION_SCHEMA_MISMATCH")
    stored=str(attestation.get("attestation_sha256") or ""); core=dict(attestation); core.pop("attestation_sha256",None)
    if not stored or stored != _canonical_sha256(core): errors.append("MODEL_RUNTIME_ATTESTATION_SHA256_INVALID")
    identity=attestation.get("runtime_identity") or {}
    if attestation.get("runtime_identity_sha256") != _canonical_sha256(identity): errors.append("MODEL_RUNTIME_IDENTITY_SHA256_INVALID")
    if str((identity.get("model") or {}).get("label") or "") != str(expected_model_id): errors.append("MODEL_ARTIFACT_ID_MISMATCH")
    if not str((identity.get("model") or {}).get("revision") or "").strip(): errors.append("MODEL_REVISION_MISSING")
    if not str((identity.get("tokenizer") or {}).get("revision") or "").strip(): errors.append("TOKENIZER_REVISION_MISSING")
    if not str(identity.get("model_identity_sha256") or ""): errors.append("MODEL_CONTENT_IDENTITY_MISSING")
    elif identity.get("model_identity_sha256") != _canonical_sha256(identity.get("model") or {}): errors.append("MODEL_CONTENT_IDENTITY_INVALID")
    if not str(identity.get("tokenizer_identity_sha256") or ""): errors.append("TOKENIZER_CONTENT_IDENTITY_MISSING")
    elif identity.get("tokenizer_identity_sha256") != _canonical_sha256(identity.get("tokenizer") or {}): errors.append("TOKENIZER_CONTENT_IDENTITY_INVALID")
    server=identity.get("server") or {}
    if str(server.get("implementation") or "").lower() != "vllm": errors.append("MODEL_SERVER_IMPLEMENTATION_MISMATCH")
    if not bool(server.get("strong_binary_identity_present")): errors.append("SERVER_BINARY_IDENTITY_NOT_CONTENT_BOUND")
    if not str(server.get("launch_config_sha256") or ""): errors.append("SERVER_LAUNCH_CONFIG_IDENTITY_MISSING")
    elif server.get("launch_config_sha256") != _canonical_sha256(server.get("launch_config") or {}): errors.append("SERVER_LAUNCH_CONFIG_IDENTITY_INVALID")
    accelerator=identity.get("accelerator") or {}
    if accelerator.get("status") != "NVIDIA_GPU_OBSERVED" or not accelerator.get("gpus"): errors.append("NVIDIA_GPU_RUNTIME_IDENTITY_NOT_OBSERVED")
    if not str(accelerator.get("torch_cuda_version") or "").strip(): errors.append("CUDA_RUNTIME_IDENTITY_MISSING")
    if not str(identity.get("python_environment_sha256") or ""): errors.append("MODEL_SERVER_PYTHON_ENVIRONMENT_IDENTITY_MISSING")
    audit=attestation.get("audit") or {}
    audit_env=audit.get("python_environment_lock") or {}
    if audit_env and identity.get("python_environment_sha256") != audit_env.get("lock_sha256"): errors.append("MODEL_SERVER_PYTHON_ENVIRONMENT_IDENTITY_INVALID")
    if not str(identity.get("cyclonedx_sbom_sha256") or ""): errors.append("MODEL_SERVER_SBOM_IDENTITY_MISSING")
    audit_sbom=audit.get("cyclonedx_sbom") or {}
    if audit_sbom and identity.get("cyclonedx_sbom_sha256") != _canonical_sha256(audit_sbom): errors.append("MODEL_SERVER_SBOM_IDENTITY_INVALID")
    if identity.get("deployment_kind") == "CONTAINER" and not str(server.get("container_image_digest") or "").strip(): errors.append("CONTAINER_IMAGE_DIGEST_MISSING")
    if identity.get("readonly_artifacts_required"):
        if str((identity.get("model_mount_access") or {}).get("status")) != "READ_ONLY": errors.append("MODEL_ARTIFACT_NOT_READ_ONLY")
        if str((identity.get("tokenizer_mount_access") or {}).get("status")) != "READ_ONLY": errors.append("TOKENIZER_ARTIFACT_NOT_READ_ONLY")
    if attestation.get("status") != "VALID": errors.extend(str(item) for item in (attestation.get("blockers") or []))
    return {"schema_version":"astrograph-confirmatory-model-runtime-attestation-check.v2","status":"STABLE" if not errors else "DRIFTED","drift_codes":sorted(set(errors)),"runtime_identity_sha256":attestation.get("runtime_identity_sha256"),"attestation_sha256":stored or None,"origin_authentication":attestation.get("origin_authentication") or {}}


def _source_integrity_check(root: Path) -> dict[str, Any]:
    script = root / "scripts/verify_source.py"
    if not script.is_file():
        return {"status": "FAIL", "errors": ["SOURCE_VERIFY_SCRIPT_MISSING"]}
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--root", str(root)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        payload = json.loads(completed.stdout) if completed.stdout.strip() else {}
    except Exception as exc:
        return {"status": "FAIL", "errors": [f"SOURCE_VERIFY_EXECUTION_ERROR:{type(exc).__name__}:{exc}"]}
    if completed.returncode != 0 or payload.get("status") != "PASS":
        return {
            "status": "FAIL",
            "errors": list(payload.get("errors") or ["SOURCE_VERIFY_FAILED"]),
            "returncode": completed.returncode,
        }
    return {"status": "PASS", "verified_files": payload.get("verified_files"), "errors": []}


def build_confirmatory_client_identity(root: str | Path, *, environment_lock: dict[str, Any], vllm_url: str, vllm_model: str) -> dict[str, Any]:
    root=Path(root).resolve(); env_check=verify_confirmatory_python_environment_lock(environment_lock)
    source_manifest=root/"SOURCE_MANIFEST.json"; checksums=root/"SHA256SUMS.txt"; prompt_sources=[root/"src/astrograph/llm/openai_compatible.py",root/"src/astrograph/llm/task_prompts.py"]; inference_profile_path=root/"configs/evaluation/llm_method_confirmatory_inference.v1.json"
    missing=[str(path.relative_to(root)) for path in (source_manifest,checksums,inference_profile_path,*prompt_sources) if not path.is_file()]
    inference=build_confirmatory_inference_runtime_snapshot(vllm_url=vllm_url,vllm_model=vllm_model,profile_path=inference_profile_path if inference_profile_path.is_file() else None); source_integrity=_source_integrity_check(root)
    prompt_entries=[{"relative_path":path.relative_to(root).as_posix(),"sha256":_sha256_file(path)} for path in prompt_sources if path.is_file()]
    sbom=build_confirmatory_cyclonedx_sbom(environment_lock,component_name="AstroGraph confirmatory orchestrator")
    identity={"source_manifest_sha256":_sha256_file(source_manifest) if source_manifest.is_file() else None,"source_checksums_sha256":_sha256_file(checksums) if checksums.is_file() else None,
              "prompt_sources":prompt_entries,"prompt_implementation_sha256":_canonical_sha256(prompt_entries),"inference_profile_file_sha256":inference.get("inference_profile_file_sha256"),"inference_profile_canonical_sha256":inference.get("inference_profile_canonical_sha256"),
              "environment_lock_sha256":environment_lock.get("lock_sha256"),"environment_packages_sha256":environment_lock.get("packages_sha256"),"cyclonedx_sbom_sha256":_canonical_sha256(sbom),"environment_cyclonedx_sbom_sha256":_canonical_sha256(sbom),
              "python_executable_sha256":_sha256_file(Path(sys.executable)) if Path(sys.executable).is_file() else None,"python_version":platform.python_version(),"python_implementation":platform.python_implementation(),"platform":platform.platform(),"machine":platform.machine(),
              "inference_config_sha256":inference["inference_config_sha256"],"prompt_profile":inference["prompt_profile"]}
    blockers=[f"CLIENT_IDENTITY_FILE_MISSING:{item}" for item in missing]
    if source_integrity.get("status") != "PASS": blockers.extend(f"SOURCE_INTEGRITY:{item}" for item in (source_integrity.get("errors") or ["FAILED"]))
    if env_check["status"] != "STABLE": blockers.extend(env_check["drift_codes"])
    return {"schema_version":"astrograph-confirmatory-client-runtime-identity.v2","status":"VALID" if not blockers else "INVALID","identity":identity,"identity_sha256":_canonical_sha256(identity),"environment_check":env_check,"source_integrity":source_integrity,"inference_runtime":inference,"cyclonedx_sbom":sbom,"blockers":sorted(set(blockers)),"claim_boundary":"Client/source/runtime identity only. It does not attest the remote model server or model bytes."}


def build_confirmatory_runtime_identity_admission(
    root: str | Path,
    *,
    environment_lock: dict[str, Any] | None,
    model_runtime_attestation: dict[str, Any] | None,
    vllm_url: str,
    vllm_model: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    client = None
    model_check = None
    model_runtime_identity_sha256 = None
    if environment_lock is None:
        blockers.append("CLIENT_ENVIRONMENT_LOCK_MISSING")
    else:
        client = build_confirmatory_client_identity(
            root,
            environment_lock=environment_lock,
            vllm_url=vllm_url,
            vllm_model=vllm_model,
        )
        if client["status"] != "VALID":
            blockers.extend(f"CLIENT:{item}" for item in client.get("blockers") or [])
    if model_runtime_attestation is None:
        blockers.append("MODEL_RUNTIME_ATTESTATION_MISSING")
    else:
        model_check = verify_model_runtime_attestation(model_runtime_attestation, expected_model_id=vllm_model)
        model_runtime_identity_sha256 = model_runtime_attestation.get("runtime_identity_sha256")
        if model_check["status"] != "STABLE":
            blockers.extend(f"MODEL_RUNTIME:{item}" for item in model_check.get("drift_codes") or [])
    identity = {
        "client_identity_sha256": (client or {}).get("identity_sha256"),
        "model_runtime_identity_sha256": model_runtime_identity_sha256,
        "inference_config_sha256": ((client or {}).get("inference_runtime") or {}).get("inference_config_sha256"),
    }
    return {
        "schema_version": RUNTIME_ADMISSION_SCHEMA,
        "status": "VALID" if not blockers else "BLOCKED",
        "runtime_identity": identity,
        "runtime_identity_sha256": _canonical_sha256(identity),
        "client": client,
        "model_runtime_check": model_check,
        "model_runtime_attestation_sha256": (model_runtime_attestation or {}).get("attestation_sha256"),
        "blockers": sorted(set(blockers)),
        "claim_boundary": (
            "Pre/post confirmatory runtime identity admission. VALID binds AstroGraph source, exact Python environment, CycloneDX SBOM, "
            "frozen inference configuration, model bytes, tokenizer bytes, model-server binary/container identity and observed accelerator/runtime metadata. "
            "Remote producer authentication remains a separate governance problem."
        ),
    }


def verify_confirmatory_runtime_identity_drift(
    *,
    pre_admission: dict[str, Any],
    post_admission: dict[str, Any],
) -> dict[str, Any]:
    drifts: list[str] = []
    if pre_admission.get("status") != "VALID":
        drifts.append("PRE_RUNTIME_IDENTITY_NOT_VALID")
    if post_admission.get("status") != "VALID":
        drifts.append("POST_RUNTIME_IDENTITY_NOT_VALID")
        drifts.extend(f"POST_BLOCKER:{item}" for item in (post_admission.get("blockers") or []))
    pre = pre_admission.get("runtime_identity") or {}
    post = post_admission.get("runtime_identity") or {}
    mapping = {
        "client_identity_sha256": "CLIENT_RUNTIME_IDENTITY_DRIFT",
        "model_runtime_identity_sha256": "MODEL_RUNTIME_IDENTITY_DRIFT",
        "inference_config_sha256": "INFERENCE_CONFIG_IDENTITY_DRIFT",
    }
    for key, code in mapping.items():
        if pre.get(key) != post.get(key):
            drifts.append(code)
    if pre_admission.get("model_runtime_attestation_sha256") == post_admission.get("model_runtime_attestation_sha256"):
        # Equality is allowed; the model-host sidecar may regenerate a byte-identical attestation.
        pass
    return {
        "schema_version": "astrograph-confirmatory-runtime-identity-drift-check.v1",
        "status": "STABLE" if not drifts else "DRIFTED",
        "drift_codes": sorted(set(drifts)),
        "pre_runtime_identity_sha256": pre_admission.get("runtime_identity_sha256"),
        "post_runtime_identity_sha256": post_admission.get("runtime_identity_sha256"),
    }
