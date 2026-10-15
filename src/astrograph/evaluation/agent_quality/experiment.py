"""Frozen, paired, sequential repeated-case trials using the EXISTING MissionAgent.

Runs do not tune policies, train models or write case memory. Live requests need
an explicit flag. Error, partial, interrupted and unrun cells remain visible.
"""
from __future__ import annotations
import importlib.metadata
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from astrograph.agents.mission.models import AgentRequest, AgentSettings
from astrograph.agents.mission.modes import resolve_mode
from .io import read_json, write_new, file_sha, sha, safe_child
from .metrics import inspect_task, aggregate, review_template


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_fingerprint(project: Path) -> dict:
    files = [project / "pyproject.toml"] + sorted((project / "src" / "astrograph").rglob("*.py"))
    files += sorted((project / "scripts" / "acceptance").glob("*.py"))
    return {str(p.relative_to(project)): file_sha(p) for p in files}


def copy_asset(source: Path, target: Path) -> str:
    source = Path(source).resolve(strict=True)
    if not source.is_file(): raise ValueError("REGULAR_INPUT_FILE_REQUIRED")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as out, source.open("rb") as inp:
        shutil.copyfileobj(inp, out)
    if file_sha(source) != file_sha(target): raise ValueError("INPUT_CHANGED_DURING_COPY")
    return file_sha(target)


def create_experiment(root: Path, *, project: Path, settings_path: Path, cases: list[dict],
                      repeats: int = 10, modes: tuple[str,...] = ("thinking", "creative"),
                      model_identity: dict | None = None, check_budget: int | None = None) -> dict:
    """Host-side preparation. cases contain ONLY request paths, not answer labels.

    Host grants are not broadened to network/training; copied input roots are
    explicitly granted for read-only testing. Private configuration is not copied.
    """
    root, project, settings_path = root.resolve(), project.resolve(), settings_path.resolve()
    if root.exists(): raise FileExistsError("NEW_EXPERIMENT_DIRECTORY_REQUIRED")
    if not 1 <= repeats <= 50 or not 1 <= len(cases) <= 24: raise ValueError("INVALID_EXPERIMENT_SIZE")
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= {"normal","thinking","creative"}:
        raise ValueError("INVALID_WORK_MODES")
    settings = AgentSettings.model_validate(read_json(settings_path))
    if check_budget is not None and (type(check_budget) is not int or not 0 <= check_budget <= 20):
        raise ValueError("INVALID_CHECK_BUDGET_CAP")
    if settings.planner != "llm_assisted" or not settings.llm_config:
        raise ValueError("REAL_LLM_SETTINGS_REQUIRED_FOR_PAIRED_EXPERIMENT")
    llm = Path(settings.llm_config).resolve(strict=True)
    if not any(llm.is_relative_to(Path(p).resolve()) for p in settings.read_roots):
        raise PermissionError("LLM_CONFIG_OUTSIDE_HOST_GRANTS")
    requests = []
    for case in cases:
        if set(case) != {"case_id", "request_path", "scope"}: raise ValueError("CASE_KEYS_MUST_NOT_CONTAIN_TRUTH_OR_JUDGMENTS")
        cid = case["case_id"]
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", cid): raise ValueError("BAD_CASE_ID")
        req = AgentRequest.model_validate(read_json(Path(case["request_path"])))
        if req.task != "diagnose": raise ValueError("QUALITY_EXPERIMENT_IS_READ_ONLY_DIAGNOSIS_ONLY")
        if not set(modes) <= set(settings.allowed_work_modes): raise PermissionError("MODE_NOT_HOST_AUTHORIZED")
        requests.append((case, req))
    if len({c["case_id"] for c, _ in requests}) != len(cases): raise ValueError("DUPLICATE_CASE_ID")
    root.mkdir(parents=True, exist_ok=False)
    pins, normalized = {}, []
    for case, req in requests:
        directory = root / "inputs" / case["case_id"]
        artifacts = {}
        for key, name in req.artifacts.items():
            source = Path(name).resolve(strict=True)
            if key == "graph_dir":
                target = directory / "graph"
                for child in ("nodes.csv", "edges.csv"):
                    p = target / child
                    pins[str(p.relative_to(root))] = copy_asset(source / child, p)
                artifacts[key] = str(target)
            else:
                p = directory / (key + source.suffix)
                pins[str(p.relative_to(root))] = copy_asset(source, p)
                artifacts[key] = str(p)
        docs = []
        for i, name in enumerate(req.documents):
            source = Path(name).resolve(strict=True)
            p = directory / (f"document-{i:03d}" + source.suffix)
            pins[str(p.relative_to(root))] = copy_asset(source, p)
            docs.append(str(p))
        req = AgentRequest.model_validate({**req.model_dump(mode="json"), "task_id": case["case_id"],
                                          "artifacts": artifacts, "documents": docs, "case": None})
        name = f"inputs/{case['case_id']}/request.json"
        write_new(root / name, req.model_dump(mode="json")); pins[name] = file_sha(root / name)
        normalized.append({"case_id": case["case_id"], "request": name, "scope": case["scope"]})
    cells = []
    for repeat in range(1, repeats+1):
        # Counterbalanced AB/BA ordering, never choose the better observed seed.
        planners = ("deterministic", "llm_assisted") if repeat % 2 else ("llm_assisted", "deterministic")
        for case in normalized:
            for mode in modes:
                for planner in planners:
                    key = f"{case['case_id']}-{mode}-{'llm' if planner=='llm_assisted' else 'det'}-{repeat:03d}"
                    cells.append({"cell_id":key, "case_id":case["case_id"], "request":case["request"],
                                  "mode":mode, "planner":planner, "repeat":repeat})
    # Freeze resource budgets by mode; both planners receive the same limits.
    mode_budgets = {}
    for mode in modes:
        resolved, _ = resolve_mode(settings, AgentRequest(task_id="budget-audit", work_mode=mode), backend_available=True)
        mode_budgets[mode] = {k:getattr(resolved,k) for k in
                             ("max_checks","max_optional_actions","max_llm_calls","max_http_attempts","max_plan_revisions","task_timeout_s")}
        if check_budget is not None:mode_budgets[mode]["max_checks"]=min(check_budget,mode_budgets[mode]["max_checks"])
    raw_config=read_json(llm)
    safe_parameters={k:raw_config.get(k) for k in ("provider","model","temperature","max_tokens","context_length","response_format_strategy","repair_attempts","tokenizer_mode")}
    manifest = {"schema_version":"astrograph-agent-quality-experiment.v1", "created_at":now(),
                "project":str(project), "host_settings_path":str(settings_path), "host_settings_sha256":file_sha(settings_path),
                "llm_config_path":str(llm), "llm_config_sha256":file_sha(llm),
                "runtime_files":runtime_fingerprint(project), "input_files":pins,
                "cases":normalized, "cells":cells, "repeats":repeats, "mode_budgets":mode_budgets,
                "case_memory":"DISABLED_FOR_ALL_ARMS_NO_WRITES", "check_budget_cap":check_budget, "declared_generation_parameters":safe_parameters, "order":"ALTERNATING_AB_BA_SEQUENTIAL",
                "live_service_access":"REQUIRES_EXPLICIT_ALLOW_LIVE", "ground_truth_in_planner":False,
                "backend_identity":model_identity or {"status":"MODEL_REVISION_AND_TEMPLATE_HASH_NOT_SUPPLIED"},
                "backend_identity_is_independently_verified":False,
                "limits":"Repeated-case integration and investigation measurements, not new holdout accuracy or optimal-policy certification."}
    write_new(root / "experiment.json", manifest)
    write_new(root / "experiment-receipt.json", {"sha256":file_sha(root / "experiment.json")})
    return manifest


def verify_experiment(root: Path, *, verify_runtime=True) -> dict:
    root = root.resolve()
    manifest = read_json(root / "experiment.json")
    if file_sha(root / "experiment.json") != read_json(root / "experiment-receipt.json")["sha256"]:
        raise ValueError("EXPERIMENT_MANIFEST_CHANGED")
    for name, digest in manifest["input_files"].items():
        if file_sha(safe_child(root, name)) != digest: raise ValueError("EXPERIMENT_INPUT_CHANGED:" + name)
    if verify_runtime:
        for path_key, hash_key in (("host_settings_path","host_settings_sha256"),("llm_config_path","llm_config_sha256")):
            if file_sha(Path(manifest[path_key])) != manifest[hash_key]: raise ValueError("PRIVATE_CONFIG_CHANGED_START_NEW_EXPERIMENT")
        if runtime_fingerprint(Path(manifest["project"])) != manifest["runtime_files"]:
            raise ValueError("RUNTIME_CHANGED_START_NEW_EXPERIMENT")
    return manifest


def run_experiment(root: Path, *, allow_live=False, max_runs: int | None=None, backend_factory=None) -> dict:
    """backend_factory is only for explicit Python tests; never exposed by CLI."""
    from astrograph.agents.mission.service import MissionAgent
    root = root.resolve();manifest = verify_experiment(root)
    if not allow_live and backend_factory is None: raise PermissionError("ALLOW_LIVE_REQUIRED")
    if max_runs is not None and max_runs < 1: raise ValueError("MAX_RUNS_MUST_BE_POSITIVE")
    # Lock orchestration too; production MissionAgent separately locks each task.
    from astrograph.agents.mission.run_lock import task_lease
    with task_lease(root, "quality-experiment"):
        started = 0
        for cell in manifest["cells"]:
            d = root / "trials" / cell["cell_id"]
            if (d / "receipt.json").exists():
                _verify_cell(d); continue
            if d.exists():
                raise RuntimeError("INTERRUPTED_TRIAL_REQUIRES_REVIEW:"+cell["cell_id"])
            if max_runs is not None and started >= max_runs: break
            verify_experiment(root)
            d.mkdir(parents=True, exist_ok=False)
            write_new(d / "started.json", {**cell,"started_at":now(),"scope":"SCRIPTED_FIXTURE" if backend_factory else "CONFIGURED_LIVE_BACKEND_REQUESTED"})
            request = AgentRequest.model_validate(read_json(root / cell["request"]))
            request = request.model_copy(update={"task_id":cell["cell_id"], "work_mode":cell["mode"], "case":None})
            settings = AgentSettings.model_validate(read_json(Path(manifest["host_settings_path"])))
            settings = settings.model_copy(update={"project_root":manifest["project"], "output_root":str(d/"tasks"),
                "read_roots":list(dict.fromkeys([*settings.read_roots, str((root/cell["request"]).parent)])),
                "planner":cell["planner"], "memory_db":None,"memory_policy":None,"resume_read_only":False,
                "approved_training_manifest_sha256":[]})
            if manifest.get("check_budget_cap") is not None:
                from astrograph.agents.mission.models import ModeBudget
                cap=manifest["check_budget_cap"]
                settings=settings.model_copy(update={"max_checks":min(settings.max_checks,cap),
                    "work_mode_budgets":{m: v.model_copy(update={"max_checks":min(v.max_checks if v.max_checks is not None else settings.max_checks,cap)}) for m,v in settings.work_mode_budgets.items()}})
            # Do not copy credentials or synthesize private config; public pins suffice.
            write_new(d / "request.json", request.model_dump(mode="json"))
            t0=time.monotonic(); record={**cell,"status":"RUNNER_ERROR", "quality":None}
            try:
                backend=backend_factory(cell) if backend_factory and cell["planner"]=="llm_assisted" else None
                agent=MissionAgent(settings,backend=backend)
                result=agent.run(request)
                task_dir=Path(settings.output_root)/request.task_id
                verify_experiment(root)
                quality=inspect_task(task_dir)
                write_new(d/"quality.json",quality)
                write_new(d/"human-review.template.json",review_template(quality))
                record.update(status=result.status,quality=quality,task_dir=str(task_dir.relative_to(root)))
            except Exception as exc:
                # Do not serialize exceptions which may contain credentials / service bodies.
                record.update(error_type=type(exc).__name__, error_detail="Inspect retained task audit; no automatic retry or silent exclusion.")
            record.update(elapsed_s=time.monotonic()-t0,finished_at=now())
            write_new(d/"trial.json",record)
            write_new(d/"receipt.json",{"sha256":file_sha(d/"trial.json")})
            started+=1
        return score_experiment(root)


def _verify_cell(directory: Path) -> dict:
    r=read_json(directory/"trial.json")
    if file_sha(directory/"trial.json")!=read_json(directory/"receipt.json")["sha256"]:
        raise ValueError("TRIAL_RECEIPT_MISMATCH")
    if r.get("quality"):
        actual=inspect_task(directory/"tasks"/r["cell_id"])
        if sha(actual)!=sha(r["quality"]):raise ValueError("TRIAL_QUALITY_CHANGED")
    return r


def score_experiment(root: Path) -> dict:
    root=root.resolve();m=verify_experiment(root,verify_runtime=False);rows=[]
    for c in m["cells"]:
        d=root/"trials"/c["cell_id"]
        if (d/"receipt.json").exists(): row=_verify_cell(d)
        else: row={**c,"status":"INTERRUPTED" if d.exists() else "NOT_RUN","quality":None}
        rows.append(row)
    out=aggregate(rows)
    out.update(experiment_sha256=file_sha(root/"experiment.json"), scheduled=len(rows),
               status_counts={s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})},
               generation_requests_sent=0, model_identity=m["backend_identity"], mode_budgets=m["mode_budgets"],
               rows=rows)
    return out
