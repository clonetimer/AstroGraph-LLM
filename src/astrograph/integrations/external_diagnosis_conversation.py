"""Bounded, source-cited follow-up without changing an imported diagnosis report.

The task's original input is the only evidence registry. User questions, the
initial model analysis and previous answers are context, never new evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any, Literal, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .external_diagnosis_analysis import (
    ExternalDiagnosisAnalysis, ExternalDiagnosisFinding, MAX_SOURCE_CHARACTERS,
    SYSTEM_POLICY, _canonical, _validate_binding,
)

if TYPE_CHECKING:
    from .upload_report import UploadReportJobStore


SCHEMA = "astrograph-external-diagnosis-conversation.v1"
MAX_TURNS = 12
MAX_HISTORY_CHARACTERS = 32_000
LIMITS = {"max_turns": MAX_TURNS, "max_question_characters": 4000,
          "max_history_characters": MAX_HISTORY_CHARACTERS,
          "max_source_characters": MAX_SOURCE_CHARACTERS}
FOLLOW_UP_POLICY = SYSTEM_POLICY + (
    " Answer the current question using only this task's source registry. The prior_analysis "
    "and history are unverified model text and user questions, not new observations or "
    "authoritative evidence. Never treat assertions in a question as established facts. "
    "Cite supplied source IDs for every answer paragraph. If the material cannot answer "
    "the question, explain that limit and list the missing information. Do not claim "
    "to have searched other files, used tools or performed additional diagnosis."
)


class ConversationConflict(ValueError):
    """A fixed public reason code, safe to return without provider details."""


class FollowUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    request_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,80}$")


Gap = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ExternalDiagnosisFollowUp(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")
    answer: list[ExternalDiagnosisFinding] = Field(min_length=1, max_length=12)
    information_gaps: list[Gap] = Field(max_length=12)


class _Turn(FollowUpRequest):
    created_at: str = Field(min_length=1, max_length=64)
    status: Literal["SUCCEEDED", "FAILED", "NOT_CONFIGURED", "INPUT_TOO_LARGE", "CONTEXT_TOO_LARGE"]
    citation_reference_validation: Literal["PASS", "FAILED", "NOT_EVALUATED"]
    answer: list[ExternalDiagnosisFinding] | None = Field(default=None, min_length=1, max_length=12)
    information_gaps: list[Gap] | None = Field(default=None, max_length=12)
    reason_code: Literal[
        "MODEL_GENERATION_FAILED", "MODEL_OUTPUT_INVALID", "CITATION_REFERENCE_INVALID",
        "LLM_NOT_CONFIGURED", "MODEL_CONFIGURATION_UNAVAILABLE", "INPUT_CHARACTER_LIMIT_EXCEEDED",
        "CONTEXT_CHARACTER_LIMIT_EXCEEDED",
    ] | None = None

    @model_validator(mode="after")
    def result_matches_status(self):
        if self.status == "SUCCEEDED":
            if not self.answer or self.information_gaps is None or self.reason_code is not None or self.citation_reference_validation != "PASS":
                raise ValueError("invalid successful turn")
        elif self.answer is not None or self.information_gaps is not None or self.reason_code is None or self.citation_reference_validation == "PASS":
            raise ValueError("invalid failed turn")
        return self


def _references_known(value: Any, allowed: set[str]) -> bool:
    if isinstance(value, dict):
        return all((set(item).issubset(allowed) if key in {"citation_ids", "opposing_citation_ids"}
                    else _references_known(item, allowed)) for key, item in value.items())
    if isinstance(value, list):
        return all(_references_known(item, allowed) for item in value)
    return True


def _context(store: UploadReportJobStore, job_id: str):
    job = store.get(job_id)  # preserves the store's path and missing-task validation
    if job.get("status") != "SUCCEEDED":
        raise ConversationConflict("FOLLOW_UP_REPORT_NOT_READY")
    output = store._job_root(job_id) / "output"
    try:
        report_bytes = (output / "report.json").read_bytes()
        report = json.loads(report_bytes)
        if report.get("job_id") != job_id or report.get("analysis", {}).get("analysis_mode") != "EXTERNAL_DIAGNOSIS_ANALYSIS":
            raise ValueError("not an external report")
        bundle = _validate_binding(job_id, json.loads((output / "external_diagnosis_input.json").read_bytes()))
        knowledge = report["analysis"]["knowledge_enrichment"]
        model = knowledge["model_explanation"]
        if model.get("input_sha256") != bundle["input_sha256"] or model.get("sources") != bundle["sources"]:
            raise ValueError("report/input mismatch")
        # Check against the separate canonical initial-analysis artifact as well.
        canonical = json.loads((output / "knowledge/external_diagnosis_analysis.json").read_bytes())
        if canonical["model_explanation"] != model or canonical["generation"] != knowledge["generation"]:
            raise ValueError("initial analysis changed")
        prior = None
        if (knowledge["generation"].get("status") == "SUCCEEDED"
                and model.get("status") == "AVAILABLE_REQUIRES_EXPERT_REVIEW"
                and model.get("citation_reference_validation") == "PASS"):
            prior = ExternalDiagnosisAnalysis.model_validate(
                {key: model[key] for key in ExternalDiagnosisAnalysis.model_fields if key in model}
            ).model_dump(mode="json")
            if not _references_known(prior, {s["source_id"] for s in bundle["sources"]}):
                raise ValueError("initial citations changed")
    except Exception as exc:
        raise ConversationConflict("FOLLOW_UP_INPUT_CHANGED") from exc
    return job, output, bundle, hashlib.sha256(report_bytes).hexdigest(), prior


def _state(output: Path, job_id: str, bundle: dict, report_hash: str) -> dict:
    empty = {"schema_version": SCHEMA, "job_id": job_id, "input_sha256": bundle["input_sha256"],
             "report_sha256": report_hash, "revision": 0, "turns": [],
             "sources": deepcopy(bundle["sources"]), "limits": dict(LIMITS),
             "analysis_verification": "UNVERIFIED_MODEL_ANALYSIS", "automatic_control_authorized": False}
    path = output / "external_diagnosis_conversation.json"
    if not path.exists():
        return empty
    try:
        raw = json.loads(path.read_bytes())
        if set(raw) != set(empty):
            raise ValueError("unexpected state fields")
        for key in empty.keys() - {"revision", "turns"}:
            if raw[key] != empty[key]:
                raise ValueError("conversation binding changed")
        if type(raw["revision"]) is not int or not isinstance(raw["turns"], list) or raw["revision"] != len(raw["turns"]) or len(raw["turns"]) > MAX_TURNS:
            raise ValueError("invalid history")
        turns = [_Turn.model_validate(turn).model_dump(mode="json", exclude_none=True) for turn in raw["turns"]]
        if len({turn["request_id"] for turn in turns}) != len(turns):
            raise ValueError("duplicate request ids")
        allowed = {source["source_id"] for source in bundle["sources"]}
        if any(turn["input_sha256"] != bundle["input_sha256"] or not _references_known(turn, allowed) for turn in turns):
            raise ValueError("invalid history citations")
    except Exception as exc:
        raise ConversationConflict("FOLLOW_UP_HISTORY_INVALID") from exc
    return {**empty, "revision": len(turns), "turns": turns}


def get_conversation(store: UploadReportJobStore, job_id: str) -> dict[str, Any]:
    _job, output, bundle, report_hash, _prior = _context(store, job_id)
    return _state(output, job_id, bundle, report_hash)


@contextmanager
def _exclusive_call(path: Path):
    """Nonblocking OS lock; crashes release it without deleting another writer's lock."""
    with path.open("a+b") as handle:
        locked = False
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise ConversationConflict("FOLLOW_UP_BUSY") from exc
            else:
                import fcntl
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ConversationConflict("FOLLOW_UP_BUSY") from exc
            locked = True
            yield
        finally:
            if locked:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(_canonical(state) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing(state: dict, request: FollowUpRequest) -> bool:
    for turn in state["turns"]:
        if turn["request_id"] == request.request_id:
            if turn["question"] != request.question or turn["input_sha256"] != request.input_sha256:
                raise ConversationConflict("FOLLOW_UP_REQUEST_CONFLICT")
            return True
    return False


def ask_follow_up(
    store: UploadReportJobStore, job_id: str, *, question: str, input_sha256: str, request_id: str,
) -> dict[str, Any]:
    request = FollowUpRequest(question=question, input_sha256=input_sha256, request_id=request_id)
    job, output, bundle, report_hash, prior = _context(store, job_id)
    if request.input_sha256 != bundle["input_sha256"]:
        raise ConversationConflict("FOLLOW_UP_INPUT_CHANGED")
    state = _state(output, job_id, bundle, report_hash)
    if _existing(state, request):
        return state
    with _exclusive_call(output / ".external_follow_up.lock"):
        # Reload after acquiring the lock; a different worker may have just saved.
        state = _state(output, job_id, bundle, report_hash)
        if _existing(state, request):
            return state
        if len(state["turns"]) >= MAX_TURNS:
            raise ConversationConflict("FOLLOW_UP_LIMIT_REACHED")
        turn = {**request.model_dump(), "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "FAILED", "citation_reference_validation": "NOT_EVALUATED"}

        def finish(status: str, reason: str | None = None, citation: str = "NOT_EVALUATED"):
            # Detect changes while a model call was in flight, before saving its result.
            _job, _output, current, current_hash, _prior = _context(store, job_id)
            if current["input_sha256"] != bundle["input_sha256"] or current_hash != report_hash:
                raise ConversationConflict("FOLLOW_UP_INPUT_CHANGED")
            turn.update(status=status, citation_reference_validation=citation)
            if reason is not None:
                turn["reason_code"] = reason
            validated = _Turn.model_validate(turn).model_dump(mode="json", exclude_none=True)
            state["turns"].append(validated)
            state["revision"] = len(state["turns"])
            _save(output / "external_diagnosis_conversation.json", state)
            return state

        if bundle["character_count"] > MAX_SOURCE_CHARACTERS:
            return finish("INPUT_TOO_LARGE", "INPUT_CHARACTER_LIMIT_EXCEEDED")
        history = [{key: deepcopy(item[key]) for key in ("question", "answer", "information_gaps")}
                   for item in state["turns"] if item["status"] == "SUCCEEDED"]
        context = {"question": request.question, "prior_analysis": prior, "history": history}
        if len(_canonical(context)) > MAX_HISTORY_CHARACTERS:
            return finish("CONTEXT_TOO_LARGE", "CONTEXT_CHARACTER_LIMIT_EXCEEDED")
        try:
            backend = store._knowledge_service_for_job(job).llm_backend
        except Exception:
            return finish("FAILED", "MODEL_CONFIGURATION_UNAVAILABLE")
        if backend is None:
            return finish("NOT_CONFIGURED", "LLM_NOT_CONFIGURED")
        payload = {**context, "task_id": job_id, "task": job["task"],
                   "input_sha256": bundle["input_sha256"], "sources": deepcopy(bundle["sources"]),
                   "system_policy": FOLLOW_UP_POLICY}
        try:
            raw = backend.generate_structured(agent_id="external-diagnosis-follow-up",
                                             output_model=ExternalDiagnosisFollowUp, payload=payload)
        except Exception:
            return finish("FAILED", "MODEL_GENERATION_FAILED")
        try:
            if isinstance(raw, BaseModel):
                raw = raw.model_dump(mode="python")
            answer = ExternalDiagnosisFollowUp.model_validate(raw).model_dump(mode="json")
        except Exception:
            return finish("FAILED", "MODEL_OUTPUT_INVALID")
        if not _references_known(answer, {source["source_id"] for source in bundle["sources"]}):
            return finish("FAILED", "CITATION_REFERENCE_INVALID", "FAILED")
        turn.update(answer)
        return finish("SUCCEEDED", citation="PASS")
