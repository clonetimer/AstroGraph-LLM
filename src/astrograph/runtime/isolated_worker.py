from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from .artifacts import RuntimeArtifactStore
from .contracts import RuntimeStageContext, StageResult
from .execution_control import CancellationToken
from .isolation import IsolatedStageInvocation, IsolatedStageResponse


def _build_stage(invocation: IsolatedStageInvocation):
    if invocation.adapter_id == "whole_spacecraft.online.v1":
        from .factory import _reviewed_online_workflow_factory
        from .stages import WholeSpacecraftOnlineDagStage

        root = Path(invocation.project_root).resolve()
        return WholeSpacecraftOnlineDagStage(
            invocation.stage_id,
            _reviewed_online_workflow_factory(root),
            final_stage=(invocation.stage_id == "online.persist.v1"),
        )
    if invocation.adapter_id == "_qualification.uncooperative_sleep.v1":
        class _UncooperativeSleepStage:
            stage_id = invocation.stage_id
            stage_version = invocation.stage_version

            def execute(self, payload, context):
                time.sleep(float(payload.get("sleep_s", 5.0)))
                return StageResult.succeeded(dict(payload))

        return _UncooperativeSleepStage()
    if invocation.adapter_id == "_qualification.crash.v1":
        class _CrashStage:
            stage_id = invocation.stage_id
            stage_version = invocation.stage_version

            def execute(self, payload, context):
                os._exit(int(payload.get("exit_code", 23)))

        return _CrashStage()
    raise ValueError(f"unknown isolated stage adapter: {invocation.adapter_id}")


def run(invocation: IsolatedStageInvocation) -> IsolatedStageResponse:
    stage = _build_stage(invocation)
    if stage.stage_id != invocation.stage_id or stage.stage_version != invocation.stage_version:
        raise ValueError("isolated worker reconstructed a mismatched stage identity")
    artifacts = RuntimeArtifactStore(execution_scope=invocation.execution_scope)
    artifacts.preload(invocation.upstream_artifacts)
    deadline = None
    if invocation.stage_profile.timeout_s is not None:
        deadline = time.monotonic() + invocation.stage_profile.timeout_s
    token = CancellationToken(deadline_monotonic=deadline)
    context = RuntimeStageContext(
        project_root=Path(invocation.project_root).resolve(),
        metadata=dict(invocation.metadata),
        scratch={},
        artifacts=artifacts,
        cancellation=token,
        execution_policy=invocation.execution_policy,
        stage_profile=invocation.stage_profile,
    )
    try:
        token.checkpoint()
        result = stage.execute(dict(invocation.payload), context)
        token.checkpoint()
    except Exception as exc:
        result = StageResult.failed(
            code=getattr(exc, "code", "ASTROGRAPH_ISOLATED_STAGE_UNHANDLED_EXCEPTION"),
            message=getattr(exc, "message", f"{type(exc).__name__}: {exc}"),
            stage_id=invocation.stage_id,
            details=getattr(exc, "details", {}),
        )
    return IsolatedStageResponse(worker_pid=os.getpid(), result=result)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m astrograph.runtime.isolated_worker INVOCATION_JSON RESPONSE_JSON", file=sys.stderr)
        return 2
    invocation_path = Path(args[0])
    response_path = Path(args[1])
    invocation = IsolatedStageInvocation.model_validate_json(invocation_path.read_text(encoding="utf-8"))
    response = run(invocation)
    response_path.write_text(response.model_dump_json(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
