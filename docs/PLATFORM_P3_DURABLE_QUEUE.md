# AstroGraph-LLM P3 — durable full-platform job queue

Status: **development candidate / Draft PR #5**, pending final release qualification.
Scope: full-platform upload-to-report execution; the focused portable workbench retains its own queue implementation. The existing upstream diagnosis source is immutable: the external-analysis route **does not compute or retrain diagnosis**.

## 1. Architecture

```text
POST /v1/platform/jobs/<id>/start
  -> upload input receipt / preflight
  -> SQLite durable queue (platform_job_queue.sqlite3, WAL)
  -> up to N detached worker slots with OS-owned locks
  -> BEGIN IMMEDIATE / atomic oldest-queued claim
  -> worker heartbeat and lease
  -> source-bound analysis (original report + optional reviewed context)
  -> publication guard in SQLite
  -> immutable output/report.json, report.md + execution journal
  -> queue terminal state and UI polling
```

**Source of truth** for claim, lease, cancellation and finalization is SQLite, not Python thread maps or the legacy job index. `job.json` preserves human-readable task details and report references. A queued worker is never given new authority over candidate labels, source material or model profiles.

The worker uses `astrograph.integrations.platform_worker --root <platform-root> --queue-slot <number>`. On API startup, ASGI lifespan checks for pending work and starts worker slots as needed. Additional instances may attempt to start the same slot; the OS-owned lock allows only one process to serve it, and closes on process exit.

## 2. Runtime configuration

| Setting | Default | Behavior |
|---|---:|---|
| `ASTROGRAPH_PLATFORM_ROOT` | `~/.astrograph/platform` | Local job and queue root |
| `ASTROGRAPH_PLATFORM_QUEUE_WORKERS` | `2` | Concurrent slots, clamped to 1–8 |
| `ASTROGRAPH_PLATFORM_QUEUE_LEASE_SECONDS` | `30` | Worker ownership TTL, minimum 3 seconds |
| `ASTROGRAPH_PLATFORM_QUEUE_HEARTBEAT_SECONDS` | `5` | Cooperative cancellation poll/heartbeat interval |
| `PYTHONUTF8` | recommended `1` | UTF-8 output on Windows consoles |

Database transactions use `BEGIN IMMEDIATE` and SQLite's busy timeout. Each claim has a worker ID and attempt counter. Tokens and API keys are **not** stored in the queue.

The older `ASTROGRAPH_PLATFORM_JOB_EXECUTOR=thread/process` choice is not used by `start()` on the full-platform feature branch: `PROCESS_QUEUE` owns asynchronous dispatch. The direct synchronous `/run` route remains available for explicitly invoked legacy operations and existing source compatibility tests, but must not bypass an already queued task.

## 3. States and API

| State | Meaning | Operator action |
|---|---|---|
| `QUEUED` | Persisted; awaiting an atomic claim | Cancel immediately |
| `RUNNING` | Worker lease active | Request cooperative cancellation |
| `CANCEL_REQUESTED` | Cancellation persisted, awaiting a safe boundary | Poll for terminal state |
| `COMMITTING` | Publication guard entered; too late to cancel | Wait for terminal receipt |
| `SUCCEEDED` | Report published and execution finalized | Open/download report |
| `FAILED` | Execution error with recorded reason | Investigate; explicit retry may be available |
| `CANCELLED` | Cancellation completed, no report publication authorized | Create new task if required |
| `INTERRUPTED` | Worker disappeared or lease expired | Inspect artifacts, explicitly resume |

API:
- `GET /v1/platform/jobs/{job_id}`: job detail plus `queue` snapshot
- `GET /v1/platform/jobs/{job_id}/diagnostics`: status, logs and queue snapshot
- `POST /v1/platform/jobs/{job_id}/start`: persist one queued request
- `POST /v1/platform/jobs/{job_id}/cancel` with `{"reason":"USER_REQUEST"}`: immediate cancellation of queued jobs, cooperative request for running jobs
- `POST /v1/platform/jobs/{job_id}/resume` with `{"confirm_duplicate_risk":true}`: explicit operator-confirmed retry of an interrupted task

These mutating endpoints require operator privileges under configured authentication. A successful cancellation-request HTTP response does **not** prove that the provider call stopped; observe the terminal queue state.

## 4. Crash handling and duplicate model costs

An expired `RUNNING` / `COMMITTING` lease becomes `INTERRUPTED`; it is **never automatically replayed**. A `CANCEL_REQUESTED` lease expiry becomes `CANCELLED`. The operator can inspect original upload receipts, the execution journal, partial output, and worker logs.

If both complete report files exist and the job identity matches, a completed interrupted task can be recovered as `SUCCEEDED` without another model call. Otherwise, `resume` requires the explicit duplicate-cost confirmation. The abandoned output is archived under `interrupted-attempts/attempt-NNN` before a new attempt starts.

A bounded `knowledge/checkpoint.json` manifest is copied into the new output directory when available. The existing hierarchical analyzer must independently validate task/input/chunk/context/model fingerprints and stored summaries before reusing a completed chunk. Incomplete or mismatched chunks are regenerated, and might incur charges. No credential is reloaded from old response files.

**Cancellation is cooperative.** A provider request can finish and be billed after the operator cancels; a guard before report publication prevents later results from being promoted to the official completed report. `COMMITTING` rejects late cancel requests rather than implying success.

## 5. Recovery and safety boundaries

- Never delete `platform_job_queue.sqlite3`, `job.json`, review snapshots or `input/` while tasks are active.
- Keep the whole platform root on a filesystem supporting reliable SQLite locking and atomic local file operations. Network shares and synced folders require independent qualification.
- Keep OS/process logs and `interrupted-attempts/` for incident inspection. Do not paste API secrets or raw customer diagnosis materials into PR discussion.
- Worker startup failures are logged under `queue/worker-launch-errors.log`; verify operator permissions and that the configured Python interpreter can import AstroGraph.
- The SQLite lease prevents blind automatic retry after worker loss; it is not a guarantee that a remote model provider did not accept or charge a request.
- Win7 target status applies only to the separately staged portable Python 3.8 distribution, **not** to the full-platform API dependency stack.
- This P3 work does not establish real-provider analytical quality or production cost acceptance. Diagnosis and source citation validation are independent of model correctness.

## 6. Verification gates before PR integration

1. Ubuntu Python 3.11 Foundation smoke with persistent-queue tests.
2. Ubuntu Python 3.11 full-platform external P0–P3 regression.
3. Windows Server 2022 Python 3.11 full-platform queue/locking regression, explicitly not a native Win10/Win7 claim.
4. Portable suites on Ubuntu Python 3.8/3.11 and Windows Server 2022 Python 3.8, verifying no regression to the separate portable engine.
5. Atomic claim at two-slot cap; worker crash and expiry; cancellation before/after publish guard; process-restart worker drain; report recovery without replay; immutable original inputs; explicit high-cost recovery and checkpoint handling.
6. Independent PR review, `git diff --check` / JS syntax, and then integration into consolidated PR #4. **No Stable label until the two new portable ZIPs are rebuilt, verified and native target acceptance is documented.**
