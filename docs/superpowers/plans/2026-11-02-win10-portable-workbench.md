# Windows 10 Portable Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a self-contained Windows x64 ZIP for external diagnosis text analysis with a local browser UI.

**Architecture:** Reuse a minimal, unmodified closure of the existing analysis, report, conversation and provider modules. Add a stdlib local server and focused static UI, bundle embedded Python/Pydantic, and include source and reproducible build instructions.

**Tech Stack:** Python >=3.11,<3.15; Pydantic2; stdlib HTTP; plain HTML/CSS/JavaScript; official Windows embedded runtime.

**Spec:** `docs/superpowers/specs/2026-11-02-win10-portable-workbench.md`

## Global Constraints

- Windows10 x64; local binding127.0.0.1; no internet installation or admin rights.
- Model weights/server excluded; online compatible API or prepared loopback model server.
- Reuse existing analysis/source/conversation validations; no diagnostic computations.
- 4MiB input, 60,000 parsed characters; no silent text truncation.
- Credentials memory-only; static assets local; POST token and origin guards.
- Report Linux verification honestly; do not claim native Windows execution.

## Review Focus

- Windows embedded isolated sys.path and subprocess transport finding the same bundled modules.
- Chinese filenames/GB18030 and Windows reserved names or unwritable extraction folders.
- Browser refresh, server restart, duplicate requests and failed model calls preserving an accurate status.
- Input markup and provider errors exposing active HTML or API credentials.
- Local mode accidentally contacting cloud through application configuration or downloading dependencies.

### Task 1: Minimal runtime and server

**Files:** Create `portable/workbench.py`, `portable/build.py`, `portable/test_workbench.py`.

**Interfaces:** Stage canonical closure through `stage_app(repo_root: Path, destination: Path) -> Path`. Server provides the HTTP interface in the spec, and `Workbench(data_dir: Path)` implements the adapter required by existing follow-up functions.

- [x] Write unittest cases for staged imports, input validation, source binding, file encoding, settings validation, credential exclusion, initial report, follow-up and failure output.
- [x] Run `python -m unittest discover -s portable -p 'test_workbench.py' -v`; observe missing implementation failure.
- [x] Implement staging, store adapter, async initial calls, sync bounded follow-up and safe HTTP handlers.
- [x] Run the same test suite against local stub HTTP providers; expected all pass without external models or internet.

### Task 2: Focused Chinese UI

**Files:** Create `portable/assets/index.html`, `portable/assets/app.js`, `portable/assets/app.css`, and UI verification as needed.

**Interfaces:** Consume Task1 HTTP contract exactly; all dynamic material rendered as text. API key excluded from browser storage. Use presets, files, source-cited report, follow-up and export controls.

- [x] Implement accessible Chinese input/settings/results/history view with locally served assets.
- [x] Verify syntax and the actual input -> model-stub -> result -> citation -> follow-up flow in a browser; verify refresh/error states and settings do not persist the key.
- [x] Resolve interface discrepancies against the spec and run server tests again.

### Task 3: Windows distribution and delivery

**Files:** Create launchers, Chinese README, sample material, pinned dependencies, build/provenance manifests under `portable/` and generated ZIP outside the repository.

**Interfaces:** Build copies Task1 staged app and Task2 assets into the official runtime distribution, uses explicit embedded `_pth`, and preserves third-party licenses. `CHECK.bat` invokes the app's offline `--self-test`.

- [x] Verify runtime official checksum, dependency Windows wheel tags, and stage a Linux-equivalent dependency-isolated execution for offline smoke tests.
- [x] Write launchers and Chinese steps for both model modes, output locations, limits, troubleshooting and exact verification limitations.
- [x] Build ZIP; run CRC/member/hash checks, source-import tests and launcher/path review.
- [x] Obtain a fresh independent review, fix substantive findings and run targeted regressions.
- [ ] Synchronize the reusable source branch to GitHub. Remote mutation calls did not complete; no branch update or Windows CI run is claimed. Source is preserved in the local commits and deliverable patch.
- [x] Build the deliverable ZIP with readable source, source patch and concrete Chinese startup/model steps.

## Execution evidence

- Minimal canonical core copied unchanged: 19 modules; distribution-only package initializers isolate the engine.
- 16 portable regression tests passed on Linux with the exact pinned Pydantic dependency versions. Includes real HTTP worker transport, offline CHECK entry point, source binding, failed model output, encoding, concurrency and IPv4/IPv6 proxy bypass.
- Isolated Chromium passed import, model-list probe, analysis, citation navigation, follow-up, three exports, credential exclusion, history reload and responsive-width checks. No remote asset requests.
- Fresh independent review completed; substantive findings were fixed with regressions.
- Official Windows runtime/wheel hashes, AMD64 PE tags, unchanged core and archive integrity verified. Native Windows/Win10 execution remains unverified.
- GitHub synchronization did not return; the deliverable contains SOURCE_CHANGES.patch against 5389bfd78f8a9f62117f8804eab181fda7557046 for recovery.
