# Windows 10 portable external diagnosis workbench

## Authorized goal

Deliver a ZIP for Windows 10 x64 which starts a Chinese browser UI locally after extraction, without Python installation, pip, Node, internet installation, admin rights, or diagnostic-engine dependencies. The user's scope is external diagnosis text -> LLM analysis -> visual output, with optional source-cited follow-up. Online compatible APIs and a preinstalled offline local model server are both supported. Models and model-serving software are not bundled.

## Architecture and boundaries

Use the existing external input, analysis, conversation, report, prompt and bounded HTTP modules unchanged. Copy the required import closure into the distribution; use intentionally empty integration/retrieval package initializers to avoid eagerly importing unrelated engines. Bundle the official Windows x64 embedded Python runtime and pinned Pydantic wheels extracted into vendor packages. The thin workbench server uses Python's standard HTTP server on 127.0.0.1 and static local assets. Source, launchers, Chinese instructions, a sample and provenance are included.

No telemetry computation, model training, graph engine, model downloading, package downloading, CDN, or automatic model call occurs at startup. Files, report history and follow-up history stay in the extracted folder. API keys are used only in memory and never appear in saved config, reports or console logs. Online mode sends the selected diagnosis materials and questions to the configured API. Local mode accepts loopback HTTP(S) only; online mode requires HTTPS. No external model server is contacted until the user tests the connection or requests analysis.

## Functional contract

- Target: Windows 10 x64, with a current Edge/Chrome browser already installed. Recommend 22H2 for current model-server compatibility. Win7/32-bit/ARM are not the binary target.
- Input: pasted text and TXT/MD/JSON/CSV/LOG files; explicit diagnosis/log/reference roles. UTF-8/BOM/UTF-16 BOM and explicitly selected GB18030 supported. Preserve original bytes separately from canonical UTF-8 input, with hashes. No PDF/DOCX in this focused UI; export to text first.
- Limits: 8 diagnosis files, 12 log files, 16 reference files, 4 MiB total decoded bytes, 60,000 parsed characters, task <=4,000 characters. Reject invalid or empty inputs, unsafe Windows filenames, duplicate names, and material that cannot be parsed. Never silently truncate evidence.
- Output: existing schema-validated analysis, source IDs, original source text/locations, hypotheses and counterarguments, uncertainty, suggested checks, explicit failures, Markdown/JSON export. Citation checking is source membership, not correctness certification.
- Follow-up: existing persisted bounded conversation, idempotent request IDs, 12 attempts, 4,000 question characters, original source binding and nonblocking concurrency lock. Re-enter current connection/key after a restart.
- UI: Chinese, one focused page, no engine menus; file/paste input, model connection preset for local LM Studio and Ollama plus custom online API, connection test, start/poll progress, saved-job selector, cited result rendering, follow-up and downloads. Use text nodes/escaped text only for model and user material.
- Server: loopback binding, Host and Origin checks, random same-origin CSRF token on mutating endpoints, bounded request bodies, fixed safe public errors, single active model operation. A restart marks in-flight jobs interrupted and never replays them automatically.
- Launch: START.bat launches bundled Python and opens the browser; CHECK.bat runs offline dependency/import/self-test without a model. Relative paths allow spaces and Chinese folder names; fail clearly if extraction is incomplete or the folder is unwritable. Ctrl+C closes the service.

## HTTP interface

`GET /api/session` returns `{token, version, limits, defaults, sample}`. Every POST has `X-AstroGraph-Token` and JSON content type. `GET /api/health` performs no outbound call.

`connection = {mode: local|online, base_url, model, api_key, max_tokens, timeout_s, response_format_strategy: auto|json_schema|json_object|plain_json}`. Defaults: local LM Studio URL `http://127.0.0.1:1234/v1`, max_tokens 4096, timeout_s 180. Model ID must be provided. Test endpoint accepts an empty model ID to discover IDs.

`POST /api/probe` takes connection and returns `{status: SUCCEEDED, models: [{id}]}` or a fixed safe error. It queries `/models`, without generating a diagnosis.

`POST /api/jobs` takes `{task, files: [{filename, role, content_base64, encoding: auto|utf-8|gb18030}], connection}`, returns HTTP202 `{job}`. `GET /api/jobs` returns `{jobs}`. `GET /api/jobs/{id}` returns `{job, report: object|null}`. Job includes job_id, task, created_at, status (RUNNING/SUCCEEDED/FAILED/INTERRUPTED), model, mode and generation_status when completed. Report has the canonical `analysis.knowledge_enrichment` structure and `input_sha256`/sources in its model explanation.

`GET /api/jobs/{id}/conversation` returns canonical conversation state. `POST /api/jobs/{id}/follow-up` takes `{question, input_sha256, request_id, connection}` and returns canonical conversation state. `GET /api/jobs/{id}/report.md`, `/report.json`, `/conversation.json` export artifacts. Errors are `{error: {code, message}}`.

## Verification and delivery

Use tests against the actual staged minimal import closure and real local HTTP model stubs; verify success, invalid model citations, malformed outputs, refused/truncated responses, offline imports, unsafe files, GB18030, persisted history, no stored credentials and same-origin controls. Use browser automation if available for the actual import/analyze/follow-up/export flow. Verify official runtime checksum, x64 PE architecture, wheel tags/hashes and final ZIP CRC. Native Windows execution cannot be inferred from Linux tests; state this limitation and ship CHECK.bat for target acceptance. Save the final ZIP as the requested downloadable artifact and keep reusable source in GitHub.
