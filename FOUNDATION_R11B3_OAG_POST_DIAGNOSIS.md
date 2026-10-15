# R11-B3 — OAG independent post-diagnosis LLM explanation

## Boundary
Only the `OAG_DIAGNOSTIC` branch of the upload-report service is changed. Its original native OAG harness runs unchanged and yields the frozen `output.summary`, evidence assessment, optional OAG LLM explanation, and hash-chained replay journal. **The post-diagnosis phase is a separate model call and artifact, not a new OAG harness stage or diagnosis/fusion vote.**

## Implementation
- Immediately after `run_uploaded_oag(...)` succeeds, `UploadReportJobStore._run_unlocked` extracts the existing frozen OAG snapshot and records a SHA-256 of the complete harness result.
- Builds a dedicated `PlatformKnowledgeEnrichmentService` with original local/LLM settings but PaperSeek and A2O disabled for this **second** stage; the OAG pipeline retains exclusive ownership of external literature/ontology execution.
- Calls the R11-B2 `run_frozen_post_diagnosis_explanation(source_kind="OAG_DIAGNOSTIC", ...)` with the task's frozen candidate/evidence snapshot and uploaded reference documents.
- Because no independently approved, same-task `material_review_context` exists in the native OAG result, R11-A reviewed graph tools are **NOT available** to the model; no synthetic proof or surrogate reviewed source is created. Local RAG and bounded LLM explanation remain available.
- Writes a separate `output/oag_post_diagnosis.json`, includes `analysis.knowledge_enrichment`, and appends a bounded status receipt to the Markdown report. The native OAG `analysis.oag_result` is untouched.
- Post-explanation exceptions produce a `FAILED_CLOSED` optional-analysis receipt (exception type only). The already-verified core result remains available; a core-result hash change instead terminates the job with an explicit integrity error.

## Tests
- `tests/test_foundation_r11b3_oag_post_diagnosis.py` uses a deterministic protocol double to assert two separate model agent IDs, correct frozen candidate, no unaudited graph tools, valid native Harness journal, isolated optional failure, and no-model behavior.
- Local targeted regression: **77 passed, 1 skipped**.
- Source-only CI now includes OAG Harness compatibility and new R11-B3 tests.
- **Not covered:** real endpoint/LM Studio behavior, full Windows acceptance, original-source dataset fidelity or real in-orbit diagnostic proof.

## Next phase
R11-C should attach the **post-result** explanation adapter to a completed whole-spacecraft online decision artifact only after its replay identity and released/frozen fault candidates can be verified. It must not add tools inside telemetry detection, BiRRF or decision governance.
