# R11-C: frozen whole-spacecraft online result → opt-in LLM explanation

## Scope and public guarantees
This phase operates **only after** the canonical `online.persist.v1` stage succeeds. It does not add or reorder online DAG stages, run fault detection, change BiRRF candidate rankings, bypass the governor, vote on a fault, or grant release authorization.

- Module: `astrograph.orchestration.online_post_diagnosis`.
- Entrypoint for completed, artifact-backed online results: `run_online_post_diagnosis_explanation(result, knowledge_service, output_root, ...)`.
- Opt-in integration for direct `WholeSpacecraftOnlineDiagnosisWorkflow.invoke(...)`: configure `post_diagnosis_knowledge_service`, and require `ExecutionPolicy.allow_llm=true` **and** `deterministic_only=false`. Otherwise no extra model call is permitted.
- Explicitly stage-driven `AstroGraphRuntime` executors can invoke the standalone post-result adapter after their terminal result is complete; this phase **does not** implicitly change or add a runtime DAG stage.

## Source identity and integrity gate
The adapter accepts only a successful `ExecutionResult` from the expected whole-spacecraft online workflow.
It validates the Pydantic `StructuredDiagnosisOutput` (including its canonical `output_sha256`), checks each available lineage SHA256 against the frozen output payload, and checks cross-field task/run/episode/decision identities. Because `online.persist.v1` intentionally omits massive runtime-stage artifacts from its public output, the system does **not** rebuild that entire result and accidentally erase its runtime-stage digest map.

Before any LLM call, the adapter also confirms the artifact root equals the original output directory and checks the persisted `manifest.json`, `report/structured_diagnosis_output.json` and `diagnosis/decision_record.json` file lengths, file hashes and parsed content equality. SHA256 is a tamper-evidence consistency check, **not a digital signature or independent authentication of who created the artifact**.

A bounded frozen snapshot is derived from the governor's accepted fault plus true ranked candidates (if any) and retained evidence IDs. No guessed candidates or private evaluation labels are introduced. If the actual candidate set exceeds the 12-ID read-only material review limit, it is refused rather than silently truncated.

## RAG / GraphRAG
After validation, the adapter reuses R11-B2's `run_frozen_post_diagnosis_explanation(source_kind="ONLINE_RESULT", ...)`. In the default online case there is **no reviewed material graph** and thus no R7–R10 approved GraphRAG tools are offered. Only a separately supplied, already approved material package with the same source run ID and exact frozen candidate set can unlock task-bound read-only tools. This phase introduces no graph writes or on-orbit causal proof.

The ordinary explanatory LLM backend is optional. No real LM Studio/model endpoint or claim-quality benchmark is certified here.

## Isolation and receipts
The terminal `ExecutionResult`, structured diagnosis, original manifest, stage topology, governor decision and release policy are untouched. Sidecar path:
`<original_online_artifact_dir>/post_diagnosis/online_explanation_receipt.json`.

The receipt retains only status, SHA256/identity references, material binding, and bounded error type/codes. Optional model/retrieval failures report `FAILED_CLOSED` and cannot turn an upstream `SUCCEEDED` diagnosis into a failed diagnostic execution. An unavailable/offline model is separate from a confirmed root cause.

## Verification and limitations
- `tests/test_foundation_r11c_online_post_diagnosis.py`: host-output freeze, manifest integrity, tampering, wrong run directory, policy refusal, successful sidecar, optional model failure.
- Existing whole-spacecraft online acceptance and Runtime DAG tests are retained in expanded source-only CI.
- Local focused regression: 49 passed (prior to remote CI).
- R11-D should target real local LLM endpoint behavior and role-scoped operator invocation, plus a separate artifact-driven runtime integration test if required. No retrospective SADA root-cause label enters inference data.
