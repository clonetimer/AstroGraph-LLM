# Foundation R11-B — Frozen-material binding at the LLM boundary

## Scope

This is a post-diagnosis LLM analysis hardening change. It does **not** alter upstream fault detection, ranking, BiRRF, accepted labels or flight-control authorization.

The shared `PlatformKnowledgeEnrichmentService.enrich` entrypoint now checks that a material-review snapshot belongs to the **current frozen diagnostic candidate set** and, when source-group tracking is present, the **same task**. The result is recorded as `material_binding` in both the model payload and serialized audit artifact.

- `BOUND_FROZEN_DIAGNOSIS`: all candidates match; task-group identifiers match when present. R11-A reviewed read-only tools may be offered if separately enabled.
- `DENIED_TASK_BINDING`: invalid candidate set, missing frozen candidate identity or mismatched task group. Reviewed material is removed before LLM context construction and tool selection; only a generic reason code is sent. Ordinary document explanation remains available.
- `NOT_AVAILABLE_NO_APPROVED_MATERIAL_REVIEW`: public-case and other post-diagnosis entrypoints without an approved review use the normal LLM explanation pass but **must not** expose reviewed causal tools.

For uploaded `RESULT_LOG_REVIEW`, the host binds `material_source_group_id` to the job ID. Legacy single-candidate callers may bind exactly one material candidate without a candidate list; multi-candidate review requires an explicit frozen candidate list. The opt-in model planner remains restricted to one selection round and at most three read-only tool calls.

## Verification

`tests/test_foundation_r11b_task_bound_explanations.py` covers matching candidates, cross-task denial, foreign-fault denial, missing task-group denial and no-review entrypoints. The GitHub source-only workflow runs these tests with the earlier RAG/GraphRAG/LLM-bridge tests.

The check is **not** a cryptographic attestation of external upstream inputs. It is a fail-closed boundary within trusted host orchestration; original review approval and storage access controls remain required. R11 event-to-telemetry alignment and real-model tool compatibility remain separate work.
