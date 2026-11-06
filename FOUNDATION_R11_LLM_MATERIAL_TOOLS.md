# R11-A: material-evidence LLM tool bridge (candidate)

Scope: **frozen post-diagnosis results -> literature/GraphRAG investigation -> LLM explanation -> safe public report**. No telemetry-core or upstream diagnosis changes. This increment is *not* the planned event-level R11 telemetry alignment.

## Delivered, precisely

- `src/astrograph/tools/material_review.py` creates a **sealed task-specific registry** with four read-only calls: `material.rag.compare_hypotheses`, `material.graph.search_causal_paths`, `material.graph.check_context` (R9), and `material.graph.check_temporal` (R10). It reads only host-precomputed, independently reviewed results; it **does not** re-run graph search against arbitrary user-supplied edges, accept filesystem paths, or authorize new candidate IDs. A `ToolRegistry.invoke` permission gate requires `READ_KNOWLEDGE`, validates schemas, logs receipts, and enforces tool output size limits.
- `PlatformKnowledgeEnrichmentService.enrich` **always** passes a bounded causal-validation summary, including reviewed path IDs, R9 context results, R10 temporal results and blocking reason codes, to `PlatformKnowledgeExplanation` whenever a material review exists. Old single-generation backends remain compatible.
- **Optional model-selected calls**: set `ASTROGRAPH_PLATFORM_MATERIAL_TOOL_CALLS_ENABLED=1` (admin environment). The model makes a structured `MaterialToolPlan` selection with a maximum of three actions; host validation and registry invocation execute them; the frozen results then enter the subsequent structured explanation request. This is a two-pass **structured selector + host tool dispatch**, not provider-native OpenAI tool-call message exchange and not an unconstrained autonomous loop. Default is disabled until the chosen backend has been validated for the selector schema. Errors fail closed and do not widen scope.
- `apply_material_answer_guard()` now releases deterministic `reviewed_causal_explanations` (approved reference paths, blocking reason codes, context/temporal statuses) while preserving R4's rejection of unverifiable LLM-generated root-cause narratives. Provider receipts, raw generated claims, and model tool arguments are not published.

## Tool input/output contracts

| Tool | Input (model-selectable) | Host-owned result |
| --- | --- | --- |
| `material.rag.compare_hypotheses` | `{}` | Frozen candidate comparison |
| `material.graph.search_causal_paths` | `fault_id` | Reviewed paths, blocked reasons, reference summaries |
| `material.graph.check_context` | `fault_id`, `path_id` | Precomputed R9 approved context verdict |
| `material.graph.check_temporal` | `fault_id`, `path_id` | Precomputed R10 approved temporal verdict |

Unknown IDs, cross-candidate path IDs, arbitrary graph snapshots, and extra argument fields are rejected. Each registry instance is frozen before the LLM is consulted. Tools never run commands or grant knowledge editing, telemetry control or diagnosis voting. `current_root_cause_proven` remains false.

## Reproducible validation

```bash
export PYTHONPATH=src
python -m pytest -q tests/test_foundation_r11_material_llm_tool_integration.py
# Logged long-running groups:
python scripts/launch_logged_tests.py start --timeout 900 r11-regression -- python -m pytest -q tests/test_foundation_r11_material_llm_tool_integration.py tests/test_foundation_r10_temporal_causal.py tests/test_foundation_r9_reviewed_applicability.py
python scripts/launch_logged_tests.py status r11-regression
```

Completion is recorded by `validation_logs/<job>/done.json`. A local completion marker **does not push updates to chat**. All synthetic fault fixtures are unrelated to the known SADA retrospective reference answer.

## Residual technical limitations

1. Tools expose **cached, approved host calculations**. They are not dynamic per-query re-evaluations of untrusted paper excerpts or current telemetry observations.
2. Model-selected calls are one bounded selection round (maximum three calls), plus one final explanation. No iterative search-and-plan loop or native provider function calling has been implemented.
3. The source excerpt equality policy still cannot validate full semantic entailment; public deterministic verdicts are labelled reference-level only.
4. Registry permission validation and tool receipts are tested with synthetic/mock backend. A production LLM provider must be separately checked before enabling the flag.
5. Legacy trace-data/acceptance-file missing issues remain separate from the R11-A enhancement work.
