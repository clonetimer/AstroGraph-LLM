# R11-B2: frozen post-diagnosis LLM explanation bridge

This phase is limited to **frozen diagnosis result → LLM-assisted analysis output**. It does **not** alter telemetry processing, detection, BiRRF, candidate rankings, or online safety decisions.

## Shared invocation

`astrograph.integrations.post_diagnosis_explanation.run_frozen_post_diagnosis_explanation`
accepts a source kind, bounded source ID, frozen diagnosis snapshot, document paths, optional previously reviewed material result, and a configured `PlatformKnowledgeEnrichmentService`.

The adapter reuses the existing R11-A four read-only tools, and does **not** add new provider-native function calls. With no reviewed material, ordinary RAG and the original model explanation path remain available.

## Wired entrypoints

- `RESULT_LOG_REVIEW`: host-generated material review, frozen candidate set and job ID must all match before the reviewed material can enter LLM tooling.
- `PUBLIC_CASE`: post-diagnosis knowledge enrichment via the same adapter, with no R7–R10 reviewed graph assumed.
- `SAT_SIM_PIPELINE`: same post-diagnosis enrichment boundary, no changes to its core inference path.

`OAG_DIAGNOSTIC` and `ONLINE_RESULT` are reserved, callable source kinds for an explicit future integration, **not automatically wired** to their separate upstream diagnostic workflows in this phase.

## Fail-closed binding

For a non-null reviewed material bundle, the adapter requires:

1. Exact equality of `source_id`, `diagnosis_snapshot.material_source_group_id`, and `material_review_context.source_group_id`. All three must exist.
2. Nonempty, unique, bounded `candidate_fault_ids` and an exact candidate-set match with the reviewed material, plus a valid frozen top candidate.
3. No model-provided path, snapshot, evidence identity, candidate addition, or review authorization.

Mismatched/unproven material is **omitted from the LLM call**, and a non-voting `post_diagnosis_binding` receipt states `DENIED_TASK_BINDING` with an explicit reason code. It is never silently treated as reviewed evidence.

This is a second fail-closed entrypoint check in addition to R11-B's `bind_frozen_material_review` within the knowledge service. Missing reviewed material produces `NOT_AVAILABLE` and keeps ordinary explanation available.

## Verification / limits

- Synthetic unit and refusal tests in `tests/test_foundation_r11b2_post_diagnosis_bridge.py`.
- Existing R11-A and platform knowledge / upload report regressions are included in the expanded source-only CI.
- Tested with no real LLM endpoint. OAG and whole-spacecraft online post-diagnosis adapter integration are outstanding.
- Do not assert that reviewed reference literature proves the current spacecraft root cause; no user-revealed SADA outcome was put into the retrieval corpus.
