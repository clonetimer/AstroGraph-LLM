# AstroGraph GitHub integration status — R11-A

Repository: [clonetimer/AstroGraph-LLM](https://github.com/clonetimer/AstroGraph-LLM)

- Base branch: `main` (unchanged original source baseline).
- Development branch: `dev/foundation-r11-integration`.
- Pull request: [#1](https://github.com/clonetimer/AstroGraph-LLM/pull/1), **Draft**.
- R5–R11-A source-only migration: [`2f4c993`](https://github.com/clonetimer/AstroGraph-LLM/commit/2f4c9937d19d90a78ac319658475f0d396cf271f), 35 changed paths.
- Expanded CI coverage: [`fe76ab8`](https://github.com/clonetimer/AstroGraph-LLM/commit/fe76ab8f959782985550de98de30a89f57f850fa).

## Included code

- R1–R4: retrieval hardening, evidence admission, graph governance, competition hypotheses and claim guard.
- R5–R6: unverified reported-observation intake and balanced symptom / literature retrieval.
- R7–R8: reviewed directed causal claims, path provenance, causal differential and contradictions.
- R9–R10: reviewed context applicability and temporal process-stage compatibility.
- R11-A: task-bound, read-only material RAG / GraphRAG / R9 / R10 tools. Model-selectable structured tool planning is **opt-in**, at most one selection round and three read-only requests. It is **not provider-native function calling**.

## Verification

- Original R5–R11-A code-only patch matched all 9 existing remote targets before migration; 26 new source/test/docs paths were staged. Runtime/SQLite/caches and the disclosed SADA retrospective outcome were excluded.
- SHA-pinned development branch and its Git object tree were checked before branch update.
- GitHub Actions [PR run 38707661864](https://github.com/clonetimer/AstroGraph-LLM/actions/runs/38707661864) and [push run 38707655826](https://github.com/clonetimer/AstroGraph-LLM/actions/runs/38707655826): **SUCCESS** at `fe76ab8`, covering 11 focused test modules from R3–R11-A plus candidate selection and pre-reasoning retrieval.
- The `source-only` test profile is a scoped CI gate, not complete original-data acceptance, a Windows trial, or real LLM inference endpoint verification.

## Remaining merge conditions

1. Review the full PR diff for provenance, security and documentation accuracy.
2. Run the selected local regression and inspect completion logs; investigate any failure separately from historical missing-test-data conditions.
3. Keep `main` untouched until PR review and merge approval. Do not assert root-cause discovery merely because a literature causal chain is approved.
4. The next feature phase is R11-B: integrate equivalent **task-frozen** read-only tools into additional post-diagnosis explanation entrypoints (not upstream diagnosis), with backend compatibility and permission tests.

This status describes a **development candidate**, not a production release.
