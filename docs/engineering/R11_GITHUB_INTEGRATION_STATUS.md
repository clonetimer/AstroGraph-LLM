# AstroGraph GitHub integration status

The default branch `main` is the original `source.zip` baseline (the repository was **not empty**).
Development branch: `dev/foundation-r11-integration`.

## Confirmed uploaded in this branch

- R1 evidence retrieval and graph approval isolation hardening.
- R2 reviewed document evidence and task-window admissibility.
- R3 literature governance, approved graph search and bidirectional retrieval.
- R4 competition-hypothesis prompts and conservative statement-level report guard.
- README public demo-credential removal.

## Not yet uploaded (do not claim merged or production-ready)

R5 reported observations; R6 multi-view retrieval; R7 causal claims; R8 causal differential; R9 reviewed applicability; R10 temporal reference chains; and R11-A model-selected read-only tools exist in the **separately delivered development candidate archive**, but have not been committed to GitHub by this import batch.

The archive also contains historical run products, SQLite databases, caches, binary assets, renamed screenshots and other files that must **not** be indiscriminately synchronized to a public repository.

## Test strategy

`source-only` pytest profile explicitly distinguishes missing historical/archive evidence from code failures. The CI workflow is a **targeted smoke gate**, not a complete project acceptance or true LLM endpoint test. Expanded and backend-specific tests should be required before a production merge.

Local archived-R4 smoke baseline (without remote CI): **54 passed, 2 skipped** for the four selected test modules.

## Next import gate

Diff only Python implementation, tests, and reviewed docs from the R11-A candidate archive against the pinned source baseline. Re-check provenance and secrets; avoid copying runtime datasets or retrospective SADA answers into the retrieval corpus. Stage and verify R5 through R11-A in sequence. Do not merge this draft branch until the full intended dependency chain and CI checks are reconciled.
