# AstroGraph R34.2 — Diagnostic Workspace Production Contract Hardening

## Status

`CURRENT_DIAGNOSTIC_WORKSPACE_PRODUCTION_CONTRACT_HARDENING_COMPLETE`

Platform UI asset version: `v0.14.2`.

R34.2 closes three production-contract gaps found after R34.1. It does not add a diagnostic algorithm, physical model, Agent Probe, Tool, graph capability, SimulationProvider, Bench feature, or UI workflow.

## Atomic source retention

The retained diagnostic-workspace source bundle is published with a same-filesystem exclusive hard-link claim:

1. each contender writes complete JSON bytes to a unique sibling temporary file;
2. `os.link(temporary, final)` atomically claims the final path without replacement;
3. the first successful publisher wins;
4. a concurrent identical payload returns idempotently;
5. a concurrent different payload fails closed as immutable;
6. temporary files are removed on every exit path.

The final path never passes through a partial-file state and a later contender cannot overwrite it.

## Candidate-context binding

The internal retained-source schema advances to `astrograph-diagnostic-workspace-source-bundle.v2`. Its self-hashed payload now includes `candidate_context_sha256`, calculated from normalized `selected_fault_id` and `component_id`.

Projection recomputes this identity from the current maintenance presentation. Any drift fails closed before the graph or simulation source is marked `VERIFIED`. This prevents a trace retained for one selected candidate from being presented as a verified source after the candidate identity changes.

The public read-only presentation remains `astrograph-diagnostic-workspace-presentation.v1` because its response shape and authority semantics do not change.

## Production-store HTTP reachability

The positive browser E2E no longer replaces `UploadReportJobStore` with a prebuilt-presentation double. It now executes:

```text
persisted successful job
  → persisted report / maintenance / visualization
  → real UploadReportJobStore retention
  → retained source-bundle bytes
  → production FastAPI endpoint
  → source Pydantic/hash/provenance/candidate validation
  → Chromium SVG and observed/simulated overlay
```

The test verifies the authority snapshot remains unchanged after graph selection and trace selection.

## Verification contract

R34.2 adds explicit checks for:

- concurrent different-payload retention: one return and one immutable rejection;
- concurrent identical-payload retention: two identical idempotent returns;
- complete valid retained JSON and no orphan temporary files;
- selected-candidate drift rejection;
- legacy unbound source-bundle v1 rejection;
- real-store graph and R31 numeric-overlay HTTP/browser positive paths.

## Frozen boundary

The following R32 core areas remain unchanged:

```text
src/astrograph/diagnosis/
src/astrograph/evidence/
src/astrograph/evaluation/
src/astrograph/fusion/
src/astrograph/simulation/
src/astrograph/orchestration/
src/astrograph/investigation/diagnostic_probes.py
```

Graph selection, trace selection, candidate drill-down, Evidence links, and Agent links remain read-only display/navigation state. They do not modify formal diagnosis, EvidenceFact, release authority, or retained producer output.

## Claim boundary

R34.2 proves atomic source retention, selected-candidate binding, and a real persisted-store-to-browser positive path for existing R23/R29/R31 outputs. It does not provide new diagnostic-performance evidence and does not qualify any algorithm, model, or dataset.
