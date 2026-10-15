# AstroGraph R35.10 — Product Experience Stage Closure Review and Stable Candidate

Status: `CURRENT_R35_10_PRODUCT_EXPERIENCE_STABLE_CANDIDATE_READY`

Platform UI asset version: `v0.15.10`.

R35.10 closes the R35 product-experience implementation sequence as a stable candidate. It does not add another review state machine. Instead, it audits the R35-R35.9 operator vocabulary, makes the existing local-review and incoming-handoff states explicitly distinct, and adds invariant checks that keep those read-only state families independent.

## Closure-review finding

The R35 chain intentionally accumulated several browser-local concepts:

- current-page review coverage and attention signals;
- current-page review closure (`REVIEW_*`);
- coverage provenance and drift;
- incoming replay comparison (`MATCHED / CHANGED / NOT_REPRODUCED / NOT_PROVIDED`);
- guided re-review (`PENDING / REVISITED`);
- incoming re-review convergence (`ALL_MATCHED / DELTAS_ALL_REVISITED / REREVIEW_PENDING`).

The R35.9 implementation was behaviorally separated, but two Chinese UI titles both used “收口”: the local page review closure and the incoming re-review convergence summary. R35.10 treats that terminology collision as a product-risk because operators could read the two independent browser-local summaries as one approval gate.

## Terminology normalization

R35.10 uses distinct UI titles:

- `本页材料复核状态` for the R35.3 current-page review closure;
- `接手重新查看状态` for the R35.9 incoming replay/re-review convergence.

The top-level review section is named `本页材料复核`.

The legacy handoff text key `Review closure status:` remains present for downstream text-consumer compatibility. A more explicit alias, `Local page review status:`, is added rather than silently changing the older field.

## Stable-candidate status overview

A compact `复核状态总览` is derived only from existing state contracts. It shows four independent facets:

1. current-page material review status;
2. explicit attention-signal count;
3. whether incoming coverage replay is loaded and how many replay deltas exist;
4. incoming re-review convergence and current pending fingerprint count.

The overview carries the boundary marker:

```text
R35_PRODUCT_EXPERIENCE_STABLE_CANDIDATE_READ_ONLY
```

This marker is not a new workflow state. It states that the status overview is a read-only composition of already-existing browser-local contracts.

## Stable-candidate invariants

R35.10 adds explicit regression coverage for the following independence rules:

- loading incoming replay does not mark current-page review areas;
- current-page review closure does not convert incoming replay comparison states;
- a transient candidate browse may make local review `REVIEWED_WITH_ATTENTION` while independently making incoming re-review `REREVIEW_PENDING`;
- marking the incoming candidate delta `REVISITED` can move incoming convergence to `DELTAS_ALL_REVISITED` while local review remains `REVIEWED_WITH_ATTENTION`;
- the underlying incoming comparison remains `CHANGED` after `REVISITED`;
- the overview and handoff do not create network, storage, maintenance-authority, diagnosis, EvidenceFact, approval, signature, release, or retained-source mutation.

## Stable-candidate scope decision

R35.10 deliberately does **not** add:

- another review/replay status family;
- persistent reviewer identity;
- saved review sessions;
- discrepancy resolution;
- automatic acceptance or waiver;
- approval/signature/release actions;
- new backend endpoints;
- changes to `astrograph-diagnostic-workspace-presentation.v1`;
- changes to `astrograph-diagnostic-workspace-source-bundle.v2`;
- changes to the R32 frozen diagnosis/evidence/evaluation/fusion/simulation/orchestration core.

R35-R35.9 behavior remains available. R35.10 is a terminology and composition hardening increment rather than a capability expansion.

## Stable-candidate exit criteria

R35.10 may be called a Product Experience Stable Candidate only when a fresh final archive demonstrates:

- source integrity and documentation freshness;
- focused R34.2 production-contract regression;
- full Diagnostic Workspace browser regression including R35.10 invariants;
- Operations Console browser regression;
- production JavaScript syntax validation;
- stable-source validation;
- R32 frozen-core and presentation-backend byte-level emptiness;
- zero delivered runtime/test cache;
- unchanged source-only/strict external-archive boundary;
- no unsupported claim that localhost HTTP positive E2E passed when the execution environment blocks `127.0.0.1`.

## Claim boundary

R35.10 establishes a stable-candidate operator vocabulary and a consolidated read-only view over the already-existing R35 review/handoff states. It does not establish diagnostic correctness, discrepancy resolution, audit sufficiency, reviewer competence, formal acceptance, approval, release readiness, or qualification of any algorithm, model, dataset, software, HIL setup, or flight system.
