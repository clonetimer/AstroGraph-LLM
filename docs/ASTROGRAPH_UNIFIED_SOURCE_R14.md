# AstroGraph PATCH-R14 — Unified Source Consolidation

Status: `CURRENT_R14_SOURCE_GOVERNANCE`

## 1. Purpose

PATCH-R14 removes the duplicated Backend-vs-complete-UI source-of-truth model. AstroGraph is now maintained as one canonical source tree. The web platform, maintenance control plane, Agent integrations and native ES-module UI are capabilities inside that project rather than a second source distribution.

This change is deliberately a **source-governance consolidation**, not a diagnostic-algorithm promotion. It does not alter the existing rule that formal hypothesis state comes from deterministic Evidence semantics, nor does it give LLMs new evidence/release authority.

## 2. Trusted parents

R14 was formed only from the final, fresh-ZIP-verified R13 artifacts:

- Backend R13 SHA256: `9a46a8b6ced89170e617b332ff644415baba4a1404f20fe0f4a83ec59e90e23a`
- Complete Platform/UI R13 SHA256: `e35e849cdf3239d8a95476e74302c33facfd0b89afb0da6df2f83daa4d6f7888`
- R13 external authoritative ledger SHA256: `6a58b072023dd251fda3c8add9740af4c4399e84fcbccb0d26bc40b86c775525`

The machine-readable merge record is `governance/merge/PATCH_R14_SOURCE_CONSOLIDATION.json`.

## 3. File-level adjudication

The two final R13 source trees were compared by relative path and SHA256 before consolidation:

| Observation | Count | R14 decision |
|---|---:|---|
| Backend filesystem files | 2476 | parent inventory only |
| Complete Platform filesystem files | 2557 | structural superset |
| Backend-only files | **0** | no Backend-only implementation to preserve separately |
| Platform-only files | **81** | retain as Platform/maintenance/UI capability files |
| Same-path, different-content files | **30** | individually recorded; complete Platform version is canonical after category review |

The 30 differences are not treated as an opaque directory overwrite. The manifest stores both parent SHA256 values, a category and the adjudication. They fall into packaging/documentation metadata, Platform control-plane extensions, optional Agent/tooling extensions, or test supersets.

## 4. Canonical project model

```text
AstroGraph/
├── src/astrograph/                  # one Python source of truth
│   ├── evidence / fusion / services / orchestration ...
│   ├── upstream/                    # provider abstraction + built-in manual provider
│   ├── integrations/                # Platform/maintenance integrations
│   └── resources/                   # native ES-module Platform UI
├── configs/
├── governance/
├── scripts/
├── tests/
└── docs/
```

There is no longer a separately maintained Backend source tree. A lightweight/headless deployment, if required, must be **built from this same tree** by dependency/profile selection; it must not fork shared Python files.

## 5. Dependency boundary

One repository does **not** mean every integration is mandatory.

- Built-in core: Pydantic, pandas, NumPy, jsonschema, and the currently eager scientific-model serialization dependency `joblib`.
- `platform`: FastAPI/Uvicorn/scikit-learn/SciPy.
- `agent`: LangGraph/LangChain.
- `neo4j`, `mcp`, `secrets`, `retrieval`, `deep-learning`: explicit extras.
- `ManualImportProvider`: built-in and available without Sat Agent or MATLAB.
- `SatAgentPlatformProvider` and `MatlabSimulinkProvider`: optional upstream descriptors; neither is a startup dependency.

### `joblib` note

The R13 Backend distribution did not declare `joblib`, but its shared `astrograph.integrations` package already eagerly imported `sat_sim_scientific_model`, which imports `joblib`. The complete Platform metadata declared it. R14 therefore retains `joblib` in the base requirements to close that existing metadata/import inconsistency. A future minimal-headless profile can move the scientific-model bridge behind a lazy import and then demote `joblib` to an ML extra; that should be done as an explicit compatibility change, not during source consolidation.

## 6. Versioning policy

R14 uses one patch identity: **AstroGraph PATCH-R14**. `PLATFORM_UI_ASSET_VERSION` remains `v0.12.13` because R14 does not change browser asset behavior. The Python distribution metadata remains `astrograph-core` / `0.7.1.post49` for installation compatibility; this is a package-name compatibility detail, not a second project/source tree.

## 7. Non-regression invariants

The merged project must continue to enforce:

1. LLM output cannot directly decide formal hypothesis status.
2. Shadow Probe remains `formal_evidence_eligible=false` absent real confirmatory promotion.
3. Maintenance overlays and PUBLIC_CASE / PIPELINE_RUNTIME children remain non-release-eligible where currently specified.
4. Historical archive evidence cannot be manufactured to make source-only tests green.
5. Frozen source reports are never rewritten by reanalysis.
6. Sat Agent and MATLAB remain optional providers.
7. Manual recompute admission remains content/hash/request-identity bound and cannot certify how an external producer generated raw telemetry.

## 8. Research audit

The consolidation is followed by a separate scientific/research audit in `docs/ASTROGRAPH_RESEARCH_AUDIT_R14.md`. Open research findings are intentionally **not silently fixed** during source consolidation because several require protocol/preregistration changes rather than local code edits.
