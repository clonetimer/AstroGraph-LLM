# AstroGraph 0.7.1.post49 — Stable Software Release

状态：`CURRENT_POST49_REFERENCE`

## Summary

post49 converges the completed post48 development series into a stable software distribution. The direct functional baseline is the frozen post48.dev63 RC. No diagnostic model retraining, model-weight change, active-KG change, confirmed-memory change, BiRRF-weight change, or threshold retuning is part of the post49 transition. Fresh-unpack smoke testing found one packaging/API continuity defect: the existing `JsonRequestLifecycleStore` was not re-exported by `astrograph.lifecycle`; post49 adds that public re-export without changing store behavior.

## Stable capability scope

The stable release carries forward the machine-readable scenario 1–8 matrix in `governance/implementation/post49/RELEASE_READINESS.json`. Stable software status must not be read as universal diagnostic qualification: classification/localization remain profile-gated, exact full three-task execution is limited to the admitted 33-channel ADCS profile, and document evidence remains non-voting for committed core diagnosis.

## Remaining external qualification

Fresh ADCS independent trajectory data, real MinerU/Anything2Ontology provider-quality evidence, additional generic classifier/localizer profiles, and independent document-fusion decision truth remain external qualification inputs.

## Promotion boundary

post49 does not enable automatic production promotion. Existing governed review, rollback, and promotion preflight remain mandatory.

## Documentation freshness maintenance

The post49 distribution separates CURRENT documentation from historical/research material. `docs/CURRENT_DOCS.json` is the default current corpus; explanatory pre-post49/dev/date documents are centralized under 历史档案（本交付源码包不包含） with a content-addressed manifest. Historical evidence remains available for reproducibility but cannot override `RELEASE_READINESS` or the current metrics protocol.
