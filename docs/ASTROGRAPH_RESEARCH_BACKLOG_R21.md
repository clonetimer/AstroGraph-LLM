# AstroGraph research backlog after R21

Document status: `CURRENT_R21_RESEARCH_BACKLOG`.

## Gate for R22

R22 must reserve new independent groups before labels or outcomes are inspected. The exposure registry must assign them `FUTURE_HOLDOUT_RESERVED`, record immutable source and group identities, and reject every use except the preregistered final evaluation. The selected R21 candidates must be frozen before that gate opens.

## DXC remediation

Use DXC09 sample and DXC10 DPI/DPII only to reduce candidate-set size and improve true-fault coverage. Separate candidate generation, component mapping, mode mapping, and temporal localization so each failure can be measured independently. Freeze the ontology map and selection rule before any new DXC holdout is opened.

## OPS remediation

Package the selected length-aware fallback as one fixed candidate. Preserve operator-specific mathematical sample requirements and report coverage, abstention, selective risk, F0.5, and truth-stratified coverage. Do not replay selection over the 529 seen external groups.

## ESA remediation

Preserve event-level metrics and detection delay. Investigate temporal resolution on development data with a small, preregistered candidate family. Retain raw-to-window trace artifacts in the next run so threshold response and aggregation effects can be distinguished directly. Never relabel an R20 confirmatory/external replay as independent validation.

## Evidence and release work

- Add a sealed R22 data-admission receipt before execution.
- Keep the data-exposure gate on shared fitting, selection, calibration, optimization, and prediction-generation APIs.
- Require the contamination audit and frozen-byte audit to pass before packaging.
- Keep uncalibrated anomaly scores out of probability metrics.
- Stop at `DEVELOPMENT_REMEDIATION_READY_FOR_NEW_REPLICATION` until new independent evidence exists.
