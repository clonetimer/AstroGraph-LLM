# AstroGraph Research Backlog after R19

R19 closes the calibration fitting/admission protocol but does not create real calibration evidence.

## P0 before a real calibration claim

1. **Real frozen calibration and external-validation data** — provide independent groups with audited model/prediction/truth/context lineage; synthetic fixtures remain protocol-only.
2. **Deployment calibrator binding** — bind the runtime probability transform actually used by the deployed model adapter to the admitted fit receipt; offline qualification alone must not silently relabel untransformed runtime probabilities as calibrated.
3. **External dataset pedigree** — verify the declared dataset SHA256 against the actual frozen dataset/export manifest at admission, not only the evaluation bundle that references it.
4. **Shift coverage plan** — freeze which simulator versions, mission modes, severities and noise regimes count as external-validation claims; a single external corpus cannot support universal OOD language.
5. **Calibration method comparison** — preregister at least identity vs temperature scaling and, if sample size supports it, a more expressive calibrator such as Dirichlet/vector scaling. Selection must not use confirmatory/external groups.

## P1 research improvements

6. Replace or complement equal-width ECE with estimator-sensitivity analysis (equal-mass/adaptive/debiased alternatives) while retaining NLL/Brier as proper scoring rules.
7. Add group-block confidence intervals for **calibrated-minus-raw deltas**, not just each absolute metric separately.
8. Add minimum classwise external group counts derived from precision/power goals rather than the current engineering minimum.
9. Evaluate calibration drift over time/mission phase and after software/model updates; calibration receipt validity should be scoped to model/data/runtime identity.
10. Separate covariate, label and concept shift experiments; current strata are descriptive and may be confounded.
11. Evaluate selective policy transfer with confidence intervals on risk and coverage, including zero-retention and rare-fault edge cases.
12. Add calibration diagnostics for NORMAL/open-set behavior; closed-set class probabilities do not establish unknown-fault reliability.

## P2 engineering/reproducibility improvements

13. Produce a dedicated calibration dataset snapshot/export verifier analogous to the R8-R10 confirmatory snapshot chain.
14. Add signed/authenticated remote external-validation provider receipts; content integrity does not authenticate a remote producer.
15. Add an explicit calibrator artifact/spec to the data-driven model manifest, rather than relying only on calibration ID + receipt hash.
16. Consolidate the older `data_driven.credibility.TemperatureScaler.fit()` helper under a clearly lab-only namespace or route it through the R19 governed protocol to avoid two calibration paths with different statistical units.
17. Add deterministic reference vectors for temperature transform and cross-language/runtime parity if calibrator execution moves outside Python.
18. Add a calibration receipt revocation/supersession mechanism when external validation or model identity changes.

## Claim boundary

None of the items above authorize LLM evidence production, BiRRF probability semantics, flight release, or Probe promotion. Those remain separate governed questions.
