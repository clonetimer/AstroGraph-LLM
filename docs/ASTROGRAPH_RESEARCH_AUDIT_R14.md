# AstroGraph PATCH-R14 — Scientific and Research Audit

Status: `CURRENT_R14_RESEARCH_AUDIT`
Date: 2026-09-18  
Canonical source: one unified AstroGraph tree formed from final R13 artifacts.

## 1. Audit question

This review asks a stricter question than “does the software run?”: **if AstroGraph were used to support a research claim about spacecraft fault diagnosis, what parts of the current implementation would make that claim scientifically invalid, underpowered, poorly identified, insufficiently reproducible, or easy to over-interpret?**

The review therefore separates:

1. **implementation/governance defects** — behavior that contradicts the platform's own deterministic evidence or grounding contract;
2. **research-design blockers** — code can run, but the planned experiment cannot yet support the intended causal/statistical conclusion;
3. **implemented but not externally validated** — useful logic with correctly stated scope, but not enough evidence to generalize;
4. **engineering/auditability optimization** — not an immediate algorithmic defect, but a reliability/review risk.

No real 38×8 A0–A7 confirmatory inference was executed in R14. No shadow Probe is promoted by this audit.

## 2. Review method

The audit used:

- file-level R13 Backend/UI diff and merge adjudication;
- source inspection of evidence compilation, candidate/fusion, Agent grounding, maintenance input admission, open-set/OOD, temporal/statistical analysis and confirmatory metrology;
- deterministic counterexamples for two high-risk semantics;
- static maintainability scans;
- comparison with spacecraft V&V / PHM and statistical/LLM-grounding literature.

NASA's systems-engineering guidance distinguishes verification from validation and expects validation under realistic or simulated operational conditions, qualification/worst-case testing, known test-article pedigree and integrated/end-to-end validation. NASA software guidance likewise emphasizes target hardware/configuration and representative operational/stress scenarios. A 2023 PHM Society spacecraft fault-detection study used hardware-in-the-loop fault injection, illustrating the evidence level normally needed to move beyond simulator-only plausibility. The statistical review also checks independence assumptions because ordinary McNemar analysis assumes independent matched pairs; clustered matched-pair outcomes require an adjusted analysis.

## 3. Executive assessment

### 3.1 What is already structurally strong

The project has several unusually important research-governance properties worth preserving:

- formal hypothesis semantics are recompiled from deterministic EvidenceDefinition/EvidenceFact rather than trusted from LLM text;
- shadow Probe execution is permission-bounded and cannot silently become release evidence;
- candidate compacting has an escape path when Top-K would hide active/mandatory/open-set candidates;
- source-only versus archival-replay evidence is explicitly separated;
- confirmatory design, case identity, dataset snapshot, materialization/export and pre/post execution receipts are content-bound;
- maintenance reanalysis preserves frozen source reports and creates overlays/children with lineage;
- R13 upstream recompute returns are schema/hash/request-identity bound while explicitly declining to certify raw-recompute provenance;
- R13 separated analytic truth data from plotting downsampling and corrected append-dependent temporal baseline behavior.

These are good foundations. The principal remaining risk is not “LLM has too much authority”; it is that **some statistical, data-quality and scientific-validity semantics below/around the LLM boundary are not yet as rigorous as the governance shell**.

### 3.2 Findings count

- Total tracked findings: **21**
- Confirmed deterministic defects: **2**
- P0 research/governance findings: **5**
- P1 scientific validation/optimization findings: **10**
- P2 engineering/research debt findings: **6**

Machine-readable registry: `governance/research/PATCH_R14_RESEARCH_AUDIT.json`.

## 4. Priority matrix

| ID | Priority | State | Finding | Required before |
|---|---|---|---|---|
| `R14-RES-001` | P0-RESEARCH | OPEN | 38-case confirmatory inference records group_id but ordinary exact McNemar ignores clustering | Resolve and preregister before any A0-A7 result is used for promotion. |
| `R14-RES-002` | P0-RESEARCH | OPEN | Probe-promotion question and current primary confirmatory estimand are not the same comparison | Before formal EvidenceFact-producer review for Probe. |
| `R14-RES-003` | P0-RESEARCH | OPEN | Promotion gate lacks a preregistered effect-size confidence interval and power/minimum-detectable-effect justification | Before formal confirmatory execution if results may support promotion. |
| `R14-GOV-001` | P0-GOVERNANCE | CONFIRMED_DEFECT | Non-finite/quality-flagged FeatureResult can be compiled as ordinary NOT_SATISFIED instead of invalid/missing data | Fix before treating the unified tree as a candidate for any formal-evidence qualification. |
| `R14-GND-001` | P0-GOVERNANCE | CONFIRMED_DEFECT | Mixed evidence + knowledge citations bypass Agent paragraph numeric grounding | Fix before describing Agent narrative numeric grounding as complete. |
| `R14-REP-001` | P1-RESEARCH | OPEN | Confirmatory execution receipt freezes model ID but not full model/runtime/software identity | Close before claiming a reproducible real A0-A7 campaign. |
| `R14-VAL-001` | P1-RESEARCH | KNOWN_BOUNDARY | No real confirmatory, HIL, target-hardware or operational-shadow validation yet | Required before any stronger-than-engineering qualification claim. |
| `R14-MOD-001` | P1-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | Hand-engineered RW softmax scores are typed as PROBABILITIES despite explicit uncalibrated status | Before publishing probability-like performance or using magnitude as calibrated confidence. |
| `R14-TMP-001` | P1-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | R13 analysis/display separation fixes key bugs, but temporal/statistical thresholds remain heuristic and sampling-sensitive at scale | Optimization before research claims based on these maintenance statistics; not currently a formal Evidence vote. |
| `R14-ADM-001` | P1-GOVERNANCE | IMPLEMENTED_NEEDS_OPTIMIZATION | Supplemental capsule verifies shape/window/unit but not a full acquisition/sampling pedigree | Before supplemental inputs are ever considered for formal release evidence. |
| `R14-OOD-001` | P1-RESEARCH | IMPLEMENTED_NEEDS_EXTERNAL_VALIDATION | Open-set/abstention mechanisms have strong scope labels but limited external validity | Before using abstention thresholds as operational reliability claims. |
| `R14-FUS-001` | P1-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | BiRRF governance is stronger than its empirical weight/smoothing calibration | Before claiming fusion weights are empirically optimal or calibrated. |
| `R14-SCP-001` | P1-RESEARCH | OPEN | Formal confirmatory scope is effectively single-fault/class-mapped; compound/cascade faults are not established | Before generalizing to multi-fault/cascading diagnosis. |
| `R14-MET-001` | P1-RESEARCH | OPEN | Confirmatory metrics emphasize classification but under-specify operational fault-management costs | Before claiming operational FDIR/PHM effectiveness. |
| `R14-SAM-001` | P1-RESEARCH | OPEN | 38-case denominator yields very sparse per-class evidence and highly discrete macro metrics | Before class-specific efficacy claims. |
| `R14-PKG-001` | P2-ENGINEERING | MERGE_RESOLVED_WITH_FOLLOWUP | Unified metadata closes an existing joblib gap but minimal-headless dependency boundaries can be cleaner | Optimization only; do not reintroduce dual source trees. |
| `R14-MNT-001` | P2-ENGINEERING | OPEN | Several governance/control-plane modules are very large and broad exception handling is frequent | Maintainability/audit optimization, especially before safety-critical qualification work. |
| `R14-NUM-001` | P2-ENGINEERING | LATENT | Evidence EQUAL/NOT_EQUAL operators use exact floating-point equality | Before introducing a continuous EQUAL/NOT_EQUAL EvidenceDefinition. |
| `R14-UP-001` | P2-GOVERNANCE | IMPLEMENTED_NEEDS_OPTIMIZATION | Manual R13 response bundles prove integrity/identity, not producer authenticity or raw-recompute provenance | Before remote provider results can contribute to any formal evidence/release path. |
| `R14-KG-001` | P2-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | KG fault-universe coverage is a completeness control, not a validity/causality measurement | Before claiming KG coverage implies diagnostic knowledge quality. |
| `R14-GND-002` | P2-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | Numeric grounding is deterministic but unit/claim semantics remain narrow | Optimization before using citation-valid rate as a strong factual-faithfulness metric. |

## 5. Findings that must be resolved before formal confirmatory/promotion

### R14-RES-001 — clustered paired inference

The frozen case manifest explicitly records `group_id`: ADCS cases use `trajectory_pair_id`; PHM cases use `subsystem:sample_index`. That is a strong sign that the design already recognizes shared experimental units. However, `paired_arm_effect()` discards that structure, indexes rows only by `case_id`, then applies an ordinary exact McNemar calculation to corrected/induced case counts.

This is not a cosmetic statistical preference. Ordinary McNemar inference assumes independent matched-pair responses; multiple case outcomes from one trajectory/subsystem group form clustered matched pairs. The project should therefore decide **what the independent experimental unit actually is**, freeze that before seeing outcomes, and use a cluster-aware method. Case-level corrected/induced counts remain valuable descriptive diagnostics, but should not automatically be treated as 38 independent Bernoulli pairs.

Recommended closure:

- report `case_count` and `independent_group_count` separately;
- choose/preregister a cluster-aware primary test/interval;
- simulate/type-I-error check the chosen method under the actual small number/size of groups;
- keep the exact ordinary McNemar result only as a sensitivity/descriptive analysis if its independence assumption is not met.

### R14-RES-002 — Probe estimand mismatch

The frozen design correctly says the primary promotion comparison is `A2_CONTROLLED_FULL vs A0_ALGORITHM_ONLY`, while `A2 vs A7_NO_ACTIVE_PROBE` is secondary/descriptive unless separately preregistered. Those statements are statistically coherent. The later product roadmap, however, often describes the confirmatory campaign as the gate for deciding whether **Probe itself** may become a formal EvidenceFact producer.

Those are different scientific questions:

- A2 vs A0: incremental effect of the entire controlled LLM/agent stack over algorithm-only;
- A2 vs A7: marginal effect of active Probe with the rest of the controlled stack held closer to constant.

A2 beating A0 cannot identify Probe's causal contribution. Before any Probe promotion review, A2-vs-A7 needs its own preregistered estimand/gates or a factorial design that identifies Probe contribution.

### R14-RES-003 — effect size / interval / power

The gate freezes zero induced errors, perfect candidate recall/citation validity, net gain ≥1 and exact p≤0.05. This is conservative in some directions, but it still lacks a preregistered effect-size interval and a precision/power justification. Exact p-values are not effect sizes. An exact-McNemar-compatible confidence interval is available in the statistical literature and would make the size/uncertainty of the paired benefit explicit.

For intuition only: if induced errors are zero, the current two-sided exact test requires multiple corrected discordant pairs before p≤0.05; therefore `min_net_diagnostic_gain=1` does not define the practically important benefit once the significance gate is binding. The project should set the minimum scientifically meaningful gain explicitly rather than let it be an accidental consequence of the p-value threshold.

### R14-GOV-001 — invalid/non-finite FeatureResult semantics

Confirmed deterministic counterexample in the merged source:

```text
FeatureResult.value = NaN
quality_flags = ["NONFINITE_INPUT"]
REQUIRED evidence: GREATER_THAN 1 A
=> EvidenceFact.status = NOT_SATISFIED
=> explanation = "Observed nan A; condition GREATER_THAN 1 is not met."
```

`evaluate_evidence()` checks `None`, but it does not first reject non-finite numeric values or severe quality flags. Because formal hypothesis compilation consumes EvidenceFacts, this is a **formal evidence-layer issue**, not merely presentation logic.

Required repair should be centralized: define which FeatureResult quality states are admissible for threshold evaluation, reject/map non-finite input before comparison, and apply the same validity check to conjunctive FeatureResults. Ideally Pydantic also forbids NaN/±Inf unless a specific invalid-result type is being represented.

### R14-GND-001 — mixed-citation numeric grounding bypass

R13 added deterministic protection against a model citing a real `E1=0.2 A` while writing `900 A`. In the unified project, a mixed Agent paragraph still bypasses that check:

```text
visible citations = {E1, K1}
E1 = 0.2 A
paragraph = "Motor current is 900 A."
citation_ids = ["E1", "K1"]
=> accepted
```

The cause is a set condition: numeric grounding is invoked only when **all** paragraph citations are evidence IDs. Adding one knowledge citation disables the evidence-numeric check.

Fix the logic before representing Agent numeric grounding as closed. A mixed paragraph should still extract explicit measurements and test them against the cited EvidenceFact subset. Longer term, use claim-level attribution rather than paragraph-wide citation membership.

## 6. Reproducibility and external validation

### R14-REP-001 — execution identity

The confirmatory receipt is already much stronger than common ML experiment logging: it freezes design hash, case manifest/hash/identity, selected dataset snapshot, exact model ID reported by the server, Python/platform, arm order and thresholds, and rechecks identity after execution.

For scientific reproducibility, the remaining gap is that **model ID is a name, not model content identity**. The receipt should additionally bind:

- model weights/revision/content digest;
- tokenizer/config digest;
- serving engine/vLLM and container/build digest;
- AstroGraph `SOURCE_MANIFEST` digest/commit;
- environment lock/SBOM;
- GPU/CUDA/driver where numerical/runtime behavior matters;
- complete inference settings and prompt/profile/config hash.

### R14-VAL-001 — validation ladder

Current synthetic/simulator/test evidence is appropriate for engineering development but not for broad spacecraft diagnostic qualification. A research program should explicitly distinguish:

1. software/contract verification;
2. frozen in-simulator diagnostic validation;
3. cross-simulator and controlled distribution-shift validation;
4. HIL/PIL fault injection on representative hardware/IO;
5. resource/stress/latency and telemetry-corruption campaigns;
6. operational-shadow use with independent review;
7. only then, any stronger release/flight qualification process.

NASA guidance explicitly separates verification from validation and emphasizes realistic conditions, target hardware/configuration, qualification/stress scenarios and integrated/end-to-end validation. The 2023 PHM Society spacecraft work cited below used hardware-in-the-loop fault injection, which is a useful comparison point—not a requirement that AstroGraph copy that exact design.

## 7. Implemented logic that should be retained but optimized

### 7.1 Deterministic semantic compiler — retain

This is one of the project's strongest architectural decisions. LLM-produced `supporting/contradicting/missing/status` is not formal truth. Future optimization should focus on property-based/model-based testing of Evidence semantics, invalid-data states and compiler invariants rather than giving the LLM more authority.

### 7.2 Controlled Probe — retain shadow-only

The Probe catalog/executor enforces bounded action and separates structural agency from diagnostic contribution. The next research step is not broader permissions; it is **value-of-information evaluation**: when does a Probe change information versus merely re-express existing evidence, what is its induced-error rate, and what is its marginal effect A2-vs-A7?

### 7.3 Candidate escape / abstention — retain

Disabling compact reasoning when active/mandatory candidates overflow Top-K or open-set review is required avoids silent truncation. But it solves candidate-context completeness, not ontology completeness or calibrated unknown detection. Pair it with independently validated OOD/risk-coverage behavior.

### 7.4 Content-addressed confirmatory snapshots — retain

Snapshot receipts, materialization, export verifier, read-only policy and pre/post identity checks are strong reproducibility controls. Add model/runtime/source-environment identity rather than weakening these controls.

### 7.5 EvidenceInputContract / supplemental admission — retain bounded status

Machine-readable field/unit/window/algorithm/hash contracts are valuable. Direct capsule admission should remain narrow until sampling/acquisition pedigree is added. Do not infer that a numerically valid capsule is equivalent to a calibrated sensor/telemetry acquisition.

### 7.6 BiRRF dependence/coverage governance — retain

Correlation-group collapse and KG universe coverage are useful anti-double-counting/completeness controls. Optimize through empirical sensitivity and KG quality validation; do not convert strict-profile engineering weights into probability language without calibration evidence.

### 7.7 R13 Manual upstream provider — retain

The manual request/response bundle correctly distinguishes **content integrity** from **provenance of computation** (`raw_telemetry_recompute_verified=false`). Future HTTP providers should add authenticated producer/execution identity, not remove this distinction.

## 8. Additional scientific issues

### Heuristic probability semantics

`sat_sim_pipeline.py` normalizes hand-engineered scores with softmax and labels them `PROBABILITIES`. The code also clearly states they are untrained/uncalibrated, and downstream conflict logic avoids calibrated-probability claims, so this is not currently equivalent to a formal probability assertion. Still, the type should be corrected to an engineering score semantics to prevent future misuse.

### Temporal/statistical analysis

R13 fixed serious integrity bugs (plot decimation no longer defines analytic truth, missingness denominator is source-based, baseline is append-stable). Remaining research work is calibration: false alarms per unit time, detection delay, cadence/window sensitivity, early-fault baseline contamination and the effect of bounded sampling on change-point results.

### Open set / OOD

The code itself states the correct boundary: Basilisk/lab calibration is not flight reliability, and ADCS post-hoc OOD has a bounded threat model. Preserve that language. Evaluate under independent simulator/HIL/mission-condition shifts and report risk-coverage/uncertainty, not just a fixed rejection threshold.

### Multi-fault scope

The 38-case design is exact mapped single-fault/class data. Do not generalize this to simultaneous/cascading faults. Add a separate campaign if multi-fault support is a scientific objective.

### Operational FDIR metrics

Accuracy/F1 are insufficient for mission fault management. Add false alarms per operating time, missed events, detection/isolation delay and resource/latency distributions. A recent spacecraft telemetry anomaly-detection review likewise identifies false alarm rates and related operational metrics as common evaluation concerns.

## 9. Engineering/auditability findings

Static review of the unified source found large control-plane modules (`upload_report.py` ~3121 LOC, `api.py` ~1526, `whole_spacecraft_stage_executor.py` ~1453, `llm_method_campaign.py` ~1313) and **194** `except Exception` occurrences across **92** Python files. This does not prove hidden defects, but it increases the cost of proving absence of trust-boundary errors. For a high-governance research platform, typed failure contracts and smaller modules are themselves part of auditability.

The merge also exposed an existing packaging inconsistency: the Backend metadata omitted `joblib` even though shared integration imports could require it. The unified metadata resolves the immediate inconsistency. Future minimization should be accomplished by lazy optional imports, not by recreating two source trees.

## 10. Recommended research sequence

### Gate A — fix semantic defects first

1. Formal FeatureResult validity/NaN/quality-flag policy.
2. Mixed-citation Agent numeric-grounding bypass.
3. Add regression tests for both.

### Gate B — freeze confirmatory statistics again **before real outcomes exist**

1. Define independent group/unit and cluster-aware paired inference.
2. Define Probe-specific A2-vs-A7 estimand if Probe promotion remains a target.
3. Add effect-size CI/minimum meaningful effect/power or precision analysis.
4. Freeze source/model/runtime/environment identity fields.

### Gate C — execute a larger validation ladder

1. Frozen ADCS/PHM data + exact model run.
2. Independent trajectories / fault severities / operating modes.
3. Cross-simulator/domain-shift and telemetry corruption.
4. HIL/PIL and representative target-compute tests.
5. Operational shadow study.

### Gate D — only then consider authority expansion

Only if the relevant mechanism-specific evidence passes should any Probe be reviewed for formal EvidenceFact production. A full-stack LLM improvement result is not sufficient evidence for that permission change.

## 11. External research cross-checks

- NASA, *Systems Engineering Handbook — Verification and Validation Plan Outline*: https://www.nasa.gov/reference/system-engineering-handbook-appendix/
- NASA, *Fundamentals of Systems Engineering*: https://www.nasa.gov/reference/2-0-fundamentals-of-systems-engineering/
- NASA Software Engineering Handbook, SWE-067 *Verify Implementation*: https://swehb.nasa.gov/spaces/SWEHBVB/pages/32604539/SWE-067+-+Verify+Implementation
- Pesola, Kolcio, Prather & Ildefonso (2023), *A Hybrid Model-Based and Data-Driven Framework for Automated Spacecraft Fault Detection*, PHM Society, DOI 10.36001/phmconf.2023.v15i1.3461: https://www.papers.phmsociety.org/index.php/phmconf/article/view/3461
- Durkalski et al., *Analysis of clustered matched-pair data*, Statistics in Medicine, DOI 10.1002/sim.1438: https://onlinelibrary.wiley.com/doi/10.1002/sim.1438
- Fay & Lumbard, *Confidence intervals for difference in proportions for matched pairs compatible with exact McNemar's or sign tests*: https://pubmed.ncbi.nlm.nih.gov/33263202/
- Zhang et al. (2024), *Towards Fine-Grained Citation Evaluation in Generated Text*: https://aclanthology.org/2024.inlg-main.35/
- Cattelan & Silva (2024), *How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks*: https://proceedings.mlr.press/v244/cattelan24a.html
- *A Review of Anomaly Detection in Spacecraft Telemetry Data* (2025): https://www.mdpi.com/2076-3417/15/10/5653

## 12. Claim boundary

This audit is a source/research-method review. It does **not** estimate AstroGraph's flight fault-detection accuracy, LLM hallucination rate, Probe benefit, false-alarm rate or safety reliability. Those quantities require the real frozen datasets/test articles and preregistered experiments described above.
