# AstroGraph PATCH-R27 — DXC expected-response diagnosis architecture

Status: **CURRENT_R27_POST_CONTACT_DEVELOPMENT_PRIMARY_CANDIDATE**  
Date: 2026-09-19

## Architecture decision

R27 does **not** make the Agent the detector and does **not** delete the legacy DXC algorithm.

The diagnosis architecture is explicitly three-layered:

1. **Legacy deterministic branch retained** — the R20 statistical predictor remains available as a frozen baseline/fallback and is not silently rewritten by R27.
2. **Additional deterministic diagnosis-core branch** — R25 command-aware numeric residuals + R26 discrete PositionSensor consistency + R27 component-mode/expected-response ranking. This branch executes without any Agent and is the only layer responsible for the R27 development-performance change.
3. **Optional Agent shadow review** — the Agent may select host-bound `MINIMAL_DIAGNOSIS_REVIEW`, `SIMULATION_COUNTERFACTUAL_REVIEW`, or `DXC_EXPECTED_RESPONSE_REVIEW`. These tools only inspect already-bound deterministic artifacts; they cannot alter telemetry, thresholds, model coefficients, selected fault IDs, formal EvidenceFact state, diagnosis disposition or release authority.

Therefore the dominant implementation form is:

`legacy retained + new deterministic algorithm branch + optional read-only Agent review`

not:

`raw data -> Agent chooses detector -> whichever result looks best becomes diagnosis`.

R27 also does **not** switch the production/runtime default. The new branch is a `PRIMARY_CAPABLE_DEVELOPMENT_CANDIDATE` whose promotion remains blocked pending a genuinely new holdout.

## Why R27 moves logic into `astrograph.diagnosis`

R25/R26 remediation lived primarily in the evaluation/research layer. R27 moves the new response-consistency primitive into:

`src/astrograph/diagnosis/dxc_expected_response.py`

The core primitive is deterministic and truth-blind. It consumes admitted telemetry, the engineering catalog, and upstream deterministic candidates. The evaluation adapter:

`src/astrograph/evaluation/dxc_r27_expected_response.py`

combines the R25/R26 branches once per case and preserves R26 first-alarm timing while replacing only the final isolation/ranking semantics.

This separation is intentional: the diagnosis primitive can later be wired into a governed runtime provider without importing evaluation labels or Agent control into the diagnosis core.

## R27 expected-response semantics

R27 targets failure modes exposed by R26:

- a PositionSensor state fault can mimic a real Relay/CircuitBreaker state fault;
- `FailedOff`, `UnderSpeed` and `OverSpeed` can be confused by local waveform morphology;
- a persistent sensor plateau can be confused with a physical component failure;
- broad graph expansion increases candidate count without deciding which component-mode explains the observed downstream response.

The new ranker uses deterministic checks including:

- **discrete physical corroboration** — an uncommanded PositionSensor transition is retained as a controlled-component fault only when at least two admitted nearby numeric sensors show a strong contemporaneous physical response;
- **PositionSensor reclassification** — absent physical corroboration, an admitted `PositionSensor:Stuck` hypothesis is preferred instead of treating the controlled switch as failed;
- **fan mode direction** — SpeedTransmitter response direction distinguishes `OverSpeed`, `UnderSpeed` and `FailedOff` when those modes are explicitly present in the catalog;
- **pump flow response** — operating-to-final FlowTransmitter response can distinguish admitted `FailedOff` versus `FlowBlocked`;
- **inverter AC collapse** — `Inverter:FailedOff` is considered only with admitted adjacent AC voltage/frequency collapse rather than unconditional topology expansion;
- **sensor Stuck versus Offset** — a persistent suffix plateau supports `Stuck`; a large persistent shift without plateau supports `Offset`;
- **bounded multi-fault retention** — a secondary continuous/plateau hypothesis is retained only under narrow deterministic conditions.

All scores are ordering weights with semantics:

`DETERMINISTIC_EXPECTED_RESPONSE_PRIORITY_NOT_PROBABILITY`.

They are not calibrated probabilities.

## Development result and evidence boundary

R27 uses the same DXC09 Sample hash split already contacted by R25/R26. The R27 rules were developed after inspecting both `SELECTION` and `INTERNAL_REPLAY`. These results are therefore **post-contact development remediation evidence**, not independent validation.

### SELECTION — 11 injected-fault cases

R26:

- fault-element micro F1: `0.400000`;
- component micro F1: `0.400000`;
- exact fault-set rate: `0.272727`;
- any fault overlap: `0.454545`.

R27:

- fault-element precision/recall/F1: `0.937500 / 0.833333 / 0.882353`;
- component precision/recall/F1: `0.937500 / 0.833333 / 0.882353`;
- exact fault-set rate: **`0.818182`**;
- any fault overlap: **`0.909091`**;
- mean emitted fault count: `1.454545`.

### INTERNAL_REPLAY — 14 cases already inspected in R25

R26:

- fault-element micro F1: `0.466667`;
- component micro F1: `0.533333`;
- exact fault-set rate: `0.428571`;
- any fault overlap: `0.500000`.

R27:

- fault-element precision/recall/F1: **`0.750000 / 0.750000 / 0.750000`**;
- component precision/recall/F1: **`0.750000 / 0.750000 / 0.750000`**;
- exact fault-set rate: **`0.785714`**;
- any fault overlap: **`0.785714`**;
- mean emitted fault count: `1.142857`.

R27 deliberately retains R26 first-alarm timing. On INTERNAL_REPLAY this remains:

- early-before-injection: `1/14`;
- post-injection first alarm: `13/14`;
- median post-injection delay: `0.328 s`.

Thus the R27 development change is an **isolation/ranking change**, not a new detection-timing result.

## Remaining failures

R27 does not solve every DXC ambiguity. Remaining INTERNAL_REPLAY misses include:

- `IT267:Stuck` versus nearby `E265/ST265:Offset` explanations;
- `LGT481:FailedOff` versus current-sensor response;
- simultaneous `LGT405:FailedOff + LGT406:FailedOff` versus the observationally similar adjacent-light candidate.

These are evidence for the next step: explicit component behavior/state-transition equations and counterfactual simulation consistency, not further ad-hoc topology expansion.

## Agent review contract

`DXC_EXPECTED_RESPONSE_REVIEW` is intentionally a read-only shadow tool. The host computes and content-binds the `DXCExpectedResponseDecision` **before** the Agent sees the tool. Executing the tool validates and mirrors that decision for investigation/explanation.

The Agent cannot:

- rerun the ranker with different parameters;
- add or remove candidates;
- change `selected_fault_ids`;
- select another telemetry window;
- make the result a formal EvidenceFact;
- change diagnosis/release authority.

Turning the Agent off therefore does not remove the R27 diagnosis-core capability.

## ESA architecture remains dual-timescale

R27 does not retune ESA. The intended architecture remains:

- legacy 5-minute branch retained as the slow/trend path;
- R26 frozen high-resolution transient detector retained as a parallel fast candidate;
- fast-path promotion remains blocked until a genuinely new first-contact holdout is available;
- neither branch requires Agent execution.

## Claim boundary

R27 does not:

- regenerate DXC09 Competition or Rematch predictions;
- claim new external validation;
- change the production runtime default;
- remove the legacy statistical DXC branch;
- convert expected-response weights to posterior probabilities;
- make Agent review a prerequisite for detection/diagnosis;
- create formal EvidenceFacts or release authority;
- claim HIL, target-hardware or flight qualification.

## Combined post-contact development view

Across the 25 already-contacted injected-fault cases in `SELECTION + INTERNAL_REPLAY`, using the same R25/R26 detection timing:

- R26 fault-element F1: `0.433333`; component F1: `0.466667`; exact fault-set rate: `0.360000`; any overlap: `0.480000`.
- R27 fault-element F1: **`0.818182`**; component F1: **`0.818182`**; exact fault-set rate: **`0.800000`**; any overlap: **`0.840000`**.
- R27 mean predicted fault count: `1.280000`.
- detection timing is unchanged by construction: early-before-injection `3/25`, post-injection first alarm `21/25`, no alarm `1/25`, median post-injection delay `0.345 s`.

This combined view is descriptive development evidence only. It is not a new holdout because both constituent slices had already been inspected during remediation.

## Relationship to simulation

R27 does **not** claim that its measured development gain comes from a simulator. The gain comes from deterministic expected-response rules applied to admitted telemetry and the engineering catalog. The R23 `SIMULATION_COUNTERFACTUAL_REVIEW` capability remains available as an optional, separately admitted shadow investigation mechanism.

The next simulation-diagnosis step should add a governed forward behavior provider only where a real component model can produce candidate-specific expected trajectories. Until such a provider is admitted, AstroGraph must not relabel heuristic response rules as simulated physics.
