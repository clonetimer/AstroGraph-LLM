# AstroGraph PATCH-R28 — DXC qualitative component-mode forward simulation

Status: **CURRENT_R28_POST_CONTACT_DEVELOPMENT_PRIMARY_CANDIDATE**  
Date: 2026-09-19

## Architecture decision

R28 keeps the architecture decision established in R27:

1. the legacy R20 statistical DXC branch is retained;
2. the R25/R26/R27 deterministic hybrid/expected-response branch is retained;
3. R28 adds a second deterministic refinement stage based on explicit component/fault state transitions and qualitative forward-response consistency;
4. Agent tools remain optional read-only investigation surfaces and are not required to produce the R28 prediction.

The implementation form is therefore:

`legacy retained + R27 deterministic core + R28 deterministic forward-model refinement + optional Agent shadow review`

not:

`raw telemetry -> Agent chooses a detector -> selected tool output becomes the diagnosis`.

R28 does not switch the production/runtime default. It remains a `PRIMARY_CAPABLE_DEVELOPMENT_CANDIDATE` pending genuinely new holdout evidence.

## Relationship to published model-based diagnosis

NASA's Hybrid Diagnosis Engine (HyDE) describes system models using component operating modes, mode transitions and behavior relations, then maintains/repairs candidates by checking consistency with observations. NASA descriptions of HyDE on ADAPT also emphasize the combination of discrete component modes, continuous behavior and command/condition-triggered transitions.

R28 adopts that *structural idea* but is not a reimplementation of HyDE. The R28 model is deliberately narrower:

- it implements a qualitative forward model only for fault modes with an explicit telemetry response contract in the currently admitted ADAPT catalog/data;
- it does not integrate differential/algebraic electrical equations;
- it does not assign transition probabilities;
- it does not claim complete system-state estimation;
- unsupported modes retain the upstream R27 result rather than receiving invented physics.

Public background used for the design review:

- NASA NTRS, *HyDE Framework for Stochastic and Hybrid Model-Based Diagnosis*, document 20120007370;
- NASA/AIAA, descriptions of HyDE component modes and command/condition transitions in integrated autonomy demonstrations;
- NASA DXC/ProDiagnose publications describing explicit component/sensor health and command/sensor evidence for ADAPT.

## R28 modeled state transitions

The first R28 behavior library covers four bounded contracts.

### 1. Numeric sensor `Stuck`

Transition:

`nominal observation -> STUCK_OBSERVATION`

Expected response:

- abrupt own-channel change;
- persistent suffix freeze;
- no need to reinterpret neighboring physical components merely because their waveform morphology changed.

### 2. Numeric sensor `Offset`

Transition:

`nominal observation -> OFFSET_OBSERVATION`

Expected response:

- a large additive step;
- post-step local dynamics remain non-zero;
- the shift persists;
- the event is not admitted as a sensor offset when it is too close to a commanded plant transition;
- early startup transients are excluded by a frozen guard.

This branch is intentionally conservative. A sensor-only R27 `Offset` candidate that cannot be reproduced by the explicit observation-fault model becomes `UNRESOLVED` rather than being silently retained as a false sensor diagnosis.

### 3. `LightBulb25W:FailedOff`

Transition:

`energized -> OFF`

The engineering graph supplies the sibling light group, local TemperatureSensor and shared LightSensor relationships. The forward response requires:

- shared light-output decrease; and
- local cooling relative to sibling bulbs on the same branch.

This enables candidate-specific multi-load isolation without selecting a neighboring bulb solely because a common current sensor changed.

### 4. Unmodeled component/fault modes

Transition:

`UNMODELED_PASSTHROUGH`

R28 does not invent behavior. The R27 candidate remains unchanged unless one of the explicitly modeled contracts supports or contradicts it.

## Forward-model result semantics

The core result is `DXCForwardResponseDecision` with:

- selected fault IDs;
- R27 base fault IDs;
- explicit state transitions;
- modeled response fits;
- vetoed fault IDs;
- unresolved reason codes;
- content-bound result SHA256.

Score semantics are:

`DETERMINISTIC_FORWARD_MODEL_FIT_NOT_PROBABILITY`.

Simulation semantics are:

`QUALITATIVE_COMPONENT_MODE_FORWARD_SIMULATION`.

The object explicitly records `continuous_physics_simulator=false`.

## Development experiment boundary

R28 consumes only the already-contacted DXC09 Sample `DEVELOPMENT` groups and the frozen R27 development predictions. No DXC Competition or Rematch prediction is regenerated.

Both `SELECTION` and `INTERNAL_REPLAY` had already been inspected during R25-R27 remediation. Therefore every R28 metric below is **post-contact development evidence**, not independent validation.

### SELECTION — 11 cases

R28:

- fault-element precision / recall / F1: `1.000000 / 1.000000 / 1.000000`;
- component precision / recall / F1: `1.000000 / 1.000000 / 1.000000`;
- exact fault-set rate: `1.000000`;
- diagnostic resolution rate: `1.000000`.

### INTERNAL_REPLAY — 14 cases

R28:

- fault-element precision / recall / F1: `1.000000 / 0.937500 / 0.967742`;
- component precision / recall / F1: `1.000000 / 0.937500 / 0.967742`;
- exact fault-set rate: `0.928571`;
- diagnostic resolution rate: `0.928571`;
- resolved-case exact fault-set rate: `1.000000`.

### Combined contacted development — 25 cases

R27:

- fault-element F1: `0.818182`;
- component F1: `0.818182`;
- exact fault-set rate: `0.800000`;
- any-overlap rate: `0.840000`.

R28:

- fault-element precision / recall / F1: **`1.000000 / 0.970588 / 0.985075`**;
- component precision / recall / F1: **`1.000000 / 0.970588 / 0.985075`**;
- exact fault-set rate: **`0.960000`**;
- any-overlap rate: **`0.960000`**;
- diagnostic resolution rate: **`0.960000`**;
- resolved-case exact fault-set rate: **`1.000000`**;
- mean emitted fault count: `1.320000`.

The resolved-case `1.0` figure must never be reported without the `0.96` resolution/coverage figure. One case is deliberately unresolved.

## Changed cases and stop rule

R28 changes exactly five contacted-development cases relative to R27:

- `Exp_665_pb_t2ff`: restores the second `ST265:Offset` observation fault beside `E281:Stuck`;
- `Exp_676_pb_t2`: resolves the two-bulb `LGT400 + LGT402 FailedOff` response from local thermal/shared-light behavior;
- `Exp_644_pb_t2f`: replaces neighboring Offset explanations with the directly observed `IT267:Stuck` freeze transition;
- `Exp_692_pb_t2`: resolves the two-bulb `LGT405 + LGT406 FailedOff` response;
- `Exp_658_pb_t2`: vetoes an uncorroborated `IT161:Offset` and returns unresolved rather than inventing which uninstrumented branch load failed.

The last case is important. The current admitted telemetry shows a coherent physical load-loss response but does not provide a uniquely modeled local response for `LGT481`. R28 therefore chooses abstention over a false sensor diagnosis.

The R28 development stop rule is:

`STOP_AFTER_FIXED_R28_MODEL_AUDIT_NO_FURTHER_CASE_SPECIFIC_RULE_OPTIMIZATION`.

No additional rule is added to force `LGT481` from the already-observed development case.

## Agent surface

R28 adds the read-only probe kind:

`DXC_FORWARD_SIMULATION_REVIEW`.

The host computes and hashes the `DXCForwardResponseDecision` before the Agent receives the probe catalog. The Agent can inspect:

- selected/base/vetoed fault IDs;
- modeled transitions and response fits;
- unresolved reason codes;
- result identity and score semantics.

The Agent cannot:

- rerun the forward model;
- edit thresholds or transition semantics;
- add/remove candidates;
- turn unresolved into a diagnosis;
- create a formal EvidenceFact;
- change diagnosis/release authority.

Therefore Agent OFF and Agent ON use the same core R28 prediction.

## Relationship to the R23 simulation provider

R23's generic `SIMULATION_COUNTERFACTUAL_REVIEW` remains useful when an independently admitted simulator can provide candidate-specific traces. R28 does not replace it. R28 instead supplies a built-in zero-extra-dependency qualitative behavior model for a bounded subset of ADAPT component modes.

A future continuous simulator may implement electrical/thermal dynamics and feed counterfactual traces through the existing simulation-provider interface. Such a provider must be validated independently; R28 development accuracy does not establish its correctness.

## Claim boundary

R28 does not claim:

- DXC Competition or Rematch improvement;
- independent external validation;
- complete HyDE/ProADAPT equivalence;
- continuous electrical/thermal simulation;
- posterior probabilities;
- production-default promotion;
- formal EvidenceFact authority;
- Agent-dependent diagnostic improvement;
- HIL, target-hardware or flight qualification.
