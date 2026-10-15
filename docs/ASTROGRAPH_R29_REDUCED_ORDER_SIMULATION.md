# AstroGraph PATCH-R29 — ADAPT reduced-order electrical load simulation diagnosis

Status: **CURRENT_R29_POST_CONTACT_DEVELOPMENT_CANDIDATE_REQUIRES_NEW_HOLDOUT**  
Date: 2026-09-19

## Purpose

R29 moves one bounded DXC path beyond R28 qualitative expected-response rules into a candidate-specific numerical time-series simulation. The initial scope is deliberately narrow: `LightBulb*W:FailedOff` on an ADAPT AC branch with admitted voltage/current telemetry and optional local thermal/relay-state evidence.

The architecture remains:

`legacy retained + R28 deterministic core + selective R29 reduced-order physical resolver + optional read-only Agent review`.

R29 does not replace the legacy statistical branch, does not require Agent execution, and does not change the production/runtime default.

## What R28 did

R28 was a deterministic qualitative component-mode forward model, not a new ML classifier and not an Agent detector. It mapped a bounded candidate fault to an explicit state transition and expected telemetry response, then supported, vetoed, or left the hypothesis unresolved according to observed response consistency. Its first bounded contracts covered numeric sensor `Stuck`, large additive sensor `Offset`, and `LightBulb25W:FailedOff`. Score semantics were `DETERMINISTIC_FORWARD_MODEL_FIT_NOT_PROBABILITY`; it explicitly recorded `continuous_physics_simulator=false`.

On the already-contacted 25-case DXC09 Sample development corpus, R28 raised fault/component micro F1 from R27 `0.818182` to `0.985075`, exact-set from `0.80` to `0.96`, while deliberately leaving one case unresolved. These are post-contact development metrics only.

## R29 reduced-order physical model

R29 adds `ADAPT_RESISTIVE_LOAD_BRANCH_REDUCED_ORDER_V1` through the existing governed `DiagnosticSimulationProvider` interface.

For each admitted `LightBulb*W:FailedOff` hypothesis the host derives, without ground truth or case-ID rules:

1. rated load power from the ADAPT component type (`LightBulb25W`, `LightBulb55W`, `LightBulb60W`, ...);
2. the nearest admitted branch AC current and AC voltage sensors from the engineering graph;
3. optional local temperature and relay-position sensors;
4. related relay command bindings;
5. the already-frozen R28 first-alarm time as the candidate state-transition time.

The observed bus voltage is treated as an exogenous boundary condition. For a candidate failed-load set, the reduced-order branch model removes the candidate nominal current and predicts the remaining branch-current trajectory under the observed voltage trajectory.

For a candidate set with total rated power `P`, pre-transition voltage `V_pre` and pre-transition branch current `I_pre`:

`I_candidate ~= P / V_pre`

`I_other_pre = max(I_pre - I_candidate, 0)`

`I_pred(t) = I_other_pre * V(t) / V_pre`

The predicted trace is compared with the observed branch current using normalized RMSE/MAE. Admission also checks:

- stable-enough AC voltage;
- significant uncommanded current loss;
- rated-power/current-drop consistency;
- expected local cooling when a candidate has a local temperature sensor;
- unexplained cooling in an unmodeled sibling load;
- relay-position consistency;
- related command activity around onset.

Only one surviving physically supported candidate set may resolve an R28 unresolved case. Multiple surviving candidates remain unresolved rather than being separated by a small post-contact distance difference.

## Simulation semantics and limits

R29 records:

- `numerical_time_series_simulation=true`;
- `simulation_semantics=REDUCED_ORDER_ELECTRICAL_LOAD_TIME_SERIES_FORWARD_SIMULATION`;
- `continuous_network_solver=false`;
- score semantics `REDUCED_ORDER_TRACE_DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY`.

This is not NASA VirtualADAPT/Simulink and not a full electrical/thermal network solver. It does not integrate differential/algebraic system equations, model converter transients, identify unobserved physical parameters, or produce posterior fault probabilities.

Simulation hypotheses use the neutral `SimulationHypothesisCandidate` contract. They do not assert that the candidate is a conflict-hitting or minimal diagnosis merely because it is being counterfactually simulated.

## Agent boundary

The R29 core executes with Agent OFF. A new read-only probe `DXC_REDUCED_ORDER_SIMULATION_REVIEW` may mirror the already-computed and content-bound decision to the Agent. The Agent cannot alter telemetry, candidate generation, component ratings, physics parameters, transition time, thresholds, selected faults, formal EvidenceFacts, or release state.

## Development result and contamination boundary

R29 was implemented after the sole R28 unresolved development case had already been inspected. Therefore no R29 DXC09 Sample number is independent validation.

The fixed R29 design was executed once over the same 25 already-contacted DXC09 Sample development cases. R28-resolved cases are passed through unchanged; the physical resolver applies to only the generic R28 unresolved physical-response disposition.

Observed post-contact development result:

| Metric | R28 | R29 |
|---|---:|---:|
| fault-element micro F1 | 0.985075 | 1.000000 |
| component micro F1 | 0.985075 | 1.000000 |
| exact fault-set rate | 0.960000 | 1.000000 |
| diagnostic resolution | 0.960000 | 1.000000 |

Exactly one case changed: `Exp_658_pb_t2`, from R28 unresolved to `LGT481:FailedOff`. The resolver used the general graph/rated-power/branch-current/temperature/command contract; nevertheless, because implementation followed inspection of that unresolved case, the apparent 1.0 development score is **post-contact closure evidence only** and cannot support a SOTA, external-validation, production-default, or flight claim.

Stop rule after this fixed audit:

`NO_PARAMETER_SEARCH_ON_DXC09_SAMPLE_AFTER_FIXED_R29_DESIGN`.

Further validation must use genuinely new holdout data or another independently admitted physical/simulation corpus.

## Promotion boundary

R29 remains:

`FROZEN_POST_CONTACT_DEVELOPMENT_CANDIDATE_REQUIRES_NEW_HOLDOUT`.

It does not claim Competition/Rematch improvement, independent external validation, VirtualADAPT equivalence, calibrated probability, formal EvidenceFact authority, HIL/target-hardware qualification, or flight qualification.
