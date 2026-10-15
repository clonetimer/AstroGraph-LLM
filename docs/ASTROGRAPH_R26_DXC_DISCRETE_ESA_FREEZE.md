# AstroGraph PATCH-R26 — DXC discrete-state consistency / ESA frozen high-resolution detector

Status: **CURRENT_R26_DEVELOPMENT_REMEDIATION_AND_FROZEN_REPLICATION_CANDIDATE**  
Date: 2026-09-19

## Scope

R26 continues from the final PATCH-R25 unified source artifact. It does not reopen DXC09 Competition/Rematch or ESA seen confirmatory/external partitions for fitting, tuning, selection or prediction regeneration.

R26 closes two concrete gaps exposed by R25:

1. DXC continuous residual diagnosis excluded bool `PositionSensor` observations because that path intentionally requires verified engineering units and numeric values. R26 adds a **separate discrete-state consistency contract** rather than weakening the continuous unit gate.
2. ESA R25 selected a high-resolution development candidate but had not yet frozen it as a content-bound future-holdout artifact. R26 freezes that exact reference/config and adds truth-free prediction and new-holdout binding contracts.

No output in this patch is calibrated probability, formal EvidenceFact, external validation or flight qualification.

## DXC discrete-state consistency

ADAPT XML contains 33 `PositionSensor` instances. Each admitted position sensor is bool-valued and has exactly one direct engineering-topology neighbor: a Relay or CircuitBreaker instance. The branch consumes only this exact XML structure and exact instance-level command bindings.

The deterministic rule is deliberately narrow:

- position state is handled as a discrete semantic value, so missing engineering units do not make it a continuous measurement;
- transitions within `1.0 s` after an explicit command to the same component are suppressed as commanded behavior;
- after a `30 s` startup guard, an uncommanded closed→open transition maps only to an adjacent admitted `StuckOpen` or `FailedOpen` mode;
- an uncommanded open→closed transition maps only to adjacent admitted `StuckClosed`;
- the branch remains development-only and `formal_evidence_eligible=false`.

R26 keeps the R25 numeric residual/MBD branch unchanged and evaluates four finite fusion policies on the R25 frozen hash split. Selection is explicitly **post-development-contact**, not a preregistration claim.

Primary selection criterion is strict DXC isolation: exact fault-set rate, then fault-element F1, component F1, then fewer emitted faults. The selected policy is:

`DISCRETE_FIRST_SINGLE_CONTINUOUS`

Semantics:

- if exact discrete-state inconsistencies exist, they have higher engineering authority than the continuous residual heuristic;
- otherwise only the highest-ranked continuous component hypothesis is emitted;
- this intentionally limits the R20/R25 over-isolation failure mode.

### Selection slice — 11 cases

- fault-element precision / recall / F1: `0.500000 / 0.333333 / 0.400000`;
- component precision / recall / F1: `0.500000 / 0.333333 / 0.400000`;
- exact fault-set rate: **0.272727**;
- any fault-element overlap: `0.454545`;
- mean predicted fault count: `1.090909`.

### Internal replay — 14 cases, already seen in R25

R26 selected fusion:

- fault-element precision / recall / F1: **0.500000 / 0.437500 / 0.466667**;
- component precision / recall / F1: **0.571429 / 0.500000 / 0.533333**;
- exact fault-set rate: **0.428571**;
- any fault-element overlap: **0.500000**;
- post-injection first-alarm rate: **0.928571**;
- median post-injection delay: **0.328 s**;
- mean predicted fault count: **1.0**.

The unchanged R25 continuous branch on the same 14 already-seen cases was:

- fault-element F1: `0.095238`;
- component F1: `0.190476`;
- exact set: `0`;
- any overlap: `0.142857`;
- post-injection first-alarm rate: `0.857143`;
- median delay: `0.4455 s`;
- mean predicted fault count: `1.857143`.

This is strong **development remediation evidence**, not independent validation. The 14-case internal replay had already been inspected in R25.

## Remaining DXC ambiguity

The new branch also exposes the next model-based diagnosis limit more precisely. A `PositionSensor:Stuck` fault can produce a frozen bool value that is observationally confusable with a real adjacent Relay/CircuitBreaker state fault unless downstream electrical/physical response equations are available. R26 therefore does not claim that the discrete rule alone is a complete HyDE/GDE-style hybrid component model.

Next DXC work should add explicit component behavior equations / expected downstream response, particularly for:

- `PositionSensor:Stuck` versus actual Relay/CircuitBreaker state faults;
- `FailedOff`, `FlowBlocked`, `OverSpeed`, `UnderSpeed` load faults;
- multi-fault consistency and diagnosis ranking.

## ESA frozen high-resolution detector

R26 performs **no new ESA model selection**. It freezes exactly the R25 selected development detector:

- robust lag-one delta z threshold = **8**;
- minimum coincident lightweight channels = **2**;
- alarm grouping gap = **40 s**;
- reference SHA256 = `7f3d73247e3ec2bed5c1b0461026785fa87a4a40405c47a81f890e0284b6553e`;
- R25 development receipt SHA256 = `f3267ca7de06ed1a96b364991c07cd725a7dc23bf468ee9249f6ccae2959ab55`.

`astrograph-r26-esa-frozen-detector-artifact.v1` binds the development receipt, reference, frozen config and exact implementation-source bytes.

The future holdout flow is:

```text
new raw source bytes
  -> R22 first-contact intake/source novelty
  -> R22 group/task/label/universe identity
  -> R26 frozen detector binding BEFORE truth
  -> raw-grid prediction receipt
  -> optional truth join / evaluation
```

The prediction receipt binds the time array, every channel value array, detector identity, score hash and alarm intervals while explicitly recording `truth_joined=false`.

A synthetic protocol smoke verifies this execution path only. It is not performance evidence.

No genuinely new ESA holdout was supplied in R26, so the preserved seen confirmatory result remains:

- decision points: `73,641`;
- alarm points: `0`;
- true anomaly events: `4`;
- TP: `0`;
- FN: `4`.

## Governance boundary

R26 does not:

- regenerate DXC Competition/Rematch predictions;
- regenerate ESA seen confirmatory/external predictions;
- use a bool state observation to weaken numeric engineering-unit admission;
- create formal EvidenceFact;
- change Agent authority;
- claim calibrated probability, external validation, HIL or flight qualification.

The machine-readable protocol and result artifacts are under:

`governance/evidence/research/r26_remediation/`
