# AstroGraph PATCH-R30 — Mechanism-model registry and blind simulation protocol

Status: **CURRENT_R30_MECHANISM_MODEL_LIBRARY_AND_SYNTHETIC_BLIND_PROTOCOL**  
Date: 2026-09-19

## 1. Classification of R27–R29 diagnosis methods

The R27–R29 chain can be classified broadly as **model-based / mechanism-model-based diagnosis**, with materially different fidelity levels:

- **R27** — deterministic expected-response consistency: qualitative mechanism/causal-response rules over command, topology and telemetry. It is model-based but not a numerical physics simulator.
- **R28** — qualitative component-mode forward simulation: explicit fault mode -> state transition -> expected response -> observation consistency. It is a qualitative hybrid/model-based diagnosis layer.
- **R29** — numerical reduced-order physical simulation: candidate-specific electrical-load time-series simulation using rated power, branch voltage/current boundary conditions and physical consistency checks. This is a reduced-order physics-based diagnosis method.

The broader label “基于机理模型的方法” is therefore appropriate, provided the implementation is **not** described as a full high-fidelity digital twin, full HyDE/ProADAPT equivalent, or continuous multi-domain plant solver.

## 2. R30 ADAPT mechanism-model registry

R30 introduces `astrograph-adapt-mechanism-model-registry.v1`. The registry covers the admitted ADAPT fault-mode instance universe with an explicit distinction between:

1. `NUMERICAL_REDUCED_ORDER` — an executable numerical provider exists;
2. `QUALITATIVE_STATE_RESPONSE` — state/response semantics are explicit but no numerical trajectory is claimed;
3. `DECLARED_ONLY_NOT_EXECUTABLE` — the mechanism is recorded for coverage/governance only.

On the supplied `AdaptDXC.xml` catalogue:

- components: **173**;
- fault-mode instances: **222**;
- mechanism registry entries: **222**;
- qualitative state-response entries: **207**;
- declared-only entries: **5**;
- numerical reduced-order entries: **10**;
- numerical reduced-order coverage of the fault universe: **4.5045%**.

The 100% mechanism-registry coverage must not be misreported as 100% numerical simulation coverage. R30 currently has numerical execution only for `LightBulb*W:FailedOff` through `adapt-resistive-lightbulb-reduced-order-v1`.

The registry also declares mechanism families for switch/position transitions, rotating machines, water-pump flow response, inverter output collapse, sensor measurement faults, battery degradation and passive-load failures. These declarations are intended to drive future provider implementation and coverage auditing.

## 3. Blind synthetic simulation campaign

R30 adds a truth-free prediction protocol for the reduced-order provider. The generator creates hidden single- and double-load failures under deliberate model mismatch:

- actual lamp power perturbed ±12%;
- variable bus voltage;
- background conductance plus constant-current nuisance load;
- voltage/current bias and measurement noise.

The diagnosis side receives only observed voltage/current traces and a fixed candidate universe. It does **not** receive `hidden_fault_ids`. Predictions are serialized to `predictions_before_truth_join.json` before truth is joined.

Frozen seed `3001` produces **48** cases. Result:

- exact fault-set accuracy: **0.7500**;
- fault-element precision: **0.8310**;
- recall: **0.8194**;
- F1: **0.8252**;
- single-fault exact accuracy: **0.7500**;
- double-fault exact accuracy: **0.7500**.

This result is intentionally not “perfect”. It demonstrates that R29’s reduced-order model loses discrimination under parameter/model mismatch and therefore needs uncertainty-aware/multi-model extensions. It is `SYNTHETIC_BLIND_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION`.

## 4. Architecture after R30

```text
legacy DXC statistical branch (retained)
          |
R25 command-aware residual
          |
R26 discrete-state consistency
          |
R27 expected-response core
          |
R28 qualitative component-mode forward simulation
          |
R29 selective numerical reduced-order resolver
          |
R30 mechanism-model registry / coverage governance
          |
future executable providers: pump / fan / inverter / switch / battery

Agent layer: optional read-only review only
```

The deterministic core remains independent of Agent execution.

## 5. Next implementation gate

R31 should not optimize the 48 synthetic R30 cases. The next useful work is to implement at least one **different physical family** with parameters frozen from system/model documentation rather than from R30 outcomes. Recommended order:

1. `WaterPump:FailedOff / FlowBlocked` using flow + electrical response;
2. `LargeFan/SmallFan` speed dynamics and electrical response;
3. `Inverter:FailedOff` AC voltage/frequency collapse;
4. switch/circuit-breaker command/state transition model;
5. only then multi-domain coupled simulation and real new-holdout evaluation.

## 6. Claim boundary

R30 does not produce new DXC Competition/Rematch validation, does not use seen ESA external data, does not establish SOTA, does not make synthetic blind cases statistically independent mission evidence, and does not promote Agent probes to formal evidence authority. The blind campaign is a protocol/generalization smoke in a related physics family, not an external benchmark.
