# AstroGraph PATCH-R31 — WaterPump reduced-order mechanism simulation

Status: **CURRENT_R31_SECOND_NUMERICAL_MECHANISM_FAMILY**  
Date: 2026-09-19

## 1. Scope

R31 implements the second executable numerical ADAPT mechanism family after R29's resistive light-load model. The scope is the public ADAPT `WaterPump` component and its two documented fault modes:

- `FailedOff` — pumps no water and draws no current when electrically supplied;
- `FlowBlocked` — water throughput falls because of an obstruction.

The public ADAPT XML also supplies `nominalWattage=35` and describes nominal flow as approximately `250 GPH`. R31 extracts those values from the admitted XML and binds them to the XML SHA256; they are not LLM-provided constants.

## 2. Reduced-order model

R31 couples electrical and hydraulic observables.

For pre-transition voltage \(V_0\), branch current \(I_0\) and pump rated power \(P\), the pump rated-current contribution is

\[
I_{pump,0}=P/V_0.
\]

The residual branch load is

\[
I_{bg,0}=\max(I_0-I_{pump,0},0).
\]

For `FailedOff`, pump electrical power is removed and post-transition current is the background-load response under the admitted bus-voltage trace. For `FlowBlocked`, the pump remains electrically energized in the reduced-order hypothesis and the pump current is approximated by \(P/V(t)\).

Flow uses a first-order transition

\[
F(t)=F_0\left[\alpha +(1-\alpha)e^{-(t-t_0)/\tau}\right].
\]

- `FailedOff`: \(\alpha=0\);
- `FlowBlocked`: \(\alpha\in[0,0.95]\) is a truth-blind nuisance fit;
- \(\tau\) is selected only from a fixed grid derived from admitted sample spacing: `1x/2x/4x/8x` median sample interval.

The fitted \(\alpha\) and \(\tau\) are trace-fit nuisance parameters. They are not posterior probabilities and are not claimed to be calibrated pump constants.

## 3. Diagnosis authority

R31 is deliberately a **corroborator**:

```text
R29 selected pump fault mode
        |
        v
R31 simulate FailedOff and FlowBlocked
        |
        v
current + flow trace-distance comparison
        |
        +-- supports base mode
        |
        +-- contradicts base mode
                |
                v
       NO AUTOMATIC OVERRIDE
```

`DXCWaterPumpPhysicsDecision.selected_fault_ids` remains identical to the base diagnosis. A physical disagreement is surfaced deterministically for later governed review rather than silently rewriting the diagnosis. `auto_override_enabled=false` is a schema invariant.

## 4. Agent tool boundary

R31 adds `DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW`.

The Agent can select only a host-generated `probe_id`. The probe mirrors an already computed, hash-bound `DXCWaterPumpPhysicsDecision`. The Agent cannot change:

- pump ratings or nominal flow;
- current/voltage/flow channels;
- event window or transition time;
- topology traversal;
- nuisance-fit grid;
- base or preferred fault mode;
- EvidenceFact, diagnosis authority, or release state.

Core prediction does not depend on Agent execution.

## 5. Mechanism-registry coverage

With R31, the supplied ADAPT catalog still has 222/222 fault-mode instances represented in the mechanism registry, while executable numerical reduced-order coverage increases from R30's 10/222 to:

- numerical reduced-order entries: **14**;
- qualitative state-response entries: **203**;
- declared-only entries: **5**;
- numerical coverage: **14/222 = 6.3063%**.

The four newly numerical entries are:

- `PMP420:FailedOff`;
- `PMP420:FlowBlocked`;
- `PMP425:FailedOff`;
- `PMP425:FlowBlocked`.

100% mechanism-registry coverage still must not be reported as 100% numerical simulation coverage.

## 6. Public DXC09 Sample mechanism-fit audit

Two already-contacted DEVELOPMENT examples contain WaterPump faults (`PMP420:FailedOff` and `PMP420:FlowBlocked`). R31 uses their known fault-injection timestamp only to align the physical-fit window. With a fixed base activation, the reduced-order model orders the correct pump mode first in both cases.

This is **TRUTH_ALIGNED_PUBLIC_SAMPLE_MECHANISM_FIT_AUDIT_NOT_ONLINE_DIAGNOSIS**. It cannot support detection-delay, online localization/classification, independent-validation, or SOTA claims.

## 7. Blind synthetic campaign

R31 freezes a separate truth-separated synthetic mode-classification campaign:

- hidden modes: `FailedOff` / `FlowBlocked`;
- fixed seed: `3101`;
- 40 cases per mode, 80 total;
- prediction bytes frozen before hidden mode truth is serialized;
- transition time is host-admitted, so this tests event-conditional mode classification, not detection;
- generator differs from the diagnosis model through pump-power perturbation, nominal-flow perturbation, electrical transient inertia, flow time constant, blockage severity, background branch load, voltage variation and sensor noise.

Frozen result:

- accuracy: **1.0000**;
- macro F1: **1.0000**;
- `FailedOff` recall: **1.0000**;
- `FlowBlocked` recall: **1.0000**.

This perfect synthetic result must not be generalized to DXC or mission data. The two modes have a strong electrical discriminant in this simulator family: `FailedOff` removes rated electrical load whereas `FlowBlocked` preserves pump energization. The campaign is therefore protocol/model-family evidence only, not external validation.

Stop rule: `NO_PARAMETER_OR_RULE_OPTIMIZATION_ON_R31_PUMP_BLIND_OR_DXC09_SAMPLE_OUTCOMES`.

## 8. Claim boundary

R31 does not claim end-to-end pump detection performance, DXC Competition/Rematch improvement, independent external validation, calibrated probability, full electrical/hydraulic network simulation, VirtualADAPT/Simulink equivalence, HIL evidence, production-default promotion, or flight qualification.
