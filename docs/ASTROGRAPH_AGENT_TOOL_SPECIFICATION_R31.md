# AstroGraph Agent Tool Governance Specification — R31

Status: **CURRENT_R31_AGENT_TOOL_GOVERNANCE**  
Date: 2026-09-19

R31 preserves the R30 governance model and adds one read-only WaterPump mechanism review probe. The machine-readable authority remains `astrograph-agent-tool-governance-catalog.v1`, policy version `R31-v1`.

## Governed static surfaces

The current static catalog contains **46** entries:

- Mission / Platform tools: **17**;
- Controlled Diagnostic capabilities: **10**;
- Diagnostic Probe kinds: **12**;
- host-orchestrated low-level ToolRegistry tools: **7**.

Dynamic OAG EvidenceCheck tools and trusted plugin extensions remain governed by their separate runtime registry/snapshot policies and are not counted as static entries until admitted.

## Diagnostic Probe kinds

1. `EVIDENCE_THRESHOLD_MARGIN`
2. `PRE_POST_CHANGE`
3. `COMMAND_RESPONSE_RESIDUAL`
4. `LAG_CORRELATION`
5. `OSCILLATION_SIGNATURE`
6. `SATURATION_LIMIT_CHECK`
7. `MINIMAL_DIAGNOSIS_REVIEW`
8. `SIMULATION_COUNTERFACTUAL_REVIEW`
9. `DXC_EXPECTED_RESPONSE_REVIEW`
10. `DXC_FORWARD_SIMULATION_REVIEW`
11. `DXC_REDUCED_ORDER_SIMULATION_REVIEW`
12. `DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW`

The R31 WaterPump probe mirrors one already computed `DXCWaterPumpPhysicsDecision`. It is not a simulator configuration interface.

## Global invariants

All probe kinds remain:

```text
invocation authority    = AGENT_SELECTS_HOST_BOUND_PROBE_ID_ONLY
evidence authority      = SHADOW_DERIVED_OBSERVATION_ONLY
diagnosis authority     = SHADOW_REVIEW_ONLY
formal release authority= false
truth access            = FORBIDDEN
core prediction depends on probe = false
```

The Agent cannot supply arbitrary Python, file paths, channels, time windows, thresholds, database queries, model parameters or simulator parameters.

A new tool or probe must declare at minimum:

- stable tool identity and runtime ID;
- surface and purpose;
- agent-selectability;
- invocation/parameter authority;
- JSON input/output contract;
- permissions and side effects;
- task/work-mode admission;
- truth-access policy;
- Evidence authority;
- Diagnosis authority;
- Release authority;
- provenance requirement;
- fail-closed policy.

The catalogue is governance metadata and does not itself make a tool executable. Runtime registries, task snapshots, telemetry/evidence admission and independent governors remain authoritative.
