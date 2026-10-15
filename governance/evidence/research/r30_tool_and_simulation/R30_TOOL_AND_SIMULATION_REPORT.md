# R30 Agent Tool Governance + Mechanism Model / Blind Simulation Report

Status: `ENGINEERING_GOVERNANCE_AND_SYNTHETIC_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION`

## Tool governance

- governed static entries: 45
- Mission/Platform: 17
- controlled diagnostic capabilities: 10
- Diagnostic Probe kinds: 11
- low-level host-orchestrated ToolRegistry entries: 7
- trusted plugins and dynamic OAG evidence checks remain runtime extensions governed by separate allow-list/SHA-pin contracts.

Global policy: no Agent release authority, no arbitrary Python/path/channel/window/threshold/database query authority, and optional probes cannot be required for deterministic core detection/diagnosis.

Machine catalog: `agent_tool_governance_catalog.json`.

## Mechanism-model classification

R27–R29 are appropriately classified broadly as mechanism/model-based diagnosis, with fidelity qualifiers:

- R27: qualitative expected-response consistency;
- R28: qualitative component-mode/state-transition forward simulation;
- R29: numerical reduced-order electrical-load forward simulation.

R30 creates a formal ADAPT mechanism-model registry. On supplied AdaptDXC.xml: 173 components, 222 fault-mode instances, 222 mechanism entries, 207 qualitative-state entries, 5 declared-only entries, 10 executable numerical reduced-order entries. Numerical coverage is 4.5045% of the fault-mode instance universe.

## Synthetic blind simulation campaign

Fixed seed: 3001. Cases: 48. Prediction bytes are persisted before hidden truth join. Generator includes ±12% actual load-power perturbation, voltage variability, nuisance background load, sensor bias and noise.

Results:

- exact fault-set accuracy: 0.7500
- fault-element precision: 0.8309859155
- recall: 0.8194444444
- F1: 0.8251748252
- single-fault exact accuracy: 0.7500
- double-fault exact accuracy: 0.7500

This is intentionally `SYNTHETIC_BLIND_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION`. The generator and diagnosis provider share a broad resistive-load physics family, so the experiment is not independent mission/DXC evidence. The non-perfect result demonstrates model-mismatch sensitivity and motivates model uncertainty/multiple mechanism providers instead of development-case rule accumulation.

## Stop / next gate

Do not tune R29/R30 on these 48 synthetic cases. The next numerical provider must cover a different component family (recommended pump/flow, rotating machine, or inverter) using frozen engineering assumptions independent of these outcomes. Real promotion remains blocked on genuinely new holdout evidence.
