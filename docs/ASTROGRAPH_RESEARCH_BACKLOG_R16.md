<!-- CURRENT_DOCUMENT: 0.7.1.post49 -->
# AstroGraph Research / Engineering Backlog after PATCH-R16

Status: **CURRENT_RESEARCH_BACKLOG**  
Date: 2026-09-18  
Baseline: PATCH-R16 unified source

## Closed in R16

`R14-REP-001` is closed at the **engineering reproducibility-identity** level. Formal confirmatory execution now binds source/checksum identity, prompt and inference profile, Python environment/SBOM, model/tokenizer content and revision, model-server binary/container/launch identity, and accelerator/driver/CUDA observations, with post-run re-observation.

This closure is deliberately not an authenticity or qualification claim. Remote producer authenticity remains part of `R14-UP-001`.

`BASE-R16-01` is also closed: frozen manifest `group_id` is now propagated into live ablation rows.

`BASE-R16-02` is closed: `execution_receipt.json` is no longer mutated after its self-hash is computed, and post-run identity verification rejects a tampered receipt.

## Remaining findings

| ID | Severity | Status | Next action |
|---|---|---|---|
| R14-VAL-001 | P1-RESEARCH | KNOWN_BOUNDARY | Validation ladder: frozen simulation -> cross-simulator/domain-shift -> HIL/PIL fault injection -> target compute/resource/stress -> operational shadow -> independent release review. |
| R14-MOD-001 | P1-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | Stop representing uncalibrated normalized heuristic RW scores as probabilities; add score semantics/calibration receipts. |
| R14-TMP-001 | P1-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | Calibrate time-axis admission, baseline policy, false alarms/ARL, detection delay and cadence/window sensitivity. |
| R14-ADM-001 | P1-GOVERNANCE | IMPLEMENTED_NEEDS_OPTIMIZATION | Bind acquisition source, sensor/calibration version, cadence/gap/coverage, uncertainty and authenticated provenance. |
| R14-OOD-001 | P1-RESEARCH | IMPLEMENTED_NEEDS_EXTERNAL_VALIDATION | Evaluate risk-coverage/OOD behavior across independent simulator/HIL/corruption/mission-condition sets. |
| R14-FUS-001 | P1-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | Preregister BiRRF weight/smoothing/source-dependence sensitivity and simpler fusion baselines. |
| R14-SCP-001 | P1-RESEARCH | OPEN | Separate single-fault scope from compound/cascade/unknown-fault campaigns. |
| R14-MET-001 | P1-RESEARCH | OPEN | Add false alarms per operating time, missed-event rate, detection/isolation delays, abstention duration and action/resource costs. |
| R14-SAM-001 | P1-RESEARCH | OPEN | Expand independent trajectory/severity/mode groups and plan sample size on independent groups. |
| R14-PKG-001 | P2-ENGINEERING | MERGE_RESOLVED_WITH_FOLLOWUP | If minimal deployment matters, move scientific model integrations behind explicit optional extras with core-only import tests. |
| R14-MNT-001 | P2-ENGINEERING | OPEN | Split large control-plane modules and replace broad exception catches at trust boundaries with typed reason codes. |
| R14-NUM-001 | P2-ENGINEERING | LATENT | Add explicit tolerance/quantization semantics for continuous-value EQUAL/NOT_EQUAL. |
| R14-UP-001 | P2-GOVERNANCE | IMPLEMENTED_NEEDS_OPTIMIZATION | Add authenticated provider/attestation signatures, challenge/nonce freshness, authorization and raw-input provenance. R16 remote attestation explicitly remains unauthenticated. |
| R14-KG-001 | P2-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | Edge provenance/review grades, causal-vs-associational semantics, contradiction/age handling and blinded leave-edge/source-out utility evaluation. |
| R14-GND-002 | P2-RESEARCH | IMPLEMENTED_NEEDS_OPTIMIZATION | Claim-level evidence-span/window/provenance binding plus unit algebra and derivation lineage. |

## Recommended sequence

### R17 — score semantics / calibration / fusion sensitivity

Prioritize `R14-MOD-001`, `R14-FUS-001` and the measurement side of `R14-KG-001`. Do not call normalized engineering scores calibrated probabilities without calibration evidence.

### R18 — external validity and operational FDIR metrics

Advance `R14-TMP-001`, `R14-OOD-001`, `R14-MET-001`, `R14-SAM-001`, `R14-SCP-001`. Freeze cross-simulator/HIL protocols before qualification results are observed.

### R19 — acquisition/provider authenticity and maintainability

Advance `R14-ADM-001`, `R14-UP-001`, `R14-MNT-001`, `R14-NUM-001`, `R14-PKG-001`, `R14-GND-002`.

## Validation ladder still required

`SOFTWARE_VERIFIED -> SIMULATION_VALIDATED -> CROSS_SIM_VALIDATED -> HIL_VALIDATED -> TARGET_COMPUTE_VALIDATED -> OPERATIONAL_SHADOW_VALIDATED -> RELEASE_QUALIFIED`

R16 remains an engineering/source-governance patch. No real 38×8 result or flight qualification is created by runtime identity closure.
