# R25 DXC / ESA Remediation Report

- Protocol receipt: `80d09d002a17b1b9d921177abc9daa74dc56dcf32b79ad3e6c8df46e6fcd86df`
- Qualification: **DEVELOPMENT_REMEDIATION_ONLY_NOT_EXTERNAL_VALIDATION_NOT_FLIGHT_QUALIFIED**

## ESA

Selected development candidate: z=8.0, coincident channels=2, group gap=40.0 s.

Development engineering screen: alarm-group precision **0.900**, anomaly-ID recall **1.000**, F0.5 **0.918**, false alarm groups **3**.

Read-only audit of the four frozen confirmatory misses shows durations 3.912–9.912 s and two raw samples inside every event annotation on each lightweight channel 41–46. No new high-resolution prediction is generated on those seen events. Frozen result remains 0 alarm / 4 FN.

## DXC09 Sample development replay

Split: {'TRAIN': 49, 'SELECTION': 11, 'INTERNAL_REPLAY': 14}.

| Metric | R20 baseline on same 14-case replay | R25 hybrid |
|---|---:|---:|
| Early first-alarm cases | 14 | 1 |
| Post-injection first-alarm cases | 0 | 12 |
| Fault-element micro F1 | 0.0254 | 0.0952 |
| Component micro F1 | 0.0254 | 0.1905 |
| Any fault-element overlap | 0.3571 | 0.1429 |
| Exact fault set | 0.0000 | 0.0000 |

R25 median post-injection first-alarm delay is **0.4455 s**. Bounded minimal-diagnosis candidates overlap at least one truth element in **0.357** of replay cases and include the exact truth as a candidate in **0.143**; isolation remains inadequate.

## Boundary

These are development/remediation results. DXC Competition/Rematch and ESA seen confirmatory/external predictions are not regenerated. No external-validation, calibrated-probability, formal-EvidenceFact, HIL or flight claim is made.
