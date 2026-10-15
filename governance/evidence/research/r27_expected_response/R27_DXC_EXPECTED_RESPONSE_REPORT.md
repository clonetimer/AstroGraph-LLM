# PATCH-R27 DXC expected-response development report

## Status

`POST_CONTACT_DEVELOPMENT_ALGORITHM_AUDIT_ONLY`

R27 keeps the legacy DXC statistical branch and adds a deterministic diagnosis-core candidate. The Agent is optional and cannot create the development-performance improvement.

## Architecture

- legacy statistical DXC branch: retained, unchanged, runtime default unchanged;
- R27 deterministic candidate: R25 command-aware residual + R26 discrete state + R27 expected-response consistency;
- Agent: optional `DXC_EXPECTED_RESPONSE_REVIEW` shadow mirror of the already-computed core decision;
- ESA: R26 frozen high-resolution candidate retained; no R27 retuning.

## DXC09 Sample post-contact development results

| Slice | Algorithm | Fault-element F1 | Component F1 | Exact set | Any overlap | Mean faults |
|---|---|---:|---:|---:|---:|---:|
| SELECTION (11) | R26 | 0.400000 | 0.400000 | 0.272727 | 0.454545 | 1.090909 |
| SELECTION (11) | R27 | 0.882353 | 0.882353 | 0.818182 | 0.909091 | 1.454545 |
| INTERNAL_REPLAY (14, already seen) | R26 | 0.466667 | 0.533333 | 0.428571 | 0.500000 | 1.000000 |
| INTERNAL_REPLAY (14, already seen) | R27 | 0.750000 | 0.750000 | 0.785714 | 0.785714 | 1.142857 |

### Combined contacted development slices (25 cases)

- R26 fault-element F1: `0.433333`; component F1: `0.466667`; exact set: `0.360000`.
- R27 fault-element F1: **`0.818182`**; component F1: **`0.818182`**; exact set: **`0.800000`**; any overlap: **`0.840000`**.
- R27 preserves R26 first-alarm timing; the change is isolation/ranking, not a new detection-timing claim.

## Remaining wrong-set cases

- `SELECTION/Exp_665_pb_t2ff`: truth `['E281:Stuck', 'ST265:Offset']`; predicted `['E281:Stuck']`.
- `SELECTION/Exp_676_pb_t2`: truth `['LGT400:FailedOff', 'LGT402:FailedOff']`; predicted `['LGT401:FailedOff']`.
- `INTERNAL_REPLAY/Exp_644_pb_t2f`: truth `['IT267:Stuck']`; predicted `['E265:Offset', 'ST265:Offset']`.
- `INTERNAL_REPLAY/Exp_658_pb_t2`: truth `['LGT481:FailedOff']`; predicted `['IT161:Offset']`.
- `INTERNAL_REPLAY/Exp_692_pb_t2`: truth `['LGT405:FailedOff', 'LGT406:FailedOff']`; predicted `['LGT407:FailedOff']`.

## Boundary

The rules were developed after observing both DXC09 Sample development slices. Competition/Rematch are not consumed. No external-validation receipt is created. Promotion requires genuinely new holdout bytes. Agent review is shadow-only and does not change the core prediction.
