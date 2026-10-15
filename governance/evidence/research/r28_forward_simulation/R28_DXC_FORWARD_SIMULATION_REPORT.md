# R28 DXC forward-simulation development report

R28 adds a truth-blind qualitative component-mode/state-transition forward model above the frozen R27 deterministic core. It is not a continuous electrical simulator and Agent execution is not required.

On the already-contacted 25-case DXC09 Sample development set:

| Metric | R27 | R28 |
|---|---:|---:|
| fault-element micro F1 | 0.818182 | **0.985075** |
| component micro F1 | 0.818182 | **0.985075** |
| exact fault-set rate | 0.800000 | **0.960000** |
| any-overlap rate | 0.840000 | **0.960000** |
| diagnostic resolution | 1.000000 | **0.960000** |

R28 returns one unresolved case rather than the R27 false sensor diagnosis. Among the 24 resolved cases, exact-set accuracy is 1.0; this selective figure is inseparable from the 0.96 resolution rate.

Changed cases: `Exp_665_pb_t2ff`, `Exp_676_pb_t2`, `Exp_644_pb_t2f`, `Exp_658_pb_t2`, `Exp_692_pb_t2`.

Stop rule: no further case-specific optimization after this fixed R28 audit. Competition/Rematch were not consumed. All metrics are post-contact development evidence only.
