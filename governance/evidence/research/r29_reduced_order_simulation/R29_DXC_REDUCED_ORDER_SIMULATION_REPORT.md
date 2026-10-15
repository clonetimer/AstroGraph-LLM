# R29 DXC reduced-order simulation development report

R29 preserves all resolved R28 predictions and applies a reduced-order numerical ADAPT electrical-load simulator only to the generic R28 unresolved physical-response disposition.

On the already-contacted 25-case DXC09 Sample development corpus:

| Metric | R28 | R29 |
|---|---:|---:|
| fault-element micro F1 | 0.985075 | 1.000000 |
| component micro F1 | 0.985075 | 1.000000 |
| exact fault-set rate | 0.960000 | 1.000000 |
| diagnostic resolution | 0.960000 | 1.000000 |

Exactly one case changed: `Exp_658_pb_t2`, from unresolved to `LGT481:FailedOff`. R29 is post-contact development only: the last R28 unresolved case had already been inspected before R29 was implemented. The fixed resolver uses no truth or case-ID input, but its apparent 1.0 development result is not independent validation.

The core is candidate-specific numerical time-series simulation over a reduced-order resistive branch model. It is not VirtualADAPT/Simulink or a full continuous electrical network solver. Agent review is read-only and non-authoritative.

Stop rule: `NO_PARAMETER_SEARCH_ON_DXC09_SAMPLE_AFTER_FIXED_R29_DESIGN`.
