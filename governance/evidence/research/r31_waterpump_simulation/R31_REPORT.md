# R31 WaterPump reduced-order simulation report

R31 adds `adapt-waterpump-reduced-order-v1`, the second executable numerical mechanism family in AstroGraph. It couples the public ADAPT WaterPump 35 W rating with current, bus voltage and flow response. `FailedOff` removes electrical load and drives flow to zero; `FlowBlocked` preserves pump energization while fitting a bounded blocked-flow asymptote and first-order response time as truth-blind nuisance parameters.

The provider is integrated as a deterministic core corroborator and never requires Agent execution. `DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW` is a read-only mirror of the frozen decision.

Mechanism registry after R31: 222/222 modeled fault-mode instances, 14 numerical reduced-order, 203 qualitative state-response, 5 declared-only; numerical coverage 6.3063%.

Public DXC09 Sample truth-aligned mechanism audit: 2 WaterPump cases, 2/2 physical best-mode agreement. The injection timestamp is used to align the window, so this is not online diagnosis performance.

Blind synthetic event-conditional pump mode campaign: 80 cases, predictions frozen before truth join, accuracy/macro-F1/FailedOff recall/FlowBlocked recall all 1.0. This perfect result is simulator-family protocol evidence only and must not be represented as DXC external validation or end-to-end detection performance.

Stop rule: `NO_PARAMETER_OR_RULE_OPTIMIZATION_ON_R31_PUMP_BLIND_OR_DXC09_SAMPLE_OUTCOMES`.
