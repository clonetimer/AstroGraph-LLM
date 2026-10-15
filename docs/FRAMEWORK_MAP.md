# AstroGraph post49 框架 → 代码/能力映射

状态：`CURRENT_POST49_FRAMEWORK_MAP`

| 环节 | 当前实现 | post49 状态 | 关键边界 |
|---|---|---|---|
| 输入/语义适配 | `astrograph.agents.mission.generic_series`、source adapters | CURRENT | shape/channel/unit/hash 契约；不以字段名猜真值 |
| 异常检测 | Generic detector + admitted domain providers | SUPPORTED | generic detection 比分类/定位覆盖更广 |
| 定位 | model localization head / generic localization contract | PROFILE_GATED | affected component ≠ root cause |
| 分类 | model classification head / execution resolver | PROFILE_GATED | arbitrary single-signal classification 不声明 |
| 数据排名 | `DataDrivenPrediction` / `FaultRanking(DATA_DRIVEN)` | CURRENT | provider provenance 与 calibrated/uncalibrated 状态分开 |
| KG 正向/逆向 | knowledge query / evidence graph / backward coverage | CURRENT_GOVERNED | E0/provisional 不升级为 reviewed truth |
| 融合 | `fuse_rankings` / BiRRF / Decision Governor | CURRENT | 文献默认不作为第四正式票 |
| 文献/RAG | retrieval + reviewed-claim bridge + joint evidence | SUPPORTED_NONVOTING | relevant ≠ supports；reviewed stance ≠ formal vote |
| 连续/主动诊断 | CaseTimeline / NextBestObservation | SUPPORTED | 只建议观测，不授权控制 |
| 专家反馈 | lifecycle + failure regression | SUPPORTED | correction 不直接修改 production |
| 受控进化 | sealed challenger/evaluation/review/preflight | SUPPORTED_TO_PREFLIGHT | production promotion 仍需治理/人工 gate |

## exact ADCS profile

post49 保留冻结的 33-channel ADCS strict binding，可执行 Detection / affected-component Localization / Classification，并可叠加已资格化的 calibration/OOD shadow 信息。partial/mismatched input 必须降级。

## 当前事实源

能力状态：`governance/implementation/post49/RELEASE_READINESS.json`。
历史架构进展表：历史档案（本交付源码包不包含），仅用于追溯。
