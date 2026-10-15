# 文档状态与结论优先级策略

状态：`CURRENT_POST49_POLICY`

## 状态

- `CURRENT`：post49 当前权威说明；
- `HISTORICAL_NON_AUTHORITATIVE`：历史设计、dev/post 结果、旧实验/qualification/台账；
- `RESEARCH_NON_RELEASE`：研究候选，不是稳定 release claim；
- `GOVERNED_EVIDENCE`：内容寻址证据；其“当前含义”必须由当前 release ledger/readiness 解释。

## 结论优先级

`RELEASE_READINESS > CURRENT_METRICS > CURRENT_DOCS > GOVERNED_EVIDENCE > HISTORY/RESEARCH`

更高历史分数、旧 PASS、旧“production qualification”标题、旧 README 中的“当前”等字样，都不能覆盖 post49 当前事实源。

## 历史引用要求

引用历史指标或结论时必须同时给出：

1. 原版本/日期；
2. 数据集与数据角色（Train/Validation/Test/used replay/held truth）；
3. 分母/coverage；
4. 协议（例如 Group-CV、known-activation retrospective replay）；
5. 当时状态（research / blocked / promoted / shadow）；
6. 当前是否仍被 post49 采用。

## 默认检索规则

项目文档导航、自动摘要和维护脚本默认只读取 `CURRENT_DOCS.json` 中的 CURRENT 文档。

本交付源码包不包含历史文档；history/audit 复现必须使用独立历史归档包，历史材料不得覆盖当前 release truth。

## 路径策略

- 当前说明：`docs/`、`docs/local_acceptance/`、`docs/agent_modes/` 等 current surface；
- 历史说明：统一 历史档案（本交付源码包不包含）；
- 当前 release 治理：`governance/implementation/post49/`；
- 历史 hash-bound 治理资产保持原路径，不因文档整理改写其原始记录。
