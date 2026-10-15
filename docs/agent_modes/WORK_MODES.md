# AstroGraph-Agent 工作模式（post49）

状态：`CURRENT_POST49_WORK_MODES`

| 模式 | 目标 | 允许 | 不允许 |
|---|---|---|---|
| `normal` | 稳定诊断 | 当前 admitted tools、deterministic core、受治理 RAG | 改阈值/权重/Active KG |
| `thinking` | 增加调查规划 | 在授权预算内规划检查、检索、解释 | 绕过 tool grant、把探索当事实 |
| `creative` | 生成受约束假设/检查建议 | hypothesis proposal、文献/知识候选 | 修改 committed diagnosis、确认 memory、生产资产 |

真实 LLM 未配置时必须记录 capability degradation；不得把 deterministic fallback 描述成真实 LLM 调用。

所有模式共享同一 committed diagnostic core 和治理边界。模式差异不能成为修改模型阈值、BiRRF 权重或正式证据角色的隐式通道。
