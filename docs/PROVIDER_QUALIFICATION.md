# post49 Provider Qualification

状态：`CURRENT_POST49_PROVIDER_POLICY`

Provider 是否“可调用”与是否“可正式诊断/并行/晋升”是不同结论。

## 必备条件

- 明确 provider/profile ID；
- artifact/version/hash 可验证；
- 输入 channel/unit/shape contract；
- timeout/resource/cleanup 边界；
- calibrated probability / OOD 能力若被声明必须有独立 qualification；
- host binding 和 capability admission 分离；
- fail-closed on drift/mismatch。

## 当前主要边界

| 类型 | post49 结论 |
|---|---|
| local deterministic isolated | 可按已冻结资格使用 |
| external CLI | 需要具体工具/平台生命周期资格；不能从 adapter contract 推断领域质量 |
| Neo4j/read-only KG | 读路径受治理；并行/取消能力按具体部署资格 |
| HTTP/LLM | 需要真实 endpoint、timeout、schema 与领域验收 |
| in-process CPU/GPU model | 执行可 profile-gated；资源生命周期/硬取消需 provider-specific 证据 |
| MinerU/A2O | adapter 可用性与航天领域解析质量严格分开 |

## 执行规则

`Capability Registry → exact binding → admission → execute`。任何一步失败都不得用 KG candidate、规则或 LLM 文本冒充“模型已执行”。
