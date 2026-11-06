# 2026-11-01 源码包的外部诊断分析增量

## 目标与判断

用户授权判断并执行新源码包更新，范围固定为：外部诊断结果文本输入 → LLM 分析与输出 → 交互界面。诊断结果如何计算不属于本项目本次工作。

新包 AstroGraph_Stable_20261101_Source.zip（191994023 bytes），SHA-256:
`b126905ba4dc5600fdc2d6d489c6362eb4ccb7a94c9c7acf8e51dc77c2718a04`。
交付说明声明源码来源提交为 `14188a21a11e9c5617af28bdf1a7bfd3c3065bd4`。
比较基线是本项目 `cf37af784843609d4d459de26b3c562f121cc039`，即 PR #2 已有外部文本链路；不是直接覆盖 main 或整个旧稳定包。

结论：选择性更新。新包的专家分析与交互概念有价值，但直接执行依赖内部故障目录、冻结诊断结构、BRB/profile、图谱和资料库。保留当前独立外部文本入口，适配以下三组增量。

## 采用与排除

| 来源 | 本次采用 | 边界 |
| --- | --- | --- |
| expert_analysis.py / expert_report_contract.py / expert_condition_analysis.py | 假设反证、前提、不确定性；带依据、目的和预期观察的核查建议 | 不移植 fault_id 约束、原子数值事实编译或诊断计算 |
| expert_conversation.py / platform_expert.js | 当前报告上的真实 LLM 追问、来源定位、幂等提交、保留历史、任务切换保护 | 独立于专家资料库、图谱搜索、工具调用、正式报告改写 |
| http_worker.py | 固定 UTF-8 子进程输出 | 保留本项目 _NoRedirect，拒绝上游重新启用自动重定向的改动 |
| token_budget.py | Mapping 包装 input_ids 的递归计数 | 不宣称改善中文粗估或自动配置 tokenizer |
| openai_compatible.py | 明确 refusal/content_filter 停止格式重试 | 不输出拒绝正文或原始 provider 错误 |
| platform_llm_probe.py | 截断完成标记不能探测为 READY；拒绝不再换格式 | 正常完整结果仍兼容 |

不采用新包整树重构、R20/R21、训练、仿真、图谱融合、专家多角色审核或数值工具；不无证据提高默认模型/输出 token 上限。新包的内部推理实验与发布回归数不作为本项目质量验收。

## 报告契约增量

保留现有 v1 的必需字段与旧报告兼容性，追加可选字段。缺少原文反证不要求模型编造反证。

- Hypothesis: `opposing_citation_ids: list[str]=[]`、`counterpoints: str=""`、`assumptions: list[str]=[]`、`uncertainty: str=""`、`discriminating_check: str=""`。
- Analysis: `review_actions: list[ExternalDiagnosisReviewAction]=[]`。
- ReviewAction: `check`、`rationale`、`expected_observations` 均为非空有界字符串，`citation_ids` 必须非空且唯一。
- 所有新增引用都检查属于本任务来源；同一来源可以同时支持与反对假设，因为一句原文可能有混合信息。
- 旧 `recommended_review_actions` 保持读取，丰富核查建议和旧建议分别展示且不丢弃。
- JSON、Markdown、Mission、Legacy Platform 均展示这些字段，并转义所有文本；不将语义判断标为已验证事实。

## 受约束追问

API:
- `GET /v1/platform/jobs/{job_id}/follow-up`
- `POST /v1/platform/jobs/{job_id}/follow-up`
- POST 严格请求：`question`（trim 后 1..4000 字符）、`input_sha256`（64 位 hex）、`request_id`（8..80 位 ASCII 字母/数字/_/-）。
- GET 沿用 viewer；POST 沿用 operator，沿用当前任务的角色共享访问规则，不引入新包的半套 owner 机制。

服务端只读取任务保存的输入包与原始报告，并重新验证任务/来源/内容绑定。禁止接受客户端来源、历史、模型或工具指令。要求任务已完成且为 EXTERNAL_DIAGNOSIS_ANALYSIS；模型沿用任务引擎选择。

模型输出 `ExternalDiagnosisFollowUp`：
`answer: list[ExternalDiagnosisFinding]`（1..12），每段必须有来源引用；
`information_gaps: list[str]`（最多 12）。无工具/图谱/数值诊断调用。
源文本、问题、前次模型分析与历史回答分别标记；历史问题和模型文本不是新的证据。
只有初始成功且引用通过的分析、以及本任务已成功且引用通过的追问可作为模型历史。

独立文件 `output/external_diagnosis_conversation.json`，绝不修改 `report.json`、`report.md` 或原分析工件。
公开 state：
`schema_version=astrograph-external-diagnosis-conversation.v1`、
`job_id,input_sha256,report_sha256,revision,turns,sources,limits`；
每 turn 有 `request_id,question,status,citation_reference_validation,created_at`，
成功时有 `answer`（cited paragraphs）、`information_gaps`，失败时仅固定 `reason_code`。
状态 SUCCEEDED / FAILED / NOT_CONFIGURED / INPUT_TOO_LARGE / CONTEXT_TOO_LARGE。
公开固定 `analysis_verification=UNVERIFIED_MODEL_ANALYSIS`、`automatic_control_authorized=false`。

最大 12 次保存的尝试，问题加已成功历史的 JSON 文本最多 32000 字符，源文本沿用 60000 字符；超限明确停止，不静默截断。
同一 request_id + 同一问题 + 同一输入哈希返回已有结果，不再次调用；不同内容复用 ID 或当前输入/报告哈希不匹配返回 409。
每任务非阻塞文件锁限制一个活动调用，进程退出后操作系统释放；并发返回 409。
错误不暴露 provider 异常/原始正文。模型失败仍保存问题与固定状态，不清空旧回答或原文。

两套 UI 均有追问入口、等待状态、防重复提交、失败保留输入/历史、来源详情。
Mission 现有对话在外部模式改为真实调用，旧材料模式保持已有行为。
账户/任务切换后丢弃迟到结果；刷新可重新加载服务器已保存的历史。
同步调用不提供虚假的取消模型按钮。问题输入 maxlength 4000，提交权限与 API 一致。

## 验证

定向测试覆盖全部新增引用、无反证/旧报告、HTML/Markdown转义；真实生产 JS执行；
本地脚本化 HTTP 的中文编码、拒绝、截断；API上传→报告→追问→下载/原文保持不变。
追问测试覆盖绑定、幂等、并发、历史限额、任务切换和 viewer/operator 权限。
运行现有 foundation-smoke 门禁并加入新增测试。真实生产模型的分析质量与浏览器视觉验收不在脚本化测试的证明范围。

