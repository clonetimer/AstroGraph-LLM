# 外部诊断文本分析

本入口处理“外部诊断结果 → LLM 分析 → 模型输出 → 界面与下载报告”。
诊断结果由外部系统生成，本入口不运行诊断算法、遥测分类、故障规则、图谱构建或案例学习。
普通故障描述不需要匹配 AstroGraph 的内部故障编码。

## 使用界面

1. 打开 `/platform`，进入“新建分析”。
2. 上传至少一份**诊断结果**文件。日志和参考资料均为可选补充。
3. 选择已配置的大模型服务，填写分析目标，然后开始分析。没有模型时可以先导入材料；
   页面会明确显示尚未配置模型，不会生成或冒充 LLM 分析。
4. 在报告中查看摘要、模型对报告的解读、原因假设、缺失信息和建议核查步骤。
5. 展开引用查看对应文件、行号/文档定位及原文；下载 JSON 或 Markdown 留档。
6. 在工作台对话区或旧版报告页继续追问。追问由本任务所选模型回答，逐段附原文引用；
   页面可重新读取已保存的历史。只有具备分析权限的用户可以发送问题，查看权限可以读取历史。

新任务使用 `EXTERNAL_DIAGNOSIS_ANALYSIS`。已生成的旧报告仍可查看。
原有 `MATERIAL_REVIEW` 保留为兼容入口，继续使用原来的候选约束和材料审查逻辑。
新入口不依赖已审核图谱，参考资料直接作为本次模型分析的补充文本。

## 命令行

使用项目已有的 LLM 环境配置或默认服务配置，运行：

```bash
astrograph-platform --root runs/platform analyze-report \
  --result /absolute/path/diagnosis.txt \
  --task "解释报告中可能的原因，列出依据、相反解释与仍需补充的信息" \
  --full
```

需要补充材料时，可重复使用 `--result`、`--log`、`--reference`。
不加 `--full` 时，短摘要也包含 `analysis_mode` 和 `llm_call_status`。
进程退出码 `0` 表示导入和报告保存完成；是否生成模型分析以 `llm_call_status` 为准。
原来的 `review` 子命令继续执行材料审查模式。

## API

仍使用现有 job API：创建任务、上传文件、预检、启动、读取报告。
除了诊断文件，上传以下 `analysis_input.json` 即可：

```json
{
  "schema_version": "astrograph-result-log-review.v1",
  "analysis_mode": "EXTERNAL_DIAGNOSIS_ANALYSIS",
  "diagnosis_results": ["diagnosis.txt"],
  "log_reports": [],
  "references": []
}
```

`log_reports` 和 `references` 可以省略。若列出某个文件，该文件必须真实上传，
且每个文件只能有一个角色。路径穿越、未列入清单的文件和不可解析的材料仍被拒绝。
为保持已有客户端的行为，省略 `analysis_mode` 的清单仍默认 `MATERIAL_REVIEW`。
界面和新命令会自动填入新模式，无需用户手写清单。

报告保留 `artifact_kind: RESULT_LOG_REVIEW` 以兼容现有下载接口，并用
`analysis.analysis_mode` 区分执行路线。模型输出在
`analysis.knowledge_enrichment.model_explanation`，包括：

| 字段 | 内容 |
| --- | --- |
| `summary` | 模型分析摘要 |
| `findings` | 模型对所提供文本的解读；每项含正文与引用编号 |
| `hypotheses` | 原因假设；每项含标题、分析和引用编号 |
| `hypotheses[].counterpoints` / `opposing_citation_ids` | 模型描述的反向解释和对应原文引用；缺少反证时允许为空 |
| `hypotheses[].assumptions` / `uncertainty` / `discriminating_check` | 假设前提、不确定性和能够区分解释的核查方向 |
| `information_gaps` | 仍缺少的信息 |
| `recommended_review_actions` | 建议核查步骤 |
| `review_actions` | 可选的结构化核查建议，含 `check`、`rationale`、`expected_observations`、`citation_ids` |
| `sources` | 本次任务的源文本、角色、文件名、定位和文件 SHA-256 |
| `citation_reference_validation` | 引用是否属于本次已提供来源 |
| `input_sha256` | 本次来源集合的完整性摘要 |

成功输出的 `model_explanation.status` 为 `AVAILABLE_REQUIRES_EXPERT_REVIEW`。
新增 `claim_governance`：将 Findings、Hypotheses、跨报告共识/分歧等转换为带稳定 `claim_id` 的主张记录，标记 `REPORTED` / `DERIVED` / `INFERRED`，绑定支持与反向来源，并默认保持 `expert_review_status: PENDING`。模型不能写入专家确认状态。
可在新建任务时通过 `phase_engine_ids.verification` 选择已审核的独立审阅模型。该模型必须与主分析模型不同；审阅输入只包含主张文本和原始来源，不包含生成模型的摘要、推理过程或已选引用。未配置独立模型、模型身份相同、上下文超限或审阅失败时，会明确记录状态，不静默回退到主模型。独立模型审阅只判断给定材料是否支持主张，不认证现实事实，仍必须由专家确认。
模型对报告的解读与原因假设可以使用自然语言，不再要求逐字摘录。
引用验证只检查来源归属，不代表模型判断或原始报告已被独立证实。
外部诊断文本与文件哈希独立保留，模型不会改写上游诊断状态。
新增报告字段均为可选，旧模型响应和已有 v1 报告仍可读取。引用检查覆盖反向依据与结构化
核查建议；“反证”是模型对原文的解释，不是程序已独立验证的反证。报告没有提供反证时，
不要求模型编造。一个原文片段包含混合信息时，可以同时作为支持和反向引用。

## 针对报告追问

追问使用服务器已保存的原始输入、通过引用检查的初始模型分析和本任务已成功的历史回答。
问题、模型分析和历史回答不会成为新的原始证据。服务端不接受客户端提供的来源列表、历史
或模型配置，不执行工具、图谱查询或新的诊断计算。

读取历史：`GET /v1/platform/jobs/{job_id}/follow-up`。
提交问题：`POST /v1/platform/jobs/{job_id}/follow-up`，请求示例：

```json
{
  "question": "目前哪些信息能区分传感器延迟与真实温度变化？",
  "input_sha256": "从当前报告或 GET follow-up 返回值读取的 64 位输入哈希",
  "request_id": "本次提交的唯一编号，例如 UUID"
}
```

`input_sha256` 必须是当前任务的 64 位小写十六进制值；`request_id` 是 8 至 80 位 ASCII
字母、数字、下划线或连字符。相同编号和内容的重发返回已有结果，不重复调用模型；
改换问题必须使用新编号。并发请求、旧输入哈希、历史不一致或达到次数上限返回 `409`。

接口返回 `astrograph-external-diagnosis-conversation.v1` 状态，包含输入/原报告哈希、
`revision`、`turns`、`sources` 与 `limits`。每个成功回答的 `answer` 是带 `text` 和
`citation_ids` 的段落数组，并附 `information_gaps`。失败记录保留问题和固定原因代码，
不展示失败响应正文。刷新后可以重新读取已保存的历史；界面重试未确认请求时复用编号。

每个任务最多保存 **12 次追问尝试**，包括模型失败记录；每个问题最多 **4,000 字符**。
问题、初始分析和成功历史构成的 JSON 上下文最多 **32,000 字符**，原文仍受独立的
60,000 字符模型调用上限约束。超过上下文上限返回 `CONTEXT_TOO_LARGE`，不会静默裁掉
旧问题或来源；需要继续时可新建分析任务。具体模型还可能有更小的 token 上限。

追问记录单独保存在 `output/external_diagnosis_conversation.json`。
初始 `report.json`、`report.md` 和模型分析工件保持原样，因此原报告下载不包含后续追问。
历史读取接口提供独立的 JSON 记录。一次任务同时只有一个追问调用，等待时不提供虚假的
“停止模型”按钮；关闭页面不代表后台模型调用已取消。

## 状态与容量

`report.status: SUCCEEDED` 表示报告已保存。LLM 是否完成由
`analysis.knowledge_enrichment.generation.status` 单独表示：

| 模型状态 | 含义与处理 |
| --- | --- |
| `SUCCEEDED` | 模型返回通过结构与引用检查的分析，界面和下载报告展示该分析 |
| `NOT_CONFIGURED` | 没有可用模型，保存原始导入文本，配置模型后新建任务分析 |
| `FAILED` | 调用、结构检查或引用检查失败，显示固定原因代码，不展示失败响应的正文 |
| `INPUT_TOO_LARGE` | 全部解析后源文本合计超过 60,000 字符，保留导入、跳过模型调用，需拆分材料 |

模型服务还可能有较小的上下文限制；这类拒绝按实际调用失败处理。
此版本没有通用的长文分块归纳流程，不会静默丢弃长行末尾或只将关键词命中的部分
冒充为整份报告。沿用的导入解析上限是每文件 180,000 字符、CSV 最多 3,000 行；
超过导入上限会在预检时明确阻止。扫描 PDF 若没有可提取文本，应先提供可读转写。

诊断结果支持 TXT、Markdown、JSON、CSV、可提取文本的 DOCX/PDF；
补充日志还支持 LOG。前两种格式最适合直接导入外部文字报告。

公开结果不会包含原始 provider 异常、密钥或未经限制的后台调用收据。
HTML 页面把源文本和模型文本作为普通文字渲染。Markdown 也转义活动标记。

## 本次验证的范围

新增自动化测试覆盖单份普通文本、未知故障描述、可选材料、完整长行、任务绑定、
伪造引用、模型未配置/失败/输入过长、两套生产 JS 渲染器的实际执行及 HTML 转义。
API 集成测试执行真实上传、预检、HTTP 模型请求和报告下载，模型端使用本机脚本化
OpenAI-compatible 服务，用于验证协议和数据贯通。它不证明某个生产模型的分析质量。
实际模型质量仍需在目标服务上使用代表性诊断报告验收。

## 2026-11-01 源码包的采用范围

本次检查 `AstroGraph_Stable_20261101_Source.zip`，SHA-256 为
`b126905ba4dc5600fdc2d6d489c6362eb4ccb7a94c9c7acf8e51dc77c2718a04`。
采用专家分析中反证/不确定性/具体核查的报告设计，以及按任务保存、绑定原文的追问交互；
这些能力已适配到本项目独立的外部文本链路。未整体替换新包的专家诊断工作流。

同时选择性修复模型调用兼容性：子进程固定 UTF-8 输出，精确 tokenizer 支持 Mapping
包装的 token 结果，模型明确拒绝时终止格式重试，服务探测不把截断输出认作成功。
本项目原有 HTTP 重定向保护保留。未提高默认模型输出 token、未默认启用内部推理或引入训练。

实现依据与边界见 [源码包增量说明](docs/superpowers/specs/2026-11-02-source-update-external-analysis.md)。
新增回归覆盖真实本地 HTTP 追问、幂等/并发/权限/容量/原报告不变，以及两套生产 JS 的
提交、历史恢复、来源转义和任务切换。脚本化模型只证明协议与数据链路，不代表真实模型
的分析质量；浏览器视觉与目标模型质量需按部署环境另行验收。

本轮本地完整门禁：39 个模块，**391 passed / 3 skipped / 0 failed**。跳过的是 source-only
模式下两份不随源码交付的历史冻结证据和一个 Windows 限定测试；没有把跳过项计为通过。


## P4-C：逐条主张的专家反馈闭环

`claim_governance.claims[]` 中每条主张均可独立登记 `ACCEPT`、`REJECT`、`REVISE` 或 `NEEDS_EVIDENCE`。每次决定都绑定原报告 SHA-256、主张 ID、独立证据编号、专家账号和时间，并通过独立 revision 防止并发覆盖。`REVISE` 只生成专家修订覆盖，不改写原始模型主张或报告；接受也只表示允许保留为经专家审阅的报告主张，不等于当前设备根因确诊。专家意见使用独立 SQLite append-only 事件表，与候选故障复核和知识图谱审批分开管理。
