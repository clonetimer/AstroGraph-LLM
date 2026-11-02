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
| `information_gaps` | 仍缺少的信息 |
| `recommended_review_actions` | 建议核查步骤 |
| `sources` | 本次任务的源文本、角色、文件名、定位和文件 SHA-256 |
| `citation_reference_validation` | 引用是否属于本次已提供来源 |
| `input_sha256` | 本次来源集合的完整性摘要 |

成功输出的 `model_explanation.status` 为 `AVAILABLE_REQUIRES_EXPERT_REVIEW`。
模型对报告的解读与原因假设可以使用自然语言，不再要求逐字摘录。
引用验证只检查来源归属，不代表模型判断或原始报告已被独立证实。
外部诊断文本与文件哈希独立保留，模型不会改写上游诊断状态。

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
