# 本机模型公开案例试用（非平台整链评测）

日期：2026-11-07。端点：用户提供的本机 LM Studio `http://127.0.0.1:1234`。仅向模型发送 `../p4-e1-public-candidates-20261107/cases/{NH2015,GAL2017}/input.md`，未发送 `reviewer_only/` 或原始网页。运行脚本和逐条原始输出均保存在本目录。

## 运行设置与边界

- 模型：`qwen/qwen3.5-9b`（Q4_K_M，测试前已加载）和 `qwen3.5-0.8b`（Q8_0，测试前未加载）。本机 LM Studio 原生 `/api/v1/chat`，`reasoning=off`、`temperature=0`、`max_output_tokens=650`、`store=false`。同一案例的两个模型收到相同输入字节。
- 单独的兼容接口探针显示：9B 在 `/v1/chat/completions` 默认设置下，300/300 输出 token 均计入 reasoning，正文为空且 `finish_reason=length`；加入 `reasoning_effort=none` 后，34 token 正文正常返回。该探针是接口兼容性发现，未计入下表质量比较。LM Studio [原生 Chat API](https://lmstudio.ai/docs/developer/rest/chat) 明确支持 `reasoning=off`。
- 四份输出是公开通报解释试用。它们没有原始遥测、BiRRF 诊断结果或平台报告生成链路，不用于计算四环节诊断准确率。两例均为公开事件，不能保证模型训练时未见过。没有独立专家复核或一致性分析。

| 案例 | 模型 | 请求状态 | 输出 token / 上限 | 耗时 | 初步可复核问题 |
|---|---|---:|---:|---:|---|
| NH2015 | 9B | 200 | 650/650 | 118.644 s | 正确保留“根因未确定”，但回答在核查记录段落中截断；将 15:15 恢复通信叙述为星上自动恢复的确定时刻，超出给定材料 |
| GAL2017 | 9B | 200 | 650/650 | 107.281 s | 区分 RAFS/PHM 与 FOC/IOV，但回答在原因分析中截断；“完全失效”等表述需要回到通报核对 |
| NH2015 | 0.8B | 200 | 520/650 | 22.776 s | 首段称“联络中断约 9 小时”，把往返通信延迟与 13:54–15:15 的失联时长混淆 |
| GAL2017 | 0.8B | 200 | 650/650 | 35.047 s | 称“IOV 构型上的 5 台 RAFS 失效”，把原文的 5 台 **PHM** 失效错记为 RAFS；后续核查建议继承此错配；末尾截断 |

本表只列人工初读时可直接与输入核对的问题，并非完成六维盲评。输出上限、量化规格和初始加载状态不同，不能据此直接选型或计算公平时延优势。`NH2015` 的 9 小时是往返通信延迟，失联时间为 81 分钟；`GAL2017` 的 RAFS 失效仅见 FOC，IOV 上的 5 台是 PHM。原始来源及截止时点见[公开候选清单](../p4-e1-public-candidates-20261107/README.md)。

## 对研究计划的调整

1. 先处理生成预算和思考模式：平台的兼容接口请求体目前没有 `reasoning_effort`，而本机默认模式的短请求已经复现空正文。修复时应做成显式、按服务商可选的配置，并用平台实际结构化报告请求回归，不能把此处原生 API 结果冒充平台成功。
2. 为报告增加长度约束或分段生成，并记录 `finish_reason`。本轮三个输出触及 650 token 上限，内容评审应保留“截断”作为失败状态。
3. 把单位/时间窗和部件构型交叉核对加入 E2 关键错误清单。仅仅生成流畅中文不足以确保维护人员能安全使用。
4. E1 仍须补未暴露、可追溯的真实/仿真整链输入及独立参考答案。当前两例只适合作为 E2 评审口径试用和 E3 接口预实验。

## 平台实际调用链复核

随后用现有合成 TC01 上传 `diagnosis_result.json`、`operations.log`、`reference.md` 和 `analysis_input.json`，从 `UploadReportJobStore` 执行预检与报告生成，使用同一本机 9B、OpenAI 兼容 `/v1`、`json_schema`、1200 输出 token。首轮与修复后两轮均为 `preflight.can_start=true`、报告 `SUCCEEDED`，报告的 `knowledge_enrichment.generation.status=SUCCEEDED`；隔离任务及完整报告在 `platform_tc01/` 和 `platform_tc01_after_fix/`。

- 首轮任务的 `llm_call_status=NOT_STARTED`，与报告内实际 `SUCCEEDED` 矛盾。原因是材料复核路径最后保存了知识增强之前的旧 `job` 对象。修复后新任务 `JOB-F2D403A96798BB677E08D9FE` 的状态为 `SUCCEEDED`，与报告一致。
- 同一探针还复现 Windows 编码缺陷：HTTP 子进程向管道输出非 ASCII JSON 时沿用系统编码，父进程按 UTF-8 解码失败。现改为显式写 UTF-8 字节。针对强制 `PYTHONIOENCODING=cp1252` 的回归测试与原有后端/材料测试共 14/14 通过。
- `json_schema` 下的两次真实后端小探针都正常返回正文；先前普通短问答的默认思考耗尽预算，**不能推断平台结构化报告也出现同一错误**。LM Studio 本机版本拒绝 `json_object`（HTTP 400），平台以 `local-openai-compatible` 的默认策略优先使用 `json_schema` 能工作。
- TC01 的模型说明在 `analysis.knowledge_enrichment.model_explanation` 中，五条带引用的说明及核查动作均保留，且标记需专家复核；最终结论由证据门控给出，仍为“无法确定根因”。模型说明主体为英文，中文维护界面的可读性仍需改进。引用 ID 校验 `PASS` 不代表因果语义已获认证。

本次是**材料复核路径的真实模型调用**，但 TC01 是已经使用过的合成示例，并非 E1 独立黄金集，也未覆盖浏览器登录、原始遥测上游诊断或专家签署。不能将“报告生成成功”写成“诊断准确率通过”。

## 中文报告复验

本次仅在当前仓库中增加平台解释任务的明确语言要求：中文用户任务请求 `response_language=zh-CN`，模型的总结、逐条说明、局限和核查动作使用简体中文，同时保留编码、单位、引用 ID。相关本地 HTTP 与材料复核回归 15/15 通过。

隔离任务 `JOB-D5CF01B87DCB168D5E39903C` 的产物在 `platform_tc01_zh/`：预检、报告、任务大模型状态均为 `SUCCEEDED`。9B 本次生成 3 条中文带引用说明和 3 条中文核查动作；初读与合成 `operations.log` 的 0.34 A、38.6 °C 记录及上传手册的条件边界一致。正式结论仍表示根因未确诊，模型说明标记为需专家复核。这是一例模型行为复验，不能推断所有中文任务稳定通过。
