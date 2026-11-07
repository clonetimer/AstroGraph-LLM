# 第四环节与可视化交互功能核对表

核对对象：`AstroGraph_P4_Merge_Source_20261107_R4.zip` 的 `source/`。这里区分**已经接入主流程**、**按配置启用**和**仅有代码或演示入口**；合并时以目标项目的接口、权限和数据为准。下列相对路径均以包内 `source/` 为起点。

## 一、用户实际操作链

| 步骤与功能 | 页面或接口 | 必须一起移植的实现 | 状态与验收要点 |
|---|---|---|---|
| 进入平台与角色登录 | `/platform` → `/platform/mission`；账户初始化、登录、登出 | `src/astrograph/api.py`、`integrations/platform_accounts.py`、`platform_state.py`、`platform_secrets.py`、`resources/mission_console.*` | 已接入；viewer/operator/reviewer/admin 由服务端授权，隐藏按钮不能代替接口鉴权 |
| 事件中心 | 任务汇总、队列、筛选、任务预览 | `api.py` 的 `/v1/platform/overview`、`integrations/upload_report.py`、`resources/mission_console.*` | 已接入；点击已完成任务应进入同一任务号的研判或报告 |
| 新建分析 | 自然语言任务、可选设备范围、回形针材料菜单 | `resources/mission_console.html/.css/.js`、`integrations/upload_report.py` | 已接入；结果在报告页，输入页只显示材料、预检和处理进度 |
| 选择引擎 | 规则或管理员保存的模型配置 | `/v1/platform/analysis-engines`、`settings/llm/profiles`，`integrations/platform_state.py`、`platform_secrets.py`、`llm/openai_compatible.py` | 已接入；配置与某次任务绑定，模型失败不能伪称模型解释成功 |
| 上传与预检 | 诊断结果、运行日志、参考资料；自动或手动预检 | `/v1/platform/jobs*`、`integrations/upload_report.py`、`result_log_review.py`、`maintenance_input_admission.py` | 已接入；`examples/platform_acceptance_cases_v1/` 含 TC01–TC09，七个用例的 `operations.log` 为未提交但必要样例，已专门纳入包 |
| 异步处理与状态 | 开始、轮询、进度与失败信息 | `integrations/platform_worker.py`、`http_worker.py`、`upload_report.py` | 已接入；独立进程退出时记录失败，不让界面无限等待 |
| 故障分析报告 | 当前判断、原因、依据、核查、处置边界、模型说明 | `integrations/upload_report.py`、`platform_knowledge.py`、`resources/mission_console.*` | 已接入；冻结上游候选，明确“候选/待核验/确诊”边界 |
| 报告标签与导出 | 研判结论、日志与文献、图谱推理、审核溯源；文字或 JSON 导出 | `resources/mission_console.js`、`api.py` 报告路由、`upload_report.py` | 已接入；报告 ID、引用 ID、任务号应一致；技术字段放在详情 |
| 事件研判与任务档案 | 当前任务的时间线、证据、动作；历史任务回看 | `resources/mission_console.*`、`api.py` 的任务及历史接口 | 已接入；不能把历史案例当当前任务观测 |
| 专家复核 | 复核队列、独立依据、意见与留痕 | `integrations/manual_report_review.py`、`material_workbench_sessions.py`、`platform_accounts.py`、`api.py` | 已接入；报告复核与设备操作授权分离 |

## 二、第四环节的知识与模型能力

| 能力 | 关键文件 | 接入状态与边界 |
|---|---|---|
| 结构化大模型解释、候选锁定、可见引用与数值核验 | `integrations/platform_knowledge.py`、`llm/openai_compatible.py`、`task_prompts.py`、`candidate_selection.py`、`trace_verification.py`、`audit.py` | 主流程接入；主机校验失败可做**至多一次**定向修复，仍失败时拒收模型解释；不等于语义因果已验证 |
| 任务上下文和模型预算 | `llm/token_budget.py`、`integrations/platform_knowledge.py`、`upload_report.py` | 主流程中的有界材料投影；不要把未送给模型的原文写成模型已见依据 |
| 本次上传资料 RAG | `integrations/platform_knowledge.py`、`material_hierarchical_rag.py`、`retrieval/structured_document.py`、`document_corpus.py`、`service.py` | 可用，增强能力受知识引擎配置影响；检索命中是参考资料，不是本次遥测证明 |
| 公开文献主动检索 | `oag/paperseek_http.py`、`paperseek_cli.py`、`retrieval/paperseek.py`、`integrations/platform_knowledge.py` | 仅配置 PaperSeek 时触发；服务/CLI 本体及网络语料**不在源码包** |
| 候选文献图谱构建与审核 | `integrations/material_graph_builder.py`、`material_graph_release.py`、`material_case_memory.py`、`material_engine_settings.py`、`api.py` | 按配置启用；抽取先入待审，专家批准后才可发布/复用；图谱数据库实例不在包内 |
| 多跳机理与知识复用 | `integrations/material_multihop.py`、`material_reasoning.py`、`material_case_memory.py`、`retrieval/document_graph_bridge.py` | 已有代码和报告接口；具体可用性受已审核、同型号知识覆盖约束；缺路径时允许拒答 |
| OAG 诊断分支 | `oag/`、`integrations/upload_report.py` | 仅当上传的输入符合 `astrograph-oag-input.v1` 时选用，不是所有普通诊断的默认路径 |
| 网页分析的 OCR 与多模态候选 | `retrieval/advanced_multimodal.py`、`vlm_transcription.py`、`integrations/material_engine_settings.py`、`platform_knowledge.py` | 默认只解析 PDF 文字层；无文字层时 OCR 默认关闭。显式回退只接受 `tesseract` 或 `docling`。Qwen3-VL 4B 与 PaddleOCR-VL 1.6 的本机直调仅为选型试验，均未接入网页上传主流程；未审核转写不能直接作事实引文；外部工具与模型文件不在包内 |
| Mission agent 的 MinerU 适配器 | `agents/mission/documents.py`、`models.py`、`document_quality.py`、`scripts/acceptance/stages_tools.py` | **已有独立命令行适配器**，需启用 `allow_mineru` 并配置真实可执行文件；这条链路未接入网页“新建分析”的 `ocr_engine` 枚举。测试夹具只证明适配器契约，不证明真实 MinerU 的航天文献解析质量 |
| 向量检索与知识引擎配置 | `retrieval/persistent_vectors.py`、`qdrant_vectors.py`、`integrations/material_engine_settings.py` | SQLite/本地嵌入或 Qdrant 依配置；外部 Qdrant、Embedding 权重不在包内 |
| 文献、图谱和历史案例的来源边界 | `integrations/platform_knowledge.py`、`material_case_memory.py`、`resources/mission_console.js` | 来源弹窗标明文件、行号、审核状态与适用范围；已审核历史资料也不能直接确诊当前事件 |

## 三、界面与管理入口

| 页面 | 文件和主要数据来源 | 特别核对 |
|---|---|---|
| 首页大屏 | `resources/home.html/.css/.js`、`home_scene.js`、`home_three.js`；任务概览接口 | 任务号与状态来自平台数据；地球和轨道是演示场景，不是实时卫星位置 |
| 事件中心、新建分析、事件研判、决策包、任务档案 | `resources/mission_console.html/.css/.js`；`api.py`、`upload_report.py` | 三个界面文件和 API 路由一起移植；安装后的 `pyproject.toml` 要包含 `resources/*` package-data |
| 证据问答 | `resources/mission_console.js` 的 `materialReply()` | **当前是从已生成报告做确定性字段检索**，有聊天外观，但每轮不调用 LLM；无报告时不能自由诊断 |
| 知识图谱 | `resources/public_spacecraft_seed.v1.json`、`mission_console.js`；`/v1/platform/knowledge/public-seed` | 内置图谱是公开、未审参考种子，不是星上合格图谱；另有已审图谱管理接口 |
| 文献建图、知识审核、审校待办 | `mission_console.*`、`api.py`、`material_graph_builder.py`、`manual_report_review.py` | 专家权限和审核状态必须与界面一致 |
| 人员与访问、模型与引擎 | `mission_console.*`、`platform_accounts.py`、`platform_state.py`、`platform_secrets.py`、`material_engine_settings.py` | 管理员操作；模型配置、Neo4j/知识引擎配置及测试连接是不同设置 |
| 旧界面兼容 | `resources/platform_ui.html/.css/.js` 及 `platform_*.js`，`/platform/legacy` | 已随完整源码入包；主入口指向新版 `mission_console`，合并时不要误把旧页面当默认入口 |
| 演示案例 | `resources/mission_demo_cases.json`、`examples/platform_acceptance_cases_v1/` | 演示与真实任务分开标识；合成预设不进入真实任务/审批队列 |

## 四、交付核对与未随包交付项

`PACKAGE_COVERAGE.json` 对上述关键源码、资源、样例与回归测试给出包内路径、字节数和 SHA-256；构建时缺任一必备文件即失败。完整文件仍以 `PACKAGE_MANIFEST.json` 为准，源码可执行 `scripts/verify_source.py --root <解压后的 source 目录>` 校验。

本包不包含账户数据库、任务历史状态、API 密钥、模型权重、Embedding 模型、Qdrant/Neo4j 服务、PaperSeek 本体、OCR 可执行程序或私有卫星遥测。目标项目若依赖这些外部条件，需单独配置和复验。此前的 63/63 后端回归、Chrome 流程及截图只证明**本仓库**；目标整体项目的端到端质量仍待验证。
