# AstroGraph 航天器故障知识图谱：公开文献初始种子（v1）

> **状态：公开参考资料形成的候选知识，全部未经型号专家审核。** 这是**可离线查看、查询、导入隔离知识库、继续扩展**的初始知识图谱，不是已经核定的航天器故障根因库，不得据此自动发送遥控指令。

## 1. 覆盖范围

- **206 个实体**、**225 条带来源信息的关系**、**30 组故障候选**，覆盖 ADCS（5）、EPS（5）、热控（4）、推进（4）、通信（5）、星载计算/CDH（4）、有效载荷（3）。
- 七种实体：`COMPONENT`、`FAILURE_MECHANISM`、`SYMPTOM`、`FAULT_MODE`、`VERIFICATION`、`ACTION`、`CONDITION`。
- 七种关系：`SUSCEPTIBLE_TO`、`ASSOCIATED_WITH`、`MANIFESTS_AS`、`INDICATED_BY`、`CHECKED_BY`、`MITIGATED_BY`、`REQUIRES`。
- 除各故障本地机理链外，还提供 **15 条跨分系统的候选机理联系**，如“通信下行减弱←姿控指向误差”“载荷数据堆积←星载存储或链路受限”。这类联系均带 `ENGINEERING_INFERENCE_CROSS_SUBSYSTEM` 标记，不是已证明的因果关系。
- **13 份 NASA/ECSS 官方资料的目录**，其中 **12 份用于当前关系**；每条关系均附资料 ID、章节/小节定位和**中文转述**，没有将转述冒充英文原文逐字引句。资料原件未被打包、转售或复制。

此图以 NASA 小型航天器资料为主，不能直接假定适用于所有航天器，尤其不能自动推广到特定飞控架构、硬件型号、温度/电压保护阈值与失效等级。

## 2. 立即查看与查询

**不用安装数据库，直接打开** `graph_viewer.html`：

- 左侧可筛选分系统、输入中文故障词、选择故障候选。
- 中间是部件、可能机理、症状、检查、措施、执行条件组成的可交互小图。
- 右侧展示相关资料链接、章节定位、跨系统替代原因及风险边界。
- 浏览器可以在离线环境显示图和中文数据；打开 NASA/ECSS 在线文献本身需要联网。

**命令行检索（仅 Python 标准库）：**

```bash
python query_graph.py --stats
python query_graph.py 飞轮
python query_graph.py 欠压 --json
python query_graph.py --fault PROP-VALVE-LEAK --hops 2
python query_graph.py --domain COMM
```

**可供其他工具导入：**

| 文件 | 用途 |
| --- | --- |
| `graph.json` | 完整规范化实体、关系、来源和故障组；包含审核/解释类型字段 |
| `graph.graphml` | 可由 Gephi、yEd 或 NetworkX 读取的关系网络 |
| `nodes.csv`、`edges.csv` | 供 Excel、数据库 ETL 使用的实体/关系数据 |
| `sources.json` | 公开资料书目、来源 URL、类型与访问日期 |
| `ontology.json` | 严格实体类型和允许关系结构 |
| `expert_review_template.csv` | 专家对每一条关系逐条审定的空白记录表 |
| `build_seed.py` | 内置 curated 案例与稳定 ID 的可重建构建脚本 |
| `validate_seed.py` | 来源、格式、本体、CSV/GraphML、脱敏/未授权状态等结构检查 |
| `neo4j_import_seed.py` | 写入**单独标签与数据库**的可选 Neo4j 导入器（默认 Dry Run） |

## 3. Neo4j 隔离导入

默认：

```bash
python neo4j_import_seed.py
# DRY_RUN_NO_DATABASE_WRITE：不会连接或写入数据库
```

对于已部署并**专门分配了独立数据库及低权限账号**的 Neo4j，先由管理员创建该数据库，再显式设置：

```bash
export ASTROGRAPH_PUBLIC_SEED_NEO4J_WRITE_ENABLED=1
export ASTROGRAPH_PUBLIC_SEED_NEO4J_DATABASE=astropublicseed
export ASTROGRAPH_PUBLIC_SEED_NEO4J_URI='bolt://127.0.0.1:7687'
export ASTROGRAPH_PUBLIC_SEED_NEO4J_USER='seed_importer'
export ASTROGRAPH_PUBLIC_SEED_NEO4J_PASSWORD='通过安全凭据渠道配置'
python neo4j_import_seed.py --confirm-write
```

需预先安装 `neo4j` Python 驱动。此导入器仅创建 `AstroPublicSeedEntity` / `AstroPublicSeedClaim` 两类**候选标签**，不会写入已审核 `Active KG`、不会伪造批准记录；正式知识图谱的审核与签名发布仍应走 AstroGraph 原有管理流程。

> Neo4j Community 单数据库部署不适合这一独立数据库导入路径，可直接使用 GraphML/CSV、离线图浏览器，或另行启动隔离的 Neo4j 实例。**不应临时改成写入生产 `neo4j` 默认库来绕过隔离要求。**

## 4. 可重复构建与检查

```bash
python build_seed.py
python export_graphml.py
python build_viewer.py
python validate_seed.py
```

`build_seed.py` 的内容是公开资料**人工归纳的初始案例定义**，不是在真实线上模型运行后自动抽取的知识，不宣称有未经验证的 LLM 准确率。源页面与章节由 NASA、ECSS 的公开资源定位。具体句子的独立校核（尤其行动建议）仍需专业审核。新资料如由 LLM 自动抽取，必须经过原文匹配、型号与本体检查、专家审核后才能成为可发布知识。

## 5. 必须遵守的证据/安全边界

1. `evidence_form=CHINESE_PARAPHRASE_NOT_VERBATIM` 表示中文转述，**不是原文定位后的逐字证据**；不能直接冒充正式 AstroGraph 文献建图接口要求的 `evidence_quote`。需先获取原始来源和真正原文引句。
2. `review_status=PUBLIC_REFERENCE_SEED_UNREVIEWED`，**225 条关系全部待审核**；`is_confirmed_root_cause=false`、`automatic_control_authorized=false`。
3. `MITIGATED_BY` 不是可执行飞行指令；`ACTION` 和 `CONDITION` 只是供工程师复核的建议及前提条件，绝非 NASA/ECSS 授权操作指令。
4. 某症状可能同时对应多种候选故障，日志相关性不足以证明因果；未观察到故障特征也不代表可排除。
5. 要进入正式故障原因分析，需要为各故障建立型号适配、故障码映射、可核实的页码/原文引句、时序证据、反证及专家审批记录，并经图谱签名/版本发布。

## 6. 参考文献入口（所有来源可在 sources.json 逐条查看）

- NASA Small Spacecraft Technology State-of-the-Art 2026: ADCS、Power、Thermal、Propulsion、Communications、Avionics、Ground Data Systems 等章节：https://www.nasa.gov/smallsat-institute/sst-soa/
- NASA *CubeSat 101: Basic Concepts and Processes for First-Time CubeSat Developers*：https://www3.nasa.gov/sites/default/files/atoms/files/nasa_csli_cubesat_101_508.pdf
- NASA *Software Engineering Handbook — Fault Detection and Response*：https://swehb.nasa.gov/spaces/SWEHBVD/pages/102695798/9.07%2BFault%2BDetection%2Band%2BResponse
- ECSS-E-ST-60-30C (AOCS)、ECSS-E-ST-31C (Thermal)、ECSS-E-ST-35-01C (Propulsion)、ECSS-E-ST-70-41C (TM/TC)：https://ecss.nl/standards/

## 7. 项目集成说明

该目录可原封不动复制到 `AstroGraph/reference/public_spacecraft_knowledge_seed/`，用于**外部公开参考资料预览/检索与专家知识入库候选**。不改变已有 GraphRAG 任务的已审核知识快照，避免未经专家确认的新关系自动影响处置建议。下一步可按设备型号选取本目录条目，经独立证据验证后通过 AstroGraph 图谱审核工作台逐条确认，再进入其签名版本发布及正式图谱查询链路。
