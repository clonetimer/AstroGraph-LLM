# 各部分结果怎样看

状态：`CURRENT_POST49_REFERENCE`

结果都在 `工作区/stages/<阶段>/`。先打开 `index.html`，再看下面这些文件。

## 数据驱动检测、定位、分类

**10-data-inference** 实际执行原三任务模型。`prepared-input.json` 是无真值的诊断输入；`three-task.json` 是模型输出；`evidence-bundle.json` 是观测与机理证据。图谱投影保存在 `graph/`。

**11-detection/detection.json**：`detection_probability`、`detection_threshold`、`detected`、窗口和模型来源。阈值保持原值，不为了样例成功改阈值。单个窗口不能得出 Macro-F1。

**12-localization/localization.json**：`independent_head` 是独立定位头；`classification_derived` 是分类聚合；`deployed_distribution` 是实际定位分布。CSV 的 distribution 列区分三种结果。这里是部件组，不能据此宣称单个反作用轮编号定位。

**13-classification/classification.json**：`raw` 是原始分类；`conditioned` 是部件约束后分布；`policy` 和 `discarded_class_mass` 展示约束与信息丢弃；`fusion_input_ranking` 才是后续融合实际接收的故障排序。模型概率不等于已校准可信度或已确认根因。

## 图谱与融合

**20-graph-reasoning/reasoning.json** 包含前向/反向查询、知识版本和路径；`nodes.csv`、`edges.csv` 可直接查看或导入图工具。不强制 Neo4j，使用项目现有只读图查询实现。

**21-fusion/fusion.json** 包含 `data_ranking`、图谱分支和 `birrf_result`。`fusion-ranking.csv` 展示排序与贡献；`diagnosis.json` 是补证前的证据裁决；`explanation.json/.md` 保留原内核结构化解释。不要把倒数排名分当故障后验概率。

**22-agent-normal/decision-delta.json** 对比初始和补证后结果；`checks.json` 保留新增证据、检查次数和停止原因；`tasks/acceptance-normal/plan.json` 与 `step-*.json`、`events.jsonl` 展示实际计划和工具过程。此阶段复用10阶段输出，不冒充又训练或又执行一次模型。

`MODEL_ONLY` 表示模型假设、`SET`/未决表示保留竞争解释，`knowledge_verified` 不等于在轨真值核实。解释较长也不等于证据更充分。

## RAG、图谱构建与文献

**30/31** 输出转换内容、清洗后的文本、逐行变更、源和派生文本哈希。**32** 输出片段位置、来源、相关度、实际检索方式；**37** 才要求实际稠密编码+Cross-Encoder。稠密失败后词法回退列为 PARTIAL，不写作已通过混合检索。

**35-graph-build/graph-candidate.json** 是待审核候选，不是已发布活动知识。A2O 的原文件在对应任务工作区保留。LLM 抽取路径明确标识 `STRUCTURED_LLM_EXTRACTOR_NOT_A2O`，不混称 A2O 已运行。

**34** 当前验证题录/摘要来源获取，不保证全文下载、版权许可或引用内容真实性。**38** 是固定两正一负的小样本检索验收，不是 Ragas，也不是领域检索质量的全面证明。

## 训练、记忆与输出

**40-model-train/training-report.json** 展示 loss、真实权重变化和三头验证混淆矩阵。默认是微型合成数据，模型在该阶段 tasks 中，不覆盖 models。**41** 的 `candidate-heads.csv` 是新候选无标签推理结果。

**25-memory** 使用单独 `acceptance-only.sqlite` 和显式合成标签/评审身份，验证暂存、确认、查询和隔离，不能宣称本次真实设备案例已由专家确认。生产记忆库不接入该演示。

**23/24** 显示实际模式和规划器；创造内容单独隔离。真实 LLM 服务验收走**39**。**61** 校验结果 schema、事件链、工件哈希和幂等读取。

## 三任务指标

**51/55** 从已保存多窗口预测复算。检测应同时看可执行子集和包含不可用的全量指标；定位分开看独立头和最终系统派生；分类分开看原始头与检测/裁决后系统输出。错类、漏检、未决、不可用不得删除或混作正确。
