# AstroGraph R6 基础增强候选版 — 多视角检索和审核机理链

## 适用范围

基于 R5 候选源码，保持“文献不可生成冻结故障候选”“上传资料不可参与正式概率投票”“知识图谱关系必须具备型号和来源审核”的原有边界。本版本没有将任何已知案例的真实根因添加到系统提示词、种子知识图谱、查询词或生产数据。

## 实现

1. `retrieval/observation_balanced.py`：从已登记的编号用户现象派生独立的查询；联合查询加上各条查询，经有界 RRF 去重，为每条现象预留可用的文献指针；返回查询哈希、覆盖数和未覆盖的原始证据 ID。没有编号现象时保留 R5 单查询行为。
2. `integrations/platform_knowledge.py`：将以上检索正式接入上传材料本地索引，返回 `local_retrieval.strategy_receipt`；仍将所有命中设为非诊断证据，不允许改变冻结候选。无冻结候选或服务配置时，不改变 PaperSeek 的原有准入策略。
3. `integrations/material_graph_builder.py`：新增 `FAILURE_MECHANISM —ASSOCIATED_WITH→ FAILURE_MECHANISM` 待审提案合法三元组，**关联不是“导致”**；模型抽取的三元组必须逐条通过原文摘录、实体锚定和独立审批流程。
4. `integrations/material_multihop.py`：4 跳能力为可选项。默认 3 跳不变；禁止反向检索当作反向因果；对否定、条件、跨型号、未审核关系保持拦截。
5. `integrations/material_reasoning.py`：管理员配置 `ASTROGRAPH_MATERIAL_GRAPH_MAX_HOPS=4` 才会将新深度连接到实际材料推理的双向和多跳调用路径；其他值拒绝。
6. `llm/task_prompts.py`：显式提醒模型，多条关联边的串联不是已证明的传播过程，必须逐跳评估来源和条件。

## 不能声称的能力

- 四跳关联 **不等于** 四段因果证明，不能确诊实际设备根因。
- 词法检索覆盖 **不等于** 机制召回准确率，低相似词命中需进一步人工判断。
- 当前文本预审无正式候选时，不会自动调用候选驱动的外部文献搜索；这不是在静默完成端到端根因推理。
- 已公开结果的回溯对照不是盲测。后续要评价发现根因的能力，需要新的、开发时未知的隔离测试案例和标准答案。

## 回归与执行

```bash
PYTHONPATH=src python -m pytest -q tests/test_foundation_r6_observation_retrieval.py
python scripts/launch_logged_tests.py start --timeout 600 r6-retro-run -- python -m pytest -q tests/test_foundation_r6_observation_retrieval.py tests/test_material_multihop_reasoning.py tests/test_platform_knowledge_enrichment.py
python scripts/launch_logged_tests.py status r6-retro-run
```

`validation_logs/<job>/output.log` 和 `done.json` 分别存储完整输出与结束触发信号。`done.json` 并非聊天推送。
