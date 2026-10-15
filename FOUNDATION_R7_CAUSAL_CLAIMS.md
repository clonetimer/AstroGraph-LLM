# AstroGraph R7 — 审核因果声明与逐跳证据账本（研发候选）

## 目标与性质

R7 将文献图谱中用于“发现关联”的路径与可检索的**因果声明链**明确分离。新能力不是模型内部思维链，也不是根据真实卫星遥测自动证明故障传播，而是一种可审计的外部证据结构。

本轮严格继承 R6 的冻结候选、文献审核、型号作用域、历史快照与撤销边界，不向检索库、提示词或图谱种子输入已披露的 SADA 案例答案。合成测试使用无关的水泵设备与摩擦/冷却机理。

## 实现和边界

1. `material_graph_builder.py` 新增 `CAUSES`（文献声称发生定向因果）和 `CONTRIBUTES_TO`（贡献/风险，不构成确定传播）的合法三元组。保留旧 `ASSOCIATED_WITH`。模型提取必须提供**连续原文引用**，引用中包含两个端点词和显式因果用语；模型不能批准自己的关系。`condition_quote` 如存在，必须逐字锚定原文。可能性和条件性表述被标为 `HYPOTHETICAL_OR_CONDITIONAL`。
2. `material_case_memory.py` 在暂存阶段二次检验因果原文，之后仍需已有的独立人员、独立依据、型号范围和快照审查。旧关系结构和历史审批记录保持兼容，不改写历史收据。
3. `material_multihop.py` 提供 `evaluate_causal_chains(...)`：**只沿已审核的肯定 `CAUSES` 正向边**进行有限深度搜索（默认三跳，最多四跳），并在检索预算耗尽时明确标记为部分结果；不使用 `ASSOCIATED_WITH` 或 `CONTRIBUTES_TO` 补齐链；不把反向检索解释为反向因果。否定冲突、时序版本不匹配、条件未验证、型号不匹配、来源异常时拒绝参与。
4. 输出每一跳的关系 ID、原文、来源 SHA256、原文定位、审核人和独立核验引用，附带 `current_root_cause_proven=false`、`sensor_observation_verified=false`、`diagnostic_vote_allowed=false`。**合格图路径只代表被审核的文献因果声明链**，不代表本设备当前已经发生故障。
5. `material_reasoning.py` 将 `multi_hop.causal_chain_review` 接入正式材料推理结果，继承 R6 管理员管理的三/四跳深度开关；不改变原有故障排名、概率、控制和报告发布门禁。
6. `material_graph_release.py` 在图谱导出检查中复验因果引句，同时在独立 Neo4j 命名空间的节点属性中保留 `claim_modality`、`condition_quote` 和 `reviewer`，避免条件信息丢失。
7. `task_prompts.py` 追加明确的权限提示：审核过的因果边仍然不是本卫星当前遥测事实，不得把可能性/贡献/关联关系改写成确定因果。

## 仍需解决

- **语义验证有限**：当前仅做高精度因果语言线索和原文一致性校验。被动句导致关系反向、隐含因果、专业术语等仍需独立人工审查，不能称为自动语义蕴含判定。
- **工况条件保守拒绝**：有条件的因果声明可保存、审查和显示，但无可信环境条件核验时不会形成可用的正向传播路径；尚未实现传感器时序、工况和机理条件的自动对齐。
- **源间拼接不构成事实证明**：多份文献均属已审核，并不意味着它们在同一任务/故障模式下同时成立；逐跳路径不是经实验验证的故障传播模型。
- **输入能力边界不变**：无上游冻结候选时，自然语言文字预审不会变成自主根因确定器；更高层假设规划将在后续阶段评估。
- **评估边界**：本轮定向回归证明 API/结构/隔离行为，不证明未知设备上的诊断准确率提升，也未使用已知 SADA 根因制作检索演示。

## 回归命令

```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests/test_foundation_r7_causal_claims.py
python scripts/launch_logged_tests.py start --timeout 600 r7-audit -- python -m pytest -q tests/test_foundation_r7_causal_claims.py tests/test_material_multihop_reasoning.py tests/test_material_graph_construction.py tests/test_material_knowledge_fusion.py
python scripts/launch_logged_tests.py status r7-audit
```

日志及完成标记分别写到 `validation_logs/<job>/output.log` 和 `done.json`。命令独立运行，任务完结不等于向聊天推送消息。
