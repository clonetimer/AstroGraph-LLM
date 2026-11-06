# R8 — 因果方向门禁、跨文献适用性审查与竞争假设差异

**性质**：R7 基础上的开发候选；不得视为真实设备根因已确诊。保留 R7 审核流程、三跳默认值、管理员四跳开关、因果引用及故障排名不投票的权限边界。

## 变更

1. **方向初筛**：对 `CAUSES` 的部分明确中英文主动/被动句加入方向倒置检测（如 `A is caused by B` 与 `B causes A`）。抽取阶段拒绝反向提案；读取审核边阶段重复核验。该匹配器仅覆盖高精度表面模式，不代替专家对复杂句的因果语义审查。
2. **节点一致性**：同一已审核图谱快照内，若同一实体 ID 对应不同规范化类型/名称，则其相关因果边不能参与路径拼接；不会跨型号借边填补机理缺口。
3. **跨文献证据边界**：每条因果链记录文献 SHA256、来源批次以及每跳审核出处；使用不同文献或任务批次拼成的路径标记 `CROSS_DOCUMENT_CONTEXT_REQUIRES_INDEPENDENT_VALIDATION`。即使所有跳均来自同一文献，也不声称已验证当前设备工况。
4. **候选对照**：使用冻结候选的审核因果路径终点，构造 `competing_hypothesis_review.causal_path_differential`。返回只属于**这组候选**的症状 ID、共有症状、来源链 ID、检索截断状态与跨文献提示；不生成新候选 ID、不产生概率、不把缺失检索命中当作反证。
5. **LLM 合同**：材料任务及系统级提示词同步加入“仅对冻结候选有差异意义、跨文献适用性未证、因果链不是现役设备观测”等限制。保留 R4 逐声明严格摘录门禁。

## 可复现测试

```powershell
python -m pytest -q tests/test_foundation_r8_causal_differential.py tests/test_foundation_r7_causal_claims.py
python scripts/launch_logged_tests.py start --timeout 900 r8-extended -- python -m pytest -q tests/test_foundation_r8_causal_differential.py tests/test_foundation_r7_causal_claims.py tests/test_material_knowledge_fusion.py
python scripts/launch_logged_tests.py status r8-extended
```

测试夹具使用**通用泵装置合成案例**；已公开的南太阳翼 SADA 故障答案没有进入检索语料、提取提示词或此轮测试标签。

## 未解决的研发限制

- 缺少由专家独立核准的结构化工况兼容性本体；跨文献链现在是**显式提示不确定性**，不是自动完成条件逻辑证明。
- 对复杂被动句、指代、省略、可能性表达等，方向判断仍依赖专家确认；当前正则不得被宣传为完成自然语言蕴含或因果发现。
- “某候选特有症状”只表示**在提供的冻结候选集合和当次检索预算内**未被其他候选覆盖，不是它的独占症状、故障确定性或经遥测证实的现象。
- 尚未使用未知标签案例、真实文献与实际 LLM 完成准确率消融。本轮代码测试通过只能证明实现行为，不证明根因识别效果。

## 长任务与运行记录

测试输出写入 `validation_logs/r8-*/output.log`；命令结束后生成 `done.json`（退出码及运行状态）。这只是在任务所在计算机上的文件触发，不自动向聊天推送结果。重试必须使用新任务名，保留原失败日志。
