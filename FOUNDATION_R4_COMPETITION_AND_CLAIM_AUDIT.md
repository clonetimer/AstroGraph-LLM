# AstroGraph R4 | 竞争假设与声明级原文核验（候选版）

基线：R3 Candidate（2026-10-14）。本轮不涉及微调、不调整上游 BiRRF/Decision Governor、不启动卫星控制。按需求方要求，不重复旧的时间信任链/验收数据恢复/真实模型闭环专项。

## 更改

- `integrations/material_reasoning.py`: 对实际已有候选生成 `competing_hypothesis_review`，记录支持/反对日志词汇线索、独立审核的负向关系 ID、共享症状、缺失的预期症状，以及前两个候选的文献差异指针。缺失不等于反证，审核图谱并非当下遥测事实。
- `integrations/platform_knowledge.py`: 将已有文献对比、反证和双向检索进入正式材料 LLM 载荷；要求按证据区分竞争假设，不允许候选重排或反向因果。
- `llm/task_prompts.py`: 将 `PlatformKnowledgeExplanation` 的竞争假设规范加到系统级任务提示，而不是只作为用户内容。
- `retrieval/material_claim_guard.py`: 声明级确定性核验。仅保留与上下文来源原文 **完全一致** 的直接引文（允许受控来源前缀）；模型的自由解释、假设、因果性复述、历史案例、图关联、当前观测推断均不作为可发布摘录。验证摘要记录逐条 SHA256、原因和来源状态。
- `integrations/material_multihop.py`: 在用户可见结果入口执行上述校验，并屏蔽未经核验的原始 LLM 摘要、论断和自由生成的控制/处置建议；不可信的候选 ID 不向公共报告回显。保留原始服务器端生成产物供授权审查。
- `integrations/result_log_review.py`: 报告标题变更为“来源原文摘录（非诊断结论）”，只展示通过原文一致性门禁的条目。

## 限制（重要）

1. **当前是逐条“引用文本一致性”核验，不是语义蕴含/NLI 的充分验证**；`EXTRACTIVE_MATCH_NOT_SEMANTIC_ENTAILMENT` 绝不表示根因证明。不存在事实核验模型已在实机上通过的主张。
2. 匹配到的上传资料、文献摘要、日志词汇线索仍可能不真实、无设备适用性或受到提示注入。其原文只可作带来源状态的引用摘录；不能替代经校准传感器数据。
3. 高保守策略会拒绝真实但仅可转述的陈述，可能牺牲报告可读性；若要开放“安全转述”，需另行建立带金标准和拒判门禁的语义核验器。
4. 经过审批的 KG 负向声明不能自动否定当前故障，缺失症状没有被转成负向证据；两个候选共享同一文献/症状也不能视为两份独立证据。
5. 不能宣称故障分类准确率提高。定向回归仅核查架构规则、误放行反例、输出行为；需要冻结数据集的盲测与人工声明金标准量化收益。

## 执行和记录

```bash
# 轻量核心门禁
PYTHONPATH=src python -m pytest -q tests/test_foundation_r4_competition_claim_guard.py tests/test_material_multihop_reasoning.py

# 后台执行；结果保存在 validation_logs/r4-repeat/output.log 与 done.json
PYTHONPATH=src python scripts/launch_logged_tests.py start --timeout 500 r4-repeat -- python -m pytest -q tests/test_foundation_r4_competition_claim_guard.py tests/test_material_knowledge_fusion.py tests/test_platform_knowledge_enrichment.py tests/test_material_hierarchical_rag.py
PYTHONPATH=src python scripts/launch_logged_tests.py status r4-repeat
```

Windows 请使用项目环境 Python 并设置 `PYTHONPATH=src`；将其余环境要求按已有项目 README 安装。日志 runner 不会在断开的聊天连接中主动推送消息，但会落盘并生成终态文件。

## 验收建议

- 无证据/仅有未审核或条件性图边时，不应生成确定性因果结论；
- 相似故障共享症状不能重复加权；
- 反证与反向路径只作为对比或调查依据；
- 虚假或合法 ID + 伪造因果说明应被阻止公开；
- 引用一致的原文保留来源级别标记，人工复核后再发布工程结论。