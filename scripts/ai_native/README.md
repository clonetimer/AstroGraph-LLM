# scripts/ai_native — AI-Native 演进路线执行脚本

> 本目录承载 AI-Native 路线（docs/CURRENT_ENGINEERING_POLICY.md）的执行入口。代码可审计；历史训练报告所对应的 checkpoint 未随归档提供，因此不能仅凭本仓库复现该次训练结果。
> 所有脚本遵守治理纪律：候选预算=1、人工 gate、不 push、证据入 governance/evidence/ai_native/。

## 脚本清单

| 脚本 | 阶段 | 职责 |
|---|---|---|
| validate_ai_native_matrix.py | Phase 0 | 校验 AI-Native 成熟度矩阵每项评分绑定证据文件 |
| run_llm_hybrid_ci_regression.py | Phase 1 | ADCS/PHM 双数据集 llm/hybrid CI 回归（reference 后端 + 偏差注入 fail-closed） |
| run_feedback_cycle.py | Phase 2 | 训练型反馈闭环宿主（真实模型 + 反馈门 + ledger） |
| run_unified_baseline.py | Phase 3 | 统一基座训练（联合/对照/迁移探针/checkpoint） |
| run_phm_cv.py | Phase 2 | PHM 5 折 grouped CV（unified vs 独立模型同预算） |
| run_phm_cv_ensemble.py | Phase 2 | PHM 5 折决策层集成 CV（检测 OR 门 + 概率平均） |
| run_evidence_suggestion.py | Phase 4 | 单 episode 证据定义建议生成 |
| run_evidence_suggestion_aggregate.py | Phase 4 | 多 episode 聚合建议（目标类 vs 其他类分布分离） |
| run_federation_registry.py | Phase 5 | 能力联邦/洞察共享注册（append-only） |
| run_governance_health_check.py | Phase 5 | 治理健康体检（只读审计，advisory） |

## 运行约定

- Python 环境：使用项目声明兼容的 Python 3.11–3.14 环境；具体 CUDA 设备由运行者显式选择。
- 数据集环境变量：`ASTROGRAPH_ADCS_V522_DATA_ROOT`、`ASTROGRAPH_PHM_V60_DATA_ROOT`。
- 训练类脚本需 CUDA 且耗时分钟级至小时级；正式运行前应先执行冒烟测试。
- `run_feedback_cycle.py` 必须显式传入 `--phm-root` 与 `--checkpoint`；本次导入包不含历史 `.pt`。
