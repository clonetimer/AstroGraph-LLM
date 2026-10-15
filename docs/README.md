# AstroGraph post49 当前文档索引

状态：`CURRENT_AUTHORITATIVE_INDEX`

## 权威顺序

1. `governance/implementation/post49/RELEASE_READINESS.json` —— 当前能力事实源；
2. `CURRENT_DATA_DRIVEN_METRICS.md` —— 当前指标口径；
3. `THESIS_METHOD_REMEDIATION.md` —— R36 证据推理、latency 与 qualification 方法整改；
4. `CURRENT_ENGINEERING_POLICY.md` —— 当前工程/治理约束；
5. `FRAMEWORK_MAP.md` / `../ARCHITECTURE.md` —— 当前架构；
6. `implementation/POST49_STABLE_RELEASE.md` —— 当前稳定发布说明。
7. `ASTROGRAPH_SPACECRAFT_SCOPE_CONVERGENCE_R32.md` —— 航天器领域 feature-freeze 与资格边界；
8. `ASTROGRAPH_POST_R32_ROADMAP.md` —— Bench/UI 与独立验证的后续路线。

**`history/` 永远低于以上来源，不得作为“最新结论”覆盖它们。**

## 当前使用文档

| 文档 | 用途 |
|---|---|
| [从这里开始](../START_HERE.md) | 安装、core、post49 acceptance |
| [R14 单源码合并](ASTROGRAPH_UNIFIED_SOURCE_R14.md) | Backend/Platform 单一源码真相与依赖边界 |
| [R14 科研严格审查](ASTROGRAPH_RESEARCH_AUDIT_R14.md) | 统计、证据、LLM grounding、V&V 与可重复性开放问题 |
| [运行指南](../RUN_GUIDE.md) | Mission / Runtime / API / GenericSeries |
| [测试策略](../TESTING.md) | 测试、验收、独立确认的区别 |
| [当前工程策略](CURRENT_ENGINEERING_POLICY.md) | 能力、数据、治理和晋升硬边界 |
| [当前指标](CURRENT_DATA_DRIVEN_METRICS.md) | ADCS/PHM 指标口径与分母 |
| [R36 方法整改](THESIS_METHOD_REMEDIATION.md) | 论文定位、概率证据图修复、event-causal ADCS 复核与剩余阻塞 |
| [框架映射](FRAMEWORK_MAP.md) | 当前能力→代码/状态 |
| [联合诊断架构](JOINT_DIAGNOSIS_ARCHITECTURE.md) | detection/localization/classification 语义 |
| [Provider 资格](PROVIDER_QUALIFICATION.md) | 运行时 provider 生命周期与 admission |
| [目录结构](DIRECTORY_STRUCTURE.md) | Current/History/治理证据布局 |
| [工作模式](agent_modes/WORK_MODES.md) | normal/thinking/creative 权限边界 |
| [本地验收](local_acceptance/README.md) | 当前命令与外部工具 |
| [R35 产品体验 Phase 1](ASTROGRAPH_PRODUCT_EXPERIENCE_R35.md) | 诊断工作区材料概览、上下文可读性与只读交互边界 |
| [R35.1 审查身份与交接](ASTROGRAPH_PRODUCT_EXPERIENCE_R35_1.md) | 投影/来源/图谱/仿真 provenance 身份展示与只读 operator handoff 摘要 |
| [Post51 多种子神经网络研究](POST51_MULTISEED_NEURAL_BENCHMARK.md) | ADCS 五骨干长周期比较、专用专家候选与独立正常确认 |

## 历史与研究

- [历史文档总入口](history/README.md)
- `research/`：研究候选，默认不构成 release truth；
- `governance/evidence/`：证据工件，可用于证明历史/当前 claim，但应由当前 release ledger 解释其资格。

机器可读文档清单：`CURRENT_DOCS.json`、`history/MANIFEST.json`。

- `CURRENT_DELIVERY_SOURCE.md`：当前无历史交付源码包的包含/排除与功能资产边界。
