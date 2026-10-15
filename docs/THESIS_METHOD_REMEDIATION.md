# post49 / R36 硕士论文方法整改记录

状态：`CURRENT_RESEARCH_REMEDIATION_CANDIDATE`

输入源码：`AstroGraph_0.7.1_post49_R36_reconstructed_source_4a31f18.zip`

适用边界：源码修复与 DEVELOPMENT 复核；不构成 qualification、飞行或因果识别证明。

## 论文定位

AstroGraph 适合定位为“面向航天器故障诊断的受治理多源证据融合与可审计推理方法”，核心贡献限定为：

1. 具有 provenance 和 truth-use 约束的遥测准入；
2. 受证据依赖约束的物理、数据与知识融合；
3. fail-closed 的证据、工具权限和诊断评价治理。

知识图谱、LLM、低阶物理模型、RRF 或 Agent 本身不单独主张首创。当前 Reasoning Graph 也不再称为因果模型；`CAUSES/EXPLAINS` 只记录描述性拓扑，不参加数值更新。

## 已修复的方法问题

### 时效概率证据图

- `valid_from/valid_to` 必须是带时区时间，消除 naive/aware datetime 运行时异常；
- 使用数值稳定 sigmoid，三条 `LR=1e-300` 的反证不再产生指数溢出；
- 只要有一个待评估 target 没有使用任何证据，全局状态为 `PARTIAL`，不再误报 `COMPLETE`；
- 数值方法命名为 `TemperedLikelihoodReasoner`。更新式为

  $$
  \log O(F\mid E)=\log O(F)+\sum_i \tau_i\log LR_i,
  \qquad
  \tau_i=c_i q_i.
  $$

  其中 $q_i$ 是启发式 evidence screening score。该方法属于 generalized/power likelihood update，不是普通独立证据贝叶斯后验，也不是校准的真值概率。

### Evidence Quality

四因子乘积保留为明确命名的 `screening_score`；`overall_confidence` 仅作 v1 兼容别名。乘积没有概率语义。论文若把它作为核心方法，仍需在冻结开发集上比较 product、minimum、geometric mean 与 learned/calibrated aggregation，并在独立组报告敏感性；本轮没有用现有 test cohort 选择聚合式。

### 生命周期与受控工具

- 全部输入为 false 时投影为 `PENDING`，阻塞原因为 `OBSERVATION_NOT_COMPLETE`；
- ASK 工具审批升级为 v2，绑定 agent、tool、规范化参数 SHA-256、带时区失效时间和一次性 approval ID；
- 相同审批不能换参数、过期复用或重复调用。

### Qualification 授权

`QUALIFICATION` 不再能由独立 CLI 枚举值直接产生。程序化评价必须提交由外部治理门验证的 `QualificationAuthorizationReceipt`，并精确绑定 dataset、protocol、cohort 和 source artifact SHA-256；缺失、身份不一致或验证失败均拒绝。独立 CLI 只接受 DEVELOPMENT 或 fixture。

## ADCS latency 协议整改及实际结果

旧 R36 生成器先读取真实 onset，再从 onset 后阈值阳性窗口中选择“首次报警”。该数字是 oracle-onset-conditioned retrospective delay，不能解释为在线检测延迟。

新协议先在完整分数序列上、完全不读取真值地生成阈值报警 episode；之后才加入 onset，计算第一个在 onset 后新开始的报警 episode。若报警在故障前已经开始并跨越 onset，它计入故障前误报，不冒充零延迟检测。

使用原冻结 ADCS v5.2.2 划分和同一 sklearn family 重新执行，得到：

| 指标 | 修正后结果 |
|---|---:|
| 轨迹数 | 182（176 fault / 6 normal） |
| Detection TP / FP / FN / TN | 155 / 6 / 21 / 0 |
| Detection precision / recall / F1 | 0.9627 / 0.8807 / 0.9199 |
| nominal FAR | 6/6 = 100% |
| fault trajectory 故障前已报警 | 110/176 = 62.5% |
| post-onset 新报警 episode 延迟 | mean 9.15 s；median 3.00 s；p95 38.90 s |
| Localization accuracy / macro-F1 | 0.6534 / 0.6068 |
| Identification accuracy / macro-F1 | 0.5568 / 0.5271 |

证据文件位于外部运行目录 `run_artifacts/post49-r36-method-remediation/`。病例 SHA-256 为 `54cd9db35db81d70eba1ea97cdf1b377663dedb247a63d8ad2020cd5e02abad5`。

这个结果不支持“ADCS 在线 Detection F1=98.3%”的表述。当前主要矛盾是故障前和正常工况的高报警率。由于 normal 只有 6 条，100% FAR 的点估计也不足以给出稳定运营 FAR；必须增加跨任务、跨姿态阶段、跨初始条件的正常运行时长。

## BiRRF 消融现状

源码已有两类工具：`astrograph.evaluation.kg_gain` 的七臂对照，以及 `scripts/experiments/run_birrf_dependence_audit.py` 的 data/KG/leave-branch-out 依赖审计。因此“完全没有消融实现”并不准确。但当前交付没有足以支持硕士论文结论的、与本轮 ADCS 冻结 cohort 绑定的独立确认结果。论文只能写“已有可执行消融协议”，不能写“BiRRF 增益已经证明”。

下一次确认实验应至少固定并同时报告 Data only、KG only、vanilla RRF、weighted RRF、BiRRF without dependency collapse、full BiRRF；按 trajectory/pair seed 分组 bootstrap，禁止按窗口独立重采样。

## 剩余研究阻塞

1. 先扩充正常/故障前工况，冻结新的正常覆盖协议，再在 development 中调整检测策略；当前182条 test cohort不能用于继续选阈值。
2. 用独立种子确认 event-causal detection、定位和分类，报告置信区间、risk-coverage 和按任务阶段分层结果。
3. 对 Evidence Quality 聚合和 tempered likelihood 做预注册敏感性分析；没有校准前只输出 screening score / generalized estimate。
4. `DiagnosticScientistAgent` 当前执行外部给定 hypothesis/request/plan，应在论文称为 controlled diagnostic experiment executor；若要主张自主科学推理，需补信息增益选实验、假设淘汰和停止准则。
5. `LiveDiagnosticLoop` 当前是 bounded-window online-compatible loop，不具备 watermark、backpressure、late event 或增量状态语义。
6. `AppendOnlyOperatorPacketStore` 是程序约束的 JSONL 追加存储，不是密码学不可篡改日志；论文不得超出该边界。

## 重建源码验证边界

本轮不再只做文件 inventory，而是对原始重建包和整改候选分别执行全仓测试。原始包结果为3328 passed / 110 failed / 203 skipped；整改候选为3337 passed / 108 failed / 203 skipped。失败节点集合比较显示本轮新增失败为0，并修复2个既有失败，但剩余108项仍使其不能成为“全量验证通过”的稳定软件基线。

稳定源码构建器还确认重建包缺少 `sat_sim_catalog.joblib` 和 `sat_sim_open_set.joblib`，而模型 manifest 仍声明其哈希，因而以 `MODEL_HASH_MISMATCH` fail closed。候选压缩包只能标为 method-remediation candidate，不能标为 stable source。不得从其他版本猜测或伪造这两个模型文件。

## 论文实验的最低验收线

- 诊断结果同时报告 fault recall、nominal operating-time false alarms、pre-onset alarms、拒判和严格标签/效应层级；
- latency 的报警序列必须先于真值独立产生；
- 所有模型/融合选择只在 development 完成，confirmation 不再调参；
- 以 trajectory、mission 或 pair seed 为统计单位给出区间；
- qualification 必须来自可验证的外部治理回执，不能由命令行字符串声明。
