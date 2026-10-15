# LLM 主动诊断探索实验运行说明

文档状态：`CURRENT_DEVELOPMENT_EXPLORATORY`。

本文档只描述开发版探索实验。所有结果均不构成正式证据、生产资格、飞行结论或根因诊断。

## 环境与输入

- Python：`E:\tmp\AstroGraph\AstroGraph\.venv\Scripts\python.exe`
- 当前动作策略实验服务：本地 `http://127.0.0.1:8081/v1`，由 `ssh -L 8081:localhost:8000 a@10.100.182.135` 转发到远端 vLLM 8000 端口
- 当前动作策略模型：`Qwen3.8-27B`（必须以 `/models` 实际返回为准）
- 历史 A2/A8/A10/A14 运行使用过 LM Studio `http://127.0.0.1:1234/v1` 和 `qwen/qwen3.5-9b`；不得把两种后端结果混作同一模型重复实验
- 原始段：`E:\tmp\AstroGraph\segments.csv`
- 官方派生特征：`E:\tmp\AstroGraph\dataset.csv`
- `segments.csv` SHA-256：`d5201e9e751eb2a53a0ff7c11567dc4239f594ea4b479b2aa66fe67ddcbcb9ba`
- `dataset.csv` SHA-256：`b524177da6f516d5c9f63c7acbc385341f0ad42046ef20c1bed2d25e51b98f02`

脚本拒绝覆盖非空输出目录。运行前会校验输入身份、报告内容摘要、开发分区和 LM Studio 模型身份。评价真值只在模型返回后加入报告；`EXTERNAL_VALIDATION` 未被开发实验消费。

## 动作策略四臂比较

先预检数据、分区、动作合同、模型清单和 vLLM 版本；预检不创建实验结果目录：

```powershell
& 'E:\tmp\AstroGraph\AstroGraph\.venv\Scripts\python.exe' scripts/experiments/run_ops_sat_action_policy_comparator.py `
  --data-root 'E:\tmp\AstroGraph' `
  --output-dir 'E:\tmp\AstroGraph\run_artifacts\ops-sat-action-policy-preflight-unused' `
  --url 'http://127.0.0.1:8081/v1' `
  --model 'Qwen3.8-27B' `
  --cases-per-class 1 `
  --preflight-only
```

随后先用每类 1 例冒烟，再以新的独立目录运行每类 4 例。四臂分别为 `FROZEN_NO_NEW_OBSERVATION`、`TRAIN_INFORMATION_GAIN_POLICY`、`STABLE_RANDOM_LEGAL_ACTION` 和 `LLM_ACTION_POLICY`，每例最多一次新观测。训练动作合同只能读取 `MODEL_TRAIN`，评价只读 `DEVELOPMENT_SELECTION`；健康病例专门检验错误前提诱导下的误警。报告中的请求 ID是内容绑定的关联标识，不是服务端签名。

```powershell
& 'E:\tmp\AstroGraph\AstroGraph\.venv\Scripts\python.exe' scripts/experiments/run_ops_sat_action_policy_comparator.py `
  --data-root 'E:\tmp\AstroGraph' `
  --output-dir 'E:\tmp\AstroGraph\run_artifacts\ops-sat-action-policy-qwen38-new' `
  --url 'http://127.0.0.1:8081/v1' `
  --model 'Qwen3.8-27B' `
  --cases-per-class 4
```

## 分组保序候选集

`astrograph.evaluation.conformal_candidates` 用冻结概率预测和独立校准组生成内容绑定回执。校准组不得在应用阶段复用，标签全集漂移或未准入的评估组会失败关闭。输出仅是影子候选集/弃权建议；空候选集会把完整标签全集作为运行时回退，同时保留 `candidate_ids=[]`，不得解释为单一诊断。

## 实验链

| 实验 | 作用 | 关键结果 | 报告内容 SHA-256 |
|---|---|---|---|
| A2/A8 | 原始 LLM 主动影子对照 | A8 未暴露新遥测，主动效果不可识别 | `22343e3167ab9cabcd93ca77ce5655702eef8a8a55bac7ef24622560689a0fd9` |
| 探针效用审计 | 六项原始段探针信息量 | 组合信息增益 `0.026302 bits/例`，召回 `0.102041` | `2a3fd53335e72cd000be17571913f54c9430a1444fadc7c6debc6d6cc1d84d86` |
| A10 | LLM 读取六探针组合 | 仅一次可归因变化且为健康误警 | `6c062775d25cc57615b827b7cf870e96bad5e181715efc379b1e77b2d9e49124` |
| A11 | 同观测确定性嵌套校准 | 227 例 F1 `0.380952`，健康误警 `2/178` | `81384cc0960be1514d8363adc292df2d5fa82712ee4b4dc66f0c34f8d014acb5` |
| A12 | 通道条件化校准 | 多召回 1 例但新增 18 个健康误警，拒绝 | `8a3f5e2994fff40f6fb6d8b0fb62933f14ff07789341ed63952a1f4101749624` |
| A13 | 官方长度无关派生特征 | 227 例 F1 `0.689655`，健康误警 `8/178` | `8fce05f9fc7237e40bc2c593acbabebb3cc2917afca0876d6c9165da36ac399f` |
| A14 | LLM 读取 A13 观测与 OOF 摘要 | 8/8 歧义或弃权，0 次可归因细化 | `63cd3835e0ce08a3c2a24a899bed8d7eb49079b86fdd4ef7271e53780ba23fc9` |
| A15 | 四臂主动动作策略比较（Qwen3.8-27B/vLLM） | LLM 与信息增益臂同为 `0.375`、健康误警 `1/4`；未胜过稳定随机臂 `0.5` | `1ac9e46cf05b023b83e37e4e709d564ddfad6ee5956f54cfc2241675ffcd7ebe` |

报告位于 `E:\tmp\AstroGraph\run_artifacts`。A14 报告文件字节 SHA-256 为 `007b651ab2cbb003bcc01e7bf5aa1e3b6ce0c61a4ced60dd074968e8ad7fabae`。

A15 报告位于 `ops-sat-action-policy-qwen38-20260927-full/report.json`，文件字节 SHA-256 为 `0f4ee33453510b14b2e967ee33f0340939fab0230648e54d7bb3226fd8e84d03`。16 条调用历史均含不同的内容绑定请求 ID。该结果说明动作闭环可观测，不说明 LLM 优于简单策略；8 例开发样本尤其不能用于确认性比较。

## A13：观测表示审计

A13 对 227 条 `DEVELOPMENT_SELECTION` 轨迹做嵌套 OOF 评估。长度无关集合为：

`mean, var, std, kurtosis, skew, n_peaks, smooth10_n_peaks, smooth20_n_peaks, diff_peaks, diff2_peaks, diff_var, diff2_var`

排除 `duration, len, len_weighted, var_div_duration, var_div_len, gaps_squared`，避免把段构造和长度代理误当作在线诊断能力。A13 读取的是已有的回顾性整段派生特征，不是新增传感器样本，也不是早期预警结果。

```powershell
& 'E:\tmp\AstroGraph\AstroGraph\.venv\Scripts\python.exe' scripts/experiments/run_ops_sat_official_feature_observation_audit.py `
  --data-root 'E:\tmp\AstroGraph' `
  --raw-policy-report 'E:\tmp\AstroGraph\run_artifacts\ops-sat-deterministic-bundle-20260926-r4\report.json' `
  --output-dir 'E:\tmp\AstroGraph\run_artifacts\ops-sat-official-feature-observation-new'
```

## A14：派生特征上的 LLM 影子修订

A14 对 A10 冻结的 8 个病例各调用模型一次，无修复重试，每例至多一次 HTTP 尝试。模型只能输出 `decision` 和 `citation_scope` 两个枚举字段；候选与引用 ID 均由宿主映射。输入包含 A13 的长度无关特征、该病例的严格 OOF 分数、折内选择阈值及候选 ID。

结果为：8 个病例均未作单候选决定，6 个引用主动观测，0 个满足“引用主动观测且候选变化”；总耗时 `321.615 s`，prompt/completion tokens 为 `9442/6464`，模型调用 8 次。A13 在同 8 例正确 5 例，A14 按弃权计错为 0 例；McNemar 双侧精确 `p=0.0625`。这是“无可归因收益”的探索结果，不是 LLM 普遍无效的结论。

```powershell
& 'E:\tmp\AstroGraph\AstroGraph\.venv\Scripts\python.exe' scripts/experiments/run_ops_sat_feature_llm_comparator.py `
  --data-root 'E:\tmp\AstroGraph' `
  --a10-report 'E:\tmp\AstroGraph\run_artifacts\ops-sat-compact-bundle-20260926\report.json' `
  --a13-report 'E:\tmp\AstroGraph\run_artifacts\ops-sat-official-feature-observation-20260926-r2\report.json' `
  --output-dir 'E:\tmp\AstroGraph\run_artifacts\ops-sat-feature-llm-new' `
  --url 'http://127.0.0.1:1234/v1' `
  --model 'qwen/qwen3.5-9b'
```

可先添加 `--case-limit 1` 做冒烟；脚本会按实际选中病例重算 A13 对照，不会把 8 例指标错用于单例运行。

## 结果解释边界

- `active_observation_exposed` 只说明工具读到了新输入。
- `active_observation_cited` 只说明输出引用了该观测。
- `active_observation_attributable` 还要求候选集合发生变化；它仍不是因果性能证明。
- 缺少可靠连续运行时长分母时，不报告单位时间误警。
- OPS-SAT 只有二元异常真值，不能据此声称根因故障分类能力。
- A13/A14 已看开发数据，后续方法选择不得把相同数据包装成独立验证。

## 工程与分发边界收口

- 当前源码分发不伪造缺失的历史治理回执或模型二进制。依赖这些资产的历史重放测试在 `configs/historical_archive_test_manifest.v1.json` 中逐项声明所需路径和跳过理由；这只解释不可重放范围，不把缺失资产视为资格通过。
- 开发数据访问通过 `dataset_use` 记录数据集、分区、用途和病例 ID。A13/A14 只消费 `DEVELOPMENT_SELECTION`，报告明确写入 `external_validation_groups_consumed=false`，并将真值连接放在模型调用之后。
- 原子文件/目录发布保持失败关闭。Windows 对新写目录的临时句柄锁只触发有界 `PermissionError` 重试；POSIX 和其他错误立即上抛。实现不会退化为递归复制，因此不会暴露半完成事务。
- 外部程序终止使用平台分支：POSIX 进程组与 Windows `taskkill`/进程终止各自遵守总宽限期；测试和入口不得假设只存在一种 shell 或路径语义。

## 定向验证

```powershell
& 'E:\tmp\AstroGraph\AstroGraph\.venv\Scripts\python.exe' -m pytest `
  tests/test_ops_sat_probe_utility.py `
  tests/test_ops_sat_compact_bundle_comparator.py `
  tests/test_ops_sat_deterministic_bundle_comparator.py `
  tests/test_ops_sat_feature_llm_comparator.py -q
```

最后一次定向结果为 `19 passed`。全量测试应使用新的临时目录和 JUnit 输出，避免污染或覆盖历史回执。

机器可读执行画像的全量命令为：

```powershell
& 'E:\tmp\AstroGraph\AstroGraph\.venv\Scripts\python.exe' -m pytest `
  --astrograph-test-profile source-core `
  --astrograph-enforce-profile `
  --basetemp 'E:\tmp\AstroGraph\run_artifacts\pytest-post-optimization-temp' `
  --junitxml 'E:\tmp\AstroGraph\run_artifacts\pytest-post-optimization.xml' `
  --durations=50 -q
```

`source-core` 是 `source-only` 的别名。跳过项会按历史封存缺失、外部运行时、外部数据、浏览器依赖和平台不适用等类别计数；未知原因预算为 0，新增无法解释的跳过会令测试会话失败。
