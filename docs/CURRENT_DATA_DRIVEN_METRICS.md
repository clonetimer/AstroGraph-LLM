# AstroGraph post49 当前数据驱动指标口径

状态：`CURRENT_AUTHORITATIVE_METRICS_INDEX`
版本：`0.7.1.post49`

> 当前能力状态以 `governance/implementation/post49/RELEASE_READINESS.json` 为最高事实源。本文件只整理仍需引用的指标及其协议，不把历史高分自动升级为最新资格。

## ADCS

### A. post49 exact-profile runtime / used-cohort replay（当前运行能力证据，不是新独立确认）

| 指标 | 结果 | 分母/说明 |
|---|---:|---|
| Detection Macro-F1 | **0.9835** | 501 个模型可执行窗口 |
| Detection Macro-F1 | **0.9207** | 全部 546 窗口，包含 unavailable |
| Execution coverage | **501/546 = 91.76%** | 45 个窗口不可执行/不可用 |
| Localization independent head | **1.0000** | 已使用 replay cohort |
| Localization system | **0.9935** | 已使用 replay cohort |
| Classification raw head | **0.9826** | 已使用 replay cohort |
| Classification system | **0.9698** | 已使用 replay cohort |

状态：`USED_COHORT_RETROSPECTIVE_NOT_INDEPENDENT_CONFIRMATION`。

### B. ADCS Group-CV 研究基线

`0.9293 / 0.9687 / 0.9399 / 0.8974`（Detection / Localization / Classification / E2E）保留为 `CURRENT_RESEARCH_BASELINE_NOT_PROMOTED`，用于跨 group 研究对照；不能与上面的 used-cohort replay 混成同一指标。

### C. 历史 2026-08-27 production qualification

旧 qualification 的具体历史指标不随当前交付源码包分发；它们不得作为 post49 当前结果。需要历史复现时请使用独立归档包。

### D. R36 event-causal 方法复核

旧 R36 trajectory adapter 先使用真值 onset 过滤报警，不能代表在线报警延迟。按真值无关的报警 episode 重算后：Detection F1 0.9199，nominal FAR 6/6，故障前报警110/176，Localization accuracy 0.6534，Identification accuracy 0.5568。该结果为同一冻结划分上的 DEVELOPMENT 方法复核，不是新独立确认；详细协议和剩余阻塞见 `THESIS_METHOD_REMEDIATION.md`。

### E. Post51 严格因果窗口多训练种子候选（未晋升）

GRU 检测专家与 InceptionTime 定位/分类专家在三个预声明训练 seed、30 epoch 预算下取得：Detection Accuracy/Recall/F1 `0.8689/0.9103/0.8529`，Localization `0.9693/0.9676/0.9698`，Classification `0.9586/0.9474/0.9485`，E2E F1 `0.8213 ± 0.0122`。独立正常确认集的最大轨迹误警率为 `1/30 = 0.0333`。状态为 `MULTISEED_SYNTHETIC_RESEARCH_CANDIDATE_NOT_PROMOTED`；协议和限制见 `POST51_MULTISEED_NEURAL_BENCHMARK.md`。

## PHM / sat

### A. 已晋升的 known-activation retrospective replay

| 协议 | Detection | Localization | Classification | E2E | 状态 |
|---|---:|---:|---:|---:|---|
| post-activation fixed split | 0.89919 | 1.00000 | 1.00000 | 0.96711 | `PROMOTED_RETROSPECTIVE_REPLAY_DEFAULT` |
| post-activation 5-fold Group-CV | 0.94241 | 1.00000 | 1.00000 | 0.98514 | `RESEARCH_CV_SUPPORT_NOT_INDEPENDENT_CONFIRMATION` |

该能力依赖已知 activation/retrospective trace-summary，不外推为未知起点在线诊断。

### B. unknown-onset / causal-window 正式研究评估

`0.94334 / 0.96010 / 0.94904 / 0.91705`（Detection / Localization / Classification / E2E）保留为单独协议的 formal research evaluation；不能被 retrospective replay 数字替换。

## 读取规则

1. 永远同时写协议、数据角色和分母；
2. replay / validation / Group-CV / independent confirmation 不互换；
3. “模型 head”与“系统最终输出”不互换；
4. unavailable/abstain 不能从分母中静默删除；
5. 任何新数字若未进入 post49/current successor readiness，不得覆盖本页。
