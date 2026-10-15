# Post51 ADCS 长周期多随机种子神经网络研究记录

## 结论

本轮在同一因果窗口、完整机理特征、冻结数据划分和告警策略约束下，对 Residual TCN、GRU、InceptionTime、ModernTCN、六级渐进 GAT 做了三训练随机种子对比。每次训练提供 30 epoch 预算，至少执行 20 epoch；实际最优 checkpoint 多次出现在第 19–27 epoch，因此早先 3–8 epoch 只能用于冒烟测试，不能支持模型排序。

没有单一骨干同时赢得三个任务。GRU 的检测跨种子最稳定，InceptionTime 的定位和分类最好。最终候选采用两个完整多任务专家：GRU 专家的检测头作为检测输出，InceptionTime 专家的定位和分类头作为后两任务输出；两个专家保留各自三任务辅助监督、独立梯度裁剪和独立 checkpoint 选择。该设计把端到端 Macro-F1 提高到 `0.8213 ± 0.0122`，同时把最低随机种子结果提高到 `0.8063`。

六级渐进 GAT 在当前数据上没有优于简单时序骨干。它应继续承担层级一致性、传播先验和解释约束，不应仅因结构更复杂而被设为默认分类器。

## 数据位置与角色

| 角色 | 绝对路径 | seed | 轨迹/行数 | 使用边界 |
|---|---|---:|---:|---|
| 故障开发与冻结测试 | `/mnt/PRO6000_disk/Timer/sat/data/generated/whole_spacecraft_adcs_paired_dataset_v5_2_2` | 20260810 | 910 / 273,910 | 沿用 sample-index group split |
| 正常工况开发 | `/mnt/PRO6000_disk/Timer/sat/data/generated/adcs_nominal_envelope_development_seed20260923` | 20260923 | 30 / 9,030 | 只进入训练与验证选择 |
| 旧确认数据 | `/mnt/PRO6000_disk/Timer/sat/data/generated/adcs_nominal_envelope_confirmation_seed20260924` | 20260924 | 30 / 9,030 | 已暴露，不再用于新候选确认 |
| 旧确认数据 | `/mnt/PRO6000_disk/Timer/sat/data/generated/adcs_nominal_envelope_confirmation_seed20260925` | 20260925 | 30 / 9,030 | 已暴露，不再用于新候选确认 |
| 本轮独立正常确认 | `/mnt/PRO6000_disk/Timer/sat/data/generated/adcs_nominal_envelope_confirmation_seed20260926` | 20260926 | 30 / 9,030 | 候选方案冻结后读取 |

本轮新生成数据由 `sat/scripts/generate_whole_spacecraft_adcs_eps_dataset_v522.py` 和 Basilisk 后端产生，30 条轨迹全部 `PASS`。开发正常数据 manifest SHA-256 为 `ef89fa91d9ea56ad9b95d27a4292282ae36dc330b245907357c2b59bd3e878c5`；新确认数据为 `70489fdfb7ca8b22fc1d912abcf3998619ba73654cc2025657f8eeeae8889fee`。

## 五种骨干对比

以下为三随机种子均值 ± 总体标准差：

| 模型 | Det F1 | Loc F1 | Class F1 | E2E F1 | 验证正常误警约束 |
|---|---:|---:|---:|---:|---|
| Residual TCN | 0.7897 ± 0.0878 | 0.9067 ± 0.0153 | 0.8824 ± 0.0161 | 0.7025 ± 0.1307 | 全种子通过 |
| GRU | **0.8351 ± 0.0121** | 0.9192 ± 0.0011 | 0.9119 ± 0.0096 | **0.7741 ± 0.0169** | 全种子通过 |
| InceptionTime | 0.8326 ± 0.1148 | **0.9762 ± 0.0058** | **0.9562 ± 0.0036** | 0.7697 ± 0.1693 | 全种子通过 |
| ModernTCN | 0.7860 ± 0.0572 | 0.9384 ± 0.0136 | 0.9249 ± 0.0087 | 0.7215 ± 0.0887 | 全种子通过 |
| 六级渐进 GAT-TCN | 0.7065 ± 0.0770 | 0.9061 ± 0.0082 | 0.8926 ± 0.0065 | 0.5884 ± 0.1043 | 未全种子通过 |

InceptionTime 的单个检测坏种子使其端到端方差明显增大。六级 GAT 的检测均值、故障事件召回和端到端指标均落后，并有一个种子违反开发集正常误警约束。这些结果否定了“层级更深就必然更准确”的假设。

## 最终专用候选

| 任务 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | F1 跨种子最小值 |
|---|---:|---:|---:|---:|---:|
| 检测 | 0.8689 ± 0.0111 | 0.8365 ± 0.0095 | 0.9103 ± 0.0076 | 0.8529 ± 0.0113 | 0.8419 |
| 定位 | 0.9693 ± 0.0031 | 0.9725 ± 0.0046 | 0.9676 ± 0.0062 | 0.9698 ± 0.0053 | 0.9634 |
| 分类 | 0.9586 ± 0.0043 | 0.9527 ± 0.0061 | 0.9474 ± 0.0045 | 0.9485 ± 0.0054 | 0.9435 |
| 端到端 15 类 | — | — | — | 0.8213 ± 0.0122 | 0.8063 |

与五模型中端到端均值最好的单骨干 GRU 相比，最终候选的端到端 F1 提高 `0.0472`，标准差从 `0.0169` 降至 `0.0122`，最低种子从 `0.7581` 提高至 `0.8063`。检测 F1 均值比 GRU 提高 `0.0178`，最低种子提高 `0.0229`。这些比较只针对同一 post51 协议。

三个训练种子在开发验证正常轨迹上的误警率均为 0；故障事件召回为 `0.8807 ± 0.0167`。独立正常确认集上，seed 20260930 为 0/30，seed 20260931 和 20260932 均为 1/30；三种子平均轨迹误警率为 `0.0222 ± 0.0157`，最大值 `0.0333`，低于预设 5% 上限。两个种子均在同一 `attitude_acquisition:nominal:mc_04` 轨迹报警，提示姿态捕获阶段的正常瞬态仍是下一轮应优先扩充的数据区域。

## 可复现命令

```bash
python scripts/data_driven/run_adcs_multiseed_neural_benchmark.py \
  --output-dir run_artifacts/post51-adcs-multiseed-neural-benchmark \
  --models residual_tcn gru inception_time modern_tcn \
    progressive_hierarchical_gat_tcn \
  --epochs 30 --minimum-epochs 20 --early-stop-patience 7 \
  --batch-size 512 --device cpu

python scripts/data_driven/run_adcs_multiseed_neural_benchmark.py \
  --output-dir run_artifacts/post51-adcs-gru-inception-specialist-final \
  --models gru_inception_specialist \
  --epochs 30 --minimum-epochs 20 --early-stop-patience 7 \
  --batch-size 512 --device cpu \
  --external-nominal-root \
  /mnt/PRO6000_disk/Timer/sat/data/generated/adcs_nominal_envelope_confirmation_seed20260926
```

结果文件：

- `run_artifacts/post51-adcs-multiseed-neural-benchmark/report.json`；
- `run_artifacts/post51-adcs-gru-inception-specialist-final/report.json`。

对应 SHA-256 分别为 `6223a3751fbadf621d5d8aa22ac304531541e516738f71a3080913e24caeafaf` 和 `e3352756936bd97f0951d2be162aaf34be93775cc60174f4081455f0b8003fd4`。

## 任务台账

| 任务 | 状态 | 证据 |
|---|---|---|
| 定位新增仿真数据 | 完成 | 本文数据位置表与 manifest hash |
| 五种神经网络长周期比较 | 完成 | 三种子、30 epoch 预算报告 |
| Accuracy/Precision/Recall/F1 完整输出 | 完成 | benchmark JSON 与 Markdown |
| 随机种子稳定性统计 | 完成 | mean/std/min/max，三个预声明 seed |
| 检测与诊断专家解耦 | 完成 | `gru_inception_specialist` 与梯度隔离测试 |
| 独立正常工况确认 | 完成 | seed 20260926，最大 1/30 轨迹误警 |
| 新故障强度与未见任务外推 | 待后续独立研究 | 当前没有新故障确认 cohort |
| 飞行数据 qualification | 不主张 | 当前仍为 Basilisk 合成 DEVELOPMENT 证据 |

## 声明边界

本轮证明的是训练随机初始化稳定性和当前 Basilisk 合成协议内的三任务收益，不证明跨航天器、跨仿真器或飞行数据泛化。旧 post49 replay 高分、R36 event-causal 复核和本轮严格因果窗口指标使用不同分母与协议，不能直接混合。新确认数据仅覆盖正常工况，因此它只约束误警，不是故障检测、定位或分类的独立确认。
