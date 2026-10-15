# ADCS 长周期多随机种子神经网络对比

每个模型使用相同数据、完整机理特征、30 epoch 以上预算和相同训练随机种子集合。正常工况开发包只进入训练/验证；测试集不参与 checkpoint、融合或告警策略选择。

| 模型 | Det Acc | Det Recall | Det F1 | Loc Acc | Loc Recall | Loc F1 | Class Acc | Class Recall | Class F1 | E2E F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru_inception_specialist | 0.8689 ± 0.0111 | 0.9103 ± 0.0076 | 0.8529 ± 0.0113 | 0.9693 ± 0.0031 | 0.9676 ± 0.0062 | 0.9698 ± 0.0053 | 0.9586 ± 0.0043 | 0.9474 ± 0.0045 | 0.9485 ± 0.0054 | 0.8213 ± 0.0122 |

独立正常工况确认数据：`/mnt/PRO6000_disk/Timer/sat/data/generated/adcs_nominal_envelope_confirmation_seed20260926`。逐随机种子误警结果见 `report.json`。

- gru_inception_specialist: 0.0222 ± 0.0157; maximum=0.0333
