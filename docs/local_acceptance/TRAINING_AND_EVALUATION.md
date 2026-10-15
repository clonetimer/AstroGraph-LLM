# post49 训练与评价规则

状态：`CURRENT_POST49_EVALUATION_GUIDE`

## 训练

任何训练输出默认是 candidate，不覆盖 `models/` 当前批准资产。模型晋升必须经过冻结 split、评价、calibration/OOD（若声明）、failure regression、promotion review 和 rollback preflight。

## 评价角色

- Train：拟合；
- Validation/Development：选择模型、阈值、校准；
- Used replay：复现/回归，不能改称新独立确认；
- Independent held truth：新确认；必须 prediction-before-truth。

## ADCS

当前 exact 33-channel profile 的 used-cohort replay 与 Group-CV 指标分开读取，见 `../CURRENT_DATA_DRIVEN_METRICS.md`。当前数据供应没有新的未使用 trajectory group 时，结论应是 `NEW_GROUP_REQUIRED`，不是重新切分旧 group。

## PHM/sat

Known-activation retrospective replay 与 unknown-onset causal-window evaluation 是不同协议，不能互换。

历史 benchmark/runbook 已归档到 `../history/`。
