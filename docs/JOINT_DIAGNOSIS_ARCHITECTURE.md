# post49 联合诊断任务语义

状态：`CURRENT_POST49_DIAGNOSIS_SEMANTICS`

## 三任务定义

1. **Detection**：判断窗口/通道是否异常；
2. **Localization**：定位受影响 component/subsystem，可多候选；
3. **Classification**：判断 fault class；
4. **Root cause reasoning**：独立于前三项，需要 KG/机理/时序/反证等证据，不由 localization 或 classification 自动推出。

模型可以共享编码器或同时输出三头，但语义解释仍保持上述区分。

## GenericSeries

- 无参量名：可做数值异常检测；物理原因能力有限；
- 有参量名：可映射语义并生成 provisional KG cause candidate；
- 只有已资格/admitted profile 才执行分类/定位；
- 不可辨识单参量允许 `UNAVAILABLE/ABSTAIN`，不能为了能力矩阵强制分类。

## exact ADCS

冻结的 33-channel ADCS profile 是 post49 当前完整三任务执行 profile。要求 channel set/order、unit contract、model artifact hash、host binding 与 admission 精确满足。它不能外推为任意 ADCS 子集、任意多参量或其他卫星。

## 多源证据

数据模型、KG、文献、Agent hypothesis、专家确认分别保留 provenance。文献与 joint-evidence package 用于解释和审计，当前不覆盖 committed core diagnosis。
