# Post50 渐进层级图诊断研究记录

## 结论

本轮把 ADCS 诊断拓扑扩展为“参量→部件→组件→功能模块→分系统→整星”，并完成与平坦 TCN、原部件级门控 GAT 的同协议消融。六级模型作为研究候选保留；它改善定位和分类，但检测收益主要来自独立正常工况覆盖、按轨迹选择的误警约束及机理残差，而不是图消息本身。

最终候选使用 11 个参量节点、8 个部件、5 个组件、4 个功能模块、1 个 ADCS 分系统和 1 个整星节点。检测支路始终为无图全局 TCN，图消息只进入定位和分类，避免工程拓扑先验直接制造“检测正确”。拓扑只由配置和通道名确定，不使用遥测数值或评估标签拟合。

## 数据与隔离

| 用途 | 数据 | 随机种子 | 轨迹 | 使用边界 |
|---|---|---:|---:|---|
| 原故障开发/旧测试 | `whole_spacecraft_adcs_paired_dataset_v5_2_2` | 20260810 | 910 | 沿用冻结 sample-index 分组协议 |
| 正常工况开发 | `adcs_nominal_envelope_development_seed20260923` | 20260923 | 30 | 18 条加入训练，12 条加入验证选择 |
| 第一次确认 | `adcs_nominal_envelope_confirmation_seed20260924` | 20260924 | 30 | 基础特征方案一次性确认；已暴露后不再用于新方案确认 |
| 最终确认 | `adcs_nominal_envelope_confirmation_seed20260925` | 20260925 | 30 | 全特征方案冻结后一次性确认 |

三个新增数据包均由 `sat/scripts/generate_whole_spacecraft_adcs_eps_dataset_v522.py` 以 Basilisk 后端生成，每条 300 秒、采样周期 1 秒，标签与遥测分离。manifest SHA256 分别为：

- 开发包：`ef89fa91d9ea56ad9b95d27a4292282ae36dc330b245907357c2b59bd3e878c5`
- 第一次确认包：`ac8dc155aac345bea2d389dd480b0fcc5e7c2587908d0a5f92e22bdc23098fdf`
- 最终确认包：`ec533a4e01f351700c4bc169f76bef62953860c78668c6c680d6af5fcff1675e`

## 消融与决策

三模型在原冻结验证协议、3 epoch、同一随机种子下的选择分数为：

| 模型 | 验证选择分数 |
|---|---:|
| Residual TCN | 0.8519 |
| 原部件级门控 GAT | 0.8616 |
| 六级渐进 GAT | 0.8707 |

直接部署该结果是错误的。新增 30 条开发正常轨迹首次检查时，有 21 条至少一次报警。随后按以下顺序整改：

1. 正常开发包按轨迹加入训练和验证，禁止同名轨迹跨数据包合并；
2. 低容量检测器改用相对每条轨迹首个因果窗口的变化特征，减弱常量工况偏置；
3. 阈值选择从窗口 F1 改为事件召回与轨迹误警约束，要求开发验证正常轨迹误警率不高于 5%；
4. 候选策略扩展为阈值 0.30–0.995、1/3/5/10 连续窗口和滞回；
5. 加入星敏—陀螺创新量、飞轮指令执行残差、速度响应残差和指令/实测电流对。

基础特征冻结方案在第一次确认包上误报 2/30 条，因此被否决。全特征方案在开发验证上选择阈值 0.99、off 阈值 0.89、连续 5 窗口；故障事件召回 0.8068，正常轨迹误警 0/18。该方案冻结后才读取最终确认包。

## 最终研究结果

| 指标 | 结果 |
|---|---:|
| 最终确认正常轨迹误警 | 0/30 |
| 最终确认误警窗口 | 0/3060 |
| 旧测试检测 macro-F1 | 0.7900 |
| 旧测试定位 macro-F1 | 0.8829 |
| 旧测试分类 macro-F1 | 0.8642 |
| 旧测试端到端 15 类 macro-F1 | 0.7313 |

结果文件：

- `run_artifacts/post50-progressive-hierarchy/validation-ablation/report.json`
- `run_artifacts/post50-progressive-hierarchy/confirmation/report.json`（被否决的基础特征确认）
- `run_artifacts/post50-progressive-hierarchy/full-feature-selection/report.json`
- `run_artifacts/post50-progressive-hierarchy/full-feature-confirmation/report.json`

## 任务台账

| 任务 | 状态 | 证据 |
|---|---|---|
| 六级语义拓扑合同 | 完成 | `configs/data_driven/progressive_hierarchies.json` |
| 六级逐级注意力与门控 | 完成 | `src/astrograph/data_driven/progressive_hierarchy.py` |
| 图与检测支路隔离 | 完成 | 梯度测试及 `detection_uses_graph=false` |
| 平坦/部件级/六级消融 | 完成 | `validation-ablation/report.json` |
| 独立正常开发数据 | 完成 | seed 20260923，30 条 |
| 约束式事件级策略选择 | 完成 | `compare_adcs_deep_encoders.py` |
| 基础特征确认 | 完成且否决 | seed 20260924，2/30 轨迹误报 |
| 机理残差增强 | 完成 | full feature profile |
| 新种子一次性确认 | 完成 | seed 20260925，0/30 轨迹误报 |
| 多训练种子稳定性统计 | 已完成 | `POST51_MULTISEED_NEURAL_BENCHMARK.md`，三个预声明训练 seed |
| 新故障强度/未见任务迁移 | 待完成 | 当前确认包仅扩展正常工况 |
| qualification/飞行有效性 | 不主张 | 全部证据仍为 Basilisk 合成开发证据 |

## 声明边界

本轮支持的结论是：在当前 Basilisk ADCS 合成协议中，机理残差和六级渐进图表征能够形成一个低误警研究候选，并提升已检测故障的定位与分类。它不证明六级 GAT 普遍优于简单模型，不证明跨航天器泛化，也不满足 qualification 或真实飞行部署要求。30 条正常确认轨迹上观测到 0 次误警只给出有限样本证据；后续必须补多训练种子、故障强度外推、任务剔除验证和置信区间。
