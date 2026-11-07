# P4-E1 新公开事件候选：时间截断评测

状态：**两例公开事件的来源与输入已封存，仍为候选；不是独立专家金标准或完整卫星诊断集。**

## 选择与用途

在 2026-11-07 检索官方任务通报，选取不同子系统且有明确公开时间线的事件：新视野号 2015-07-05 初报与 2015-07-06 后续结论；Galileo 2017-01-19 原子钟异常通报。检索本地 CP275 工作区及当前仓库的相关文字，未发现这两个具体官方通报此前进入案例/报告种子。此项只能说明项目资料的局部未见，不能证明基础模型预训练未见。

每个 `cases/*/input.md` 是根据截止时点官方通报**重新表述**的模型可见材料；不包含后续结果。官方原始 HTML 保存在 `sources/` 供核验；网页导航、推荐链接与后续文章可能泄漏答案，因此**不直接把原始 HTML 或整个目录送给模型**。`reviewer_only/` 只能在生成后供评审使用。文件与来源 SHA-256 在 `manifest.json`，其自身哈希在 `manifest.sha256`。

| 案例 | 当时可见来源 | 评审侧材料 | 评价核心 |
|---|---|---|---|
| NH2015 | NASA 2015-07-05 初报 | NASA 2015-07-06 后续通报 | 不能把事后时序缺陷提前当已确认根因 |
| GAL2017 | ESA 2017-01-19 通报 | 同通报的来源范围核对 | 区分 RAFS/PHM、FOC/IOV 及未知根因 |

本批没有原始遥测或上游 BiRRF 输出，适于检验公开材料的证据边界、阅读与核查建议，**不适于计算真实在轨故障诊断准确率，也不能代替四环节联通验收**。两例样本也不足以选型。后续仍须补原始任务数据与独立工程评审。

## 来源

- [NASA 初报](https://www.nasa.gov/missions/new-horizons-team-responds-to-spacecraft-anomaly/)，2015-07-05。
- [NASA 后续通报](https://www.nasa.gov/missions/nasas-new-horizons-plans-july-7-return-to-normal-science-operations/)，2015-07-06，仅评审侧使用。
- [ESA Galileo 钟异常通报](https://www.esa.int/Applications/Satellite_navigation/Galileo_clock_anomalies_under_investigation)，2017-01-19。

JPL Dawn 页面虽可检索，直接下载返回 HTTP 403；未列入本批封存案例。
