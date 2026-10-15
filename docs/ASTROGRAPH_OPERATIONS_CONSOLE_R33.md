# AstroGraph Operations Console R33

**Document status:** `CURRENT_R33_OPERATIONS_CONSOLE`

## 目标

R33 将 Platform UI 升级为 AstroGraph-Agent Operations Console（UI `v0.13.0`）。它服务于航天器诊断操作流程：导入数据、服务器识别、Preflight、选择已有诊断模式、启动任务、查看后端实际阶段、复核诊断结果、执行受批准的 Agent 调查并导出结果。

R32 航天稳定核心保持冻结。R33 不修改诊断算法、Evidence 语义、仿真模型或 Agent 权限。

## 用户工作流

```text
导入数据 → 服务器识别输入 → Preflight → 选择已有模式 → 启动任务
       → 查看后端实际阶段 → 诊断结果 → Agent 调查 → 导出/维护
```

新建诊断页使用三步 Wizard。浏览器只展示服务器返回的工件识别、能力矩阵、阻断项和警告，不依据文件名猜测 Telemetry、Command log、System model 或 Runtime mapping。Preflight `BLOCKED` 时启动按钮保持禁用。

“标准诊断”和“增强诊断”是 UI 别名，分别映射到后端已经返回的 `normal` 与 `thinking` mode。后端未声明的 mode 不会出现在操作选项中；`creative` 仍保留在高级设置。

## 主导航

- 工作台：当前需处理、正在运行、最近诊断和服务状态。
- 新建诊断：上传、Preflight、真实 mode 选择和启动。
- 诊断任务：任务历史及重新打开。
- 诊断结果：结论、正式证据、图谱与机理、仿真、Agent 调查和技术详情。
- Agent 调查：模式、受控工具、执行回执和调查时间线。
- 图谱与仿真：系统图谱、故障传播、候选、仿真回执和相关知识。
- Bench：工程验证二级入口，不作为操作首页或资格声明。
- 系统与服务：LLM、Neo4j、MCP 与运行能力配置。

原有 DOM page ID 和 Router 机制保持不变；R33 只更新用户可见名称，并新增 `bench` 页面。

## 页面职责

`platform_job.js` 继续拥有上传、Preflight、start/retry、轮询和报告读取。`platform_report.js` 继续拥有 MaintenancePresentation、正式 Evidence Summary、Next Diagnostic Check、遥测与维护详情。`platform_operations.js` 只把已有 job/maintenance/report/agent 数据转换成操作员视图，不调用 predictor，不重新解释 Evidence，也不创建新的运行模式。

运行阶段只显示后端返回的 `stage_history`、`stages` 或当前 `stage`。界面不会为视觉完整性虚构 MBD、Simulation 或 Agent 阶段。

## Agent UI 权限边界

Agent Probe 只提交：

```json
{"probe_id": "..."}
```

浏览器没有 channel、threshold、window 或 algorithm 编辑器。Probe 技术 ID 在普通视图中转换为操作语言，原 ID 和 result hash 只在展开的审计详情中显示。Probe 回执不改变 formal evidence、diagnosis 或 release。

## Evidence UI 边界

证据页只读取正式 `EpisodeEvidenceBundle`/maintenance `evidence_summary`。如果正式摘要不存在，页面显示“正式证据摘要不可用”。波形统计、RAG、LLM reasoning 与图表不会被重新包装成正式 Evidence Summary。

## 图谱/仿真 UI 边界

图谱页只展示受治理的只读查询结果或报告已保存的参考机理；不提供任意 Cypher 输入。仿真页只展示已有受治理回执；不提供任意 simulator function 或参数执行。Simulation fit score 不显示为 fault probability。

## Bench 辅助定位

R33 Phase 1 的 Bench 是二级工程验证入口。工作台不展示 F1、MCC、AUPR 或模型排行榜。Bench 内容不得改变任务诊断、正式 Evidence 或发布状态。

## 测试结果

R33 新增静态合同测试和七个 Chromium 行为场景，覆盖三层状态分离、Preflight 启动门、Evidence 缺失、Probe 最小请求、Probe 后状态不变，以及 1366×768/160% 字体无页面级横向溢出。最终 package/fresh-extract 数量与日志保存在包外 R33 delivery evidence 中。

## Claim Boundary

R33 是前端和操作层升级。它不构成新的诊断算法性能证据、外部验证、概率校准或飞行资格。以下三条是展示层不变量：

```text
pipeline SUCCEEDED != diagnosis succeeded
Agent probe result != formal EvidenceFact
simulation fit score != fault probability
```
