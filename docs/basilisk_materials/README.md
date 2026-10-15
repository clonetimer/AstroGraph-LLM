<!-- CURRENT_DOCUMENT: 0.7.1.post49 -->
# Basilisk 仿真材料与审核接收

本目录是随源码发布的精简材料副本。完整源码快照、原审核文件、评估 JSON 与原始哈希记录位于项目外侧 `../Basilisk仿真材料包/`（与 AstroGraph 源码目录同级）。本目录的窗口复核见 `窗口复核.md`，工作台材料见 `rag/`。

根据项目所有者本轮说明，当前目标是 Basilisk 仿真诊断。不再把真实设备手册、在轨故障报告或真实遥测作为当前工作的前置材料。采用官方模型说明、项目机制源码、可重复仿真与原始数据记录；不据此扩大为硬件或在轨资格。

## 本轮交付

- 已校验 8 条人工审核：4 APPROVED、4 REJECTED。批准的3条检索记录按原冻结分组分别为 train/validation/test，各1条；本体仅1条训练组空三元组。
- 52 条仿真场景/故障/约束目录记录（含正常项，并非52种故障），6项项目机制FMEA，5份可上传工作台的源码RAG文档。
- 6份来源逐字绑定且结构合法的候选本体；保持 REVIEW_REQUIRED，未伪造人工批准、未写入Active KG。
- 12项原生仿真机制测试通过；30个固定窗口保留28成功和2数据门禁拒绝。详细数值在本目录的窗口复核.md。

## 如何使用

工作台可上传 `rag/` 中的 Markdown，与既有诊断工件共同使用。`evaluator-only/` 内有事后标签与注入时刻，只用于评估，不上传到诊断或检索链路。`fault-catalog.json` 是机制知识，不能把其中事件名称当作当前案例真值。`source-snapshots/` 及 manifest.json 记录当前源码和哈希；sat 目录没有Git元数据，故不编造其commit。

## 本体字段规范（补齐上轮说明不足）

每条 triple 必须含 subject、predicate、object、source_document_id、source_quote。subject/object 是原文中的非空实体字符串，本接口不要求先编造独立实体 ID；source_document_id 必须是该输入文档的 ID。predicate 仅允许 part_of、affects、associated_with、measured_by、has_symptom、requires_evidence。引文逐字取自来源，主体客体均需出现。没有原文支持时允许空 triples。

根对象固定为 `{"triples":[],"lifecycle":"REVIEW_REQUIRED","active_kg_modified":false}`；完整 JSON Schema 见 ontology-proposal.schema.json。候选文件用 associated_with 连接目录名称与其描述，不将目录关联升级为已验证物理因果。

上轮“固放→solid rocket motor”拒绝记录保持 REJECTED，即使纠正列已有新查询，也不能自动变成批准样本。本地 transmitter runner 明确包含 RF Power Amplifier、TWT、SSPA，可据此保留“固放”或采用来源明确的射频固态功率放大器术语。

## 按需训练边界

3条已批准检索样本已完成训练计划导出，原 train/validation/test 分组不变；但只有1条训练记录，不启动无法支持泛化判断的领域微调。本体仅有1条空结果，缺少正例和完整来源拆分。新增源码候选用于材料补齐和审核，不能冒充 Timer 已批准的新训练集。当前无需额外搜集真实设备材料；后续可在本平台继续生成仿真案例和检验标签质量。

## 官方资料与本地版本

本地环境为 **2.11.0+satfix1**，访问的官网为 **2.11.1**。官网说明用于模型依据，本地实现行为以源码、配置和可执行测试为准。

- [官方示例入口](https://avslab.github.io/basilisk/examples/index.html)：姿态控制、轮控、能源与故障注入示例导航。
- [轮控示例](https://avslab.github.io/basilisk/examples/scenarioAttitudeFeedbackRW.html)：轮执行器、指令映射、轮速反馈；正常机动中也可能出现饱和，不能把饱和直接等同于故障。
- [轮故障事件示例](https://avslab.github.io/basilisk/examples/BskSim/scenarios/scenario_AddRWFault.html)：通过事件处理器注入轮故障。
- [SimpleBattery](https://avslab.github.io/basilisk/Documentation/simulation/power/simpleBattery/simpleBattery.html)：净功率积分和容量限制；官网还描述故障消息接口。本项目所审路径直接修改 storageCapacity，两者不能混写。
- [轮执行器接口](https://avslab.github.io/basilisk/Documentation/simulation/dynamics/reactionWheels/reactionWheelStateEffector.html)：指令输入、轮速与日志输出。数值稳定参数不是诊断验收阈值。
