# 初始知识图谱示例路径

> 此页仅演示如何用**公开通用资料**枚举故障原因与复核措施，所有推导与建议均未经型号专家审核。箭头表达**候选关联**，不代表该条因果已在某型号上成立。

## 例 1：姿控——反作用飞轮动量饱和

- **部件**：反作用飞轮
- **可能机理**：飞轮角动量存储达到设计极限
- **异常症状**：飞轮转速接近限值，姿态控制力矩裕量不足
- **建议验证**：核对飞轮转速/角动量估算、力矩指令和姿态误差时序
- **可能措施**：评估按批准的型号程序进行动量卸载；前提是外力矩执行机构及程序被批准
- **主要资料**：[NASA 2026 GNC，第 5.2.2 节 Reaction Wheels](https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/)
- **代号**：`ADCS-WHEEL-SAT`

## 例 2：电源——母线欠压

- **候选故障**：`EPS-UNDERVOLT`
- **支持线索**：母线电压下降、负载变化、供电能力下降及电池荷电状态变化等数据需要联合判断
- **跨系统候选上游因素**：太阳电池阵列性能下降，或运行进入地影后光伏发电中断
- **反证要求**：核验是否只是正常地影工况、传感器偏差或母线瞬态而非实际失效
- **行动边界**：负载重分配和保护恢复只能依据正式飞行规则提出与批准
- **资料**：[NASA 2026 电源分系统](https://www.nasa.gov/smallsat-institute/sst-soa/power-subsystems/)

## 例 3：推进——阀门泄漏与非指令力矩

- **候选故障**：`PROP-VALVE-LEAK`
- **公开案例背景**：NASA 的推进章节记载了 MarCO-B 阀门泄漏与姿控外扰力矩的案例
- **关联链**：推进阀门 → 泄漏产生非指令微推力 → 姿态角加速度/扰动 → 核查压力与阀门指令 → 经审批的风险复核
- **不能推断**：当前姿态异常就必然是推进泄漏，或可自动执行推进补偿
- **资料**：[NASA 2026 在轨推进，第 4.2 节](https://www.nasa.gov/smallsat-institute/sst-soa/in-space_propulsion/)

## 例 4：通信——高增益天线指向偏差

- **候选故障**：`COM-RF-POINT`
- **链路机制**：窄波束定向天线依赖更精准的指向；指向偏差会降低链路裕量
- **跨系统替代假设**：星敏感器解算丢失 → 姿控指向误差 → 可能影响定向天线
- **核查思路**：联合对比姿态误差、天线指向几何、地面信号强度、载波锁定状态
- **资料**：[NASA 2026 Communications](https://www.nasa.gov/smallsat-institute/sst-soa/soa-communications/)、[NASA 2026 GNC](https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/)

## 例 5：有效载荷——下行数据积压

- **候选故障**：`PLD-DATA-STALL`
- **候选解释**：星载存储占用上升，或通信链路有效吞吐下降，两种机制都有可能造成待下行数据堆积
- **区分验证**：比对载荷生成速率、实际有效数传速率、存储写入/释放趋势和地面接收记录
- **资料**：[NASA 2026 Avionics](https://www.nasa.gov/smallsat-institute/sst-soa/small-spacecraft-avionics/)、[NASA 2026 Communications](https://www.nasa.gov/smallsat-institute/sst-soa/soa-communications/)

这些路径在 `graph_viewer.html` 和 `query_graph.py --fault <CODE>` 中均可查到。公开文献只提供一般工程机理与案例背景；若要用于正式项目，需要补充任务型号具体配置、日志证据及原文逐字定位。
