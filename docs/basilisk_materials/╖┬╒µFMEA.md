# 仿真机制 FMEA
由当前项目源码整理，非硬件故障报告；严重度、发生度、探测度与 RPN 均未编造。
|编号|故障效应|项目实现|候选可观测量|歧义/边界|源码|
|---|---|---|---|---|---|
|SIM-RW-JAM|adcs_rw_jamming|ReactionWheelCommandFaultManager:逆速度方向制动，低速容差内输出归零；有效力矩上限为零|指令/执行残差、轮速响应、姿态误差|稳态末段可能与力矩能力下降混淆；需保存早期制动过程|src/sat_sim/bsk_engine/project_native_modules.py:519-524|
|SIM-RW-TORQUE|adcs_rw_torque_authority_loss|缩放所选轮的指令和力矩上限|请求/执行力矩与轮速响应|低指令或无激励时不能仅凭小输出判故障|src/sat_sim/bsk_engine/project_native_modules.py:503-506|
|SIM-EPS-CAPACITY|eps_battery_capacity_loss|通过 manifest_start_s 门控，按剩余比例设置 storageCapacity|电池能量与 SOC 的成对变化|登记、机制生效、可观测显现时刻不同；受工况和能量边界影响|src/sat_sim/bsk_engine/project_native_modules.py:941-947,975|
|SIM-PA-L2|power_amplifier_fault|L2 参数效应声明：RF 功率乘0.35、PA效率乘0.55、热量加8W|rf_power_w、pa_efficiency、heat_generation_w|这是项目组件层声明，不能等同于整星简化降码率模型或硬件实测|src/components/transmitter/faults.py:81-88|
|SIM-TX-RATE|comm_data_transmitter_rate_loss|按剩余比例缩放下行码率请求|请求码率、执行码率、链路/供电/数据可用性|不能据降码率单独认定固放机理；与链路丢失、无数据/无供电区分|src/sat_sim/bsk_engine/project_native_modules.py:963-964,992-993|
|SIM-THERMAL-HEATER|thermal_heater_stuck_off|将选定加热器置为off，并按目标调整供电通道|加热请求、实际热输入、温度时间响应|热惯性意味着短窗口可能缺乏可分辨温升变化|src/sat_sim/bsk_engine/project_native_modules.py:959-960,978-980|
