#!/usr/bin/env python3
"""Build a provenance-preserving, UNREVIEWED public-spacecraft reference KG.

The source registry below contains official public reference pages checked on
2026-10-08. Values in `summary` are original Chinese *paraphrases*, NOT quotes.
Inferences, suggested inspections and possible mitigations are not NASA/ECSS
instructions and must not be promoted to flight procedures.
"""
from __future__ import annotations
from collections import Counter
import csv
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCES = {
 'NASA-2026-ADCS': dict(title='NASA State-of-the-Art of Small Spacecraft Technology 2026 — 5.0 Guidance, Navigation, and Control',org='NASA S3VI',year=2026,url='https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/',kind='public_technical_survey'),
 'NASA-2026-EPS': dict(title='NASA State-of-the-Art of Small Spacecraft Technology 2026 — 3.0 Power',org='NASA S3VI',year=2026,url='https://www.nasa.gov/smallsat-institute/sst-soa/power-subsystems/',kind='public_technical_survey'),
 'NASA-2026-THERM': dict(title='NASA State-of-the-Art of Small Spacecraft Technology 2026 — 7.0 Thermal Control',org='NASA S3VI',year=2026,url='https://www.nasa.gov/smallsat-institute/sst-soa/thermal-control/',kind='public_technical_survey'),
 'NASA-2026-PROP': dict(title='NASA State-of-the-Art of Small Spacecraft Technology 2026 — 4.0 In-Space Propulsion',org='NASA S3VI',year=2026,url='https://www.nasa.gov/smallsat-institute/sst-soa/in-space_propulsion/',kind='public_technical_survey'),
 'NASA-2026-COMM': dict(title='NASA State-of-the-Art of Small Spacecraft Technology 2026 — 9.0 Communications',org='NASA S3VI',year=2026,url='https://www.nasa.gov/smallsat-institute/sst-soa/soa-communications/',kind='public_technical_survey'),
 'NASA-2026-AVIONICS': dict(title='NASA State-of-the-Art of Small Spacecraft Technology 2026 — 8.0 Small Spacecraft Avionics',org='NASA S3VI',year=2026,url='https://www.nasa.gov/smallsat-institute/sst-soa/small-spacecraft-avionics/',kind='public_technical_survey'),
 'NASA-2026-GROUND': dict(title='NASA State-of-the-Art of Small Spacecraft Technology 2026 — 11.0 Ground Data Systems and Mission Operations',org='NASA S3VI',year=2026,url='https://www.nasa.gov/smallsat-institute/sst-soa/ground-data-systems-and-mission-operations/',kind='public_technical_survey'),
 'NASA-CUBESAT101': dict(title='CubeSat 101: Basic Concepts and Processes for First-Time CubeSat Developers (2017)',org='NASA CubeSat Launch Initiative',year=2017,url='https://www3.nasa.gov/sites/default/files/atoms/files/nasa_csli_cubesat_101_508.pdf',kind='public_textbook'),
 'NASA-FDIR': dict(title='NASA Software Engineering Handbook 9.07 Fault Detection and Response',org='NASA',year=None,url='https://swehb.nasa.gov/spaces/SWEHBVD/pages/102695798/9.07%2BFault%2BDetection%2Band%2BResponse',kind='public_guidance'),
 'ECSS-AOCS': dict(title='ECSS-E-ST-60-30C Satellite attitude and orbit control system requirements (2013)',org='ECSS',year=2013,url='https://ecss.nl/standard/ecss-e-st-60-30c-satellite-attitude-and-orbit-control-system-aocs-requirements/',kind='public_standard_scope'),
 'ECSS-THERM': dict(title='ECSS-E-ST-31C Thermal control (2008)',org='ECSS',year=2008,url='https://ecss.nl/standard/ecss-e-st-31c-thermal-control/',kind='public_standard_scope'),
 'ECSS-PROP': dict(title='ECSS-E-ST-35-01C Liquid and electric propulsion for spacecraft (2008)',org='ECSS',year=2008,url='https://ecss.nl/standard/ecss-e-st-35-01c-liquid-and-electric-propulsion-for-spacecraft/',kind='public_standard_scope'),
 'ECSS-PUS': dict(title='ECSS-E-ST-70-41C Telemetry and telecommand packet utilization (2016)',org='ECSS',year=2016,url='https://ecss.nl/standard/ecss-e-st-70-41c-space-engineering-telemetry-and-telecommand-packet-utilization-15-april-2016/',kind='public_standard_scope'),
}

# Schema: fault, component, mechanism, symptom, verification, action, condition.
# Claim-level source_summary is a carefully bounded Chinese paraphrase of the
# source; action and diagnostic check nodes are editorial proposals (not quotes).
CASES = []
def case(domain, code, fault, component, mechanism, symptom, verification, action, condition, sources, locator, summary, *, documented_case=False, infer_mechanism=False, aliases=None):
 CASES.append(dict(domain=domain,code=code,fault=fault,component=component,mechanism=mechanism,
                   symptom=symptom,verification=verification,action=action,condition=condition,
                   sources=sources,locator=locator,summary=summary,documented_case=documented_case,
                   infer_mechanism=infer_mechanism,aliases=aliases or {}))

# ADCS / AOCS
case('ADCS','ADCS-WHEEL-SAT','反作用飞轮动量饱和','反作用飞轮','飞轮角动量存储达到设计极限','飞轮转速逼近限值并伴随单轴力矩能力下降','比对飞轮转速、角动量估算、力矩命令与姿态误差趋势','评估按型号程序执行飞轮动量卸载','存在经批准的外力矩执行机构和卸载程序',['NASA-2026-ADCS','ECSS-AOCS'],'NASA §5.2.2 Reaction Wheels','NASA说明飞轮转速达到最大值时会丧失进一步储存动量/提供相关力矩的能力，需要外力矩执行器卸载。',aliases={'反作用飞轮':['reaction wheel'],'飞轮角动量存储达到设计极限':['reaction wheel saturation']})
case('ADCS','ADCS-STR-LOSS','星敏感器姿态解算丢失','星敏感器','高角速度或强杂散光导致星场匹配失效','星敏感器解算有效标志间歇丢失','检查星敏状态、角速度、太阳夹角与视场遮挡','评估启用经验证的惯导递推估计通路','IMU和状态估计算法可用且处于批准模式',['NASA-2026-ADCS'],'NASA §5.2.6 Star Trackers','NASA说明星敏感器的星场识别会受到卫星角速度、外部光源和眩光影响；可用IMU传播的滤波器维持解算。')
case('ADCS','ADCS-WHEEL-JITTER','飞轮扰动导致精密指向退化','反作用飞轮','转子静动不平衡引发微振动及扰动力矩','姿态误差或载荷视轴抖动随轮速变化','联合比对轮速谐波、姿态误差和载荷抖动频谱','提出飞轮工作区间及结构耦合复核建议','需经过姿控与载荷联合分析及任务允许的控制策略',['NASA-2026-ADCS','ECSS-AOCS'],'NASA §5.2.2; ECSS §Scope','NASA讨论飞轮失衡产生扰动力矩，ECSS强调姿轨控要满足载荷扰动和指向约束。')
case('ADCS','ADCS-MAG-LIMIT','磁力矩器卸载能力不足','磁力矩器','当地磁场方向或强度约束可用外力矩','飞轮卸载期间剩余角动量下降缓慢','检查轨道地磁模型、磁力矩指令与卸载前后动量','核对卸载策略并评估任务许可的其他外力矩源','当地存在可利用磁场且替代执行器获工程批准',['NASA-2026-ADCS'],'NASA §5.2.3 Magnetic Torquers','NASA指出磁力矩器提供的控制力矩受局部磁场方向制约，并用于飞轮动量卸载。')
case('ADCS','ADCS-SAFE-MODE','姿轨控系统进入安全模式','姿轨控状态机','故障检测触发自主模式转换','姿控模式状态由任务模式切换至安全模式','核对FDIR事件码、转换条件、模式序列与时间戳','保持可通信的安全状态并按批准流程复核恢复条件','任务定义的安全模式已验证且地面授权恢复',['ECSS-AOCS','NASA-FDIR'],'ECSS §Scope; NASA §3.3 flight-system safing','ECSS把紧急安全姿态及恢复列为AOCS功能；NASA提出故障保护至少应保持安全且可地面指令的状态。')

# Electrical power
case('EPS','EPS-ECLIPSE','轨道阴影期光伏发电中断','太阳电池阵列','进入地影后缺乏入射太阳辐照','太阳阵输出电流显著下降','对照轨道食预报、光伏电流与母线负载时序','复核阴影期储能预算和批准的负载策略','储能系统健康且地影时长符合任务能量预算',['NASA-2026-EPS'],'NASA §3.2 Solar Cells','NASA明确说明太阳电池阵列在食期间不能发电，系统需考虑储能和功率预算。')
case('EPS','EPS-UNDERVOLT','主电源母线欠压','电源管理与配电单元','可用供电功率与当前负载需求失配','母线电压低于设计阈值并伴随欠压事件','交叉核查母线电压、负载电流、发电功率和电池荷电状态','建议依据任务级欠压保护与恢复规程复核负载安排','禁止越权执行负载切换或解除安全保护',['NASA-2026-EPS','NASA-FDIR'],'NASA §3.6 PMAD; Fault Detection and Response §3.3','NASA将PMAD定义为电源控制/分配核心；故障管理资料要求欠压场景有可确定状态与恢复策略。',infer_mechanism=True)
case('EPS','EPS-BATTERY-COLD','蓄电池低温影响运行','蓄电池组','阴影期散热与加热不足引起低温','电池温度接近或超出任务允许下界','比对电池温度、食段、加热回路和充放电记录','核验电池热控加热与运行约束','加热动作符合电池温度边界及任务热控程序',['NASA-2026-THERM','NASA-2026-EPS'],'NASA Thermal §7.3.1 Heaters','NASA举例多型CubeSat使用电池加热器维持食期间的电池温度；具体低温限值随电池而变。')
case('EPS','EPS-SOLAR-LOW','太阳电池阵列发电能力下降','太阳电池阵列','长期退化、污染或辐照影响有效发电能力','在相近照明条件下阵列输出功率逐步下降','对比太阳角、温度、功率、电压电流曲线及历史基线','建议复核阵列退化模型及任务功率裕量','需消除姿态照明和负载变化等混杂因素',['NASA-2026-EPS'],'NASA §3.2 Solar Cells','NASA指出太阳电池性能会因老化、辐射及污染随任务时间下降。')
case('EPS','EPS-PMAD-LOAD','供配电支路异常高负荷','电源管理与配电单元','支路功耗增大导致供配电裕量降低','支路电流上升并伴随母线电压扰动','按负载清单核对各支路电流、电源开关状态及事件日志','建议核验故障隔离和优先级负载管理策略','需满足本型号允许断电清单及地面授权',['NASA-2026-EPS','NASA-FDIR'],'NASA §3.6 PMAD; NASA §3.3 resources preservation','NASA概述PMAD的负载供电控制和分配职责；过载的具体判断方法属于需型号数据核实的工程推断。',infer_mechanism=True)

# Thermal
case('THERMAL','THM-OVERHEAT','设备温度超出上限','热控辐射器与导热通路','内部耗散与外部吸热超过散热能力','关键部件温度超过任务热限','联合检查载荷功耗、受照姿态、导热路径与热敏曲线','开展热平衡复核并提出经批准的降耗或散热方案','设备热限、允许姿态与负载策略均需型号验证',['NASA-2026-THERM','ECSS-THERM'],'NASA §7.1 Introduction; ECSS §Scope','NASA给出吸热、发热、储热、向外辐射的热平衡框架；ECSS要求热控全过程分析验证。')
case('THERMAL','THM-COLD','设备温度低于运行下限','热控加热器','散热与低环境热输入造成温度下降','设备温度逼近或低于运行允许范围','检查食段、加热通断、功耗与多个温度测点','提出保温和加热回路核验建议','加热功率及操作时序应依型号规程批准',['NASA-2026-THERM'],'NASA §7.1; §7.3.1 Heaters','NASA指出设备必须在存活和工作温度范围内，电阻加热器是常用的主动热控装置。')
case('THERMAL','THM-HEATER-FAULT','温控加热回路疑似异常','热控加热器','加热控制、供电或传感反馈链功能异常','加热请求与温度响应明显不一致','比对温控门限、加热电流和冗余温度测量','对加热器与供电链进行独立状态核查','无独立温度佐证时不得将故障归因于加热器',['NASA-2026-THERM'],'NASA §7.3.1 Heaters','NASA描述有源电加热维持温度的用途；控制故障及诊断方法为需核实的工程推断。',infer_mechanism=True)
case('THERMAL','THM-HEATPIPE','热传导路径有效性下降','热管与热接口','热管或接触导热性能下降','冷热端温差增大或局部温度出现异常梯度','比较热管冷热端温度、工作功率与热模型','检查热路径与热接口的设计/验证依据','在没有试验或仿真前不判定具体热管失效',['NASA-2026-THERM'],'NASA §7.2.3; §7.2.8 Heat Pipes','NASA概述接触导热和热管承担的传热职能，具体失效模式需型号级分析。',infer_mechanism=True)

# Propulsion
case('PROP','PROP-VALVE-LEAK','推进阀门泄漏疑似引发非指令力矩','推进器阀门','阀门泄漏产生持续非指令微推力','飞行器存在持续偏置角加速度或姿态扰动','核对压力变化、阀命令、喷嘴方向和姿态动量变化','开展推进泄漏风险复核与姿控补偿可行性分析','所有推进与补偿操作需专门飞行安全审批',['NASA-2026-PROP','ECSS-PROP'],'NASA §4.2 Chemical Propulsion — MarCO-B case','NASA报道MarCO-B推进系统存在阀门泄漏，叠加泄漏对航天器产生持续力矩。',documented_case=True)
case('PROP','PROP-VALVE-STUCK','推进阀门无法按指令开启','反作用控制喷气阀','阀门卡闭阻止对应推进器产生预期推力','同一阀门指令后缺乏预期动力学响应','对照阀门状态、命令、压力与姿态响应','评估经过授权的冗余推力配置','必须验证剩余执行器足以满足姿态及安全约束',['NASA-2026-PROP'],'NASA §4.2 Chemical Propulsion — BioSentinel case','NASA记录BioSentinel一只RCS阀门停留在关闭状态，团队用其余五只推进器进行动量卸载。',documented_case=True)
case('PROP','PROP-THRUST-LOW','推进执行能力低于预期','推进系统','推进工况与供给条件变化导致有效推力下降','指令机动与估计速度增量出现偏差','复核指令时长、推进工况、轨道估计与供给状态','进行推进能力预算与限制条件复核','不得从遥测推断直接执行额外点火',['NASA-2026-PROP','ECSS-PROP'],'NASA §4.1; ECSS §Scope','NASA报告阐述小型推进的推力与工况范围；具体下降机理仍需型号推进模型。',infer_mechanism=True)
case('PROP','PROP-EP-POWER','电推进工况受供电能力限制','电推进电源处理单元','输入电功率不足以支持计划的推力工况','电推进工况与功率指令不一致','检查电推进输入功率、母线负载和推力模式记录','联合电源与推进评估允许的功率工况','需满足EPS功率裕量与推进器厂家工作边界',['NASA-2026-PROP','NASA-2026-EPS'],'NASA §4.3 Electric Propulsion; Power §3.6','NASA将电推进与电功率供给联系起来；具体受限行为随推进方案而异。',infer_mechanism=True)

# Communications / ground
case('COMM','COM-RF-POINT','高增益射频天线指向误差','高增益定向天线','窄波束指向偏差降低接收端有效增益','可见弧段下行信噪比和接收帧率降低','对照天线指向几何、姿控误差和地面信号强度','复核天线定标与合法允许的指向策略','需得到姿控及通信联合审批',['NASA-2026-COMM','ECSS-AOCS'],'NASA §9.1; ECSS AOCS §Scope','NASA指出高增益天线的定向能力提高链路性能，但窄波束要求更准确指向。')
case('COMM','COM-RF-MARGIN','射频链路余量不足','射频通信链路','传播衰减与传输速率/发射功率组合导致链路裕量下降','地面收包率下降或误码事件增多','重算链路预算并检查仰角、天气、调制编码和射频功率','评估可授权的调制速率或地面站调整','所有频率功率和数据率调整须遵守链路与频率许可',['NASA-2026-COMM'],'NASA §9.1 Introduction','NASA说明链路预算需考虑雨衰/大气衰减、编码、天线与功率，并在数据率、质量及质量功耗间权衡。')
case('COMM','COM-OPT-ACQ','星地光通信链路捕获失败','激光通信终端','粗精指向误差使窄光束无法保持捕获','光链路捕获锁定标志缺失','检查粗指向、精指向、光轴校准和终端几何','评估经批准的捕获与指向校核流程','避免未授权改变激光发射状态或危险操作',['NASA-2026-COMM'],'NASA §9.3.1 System Architecture','NASA指出星地光通信对指向精度要求高，通常需要粗精两级指向及与ADCS协同。')
case('COMM','COM-GROUND-FAULT','地面站链路配置异常','地面站射频设备','地面站天线、馈线、频率或工作模式不匹配','预报过境期间未能正确解码遥测','核对地面天线指向、馈线阻抗、接收频率与解码模式','先开展地面站端非侵入性检查与设备联测','不能仅凭无下行判定星上发射机故障',['NASA-CUBESAT101'],'CubeSat 101 §2.10 p.22 (PDF page index 29)','NASA CubeSat 101建议系统排查地面天线指向、线缆阻抗、接收频率与模式，避免遗漏地面站故障。')
case('COMM','COM-RF-CARRIER','故障安全模式下遥测下行丢失','星载射频发射通路','故障保护配置未维持可用的基本下行链路','进入安全模式后预测射频载波未检出','对照安全模式配置、地面过境机会与下行载波日志','复核安全模式应保留的地面可指令与下行能力','不得在无通信保障时未经批准退出安全模式',['NASA-FDIR','ECSS-PUS'],'NASA §3.3 Flight system safing; ECSS §Scope','NASA故障保护指南强调保持安全、可地面指令及可预测的射频信标/下行；ECSS明确遥测遥控数据服务作用。',infer_mechanism=True)

# C&DH / Avionics
case('CDH','CDH-MEM-SEU','星载存储器单粒子翻转','星载存储器','单个高能粒子沉积电荷改变存储逻辑态','存储校验或ECC纠错计数异常','比对ECC计数、事件日志与轨道辐照环境','建议核查已设计的纠错或存储刷新策略','必须确认处理器和存储器具备对应受控功能',['NASA-2026-AVIONICS'],'NASA §8.2.1 Radiation Mitigation and Tolerance Schemes','NASA区分瞬态SEE与累积辐照，并描述SEU可改变存储单元逻辑状态。')
case('CDH','CDH-SEL','器件单粒子闩锁造成异常电流','星载电子器件','单粒子事件触发寄生结构形成大电流通路','局部供电支路电流快速上升','检查电流保护事件、供电历史和辐射环境','评估已设计的电源隔离与保护状态','需排除其他过流原因，且不得越权复电',['NASA-2026-AVIONICS'],'NASA §8.2.1 Radiation Mitigation and Tolerance Schemes','NASA指出单粒子闩锁可在半导体中形成寄生导电结构并造成高电流状态。')
case('CDH','CDH-WATCHDOG','星载计算机看门狗复位事件','星载计算机','软件任务卡滞或健康监测超时触发看门狗','看门狗重启计数上升且处理任务中断','对齐重启原因寄存器、软件事件日志和供电异常','审查故障隔离、看门狗及选择性复位策略','复位相关动作须经过状态与任务影响评估',['NASA-2026-AVIONICS'],'NASA §8.1; §8.2.1','NASA报告将看门狗、选择性电源循环及软件故障隔离列为容错手段；实际触发原因需进一步验证。',infer_mechanism=True)
case('CDH','CDH-STORAGE-FULL','星载数据缓存容量不足','星载数据存储模块','输入数据产生率超过存储释放速率','缓存占用率持续上升并接近上限','比对载荷生成量、压缩率、已下传字节和剩余容量','复核受许可的数据优先级与存储保持策略','不得未经授权清除未回传载荷数据',['NASA-2026-AVIONICS','NASA-2026-COMM'],'NASA Avionics §8.2.4; Communications §9.1','NASA介绍星载非易失存储用于临时载荷缓存，链路设计需权衡下传数据率；容量演变为工程推断。',infer_mechanism=True)

# Payload (generic not mission-specific)
case('PAYLOAD','PLD-DATA-STALL','有效载荷数据下传滞后','载荷数据处理链','载荷数据率与数传窗口能力不匹配','载荷数据待下行队列持续增长','核对载荷采集计划、机上存储和地面下行窗口','提交载荷计划与数据率约束复核建议','不得擅自更改科学载荷任务计划',['NASA-2026-AVIONICS','NASA-2026-COMM'],'NASA Avionics §8.1; Communications §9.1','NASA介绍载荷数据由星载计算和数传链处理；持续堆积需要结合任务数据率和地面窗口判断。',infer_mechanism=True)
case('PAYLOAD','PLD-TEMP-LIMIT','有效载荷运行温度限制违背','有效载荷热控接口','载荷耗散/环境热输入与热控能力不匹配','载荷关键温度测点超出批准的运行范围','检查载荷开机时间、功耗、热控测点与姿态','建议暂停高风险操作并开展热控联合复核','需查本型号载荷存活和工作温度边界',['NASA-2026-THERM','ECSS-THERM'],'NASA Thermal §7.1; ECSS §Scope','NASA说明所有航天器部件必须在各自的存活与工作温度范围内，载荷也受热环境限制。',infer_mechanism=True)
case('PAYLOAD','PLD-POINT-JITTER','精密载荷观测指向品质下降','有效载荷指向链','姿控微扰与载荷视轴稳定需求不相容','有效载荷观测成像模糊或稳定度指标下降','相关分析飞轮频率、星敏解算及载荷视轴误差','提交跨姿控和载荷的扰动耦合分析','必须依据该载荷实际成像架构和姿控指标',['NASA-2026-ADCS','ECSS-AOCS'],'NASA ADCS §5.2.2; ECSS AOCS §Scope','NASA描述飞轮不平衡引起姿态扰动；ECSS指出AOCS需满足载荷扰动和指向要求。',infer_mechanism=True)

# Candidate cross-subsystem hypotheses. The intent is to show *alternative*
# knowledge links, never to turn a cross-subsystem correlation into proven cause.
CROSS_LINKS = [
 ('COM-RF-POINT','ADCS-STR-LOSS','星敏失锁可成为姿控指向误差的候选上游因素',['NASA-2026-ADCS','NASA-2026-COMM']),
 ('PLD-POINT-JITTER','ADCS-WHEEL-JITTER','飞轮扰动可能耦合到精密载荷视轴',['NASA-2026-ADCS','ECSS-AOCS']),
 ('PLD-POINT-JITTER','ADCS-STR-LOSS','星敏姿态解算丢失可能影响载荷指向',['NASA-2026-ADCS','ECSS-AOCS']),
 ('EPS-UNDERVOLT','EPS-SOLAR-LOW','光伏出力下降是母线供电不足的候选因素之一',['NASA-2026-EPS']),
 ('EPS-UNDERVOLT','EPS-ECLIPSE','日食阶段无光伏发电是母线电能收支的关键工况',['NASA-2026-EPS']),
 ('CDH-WATCHDOG','EPS-UNDERVOLT','供电瞬态问题可作为星载计算机复位的替代排查假设',['NASA-FDIR','NASA-2026-EPS']),
 ('COM-RF-CARRIER','EPS-UNDERVOLT','总线欠压可作为星载下行发射通路失效的待查因素',['NASA-FDIR','NASA-2026-EPS']),
 ('COM-RF-CARRIER','CDH-WATCHDOG','星载计算机复位或任务中断可作为下行中断的待查因素',['NASA-2026-AVIONICS','NASA-2026-COMM']),
 ('PLD-DATA-STALL','CDH-STORAGE-FULL','星载缓存释放受限可能使载荷待下行数据持续累积',['NASA-2026-AVIONICS','NASA-2026-COMM']),
 ('PLD-DATA-STALL','COM-RF-MARGIN','链路余量不足使有效数据率降低可能造成载荷缓存积压',['NASA-2026-COMM']),
 ('PROP-EP-POWER','EPS-UNDERVOLT','总线欠压可作为电推进输入功率不足的待查因素',['NASA-2026-EPS','NASA-2026-PROP']),
 ('PLD-TEMP-LIMIT','THM-OVERHEAT','整星或局部散热能力不足可影响载荷温度',['NASA-2026-THERM']),
 ('EPS-BATTERY-COLD','THM-HEATER-FAULT','加热回路异常可能导致电池在食期的热稳定不足',['NASA-2026-THERM']),
 ('ADCS-SAFE-MODE','EPS-UNDERVOLT','欠压可触发故障保护安全响应，具体触发条件取决于型号',['NASA-FDIR']),
 ('ADCS-SAFE-MODE','ADCS-STR-LOSS','姿态估计异常可触发AOCS安全策略，是否触发取决于型号配置',['ECSS-AOCS']),
]

ONTOLOGY = {
 'entity_types':['FAULT_MODE','COMPONENT','FAILURE_MECHANISM','SYMPTOM','VERIFICATION','ACTION','CONDITION'],
 'allowed_triples':[
  ['COMPONENT','SUSCEPTIBLE_TO','FAILURE_MECHANISM'],
  ['FAULT_MODE','ASSOCIATED_WITH','FAILURE_MECHANISM'],
  ['FAILURE_MECHANISM','MANIFESTS_AS','SYMPTOM'],
  ['FAULT_MODE','INDICATED_BY','SYMPTOM'],
  ['FAULT_MODE','CHECKED_BY','VERIFICATION'],
  ['FAULT_MODE','MITIGATED_BY','ACTION'],
  ['ACTION','REQUIRES','CONDITION'],
 ],
 'definitions':{'SUSCEPTIBLE_TO':'部件相关的可能失效机理，并不表示实际发生',
                'ASSOCIATED_WITH':'候选故障与可能机理关联，不代表已确诊',
                'MANIFESTS_AS':'机理可能表现的可观测征兆',
                'INDICATED_BY':'故障候选与异常征兆的线索关联',
                'CHECKED_BY':'提出可供人工审核的检查项，非强制处置',
                'MITIGATED_BY':'仅为辅助处置方案，不具备控制授权',
                'REQUIRES':'采取任何建议措施前需要确认的条件'},
}


def digest_id(prefix, value): return prefix + '-' + sha256(value.encode('utf8')).hexdigest()[:16].upper()

def build():
 nodes_by_id={}; edges=[]; clusters=[]
 pairs={('COMPONENT','SUSCEPTIBLE_TO','FAILURE_MECHANISM'),('FAULT_MODE','ASSOCIATED_WITH','FAILURE_MECHANISM'),('FAILURE_MECHANISM','MANIFESTS_AS','SYMPTOM'),('FAULT_MODE','INDICATED_BY','SYMPTOM'),('FAULT_MODE','CHECKED_BY','VERIFICATION'),('FAULT_MODE','MITIGATED_BY','ACTION'),('ACTION','REQUIRES','CONDITION')}
 for c in CASES:
  ns={}
  for key,kind in [('fault','FAULT_MODE'),('component','COMPONENT'),('mechanism','FAILURE_MECHANISM'),('symptom','SYMPTOM'),('verification','VERIFICATION'),('action','ACTION'),('condition','CONDITION')]:
   name=c[key]; nid=digest_id('KGN',kind+'::'+name)
   ns[key]=nid
   if nid not in nodes_by_id:
    nodes_by_id[nid]=dict(id=nid,name=name,entity_type=kind,domains=[c['domain']],aliases=c['aliases'].get(name,[]),status='PUBLIC_REFERENCE_SEED_UNREVIEWED')
   elif c['domain'] not in nodes_by_id[nid]['domains']: nodes_by_id[nid]['domains'].append(c['domain'])
  pairings=[('component','SUSCEPTIBLE_TO','mechanism'),('fault','ASSOCIATED_WITH','mechanism'),('mechanism','MANIFESTS_AS','symptom'),('fault','INDICATED_BY','symptom'),('fault','CHECKED_BY','verification'),('fault','MITIGATED_BY','action'),('action','REQUIRES','condition')]
  edge_ids=[]
  for idx,(src,pred,tgt) in enumerate(pairings):
   a,b=nodes_by_id[ns[src]],nodes_by_id[ns[tgt]]
   assert (a['entity_type'],pred,b['entity_type']) in pairs
   relation_kind=('PROPOSED_CHECK' if pred=='CHECKED_BY' else 'UNAUTHORIZED_RESPONSE_SUGGESTION' if pred in {'MITIGATED_BY','REQUIRES'} else
                  'ENGINEERING_INFERENCE' if c['infer_mechanism'] or pred in {'INDICATED_BY','MANIFESTS_AS'} else
                  'DOCUMENTED_CASE_DERIVED' if c['documented_case'] else 'PUBLIC_SOURCE_PARAPHRASE')
   # This is not a score, and no model confidence is fabricated.
   eid=digest_id('KGE',c['code']+'::'+pred+'::'+a['id']+'::'+b['id'])
   item=dict(id=eid,source=a['id'],target=b['id'],relation_type=pred,
       domain=c['domain'],fault_code=c['code'],interpretation=relation_kind,
       source_ids=c['sources'],source_locator=c['locator'],source_summary_zh=c['summary'],
       evidence_form='CHINESE_PARAPHRASE_NOT_VERBATIM',
       equipment_scope='PUBLIC_GENERAL_SPACECRAFT',
       review_status='PUBLIC_REFERENCE_SEED_UNREVIEWED',
       is_confirmed_root_cause=False,
       automatic_control_authorized=False,
       approval_required=(pred in {'MITIGATED_BY','REQUIRES'}))
   edges.append(item);edge_ids.append(eid)
  clusters.append(dict(code=c['code'],domain=c['domain'],title=c['fault'],sources=c['sources'],source_locator=c['locator'],summary=c['summary'],documented_case=c['documented_case'],
      source_ids=ns,edge_ids=edge_ids))
 # Alternative causal links share the SAME nodes across fault families and
 # supply a meaningful (but unapproved) multi-hop graph rather than 30 islands.
 specs={c['code']:c for c in CASES}
 by_code={cl['code']:cl for cl in clusters}
 for fault_code,mechanism_case,summary,citation_ids in CROSS_LINKS:
  assert fault_code in by_code and mechanism_case in by_code
  a=by_code[fault_code]['source_ids']['fault']; b=by_code[mechanism_case]['source_ids']['mechanism']
  cc=specs[fault_code]
  eid=digest_id('KGE',fault_code+'::CROSS::'+mechanism_case)
  edges.append(dict(id=eid,source=a,target=b,relation_type='ASSOCIATED_WITH',
       domain=cc['domain'],fault_code=fault_code,
       interpretation='ENGINEERING_INFERENCE_CROSS_SUBSYSTEM',
       source_ids=citation_ids,source_locator='跨分系统关联：请分别核对所引资料相应章节',
       source_summary_zh=summary,evidence_form='CHINESE_PARAPHRASE_NOT_VERBATIM',
       equipment_scope='PUBLIC_GENERAL_SPACECRAFT',
       review_status='PUBLIC_REFERENCE_SEED_UNREVIEWED',
       is_confirmed_root_cause=False,automatic_control_authorized=False,approval_required=False))
  by_code[fault_code]['edge_ids'].append(eid)
 nodes=sorted(nodes_by_id.values(),key=lambda x:(x['entity_type'],x['name'],x['id']))
 edges.sort(key=lambda x:(x['domain'],x['fault_code'],x['relation_type'],x['id']))
 return nodes,edges,clusters

def dumpjson(path, obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
def csvw(path, cols, rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
  for x in rows: w.writerow({k:x.get(k,'') for k in cols})

if __name__=='__main__':
 nodes,edges,clusters=build()
 sources=[dict(id=id,**s,verified_via_public_page=True,accessed_on='2026-10-08',note='文献原文未随包再分发；标注的摘要均非逐字引文') for id,s in SOURCES.items()]
 graph=dict(schema_version='astrograph-public-spacecraft-seed.v1',built_on='2026-10-08',
    graph_kind='UNREVIEWED_PUBLIC_REFERENCE_SEED',
    scope='PUBLIC_GENERAL_SPACECRAFT',is_flight_qualified=False,
    note='此图是根据公开教材/技术文献编制的候选通用知识，不能作为已审核的型号知识或实际故障确诊；修改、操作或发布前需要型号级专家批准。',
    ontology=ONTOLOGY,entities=nodes,relations=edges,clusters=clusters,sources=sources)
 dumpjson(HERE/'graph.json',graph)
 dumpjson(HERE/'sources.json',sources)
 dumpjson(HERE/'ontology.json',ONTOLOGY)
 csvw(HERE/'nodes.csv',['id','name','entity_type','domains','aliases','status'],
      [{**n,'domains':';'.join(n['domains']),'aliases':';'.join(n['aliases'])} for n in nodes])
 csvw(HERE/'edges.csv',['id','source','target','relation_type','domain','fault_code','interpretation','source_ids','source_locator','source_summary_zh','equipment_scope','review_status','approval_required'],
      [{**e,'source_ids':';'.join(e['source_ids'])} for e in edges])
 csvw(HERE/'expert_review_template.csv',['relation_id','fault_code','relation_type','source_name','target_name','references','reference_locator','proposed_meaning','suggested_review_decision','independent_truth_source','applicable_model','reviewer','notes'],
      [dict(relation_id=e['id'],fault_code=e['fault_code'],relation_type=e['relation_type'],source_name=next(n['name'] for n in nodes if n['id']==e['source']),
            target_name=next(n['name'] for n in nodes if n['id']==e['target']),references=';'.join(e['source_ids']),reference_locator=e['source_locator'],proposed_meaning=e['source_summary_zh']) for e in edges])
 summary=dict(schema_version=graph['schema_version'],node_count=len(nodes),edge_count=len(edges),fault_case_count=len(clusters),source_count=len(sources),used_source_count=len(set(k for c in clusters for k in c['sources'])),
              domains=dict(Counter(c['domain'] for c in clusters)),entity_types=dict(Counter(n['entity_type'] for n in nodes)),relation_types=dict(Counter(e['relation_type'] for e in edges)),interpretation_kinds=dict(Counter(e['interpretation'] for e in edges)),all_unreviewed=all(e['review_status']=='PUBLIC_REFERENCE_SEED_UNREVIEWED' for e in edges))
 dumpjson(HERE/'build_summary.json',summary)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
