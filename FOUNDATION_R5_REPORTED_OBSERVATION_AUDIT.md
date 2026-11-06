# AstroGraph R5 — 自然语言故障现象输入稳固性审查（开发候选）

基线：R4 Candidate（2026-10-14）。本轮**没有恢复端到端真实模型验收、也没有扩大正式故障候选**。不进行微调、不改变历史时序证据门禁。

## 背景：用户提供的故障现象样例

1. 输出总电流和主误差电压3次跌落
2. 轴温最高升至181°C，壳温最高升至150°C
3. 调整南太阳翼角度时出现堵转

**这只是一份尚未得到仪表读数与时标核验的人工报告，不是真值标签。** 用户稍后提供的实际项目输出应与此原文逐条核对；不能提前填入具体故障 ID。

## 工程审查发现

1. `platform_cli review` / 材料复核要求 `diagnosis_results`、`log_reports` 和 `references` 三种角色资料：不能把三行自由文本冒充完整诊断任务。单独给出预审命令，正式任务仍需相应资料。
2. 原 R4 `PlatformKnowledgeEnrichmentService.enrich()` 将 `task` 作为检索/LLM 用户请求，不为编号输入产生独立的来源收据。R5 把**明确编号的条目**保留成独立 ledger，加入服务返回对象和 LLM 载荷，标记 `USER_REPORTED_UNVERIFIED`、非投票、未对齐时标、无已验证物理根因。
3. R4 声明级原文核验仍严格阻止因果性自由生成结论出现在正式材料报告。R5 加入具体太阳翼案例的“虚假同步跌落”“已确诊机械卡死”“擅自下达复位指令”回归用例。
4. 不能把总电流和主误差电压的“3次跌落”解析为两路信号每路独立 3 次，更不能直接认为二者同步；没有原始电流/电压波形、采样率和时标。
5. 不能把温度峰值当作超过限值的确认：缺少型号、测点、允许限值、工况、时序、传感器校准与历史轨迹。
6. “调整南太阳翼角度时出现堵转”是报告的操作过程关联，不等于已确认机械卡死，更不能证明其引起前两条现象。

## 本轮实现范围

- `src/astrograph/retrieval/reported_observations.py`：有界提取编号原文，保存行号、原文 SHA256、数值字面量和来源状态；阻止自动诊断投票。
- `src/astrograph/integrations/platform_knowledge.py`：正式 LLM 载荷新增 `user_reported_observation_ledger`，返回对象也保留可追溯结构，不改变候选集合。
- `src/astrograph/llm/task_prompts.py`：将“非同步、无先后关系、不可直接下机械根因结论”的安全约束加入系统级任务提示。
- `scripts/review_reported_observations.py`：无需模型的文本预审 CLI；不会假装已经执行遥测诊断。
- `examples/r5_south_solar_wing_observations.txt`：用户风格的回归样例。
- `tests/test_foundation_r5_reported_observations.py`：编号识别、数值保真、来源 ID 不重复、截断披露、非法因果输出拦截、实际 LLM 载荷验证。

## 可复现命令

```bash
python scripts/review_reported_observations.py --file examples/r5_south_solar_wing_observations.txt --output observation_preview.json
PYTHONPATH=src python -m pytest -q tests/test_foundation_r5_reported_observations.py

# 后台执行、落盘日志、完成触发，Windows/Linux 均支持
python scripts/launch_logged_tests.py start --timeout 600 r5-repeat -- python -m pytest -q tests/test_foundation_r5_reported_observations.py tests/test_foundation_r4_competition_claim_guard.py tests/test_platform_knowledge_enrichment.py
python scripts/launch_logged_tests.py status r5-repeat
```

输出 `observation_preview.json` **不是诊断报告**，不含确认根因或自动处置许可。

## 下一步：拿到项目实际输出后如何验收

对每条“根因、故障位置、温度超限、同步性、动作建议”，检查：

- 是否引用到**当前任务对应**的已核实遥测、原始日志或专家审核的适用文献；
- 是否说明电流/电压波动与温度峰值/堵转事件的时间重合性，未获得时标则保留未知；
- 是否区分观测、假设、反证、待测数据和证据不足；
- 是否给出竞争解释及鉴别所需的电源母线电压、两路信号波形、驱动电流/扭矩、指令/反馈角度、机械限位状态、温度连续曲线和阈值，而不先行确定根因；
- 是否错误地把文献中出现的类似症状作为当前卫星的已证实根因。

**限制**：本轮不是 NLI 语义蕴含模型，没有做弱故障识别准确率或工业现场能力证明。只有在实际输出到达后，才能进行样例对照和独立的声明级金标准评估。
