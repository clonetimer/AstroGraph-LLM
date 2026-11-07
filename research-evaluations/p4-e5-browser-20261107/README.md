# P4-E5 原生 Chrome 浏览器联通复验

日期：2026-11-07。项目工作目录仅为 `D:/Project/AstoGraph-LLM/AstroGraph-LLM`。

## 范围与环境

- 使用 Windows 安装的 Chrome、真实 HTTP 服务、隔离的临时账户数据目录和仓库内 TC01 的三个真实文件（诊断结果、运行日志、参考资料）。
- 浏览器逐步完成：初始化管理员、创建领域专家、上传文件并自动预检、生成和打开报告、专家重新登录并提交复核意见。每次运行保留 `04` 至 `08` 的真实页面截图及 `workflow_receipt.json`。
- 规则引擎流程证据位于 `account_rules/`。本机 9B 模型经 `http://127.0.0.1:1234/v1` 的流程证据位于 `account_model_9b/`；该次任务 `JOB-749861183E1F8D13E1FC1753` 的报告、模型调用均为 `SUCCEEDED`，专家复核计数为 1。
- `screenshots/` 中 `01` 至 `03` 是默认平台模式下的首页、上传预检和分析报告截图。
- 页面文案与折叠区调整后，规则流程在 `account_rules_after_ui/` 通过（10.42 秒），9B 流程在 `account_model_9b_after_ui/` 通过（103.91 秒）；后者任务 `JOB-F89BAD251FD49ED6299E9BA2` 的模型状态与报告状态均为 `SUCCEEDED`，专家复核计数为 1。
- 输入区压缩后，`account_rules_compact_input/` 与 `account_model_9b_compact_input/` 再次通过；`06a_新建分析输入页.png` 显示初始输入和“开始分析”按钮在 1440 × 900 首屏内，处理步骤初始收起，自动预检时展开。9B 流程的任务号为 `JOB-C90A0E52B166E05E7EF32CC5`，报告与模型状态均为 `SUCCEEDED`。
- 增加 `07a_任务总览.png`：同一管理员会话中，首页从真实任务概览读取当前任务号、状态与计数，1440 × 900 时无纵向滚动。地球与轨道标明为演示场景，不能当作该任务的实时卫星遥测。
- `account_process_rules/` 与 `account_process_model_9b/` 采用账户模式默认的独立进程执行器，均完成相同浏览器流程。后者任务 `JOB-EBE2B2289ED6BD5D7589AEDF` 的报告及模型调用均为 `SUCCEEDED`，专家复核计数为 1。
- `account_process_model_9b_citations_final/` 在同一独立进程路径复验了模型引用。`07b_模型引用来源.png` 展示 `L1-0002` 对应 `operations.log` 第 2 行；`07c_上传资料引用.png` 展示 `reference.md` 的行范围和“不是当前卫星遥测”的边界提示。截图均来自原生 Chrome 的可见视口。
- `citation_sources_synthetic.json` 保存了仓库图谱与历史记忆后端测试产生的引用记录，明确标为**合成测试输出**。`citation_contract/` 的 Chrome 截图覆盖已审图谱关联、历史已审资料、历史案例和上游误判反例；这检验了弹窗的来源分类与字段呈现，不构成真实任务的图谱或历史案例命中。

## 可复验命令

在本仓库目录运行，解释器路径为 `D:/Project/AstoGraph-LLM/.venv/Scripts/python.exe`：

```powershell
$env:ASTROGRAPH_CHROMIUM_EXECUTABLE = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
$env:ASTROGRAPH_UI_EVIDENCE_DIR = Join-Path $PWD 'research-evaluations\p4-e5-browser-20261107\account_rules'
Remove-Item Env:ASTROGRAPH_P4_BROWSER_MODEL -ErrorAction SilentlyContinue
D:\Project\AstoGraph-LLM\.venv\Scripts\python.exe -m pytest tests/ui/test_p4_account_workflow_browser.py -q

$env:ASTROGRAPH_P4_BROWSER_MODEL = '1'
$env:ASTROGRAPH_UI_EVIDENCE_DIR = Join-Path $PWD 'research-evaluations\p4-e5-browser-20261107\account_model_9b_after_ui'
D:\Project\AstoGraph-LLM\.venv\Scripts\python.exe -m pytest tests/ui/test_p4_account_workflow_browser.py -q

$env:ASTROGRAPH_P4_BROWSER_EXECUTOR = 'process'
$env:ASTROGRAPH_UI_EVIDENCE_DIR = Join-Path $PWD 'research-evaluations\p4-e5-browser-20261107\account_process_model_9b'
D:\Project\AstoGraph-LLM\.venv\Scripts\python.exe -m pytest tests/ui/test_p4_account_workflow_browser.py -q

D:\Project\AstoGraph-LLM\.venv\Scripts\python.exe -m pytest tests/test_upload_report_platform.py tests/test_platform_knowledge_enrichment.py tests/test_openai_compatible_backend.py tests/test_result_log_review.py -q

D:\Project\AstoGraph-LLM\.venv\Scripts\python.exe -m pytest tests/test_material_knowledge_fusion.py::test_only_independently_confirmed_equipment_scoped_cases_are_retrievable tests/test_material_knowledge_fusion.py::test_historical_misclassification_is_counterexample_not_action_advice tests/test_material_knowledge_fusion.py::test_bound_csv_graph_is_readonly_and_detects_hash_drift tests/ui/test_p4_citation_source_contract.py -q
```

## 观察与边界

- 9B 模型报告的引用编号检查为 `PASS`，浏览器确实显示模型的观测、假设与建议；这只证明调用、结构与引用编号联通，**不等于引用内容的语义真实性或故障根因准确率得到专家验证**。
- 模型声明已按“任务观测、待核验原因、上传资料、图谱关联”等来源类型标注；可解析的引用编号可以打开报告中保留的来源片段。运行日志和上传资料已用真实 TC01 浏览器流程点击验证；图谱路径和历史案例仅由合成后端输出驱动的浏览器契约测试验证；公开文献仍待有实际命中的案例验证。
- 同一浏览器契约测试还把已保存的 TC01 模型报告交给现行前端渲染函数，验证界面显示 `reference.md` 等可读标签、保留完整引用 ID 供定位，不在按钮上平铺长哈希。
- 账户浏览器用例默认用线程执行器隔离界面流程；将 `ASTROGRAPH_P4_BROWSER_EXECUTOR=process` 可覆盖默认账户部署路径。两种执行器均已在本机 Chrome 中通过规则与 9B 模型流程。此前出现过子进程启动前的 OpenBLAS 内存分配失败，已增加“退出码 + worker.log”失败状态处理；这次通过不排除资源压力下的偶发失败，目标部署环境仍需复验。
- 这里的 TC01 是已有合成夹具，不是独立留出集。真实遥测采集、上游 BiRRF 诊断、型号级来源人工核验和双专家盲评不在本次浏览器通过范围内。
- 根据截图，报告标题和首屏重复的规则结论已作局部调整；`account_model_9b_after_ui/` 记录调整后的复验。事件中心零任务时的空白来自没有真实任务数据，首页实时任务展示仍需目标系统的数据源支持。
- 新建诊断页已删除重复的对话开场文案，压缩输入前区域，并以“处理中自动展开”的方式保留透明度。事件中心零任务时仍保留真实空状态；没有接入任务或遥测时不填充假实时信息。
- 四组 P4 后端测试在具备本机回环网络权限的环境中为 **58/58 通过**，包括新增的真实账户进程执行器用例。此前停顿用例的堆栈显示测试客户端阻塞在 Windows socketpair 建立；同一用例给予回环权限后 **1/1 通过**。因此先前中断不能解释为算法失败。
