# P4 模型来源校验与纠错回归

日期：2026-11-07。代码与全部新产物仅在 `D:/Project/AstoGraph-LLM/AstroGraph-LLM`。本机端点为 `http://127.0.0.1:1234/v1`，模型 `qwen/qwen3.5-9b`。输入是已使用过的合成 TC01 或基于它扩写的合成材料，不计入独立盲测或诊断准确率。

| 隔离目录 | 输入 | 原始模型响应 | 平台结果 | 定位 |
|---|---|---|---|---|
| `platform_tc01_grounding_capture/` | 原 TC01 | 候选编号为 `null`、`abstained=true` | 报告成功，模型解释拒收 | 把“根因未确诊”误当作“上游无候选” |
| `platform_tc01_candidate_contract/` | 原 TC01 | 保留 `ADCS_RW_FRICTION_INCREASE`，`abstained=false` | 报告和模型解释成功 | 在任务说明中明确区分候选和确诊后，此例通过 |
| `platform_tc01_clean_chinese/` | `clean_synthetic_tc01/` | `0.34 A` 引到 `D1-0006` | 模型解释拒收 | 正确数值搭配错误来源 ID |
| `platform_tc01_clean_retry/` | 同上 | 首轮与泛泛重试仍引用 `D1-0006` | 模型解释拒收 | 仅要求“核对 ID”不足以纠正本机 9B |
| `platform_tc01_clean_guided_retry/` | 同上 | 首轮 `D1-0006`；带匹配 ID 反馈后改为 `L1-0002`、`L1-0003` | `JOB-906D7A317ABA235F9969250C` 报告和模型解释成功 | 数值来源校验保持严格；一次有界重试可修正此例 |

最后一例内部 `generation.host_validation_attempts=2`，第一次失败记录指向 `0.34 A` 对应的可见证据 `L1-0002`，第二次响应 `finish_reason=stop`，引用校验 `PASS`。两次原始响应分别保存在该目录的 `raw_model_response_01.json`、`raw_model_response_02.json`。原始响应是服务器侧评测资料，不应直接展示为已审核工程结论。

`clean_synthetic_tc01/` 不是编码修复。原 TC01 文件与模型输出的 Unicode 码点核查表明中文内容有效；先前 PowerShell 工具输出乱码导致了错误判断。新合成材料增加了工况比较和操作边界文字，用来测试引用与可读性。名称沿用了建档时的 `clean`，不表示原文件损坏。

回归命令（在本仓库执行）：

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
D:\Project\AstoGraph-LLM\.venv\Scripts\python.exe -m pytest tests/test_upload_report_platform.py tests/test_platform_knowledge_enrichment.py tests/test_openai_compatible_backend.py tests/test_result_log_review.py -q
```

最终四组后端测试为 **63/63 通过**。中途曾因 C: 临时目录空间不足，以及把 pytest 任务根放在源码树内而失败；两种环境原因已定位，最后一次运行使用了源码树外的系统临时目录。本机 9B 调用证明的是具体合成案例的链路行为；时间量、构型错配、真实因果正确性与独立专家复核仍未完成。
