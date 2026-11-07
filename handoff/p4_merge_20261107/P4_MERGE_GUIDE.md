# 第四环节与可视化交互：人工合并说明

## 1. 合并范围

`source/` 提供完整当前仓库，便于查依赖。请以目标整体项目的稳定版为基底，逐项比较契约后再接入。不要整体覆盖目标项目的 `release.json`、`pyproject.toml`、治理清单、已有模型、数据集和上游诊断逻辑。

| 接入点 | 主要文件 | 合并时核对 |
|---|---|---|
| 平台 API 与账户 | `src/astrograph/api.py`、`integrations/platform_accounts.py`、`platform_state.py`、`platform_secrets.py` | `/platform`、`/v1/platform/*` 路由、角色权限、模型配置与任务状态的目标版契约 |
| 上传、预检、报告编排 | `integrations/upload_report.py`、`result_log_review.py`、`platform_worker.py`、`http_worker.py` | 上游诊断快照不可由任务文本或文献改写；进程退出应有失败状态；报告与模型状态一致 |
| 大模型解释 | `integrations/platform_knowledge.py`、`llm/openai_compatible.py`、`llm/task_prompts.py` | 保留候选与确诊的区别；结构化响应、引用 ID 与数值校验；主机校验失败至多一次纠错重试 |
| 文献与知识 | `integrations/material_*.py`、`retrieval/`、`oag/`、`resources/*.json` | 上传资料检索、待审图谱候选、已审图谱与历史案例的来源边界；不要把检索命中当本次根因证明 |
| 浏览器界面 | `resources/mission_console.html`、`.css`、`.js`，以及首页与平台资源文件 | `api.py` 的静态资源路由、`pyproject.toml` 的 package-data、页面入口和模型引用弹窗必须一起核对 |
| 回归样例 | `examples/platform_acceptance_cases_v1/`、`tests/test_*platform*.py`、`tests/test_result_log_review.py`、`tests/ui/test_p4_*.py` | `operations.log` 不可漏打包；合成案例仅证明流程和契约，不是诊断质量金标准 |

上表是合并导航，不代表列出的目录可无差别覆盖目标版。特别是 `api.py` 和 `upload_report.py` 承载多环节接口，宜做逐块差异审阅。当前包内的完整 `source/` 可用于追踪导入和资源依赖；实际人工移植以目标稳定版的接口为准。

逐页面和逐能力的核对表见 `P4_COMPLETE_FUNCTION_INVENTORY.md`；`PACKAGE_COVERAGE.json` 对关键文件给出包内路径与哈希，构建时缺文件即失败。

## 2. 建议执行顺序

1. 在目标整体项目创建独立合并分支，记录目标版版本号和基线测试结果。
2. 先对齐诊断快照、任务 ID、角色与来源 ID 的数据契约，再接入大模型解释和知识增强。保持原上游 BiRRF/融合结论的归属与审批边界。
3. 接入报告 API 和三个 `mission_console` 资源；检查 `source/pyproject.toml` 中资源打包配置，确认安装态页面加载的是合并后的文件。
4. 用目标版已有账户和新建隔离数据目录分别复验：建号、登录、真实文件上传、预检、生成、报告、引用原文、专家复核。核对同一任务编号贯穿页面与报告。
5. 对规则引擎和至少一个已配置模型分别跑流程；模型失败时必须明确显示失败及确定性回退来源，不得把回退文字标为模型分析。
6. 再接入文献检索、已审图谱与历史案例；先用合成引用契约验证呈现，再用目标版真实来源验证语义。保持未审资料只作为参考。
7. 合并后生成**目标项目自己的**源码清单与哈希，并在目标 Windows 浏览器中保存新截图。包内截图只能证明此源码仓库在本机的行为。

## 3. 核心回归命令（在目标项目中按其环境路径调整）

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m pytest tests/test_upload_report_platform.py tests/test_platform_knowledge_enrichment.py tests/test_openai_compatible_backend.py tests/test_result_log_review.py -q
python -m pytest tests/ui/test_p4_account_workflow_browser.py tests/ui/test_p4_citation_source_contract.py -q
```

账户浏览器测试会创建隔离测试数据并启动本机 HTTP 服务，需要 Chrome/Playwright 和可访问的本机回环接口。真实模型测试要先配置目标版所用模型服务；不要把本包的本机 `1234` 地址当目标设备的固定配置。

## 4. 交付边界

- 包含源码、测试、合成案例、少量真实浏览器截图和方法对比；不含账户数据库、运行任务状态、API 密钥、远程模型权重和原始私有遥测。
- 现有 `release.json` 与旧资格声明来自仓库原版本，**不为本次人工合并重新签署整体项目资质**。
- 独立金标准、双专家盲评、目标四环节平台的原始遥测端到端验收尚未完成，见测试说明。
