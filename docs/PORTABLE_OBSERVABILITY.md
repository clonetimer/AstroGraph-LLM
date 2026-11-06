# AstroGraph 外部诊断分析：执行记录与源码对齐（P0/P1 第一批）

日期：2026-11-04。开发版：`2.8.0-dev.1`。本次交付是完整仓库源码增量，不是新一版 Win10/Win7 安装包。

## 用户可见变化

继续使用原有“外部诊断结果 → LLM 分析 → 报告与追问”流程。在结果页新增“执行记录、材料与报告版本”区域。完整源码的 Mission 和 Legacy 平台以及便携工作台，都使用同一份 `src/astrograph/resources/execution_view.js` 和 CSS。

- 每次分析、重试、实际新增追问有独立运行编号；重放相同追问请求编号不会新增付费调用或追问运行。
- 展示真实记录的材料校验、片段分析、片段复用、综合分析、结构/引用检查、报告写入等阶段。导入和解析发生在运行记录创建之前；本版没有把解析全过程伪造为阶段记录。
- HTTP 请求尝试逐次记录。传输响应成功不等于模型分析成功；格式修复重试、传输失败分开保留。
- 输入、输出 token 来自服务端返回；任何请求缺少用量时，汇总显示不完整，并列出已知部分。没有从字符数猜测实际 token。费用未配置计价，显示未估算。
- 本次材料清单来自任务原始来源登记：文件名、角色、分段数、字符数、定位和哈希。长文最终综合使用片段摘要，清单不声称每个字都被模型理解。
- 保存每次运行的报告 JSON/Markdown；追问保存对应对话快照。下载校验哈希，不覆盖之前的运行产物。现阶段提供版本下载，不提供任意两个版本的语义比较。
- 旧任务仍可打开，没有记录的历史调用不补造、不计成零费用。

## 实现边界

记录模块只依赖 Python 标准库。未复制 PilotDeck 代码，未引入其 Node/Agent OS 运行时，也未变更项目许可声明。借鉴的是任务级可见、可追溯和可恢复的组织方式，不引用其性能或降本结果作为本项目测试结论。

当前传输观察覆盖 `OpenAICompatibleStructuredBackend` 的每次 `request_json` 调用尝试；不能证明请求必然到达供应商，也不能重建未接入此传输器的后端调用。用户问题、材料正文、模型正文、API Key、请求头、服务地址及原始供应商异常不写入执行记录。模型名、文件名和时间仍可能属于业务信息，应按本地任务数据保护。版本报告本身仍含诊断材料/模型输出，不是脱敏文件。

每次请求前先保存开始记录，结束后保存响应状态与服务端用量。记录写盘失败是终止错误，不作为格式错误继续付费重试。原子替换保证进程级写入一致性；不声称覆盖所有存储设备断电故障。

启动便携工作台、恢复备份时，遗留 RUNNING 运行转为 INTERRUPTED，不自动重放模型请求。完整平台的多进程崩溃接管与统一队列属于后续工作；本版不把正在其他工作进程执行的记录擅自判为中断。

## 数据目录与接口

```text
<task-root>/execution/runs/<run-id>/run.json
<task-root>/execution/runs/<run-id>/artifacts/report.json
<task-root>/execution/runs/<run-id>/artifacts/report.md
<task-root>/execution/runs/<run-id>/artifacts/conversation.json
```

原 `output/report.json` 等接口保持兼容，版本副本存放在独立目录。记录与备份一起迁移；还原后会过滤未知字段并标识不可读记录。SHA-256 是完整性检查，不是签名或对本机管理员篡改的防护。

| 部署入口 | 记录 API | 版本产物 API |
|---|---|---|
| 完整源码平台 | `GET /v1/platform/jobs/{job_id}/execution` | `GET /v1/platform/jobs/{job_id}/runs/{run_id}/{artifact_name}` |
| 便携服务 | `GET /api/jobs/{job_id}/execution` | `GET /api/jobs/{job_id}/runs/{run_id}/{artifact_name}` |

产物名只允许 report.json、report.md、conversation.json，未完成或哈希不一致的产物不提供下载。完整平台沿用现有 viewer/bearer/会话权限，不新增跨账号租户隔离承诺；便携服务仍只绑定回环地址并使用原有来源检查。

## 同源构建

`portable/build.py` 的 `stage_app`、`stage_source`、`stage_bootstrap` 是当前共用装配入口。

- `stage_app` 只拷贝所需模块；原完整仓库的包初始化导出保持不变。
- 共用组件由 `src/astrograph/resources` 装配到便携 assets，不在两处维护副本。
- `build.py` 使用 Win10 运行时/依赖配置；`build_win7.py` 使用 Win7 Python3.8 锁定材料和 `profiles/win7` 启动脚本。两者复用相同业务源码和前端。
- `stage_source` 导出最小便携源码，不把整个诊断/训练研究目录放进便携包；不复制用户 data、密钥或开发环境。
- 构建版本从应用 VERSION 读取。Win7 环境检测、运行时和 wheel 不能混用 Win10 版本。
- `refresh_v27.py` / `refresh_v271.py` 等保留为历史版本重建参考，不是当前开发版发布入口。不要用历史刷新脚本给当前源码套用旧版本标签。

本轮没有重新下载/升级 PDF 依赖，也没有重新发布原便携包。Win7 与 Win10 的原生执行验收和真实模型质量评价仍需独立进行。

## 在开发机体验

完整平台可沿用仓库 RUN_GUIDE.md 中的环境与启动方式。体验便携源码入口时，在已经装好该 Python 对应依赖的开发环境、仓库根目录执行：

```shell
python -c "from pathlib import Path; from portable.build import stage_app; stage_app(Path('.'), Path('.local/observability-app'))"
python .local/observability-app/workbench.py --data-dir .local/observability-data --open-browser
```

`.local/observability-app` 必须为空或不存在；该步骤是源码装配，不下载运行时或依赖。没有 Python 的最终用户应继续使用已经交付的便携 ZIP，不能把源码 ZIP 当成含运行环境的安装包。不要覆盖使用中的旧 data 目录。

## 后续范围

P2 工作区/经复核上下文，P3 多模型路由/任务队列尚未实施。价格配置、完整平台的崩溃接管、报告版本差异比较、超大记录分页也尚未实施。当前分析链路不增加诊断计算、自动执行设备操作或默认跨任务记忆。
