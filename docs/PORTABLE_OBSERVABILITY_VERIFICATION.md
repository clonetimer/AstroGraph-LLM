# P0/P1 源码改造验证记录

日期：2026-11-04。版本：2.8.0-dev.1。基础提交：187453966fd5e0a5ce488f81e27dc01376982a75。

本记录针对本次完整仓库源码修改。导入的 V2.6/V2.7/Win7 历史说明仅用于恢复来源，不是本次运行结果。没有重发或覆盖任何用户已下载的便携 ZIP。

## 本次实际执行

| 范围 | 命令/方法 | 结果 |
|---|---|---|
| 原便携基线 | 原仓库 unittest portable | 41 项通过 |
| 最终共享便携源码 | `python -B -m unittest discover -s portable -p 'test*.py' -v` | 94 项，93 通过、1 按源码目录条件跳过、0 失败；50.899 秒 |
| 完整平台相关回归 | `.github/workflows/foundation-smoke.yml` 的 39 个模块，另加 `tests/test_external_execution_observability.py` 和 `tests/ui/test_execution_views_integration.py`；`--astrograph-test-profile source-only` | 398 通过、3 跳过，129.20 秒 |
| 浏览器组件 | `browser_navigation.py`、`browser_smoke.py`、`browser_polish.py`、`browser_execution.py` | 9+3+10+4=26 组通过 |
| Python 3.8 新记录逻辑 | Linux Python 3.8.20，`test_execution_observability.py` | 15 项通过 |
| Python 3.8 实际装配应用 | Linux Python 3.8.20，`workbench.py --self-test` | PASS；中文路径、本机 HTTP 子进程、引用、追问持久化、Markdown 导出 |
| 兼容语法 | Linux Python 3.8 解析最终装配闭包 | 24 个 Python 文件通过 |
| 代码检查 | compileall、5 份修改/新增 JS 的 `node --check` | 通过 |
| 行尾检查 | `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check` | 通过，Windows BAT 使用 CRLF |

便携源码套件跳过项是针对已发行目录 MANIFEST.json 的检查；本轮未构建新二进制发行包，因此没有将其计为通过。相关平台的三项跳过沿用源代码测试配置，不等于全部仓库或真实模型验收。

浏览器测试运行真实生产 HTML/CSS/JS，但 API 是脚本化测试返回；后端独立回归使用真实本机 HTTP 模型桩。没有调用生产模型。Windows BAT/Windows PE、Win7/Win10 原生系统没有在本轮执行。Windows Server 上曾经验证的 Python 3.8 依赖属于历史证据，未冒充本次证据。

Linux Python 3.8.20 是从官方源码材料在本机编译，并搭配兼容 wheel 执行。该开发解释器缺少宿主 libffi 对应的 `_ctypes`，本次测试与应用自检不使用它；这不是随包 Windows 运行时，也没有将其宣称为完整 Win7 验收。

## 全仓库范围与未通过项

全仓库 source-only 命令也实际执行，不以专项回归代替：

```shell
PYTHONPATH=src python -B -m pytest -q --astrograph-test-profile source-only --maxfail=1
```

第一次在 154 通过、3 跳过后停在 `tests/implementation/acceptance/test_local_acceptance.py::test_source_structure_links_and_no_product_core_import_cycles`：缺少 `START_HERE.md`。基线 Git 对象也没有该文件。随后补齐根目录入口文档，该失败用例单独重跑通过（1 passed）。重新运行全仓库命令，在 303 通过、3 跳过后，于 supervisor 的长时执行流程处主动中止（KeyboardInterrupt，229.81 秒）；没有计为全量通过，后续未执行的项目仍未验证。该次中止不属于生产代码异常，也不隐去全量未完成事实。

不能把新增记录功能的专项通过称为完整研究仓库所有测试均已通过。完整仓库还包含不属于本次外部诊断文本链路的仿真、训练和历史验收模块。

## 先失败后修复与真实发现

- 恢复基线的导航、Python3.8 导入/路径与文档进程保护检查先失败，再补回已交付代码。
- 新记录测试先失败，补入 RunJournal、逐次 HTTP 观察、材料清单、报告版本、API 和三处前端集成。
- 观察到格式修复的第二次实际请求也消耗 token，改为逐次记录；未返回的用量保持未知，不按零统计。
- 恢复备份中的 RUNNING 记录、嵌套未知字段及损坏材料记录出现失败测试，修复后通过；旧报告不补造用量。
- 第一次浏览器集成发现生产 HTML 漏载公共记录脚本，新增装配检查并修复；没有把缺失组件导致的失败隐藏。
- 新磁盘故障回归发现响应收到后记录写入失败可能触发额外付费重试；ExecutionJournalError 现在是终止错误，新测试先失败后通过。失败的存储介质仍可能无法写下终止状态，不能承诺断电不丢失。
- 两个旧容量测试按 60,000 字符作全局断言，与已交付 180,000 字符输入容量不一致。测试改为读取当前上限或显式设置该测试的旧策略，仍验证超限拒绝；没有移除该保护。
- 原生产 CHECK.bat 引号修复保留。最初普通 diff --check 把 CRLF 当作尾空格；最终明确启用 cr-at-eol 检查，其余空白规则保留。

## 审查结论与明确取舍

作者自审，无独立审查工具；没有伪称独立 reviewer 通过。

P0 按文件合并，不覆盖完整包初始化和不相关业务模块；Win10/Win7 的源码装配共用 `stage_app/stage_source/stage_bootstrap`，独立依赖和启动配置保留。三处前端仅共用新增记录组件，不宣称整个完整平台已经与便携 UI 完全相同。

P1 已覆盖外部分析、长文片段、实际新增追问与逐次兼容 HTTP 调用；没有审计解析开始前的全部行为、其他未经此传输器的后端，也没有保证请求必然到达/被供应商计费。

日志只保存允许的元数据；版本报告可能含原文和模型输出，仍按诊断资料保护。哈希不等于签名，备份内容不等于已确认事实。

暂缓项：模型计价配置、报告版本语义差异、超大运行历史分页、完整平台多进程崩溃接管、P2 工作区/复核上下文、P3 路由/任务队列。大记录的全部读取及轮询时展开状态保持仍有可优化空间，本轮不新增复杂交互。

借鉴 PilotDeck 的任务级可追溯设计，独立实现，没有复制/翻译其源码，也没有改动完整项目许可。用户授权必要许可不被解释为静默更改项目许可或开放任意外部操作。

## 发布状态

这是源码增量；新增 CI 工作流只是已经写好的配置，不代表远程运行成功。提交、推送、补丁应用与源码 ZIP 完整性以本次交付清单的实测结果为准。


## 交付核验说明

最终生成完整源码 ZIP 和相对基线的 Git 增量补丁；在独立干净工作树上检查并应用补丁后，将结果 tree SHA 与提交 tree SHA 对照。具体 SHA、文件数及实际推送结果在交付目录的 `AstroGraph_P0P1_DeliveryVerification.json` 中记录，避免把未经执行的远程 CI 写成通过。
