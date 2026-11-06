# 外部诊断分析源码包增量 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. 用户已明确授权“判断…然后执行”，本次直接实施。

**Goal:** 把新包可独立使用的 LLM 分析、追问和兼容性增量整合到 PR #2。
**Architecture:** 保留外部文本入口；追加兼容报告字段和独立任务追问记录；选择性移植传输修复。三项并行实现使用互不重叠文件，根代理负责追问后端和整合。
**Tech Stack:** Python/Pydantic/FastAPI，原生 JavaScript，pytest/Node。
**Spec:** docs/superpowers/specs/2026-11-02-source-update-external-analysis.md

## Global Constraints

- 不执行诊断计算、图谱、R20/R21、训练或工具调用。
- 保留完整源文本、任务绑定、SHA-256、旧模式与旧报告兼容性。
- 原始报告不被追问改写；来源归属通过不等于语义正确。
- 所有公开错误为固定代码；模型和源文本以普通文本展示。
- 不整体覆盖新包共享文件；保持 _NoRedirect。
- API和schema数值限制以 spec 为准。

## Review Focus

1. 外部文本混合支持/反对信息：同一来源可出现于两种列表，拒绝各列表内部重复和外来 ID。
2. 模型输出看似完整但 finish_reason 截断/拒绝：探测和正式调用不误报通过。
3. 请求重发、并发、进程退出：不重复计费、不覆盖历史、不留下永久活动锁。
4. 账户/任务切换期间迟到响应：不显示其他会话结果。
5. 新字段包含活动标记、旧报告没有新字段：安全完整展示、兼容读取。

## Task 1: 传输兼容修复

**Files:** llm/token_budget.py, llm/openai_compatible.py, integrations/http_worker.py, integrations/platform_llm_probe.py；对应定向测试。
**Interfaces:** 保留公开调用接口，增加 ModelRefusalError 终止信号；外部 runner 继续固定公开错误。

- [x] 新行为测试先失败：GBK进程输出中文、Mapping/tensor、refusal/content_filter一次请求、截断probe。
- [x] 仅移植 spec 所列差异；probe拒绝也终止fallback。
- [x] 运行相关测试和已有 backend/probe 回归；记录证据。

## Task 2: 分析与报告契约

**Files:** integrations/external_diagnosis_analysis.py, integrations/external_diagnosis_report.py, tests/test_external_diagnosis_analysis.py。
**Interfaces:** spec 中的可选 hypothesis 字段与 review_actions；对原 API 和 output model 类名保持兼容。UI由Task3实现，task_prompts由根代理更新。

- [x] 增加外来/重复反证和动作引用、空反证、旧输出、Markdown转义失败测试。
- [x] 实现字段和所有引用检查、Markdown输出。
- [x] 运行定向测试，交付精确JSON字段给Task3。

## Task 3: 两套界面追问与分析字段

**Files:** resources/mission_console.{js,html,css}, resources/platform_report.js, resources/platform_ui.{html,css}；必要时小型共享模块；tests/ui 中新增/扩展测试。
**Interfaces:** 使用 spec GET/POST follow-up 与 state/turn；不要编辑Python。状态/权限与服务端一致。

- [x] 添加两套真实JS的渲染、提交、防重复、失败保留和任务切换失败测试。
- [x] 展示全部新假设字段与带引用核查；接入真实追问和历史恢复。
- [x] 验证旧报告、未配置、错误、HTML转义和迟到响应。

## Task 4: 追问后端与整合

**Files:** 新 integrations/external_diagnosis_conversation.py；api.py, upload_report.py, llm/task_prompts.py；新 tests/test_external_diagnosis_conversation.py；EXTERNAL_DIAGNOSIS_ANALYSIS.md 与CI。
**Interfaces:** 按 spec 定义 follow-up API、服务端状态、幂等性、锁、预算和输出模型；供Task3使用。

- [x] 测试先失败：成功引用、无模型、错误绑定/报告修改、伪造引用、幂等、并发、历史超限、权限、旧报告拒绝。
- [x] 实现独立会话文件和原报告完整性校验，调用任务所选backend；所有历史来自服务器。
- [x] 添加专用task prompt，更新原分析prompt，API请求和固定异常映射。
- [x] API集成使用本地真实HTTP模型服务，验证原报告字节不变。
- [x] 更新使用文档、上游采用记录及CI门禁。

## Task 5: 审查、验证和提交

- [x] 独立完整diff审查，并修复重要问题。
- [x] 运行新增测试和 foundation-smoke 全部门禁：391 passed, 3 skipped in 248.53s；39 个CI模块，日志保留。
- [ ] 核对差异范围与干净状态；通过 GitHub API 更新已授权分支（expected_sha 防覆盖并发修改）。
- [ ] 更新PR #2说明，核对远程tree与CI；交付链接、采用理由、测试结果与实际限制。


## 本地验收回执

2026-11-02：39个CI模块共391 passed、3 skipped、0 failed。跳过项是source-only下缺失的2份历史冻结证据夹具和1个Windows限定测试。独立审查发现并修复同subject不同bearer凭据的未发送草稿隔离问题，修复后无重要阻塞。生产JS语法与git diff --check通过。远程提交和CI最终回执以PR #2为准。
