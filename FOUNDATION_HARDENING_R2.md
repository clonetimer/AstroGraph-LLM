# AstroGraph R2 — 前/后增强基础加固与因果时间闭环

日期：2026-10-14。基线：上传的 `astrograph_source_foundation_hardening_candidate_20261014.zip` (`0.7.1.post49`)。

**交付状态：加固候选，非正式发布。** 此轮不加入新的模型、Agent、RAG 框架和知识图谱能力；不改变 LLM 的诊断决策权限。Git 仓库、CI、实机在线推理和原正式验收数据不可用，不能声称这些验收已通过。

## 1. 本轮进度

| 内容 | 改动 | 风险控制 |
|---|---|---|
| 正式文档证据准入 | `evaluate_document_claim_admissions` 增加独立 `observation_time_s` 入参；无时间、非有限时间、未来证据拒绝 | 不改变签名的 v1 policy Schema 或 `policy_sha256` |
| 文档 KG_FORWARD 排名 | `build_document_claim_forward_ranking` 二次检查时刻、episode、来源审核、权限及证据类型 | 即使拿到旧的 eligible admission，也无法在较早的窗口使用未来观察 |
| 在线影子融合 | 从 `evidence.bundle.episode.end_s` 传入窗口截止时刻；缺失时 fail-closed | 影子输出仍不能修改正式融合结论 |
| 预推理检索 | 从合法 `RUNTIME_DIAGNOSIS` 的 `DocumentAccessContext` 获取窗口时刻，失败时不可形式化准入 | 不依赖调用方仅提交某个文本标签 |
| Mission RAG 审核增强 | 在取得 OAG 已绑定窗口后再初始化审核证据增强器，加入 episode 检查 | 非诊断任务缺少窗口时仅展示一般检索上下文，不输出受审核时间证据立场 |
| 可审计日志 | 新增 `scripts/launch_logged_tests.py`；启动后立即返回，自动写 `.done.json` | 日志输出不占用交互会话；Windows/Linux 可用 |
| README | 删除与“没有默认账号”声明矛盾的演示账号密码；补充耗时任务启动/查询方法 | 避免误把仓库示例密码当成安全凭据 |

## 2. 时间门禁的操作语义

- `observation_time_s` 表示**本次推理可见的观测截止时刻**，不是文档发布时刻、当前计算机墙上时间或人工审批时间。应由经过验证的数据/episode 窗口提供。
- 任一**时序** claim 的 `valid_time_end_s` 或 `observable_cutoff_time_s` 晚于推理截止时刻，形式化准入拒绝，代码 `FUTURE_DOCUMENT_EVIDENCE_BLOCKED`。
- 未传可信时刻时，时序 claim 准入拒绝，代码 `INFERENCE_TIME_REQUIRED`。直接排名构造器即使收到曾经 eligible 的 admission，也会独立拦截。
- 当前正式文档直接融合仍未默认开启；在线的 `DocumentClaimOnlineShadowConfig` 强制 `SHADOW_ONLY`，保留上述限制。
- 冻结 trace 的 admission 仍由 hash-pinned policy 在在线时重算。历史 trace 若当时以无时刻方式批准了时序 claim，升级后会失败，需要按原审批流程、实际窗口重新生成并签发证据追踪资产，**不能批量跳过门禁**。
- 缺少因果窗口的 Mission Research/Build Knowledge 不应获得时间相关的审核故障立场；诊断过程从已绑定 episode 取窗口后才允许尝试该映射。

## 3. 测试结果（各分组有重叠，不应相加作为独立用例数）

- `r2_final_core.log`：**33 passed**（时间边界、候选/排名、GraphRAG 文档证据、Mission 受审核映射）
- `r2_regression2.log`：**101 passed, 2 skipped**（预推理检索、材料治理、知识融合、LLM 受控循环等）
- `validation_logs/foundation-integration/output.log`：**65 passed**（下一代 RAG、任务控制台、权限、知识增强和平台工作台）
- `validation_logs/r2-cli-smoke-fixed/done.json`：**PASSED**，退出码 0；标准输出落盘且完成标记正常生成。
- `time_gate_baseline.log` 和 `r2_core.log` 中初始失败为新门禁后兼容性测试未提供可信时刻；`r2_core_fixed.log` 与 `r2_final_core.log` 已修正并重跑。`r2_regression.log` 中为错误文件路径造成 pytest exit 4；随后使用正确路径 `r2_regression2` PASS。以上日志用于审计，不可挑选性删除。

## 4. 仍未消除的 P0 / P1 风险

1. **正式交付清单未核实**：原 `SHA256SUMS.txt` 包含 7 个不存在的 `platform_acceptance_cases_v1/TC*/operations.log`，原 `result_log_review/operations.log` 被合成夹具代替，哈希不一致。不可伪造日志或修改历史清单冒充通过。原清单另有部分历史 profile 排除文件与本轮修改文件，与这 7 个独立区分。
2. **时刻信任链**：本轮在代码上闭合 `as-of` 检查，仍依赖上游 `EpisodeWindow.end_s` 与实际使用的测量窗口一致。需要补端到端防篡改证据，包括数据 hash、窗口 cut、请求 hash。
3. **跨型号外部数据**：GraphRAG 关联不等于因果；尚未用真实不同型号和故障相似样本评估误诊率。不能称其已在航天任务中完成现场验证。
4. **真实模型推理**：尚未用 LM Studio / vLLM 的长期并发、长上下文、断网、超时、异常重定向、提示词注入及权限分离做联机全回归。
5. **跨平台与性能**：本轮在 Linux 的项目 Python 环境运行，Windows PowerShell 和当前用户笔记本上的时延、进程终止行为还需实测。
6. **GraphRAG 与材料立场适用性**：时间门禁现阻止未来投票，但仍需外部测评验证因果方向、反证、证据来源独立性；不可自动据此向卫星下发控制动作。
7. **Git/CI**：交付 ZIP 不带 `.git`，尚未真正建立 feature 分支、commit、PR、Actions、main merge 或 tag。

## 5. 后续严格顺序

P0：原签名验收数据找回/清单校验 → 集成 API/CLI 的真实时刻信任链验证 → Windows + 本地 LLM 端到端回归 → Git/CI 绿灯。

P1：弱故障、并发故障、跨型号和不相关文档压力与反例测试；量化 Recall、证据泄漏率、误诊率、拒判与推理成本。

P2：只有上述基础达到验收门槛后，考虑引入新的重排序、反证 RAG、GraphRAG 动态规划或自适应 Agent 检索。

## 6. 不阻塞前端的测试命令

```powershell
python scripts/launch_logged_tests.py start --timeout 900 r2-smoke -- python -m pytest -q tests/test_document_claim_causal_time_hardening.py tests/test_document_claim_fusion_adapter.py
python scripts/launch_logged_tests.py status r2-smoke
Get-Content validation_logs\r2-smoke\output.log -Tail 50
```

独立进程在完成后原子写出 `validation_logs/r2-smoke/done.json`，可作为本地文件监视器或 CI 的完成触发信号。**它不是 ChatGPT 的主动推送**；聊天连接断开后，脚本继续运行不代表模型能够自动回到同一会话发送完成通知。

## 7. R2 补丁在干净候选基线上的复放验证

将本轮统一 diff 应用至重新解压的 R1 candidate 后，执行 `git apply --check` 为 PASS；应用后 `src/astrograph/retrieval/document_claims.py` 的 SHA256 与 R2 工作目录一致；再次运行时间门禁测试得到 **13 passed**。此次仅验证补丁可应用与核心测试，不等同仓库正式 Git 提交。当前已连接 GitHub 的仓库列表未发现名称明确对应 AstroGraph 的项目，因此未向其他项目仓库误提交。