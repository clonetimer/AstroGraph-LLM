# AstroGraph Agent Tool Specification — R30

Status: **CURRENT_R30_TOOL_GOVERNANCE_SPECIFICATION**  
Date: 2026-09-19

## 1. Scope

This document defines the governance contract for tools that an AstroGraph Agent can select directly, capabilities it can invoke inside the controlled diagnostic loop, deterministic probes it may request by `probe_id`, and lower-level tools orchestrated by the host. It is descriptive and restrictive: inclusion here does **not** grant runtime permission. Task-bound registries, permission scopes, admission gates and independent governors remain authoritative.

Global invariants: no arbitrary Python, arbitrary paths, arbitrary telemetry channels/windows/thresholds, arbitrary database queries, or Agent release authority. Optional Agent probe execution must not be required for the deterministic detection/diagnosis core.

## 2. Tool surfaces

| Surface | Count | Agent selection semantics |
|---|---:|---|
| `MISSION_PLATFORM` | 17 | Mission/task tools exposed through the bounded task planner and `ToolRegistry`. |
| `CONTROLLED_DIAGNOSTIC_CAPABILITY` | 10 | Fixed read-only capabilities in the controlled LLM diagnostic loop. |
| `DIAGNOSTIC_PROBE` | 11 | Host-bound deterministic probes; Agent selects only an admitted `probe_id`. |
| `LOW_LEVEL_AGENTIC_REGISTRY` | 7 | Deterministic workflow tools orchestrated by the host; not automatically optional MissionAgent tools. |

## 3. Mission/platform tool catalogue

| Tool ID | Purpose | Tasks | Modes | Diagnosis authority | Truth access | Side effect |
|---|---|---|---|---|---|---|
| `dataset.replay` | 离线回放带真值数据集，用于实验评估，不作为在线诊断。 | replay | — | `NONE` | `OFFLINE_EVALUATION_ONLY` | `READ_ONLY` |
| `diagnosis.fuse` | 综合遥测、知识与规则证据，形成维护人员可复核的故障候选；不会绕过人工复核门槛。 | diagnose | normal, thinking, creative | `DETERMINISTIC_CORE_EXECUTION` | `FORBIDDEN` | `READ_ONLY` |
| `document.clean` | 保守清洗文本，保留数字、单位、否定词和来源。 | build_knowledge, diagnose, infer_candidate, replay, research, train_model | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |
| `document.convert` | 把受支持文档转换为可检索文本，同时保留来源工件。 | build_knowledge, diagnose, infer_candidate, replay, research, train_model | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |
| `exploration.propose` | 仅在创造模式生成隔离的、可证伪的离线假设；不进入诊断投票。 | — | creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `WRITE_ARTIFACT` |
| `graph.build` | 把来源约束的信息整理为待审核图候选，不直接发布到 Active KG。 | build_knowledge | — | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `WRITE_ARTIFACT` |
| `graph.reason` | 检查诊断证据之间的已登记关系，主要用于工程审计。 | diagnose | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |
| `knowledge.graph.match_observations` | 根据已观测到的异常现象，在本机知识图谱中查找相关故障候选，作为复核线索。 | diagnose | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |
| `knowledge.graph.mechanism` | 沿设备、现象和故障关系查看可能的传播路径，帮助解释为什么当前现象与候选故障相关。 | diagnose | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |
| `literature.search` | 通过主机配置的 PaperSeek 查询外部文献；不接触遥测真值。 | build_knowledge, diagnose, infer_candidate, replay, research, train_model | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `EXTERNAL_IO` |
| `model.capabilities` | 检查当前通道与已登记模型合同是否兼容；发现兼容性不等于授权执行。 | diagnose, replay | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |
| `model.execute_compatible` | 只通过主机绑定的兼容提供方执行已准入模型。 | diagnose | normal, thinking, creative | `DETERMINISTIC_CORE_EXECUTION` | `FORBIDDEN` | `READ_ONLY` |
| `model.infer` | 准备受治理的原生 ADCS/PHM 推理，或对通用时序执行带语义边界的检测。 | diagnose | normal, thinking, creative | `DETERMINISTIC_CORE_EXECUTION` | `FORBIDDEN` | `READ_ONLY` |
| `model.infer_candidate` | 用离线候选模型和精确通道合同执行无真值推理。 | infer_candidate | — | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |
| `model.train` | 仅在明确审批的离线三任务训练清单上训练候选模型。 | train_model | — | `NONE` | `FORBIDDEN` | `READ_ONLY` |
| `output.structured` | 把已冻结的诊断结果整理成可保存、可审计的诊断记录。 | build_knowledge, diagnose, infer_candidate, replay, research, train_model | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `WRITE_ARTIFACT` |
| `rag.retrieve` | 只在当前任务已转换的本地文档中检索来源片段。 | build_knowledge, diagnose, infer_candidate, replay, research, train_model | normal, thinking, creative | `CANDIDATE_OR_CONTEXT_ONLY` | `FORBIDDEN` | `READ_ONLY` |

## 4. Controlled diagnostic-loop capabilities

| Capability | Purpose | Authority |
|---|---|---|
| `execute_diagnostic_probe` | Execute exactly one host-bound probe by probe_id; model-defined channels, windows and thresholds are forbidden. | `SHADOW_REVIEW_ONLY` / `SHADOW_DERIVED_OBSERVATION_ONLY` |
| `inspect_candidate_fusion` | Inspect the governed data/rule/KG candidate fusion result. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `inspect_deterministic_core` | Inspect the independent deterministic diagnosis and its evidence references. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `inspect_evidence_status` | Summarize executable evidence facts, roles, statuses, and missing items. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `inspect_experience_context` | Inspect separately approved experience and memory reuse context. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `inspect_kg_support` | Inspect episode-bound KG evidence and directional associations. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `inspect_probe_catalog` | Inspect host-approved deterministic diagnostic probes available for this episode. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `inspect_retrieval_context` | Inspect admitted pre-generation document and ontology context. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `inspect_verification_plan` | Inspect unresolved evidence and the deterministic verification plan. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |
| `synthesize_hypotheses` | Stop investigation and synthesize hypotheses from the recorded trajectory. | `CANDIDATE_OR_CONTEXT_ONLY` / `READ_FORMAL_EVIDENCE` |

## 5. Diagnostic probe catalogue

Every probe below is `SHADOW_DERIVED_OBSERVATION_ONLY`. The host fixes source identity, channels, windows, thresholds, algorithms and model artifacts before the Agent sees the catalogue. Probe execution cannot create formal `EvidenceFact`, change the deterministic diagnosis, or change release state.

| Probe kind | Purpose |
|---|---|
| `COMMAND_RESPONSE_RESIDUAL` | Review host-bound commanded versus observed response residual. |
| `DXC_EXPECTED_RESPONSE_REVIEW` | Mirror the frozen R27 expected-response ranking. |
| `DXC_FORWARD_SIMULATION_REVIEW` | Mirror the frozen R28 qualitative forward-model decision. |
| `DXC_REDUCED_ORDER_SIMULATION_REVIEW` | Mirror the frozen R29 numerical reduced-order simulation decision. |
| `EVIDENCE_THRESHOLD_MARGIN` | Recompute distance from an already-frozen evidence threshold. |
| `LAG_CORRELATION` | Review a host-bound lagged association over admitted arrays. |
| `MINIMAL_DIAGNOSIS_REVIEW` | Review deterministic conflicts and bounded minimal diagnoses. |
| `OSCILLATION_SIGNATURE` | Review a host-bound oscillation signature. |
| `PRE_POST_CHANGE` | Review a host-bound before/after time-series change. |
| `SATURATION_LIMIT_CHECK` | Review whether admitted values approach a frozen saturation limit. |
| `SIMULATION_COUNTERFACTUAL_REVIEW` | Compare host-admitted observed and counterfactual simulation traces. |

## 6. Lower-level host-orchestrated ToolRegistry tools

| Tool ID | Evidence authority | Diagnosis authority | Permissions |
|---|---|---|---|
| `d3_rw.evidence.build` | `BUILD_FORMAL_EVIDENCE_DETERMINISTICALLY` | `NONE` | READ_DATA, READ_KNOWLEDGE |
| `d3_thermal_star.evidence.build` | `BUILD_FORMAL_EVIDENCE_DETERMINISTICALLY` | `NONE` | READ_DATA, READ_KNOWLEDGE |
| `diagnosis.invoke_core` | `READ_FORMAL_EVIDENCE` | `DETERMINISTIC_CORE_EXECUTION` | RUN_DIAGNOSIS |
| `diagnosis.invoke_thermal_star_core` | `READ_FORMAL_EVIDENCE` | `DETERMINISTIC_CORE_EXECUTION` | RUN_DIAGNOSIS |
| `evidence.catalog.read` | `READ_FORMAL_EVIDENCE` | `NONE` | READ_KNOWLEDGE |
| `evidence_graph.build` | `NONE` | `CANDIDATE_OR_CONTEXT_ONLY` | READ_DATA, READ_KNOWLEDGE |
| `semantics.validate` | `NONE` | `CANDIDATE_OR_CONTEXT_ONLY` | READ_DATA |

## 7. Tool contract requirements

Each Agent-callable or Agent-selectable tool should have the following frozen fields before production admission:

1. **Identity** — stable `tool_id`, version and implementation/source hash where applicable.
2. **Purpose and surface** — why it exists and whether it is core, context/retrieval, experimental, or shadow review.
3. **Input schema** — `additionalProperties=false` where feasible; no model-authored Python/callable/path escapes.
4. **Host-bound parameters** — channels, windows, thresholds, model IDs, source identity and simulation parameters must be frozen by the host for diagnostic probes.
5. **Permissions and side effects** — explicit `PermissionScope`, `SideEffectLevel`, timeout, retry and execution mode.
6. **Truth policy** — online diagnosis forbids ground truth; truth-aware `dataset.replay` is segregated as offline evaluation.
7. **Evidence authority** — distinguish formal deterministic evidence construction from shadow/derived observations and context retrieval.
8. **Diagnosis authority** — distinguish deterministic core execution from context/candidate generation and shadow review.
9. **Release authority** — always false for Agent tools; release remains under independent governance.
10. **Provenance** — task registry snapshot, artifact SHA256, probe source hash or plugin SHA256 pin.
11. **Failure policy** — fail closed; missing/invalid inputs must not silently become nominal or supported evidence.
12. **Auditability** — invocation, result identity and contribution status should be persisted where the workflow is auditable.

## 8. Runtime extension policy

Trusted plugins remain optional. Only `APPROVED` plugins matching the task/work-mode policy can enter a future sealed registry snapshot. Dynamic plugin execution is isolated and SHA-256 pinned. Hot reload never mutates a running task registry. Dynamic OAG evidence checks are prebuilt registered checks; their raw tool output alone is not formal release authority.

## 9. Machine-readable source

Catalog schema: `astrograph-agent-tool-governance-catalog.v1`; current catalog SHA256: `b360b16ab2049cc08747eb6249dd8f8a25f21239a8a80b0d20c96782abf379b3`.  
Generated artifact: `governance/evidence/research/r30_tool_and_simulation/agent_tool_governance_catalog.json`.  
Runtime read-only endpoint: `GET /v1/platform/agent-tool-specification`.

## 10. Claim boundary

This specification normalizes tool semantics and authority. It does not imply that every tool is configured or available in every deployment, does not promote shadow probes to formal evidence producers, and does not make Agent participation necessary for core diagnosis.
