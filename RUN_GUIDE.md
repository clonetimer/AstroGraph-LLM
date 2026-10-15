# AstroGraph-Agent post49 运行指南

状态：`CURRENT_POST49_RUNTIME_GUIDE`

## Mission Agent

```bash
python -m astrograph.agents.mission doctor --settings configs/agent/mission.default.json
python -m astrograph.agents.mission run --settings configs/agent/mission.default.json --request examples/mission_agent/diagnose.normal.json
```

Thinking / creative 模式使用对应示例配置与请求；工作模式的当前边界见 `docs/agent_modes/WORK_MODES.md`。

## Runtime / API / CLI

稳定软件入口由 `release.json` 声明：

- Runtime：`astrograph.runtime.AstroGraphRuntime`
- Service：`astrograph.service`
- API：`astrograph.api`
- CLI：`astrograph.cli`

兼容诊断引擎只能位于上述稳定入口之后，不能形成另一套生产编排事实源。

## GenericSeries

- 任意有限数值时序：可进入通用异常检测；
- 分类/定位：只在 capability + exact input contract + strict admission 同时满足时执行；
- exact 33-channel ADCS profile：可执行 detection / affected-component localization / classification；
- partial、顺序/单位/哈希不匹配：必须降级或拒判。

## 文献与知识

文本/RAG/reviewed claim/knowledge proposal 都是受治理证据。普通文献保持 `diagnostic_vote_allowed=false`；未经独立资格与治理晋升不得成为正式 BiRRF 票。

## 专家反馈

专家纠错可形成 failure case、improvement candidate、sealed challenger、evaluation receipt、promotion review packet 和 rollback-covered preflight。**post49 不启用自动 production promotion 或在线自改源码。**

## Web 平台后台任务执行器

本地桌面运行默认使用同进程后台线程：

```bash
ASTROGRAPH_PLATFORM_JOB_EXECUTOR=auto   # local auth 下等价于 thread
```

如服务器部署希望每个任务使用独立 Python worker，可显式配置：

```bash
ASTROGRAPH_PLATFORM_JOB_EXECUTOR=process
```

`/platform -> 系统与服务 -> 检测服务` 的 READY 需要模型目录、普通 Chat 和结构化 JSON-Schema 三条链全部通过。核心诊断不会调用 LLM；只有进入 `POST_DIAGNOSIS_KNOWLEDGE` 后才会向结构化模型发起请求。

## 本地验收

见 `START_HERE.md` 和 `docs/local_acceptance/README.md`。

## 历史命令

历史 runbook、dev 修复说明、旧 benchmark 包说明均在 历史档案（本交付源码包不包含）。只有在复现对应旧证据时才使用，不能作为 post49 默认操作指南。

## UI v0.7: LM Studio structured-output troubleshooting

If `/models` and normal Chat pass but structured output times out:

1. In `系统与服务`, keep provider as `LM Studio` and choose `结构化输出策略 = 自动兼容`.
2. Apply configuration, then run `检测服务` again. The result now shows the actual strategy attempts and the effective validated strategy.
3. For `qwen/qwen3.5-9b`, LM Studio's model card exposes `Enable Thinking` and defaults it to enabled. If a tiny structured probe is still slow, turn **Enable Thinking off in LM Studio** for that loaded model and retry.
4. If needed, lower `结构化最大 Tokens` for post-diagnosis explanation (for example 512–1200 depending on the task) rather than increasing the deadline without bound.
5. `严格 JSON Schema` is available explicitly, but local reasoning models may be slower under grammar-constrained generation. `JSON Object + 主机 Schema 校验` keeps Pydantic validation on the AstroGraph side without requiring strict constrained decoding.

The core diagnostic candidate remains frozen before the LLM stage regardless of these settings.

## UI v0.7: tool plugin manager

Enable the trusted plugin runtime before using the GUI installer:

```bash
export ASTROGRAPH_AGENT_TOOL_PLUGINS_ENABLED=1
export ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR=/srv/astrograph/reviewed-agent-tools
```

Then open `Agent 中心 -> 插件管理`. Managed install/update performs syntax + BaseClass/manifest/ToolSpec/dependency validation before commit and rolls back failed updates. Hot reload changes future task registry snapshots only.


## UI v0.8: CORE_DIAGNOSIS vs CORE_DIAGNOSIS_PERSIST

- `CORE_DIAGNOSIS`: deterministic model/KG/BiRRF computation; LM Studio is not called yet.
- `CORE_DIAGNOSIS_PERSIST`: writes governed diagnostic audit artifacts. v0.8 contains the Windows/CRT EBADF compatibility repair here.
- `POST_DIAGNOSIS_KNOWLEDGE`: structured LLM / document post-processing begins here.

If a v0.8 task still fails, report the exact failed stage and `error.traceback.log`; do not infer an LLM failure from a pre-LLM stage.

## v0.9：持久配置、Neo4j 与 MCP

### LLM / Neo4j 配置

“系统与服务”现在可以把已经验证成功的配置保存为 Profile，并设为默认。启用状态的 LLM Profile 只有服务探测 READY 后才能保存。Neo4j Profile 也会在保存前执行连接检测。

密钥不写入 SQLite 明文：优先使用 OS keyring；没有可用 keyring 时，配置的非敏感部分仍可保存，但重新激活涉及密钥的 Profile 时应重新输入密钥。

### SQLite Control Plane

默认数据库：

```text
$ASTROGRAPH_PLATFORM_STATE_DB
# 或
$ASTROGRAPH_PLATFORM_ROOT/control_plane.db
```

SQLite 只保存配置/索引/维护状态；原始数据和完整诊断工件继续保留在任务目录。

### Neo4j

推荐建立只读用户：

```text
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=astrograph_reader
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

维护 Agent 默认只暴露领域查询，不开放任意 Cypher。

### MCP

安装：

```bash
pip install -e '.[mcp]'
```

注册 Streamable HTTP Provider 后，只读 Provider 的 allowlisted 工具会在**下一次** Mission Agent 任务创建时快照进 sealed ToolRegistry。写 Provider 当前只允许注册/检测，不进入任务。

详见 `TOOL_PROVIDER_ARCHITECTURE_V0.9.md` 和 `NEO4J_AGENT_TOOL_V0.9.md`。
