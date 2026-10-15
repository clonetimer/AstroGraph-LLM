# post49 LLM 服务接入

状态：`CURRENT_POST49_EXTERNAL_LLM_SETUP`

AstroGraph 稳定源码不包含已部署 LLM 服务。真实 LLM 能力只有在**实际 endpoint + model ID + auth + schema 请求/响应**通过后才能声明。

## 配置原则

- 使用宿主私有 settings，不把密钥写入共享源码/receipt；
- `doctor`/静态配置检查不等于在线调用成功；
- deterministic fallback 必须显式标记，不能冒充 LLM；
- LLM 不能绕过 tool grants、committed diagnosis、knowledge/memory promotion gate。

查看当前可用接入工具：

```bash
python scripts/acceptance/llm_service.py --help
python scripts/acceptance/run.py list
```

具体 endpoint/provider 参数以当前 CLI `--help` 和宿主配置 schema 为准，避免复制历史 dev runbook 中可能已变化的字段。
