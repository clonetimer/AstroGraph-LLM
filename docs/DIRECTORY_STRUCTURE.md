# post49 目录结构与文档规范

状态：`CURRENT_POST49_DIRECTORY_POLICY`

```text
AstroGraph/
├── README.md / START_HERE.md / RUN_GUIDE.md / TESTING.md
├── src/astrograph/                    产品代码
├── configs/                           当前运行/验收配置
├── models/ data/ reference/ benchmarks/ 受治理资产
├── docs/
│   ├── README.md                      当前文档唯一入口
│   ├── CURRENT_*.md/json              当前事实/策略
│   ├── local_acceptance/              当前运行手册
│   ├── agent_modes/                   当前模式边界
│   ├── agentization/                  当前 Agent 化合同
│   ├── implementation/POST49_STABLE_RELEASE.md
│   └── history/                       **所有历史说明性文档统一归档**
├── governance/
│   ├── implementation/post49/         当前 release 治理
│   ├── evidence/                      hash-bound 证据（含历史，保持原路径）
│   └── ...
├── research/                           研究候选，非 release truth
└── scripts/acceptance/                 验收工具
```

## 文档移动规则

- dev/post/date/旧实验/旧 qualification/旧修复 runbook：进入 历史档案（本交付源码包不包含）；
- 当前 release 文档不得以旧版本标题自称“当前”；
- hash-bound governance evidence 不为“看起来整齐”而移动或重写；通过 `governance/README.md` 标识 current vs history；
- 旧文档原始路径与新归档路径记录在 独立历史归档包的 manifest（本交付源码包不包含）；
- 新 current 文档必须进入 `docs/CURRENT_DOCS.json`。

## 运行产物

Acceptance workspace 应位于源码目录之外，例如 `../AstroGraph-local/post49/`。源码树不存放临时运行输出、pytest cache 或本机密钥。
