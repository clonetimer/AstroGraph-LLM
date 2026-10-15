# post49 Provider Qualification 验证指南

状态：`CURRENT_POST49_PROVIDER_VALIDATION`

验证至少覆盖：

1. artifact/hash 漂移；
2. channel/order/unit mismatch；
3. timeout / crash / cleanup；
4. schema/response invalid；
5. capability discovery 不等于 execution authorization；
6. 未校准 probability 不标成 calibrated；
7. OOD qualification 只在预声明 threat model 内成立；
8. external adapter fixture PASS 不等于真实 provider 领域质量 PASS。

当前运行/验收入口：`python scripts/acceptance/run.py list`。历史 provider qualification 报告统一在 历史档案（本交付源码包不包含），不再从旧 post 说明推断 post49 当前状态。
