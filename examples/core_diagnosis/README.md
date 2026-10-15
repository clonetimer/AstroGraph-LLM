# 合成数值补证样例

此目录由 `tests/implementation/conftest.py` 的明确合成夹具生成，仅验证代码链路，不表示真实 ADCS 故障准确率。输入模型首选 A，数值检查提供 B 的支持及 A 的反证，原版数值算子和真实图查询/BiRRF 将结论改为 B。`memory-policy.json` 只用于示范独立确认，不是生产审核配置。

运行方法见 `docs/implementation/RUN_CORE_POST48_DEV2.md`。执行证据见 `governance/implementation/post48_dev2/examples/`；数据库不随源码交付。
