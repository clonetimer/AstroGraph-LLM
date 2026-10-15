# 文献图谱构建最小样例（合成数据）

此目录仅包含虚构的实验台资料，非航天器真实失效/处置依据。先设置 `ASTROGRAPH_MATERIAL_GRAPH_BUILD_ENABLED=1` 并配置本地受控的 LLM 服务：

```bash
astrograph-platform --root ./runs/platform graph-build \
  --reference examples/material_graph_builder/synthetic_manual.md \
  --equipment-scope LAB_RW_DEMO_V1

astrograph-platform --root ./runs/platform graph-pending
```

生成的候选三元组必须含真实材料引句（不能杜撰），需领域专家审查。未经审核的草稿不会进入已审图谱；审核/签名发布详细见 `MATERIAL_GRAPH_BUILDER_DELIVERY.md`。
