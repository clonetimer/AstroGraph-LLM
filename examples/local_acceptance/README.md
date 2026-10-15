# 本地验收输入说明

执行 `scripts/acceptance/run.py init` 后，工作区 `inputs/` 中生成：`demo-training.npz`、`demo-training.json`、`demo-inference.npz`、`demo-provenance.json`、清洗样例 `acceptance-note.txt` 和检索金标夹具 `retrieval-gold.md`。

训练/验证/推理样本均为合成，确认案例也使用独立隔离的合成库；不表示真实故障准确率或专家核实。初始化不会批准训练。所有真正训练、推理、RAG和记忆操作仍调用项目已有实现。

真实ADCS样例在历史已用片段目录，不在此复制。图谱、融合和模型结果请看本地阶段10—24；外部工具没有真实服务时只显示BLOCKED。
