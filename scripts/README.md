# 脚本导航

当前本地验收统一入口为 `python scripts/acceptance/run.py`。完整命令在 `docs/local_acceptance/STAGE_COMMANDS.md`。

| 类别 | 位置 | 说明 |
|---|---|---|
| 单阶段验收与结果导出 | `acceptance/` | 不复制诊断算法，不是产品DAG |
| 三任务原始回放与复算 | `evaluate_three_task_agent_zip.py`、`score_three_task_replay.py` | 保留既有协议 |
| 参考架构实验 | `prepare_mode_benchmark.py`、`run_mode_benchmark.py`、`replay_normal_mode_benchmark.py`、`score_mode_benchmark.py` | 对比不冒称最新SOTA完整复现 |
| 源码与发布审计 | `verify_source.py`、`verify_release.py`、`tools/` | 与功能及性能验收分开 |
| 历史研究与集成 | 其余既有脚本 | 为复现保留原路径，不默认全部运行 |

新验收功能放入 `acceptance/`；新业务算法应放入 `src/astrograph/`，不要在验收脚本复制第二份裁决或概率计算。
