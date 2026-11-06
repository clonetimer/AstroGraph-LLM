# AstroGraph-LLM 源码使用入口

## 选择合适的运行方式

本仓库包含完整平台与外部诊断分析的轻量入口。两者复用 `src/astrograph` 中的分析模块，启动方式和运行时依赖不同。

- **源码平台**：从 [README](README.md) 与 [运行指南](RUN_GUIDE.md) 开始，按项目依赖配置开发环境。
- **外部诊断材料分析**：见 [外部诊断文本分析](EXTERNAL_DIAGNOSIS_ANALYSIS.md)。输入是已有诊断结果；这个入口不运行遥测检测或训练。
- **本次执行记录与用量升级**：见 [共享源码与执行记录](docs/PORTABLE_OBSERVABILITY.md) 及 [验证记录](docs/PORTABLE_OBSERVABILITY_VERIFICATION.md)。这是源码增量，不是新的 Win7/Win10 二进制发行包。
- **便携版维护与构建**：见 [便携工作台说明](portable/README_zh-CN.md)。Win7 与 Win10 采用独立运行时/依赖配置，不能混用两种发行包的二进制。
- **完整项目验收命令**：见 [阶段验收命令](docs/local_acceptance/STAGE_COMMANDS.md)。历史阶段证据不等于本轮重新执行成功。

## 使用边界

开发环境的 Python/pip 操作与目标设备的便携部署不同。现有便携 ZIP 使用随包解释器；GitHub 的源码 ZIP 不包含完整的运行时安装介质，也不能靠改扩展名当作安装包。

优先以一份短诊断文本测试：导入 → 实际模型分析 → 引用 → 追问 → 导出。模型列表可读只证明连接探测，不等于生成接口或诊断分析质量验收。

不要覆盖现有用户 `data/`。升级前备份，使用独立新目录或分支。模型密钥不应提交仓库、写入执行记录或通过日志发布。
