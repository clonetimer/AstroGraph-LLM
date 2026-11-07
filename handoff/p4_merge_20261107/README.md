# AstroGraph 第四环节源码合并包

快照日期：2026-11-07。包内 `source/` 是 `D:/Project/AstoGraph-LLM/AstroGraph-LLM` 的当前源码快照，包含未提交的第四环节与界面改动。`handoff/` 是供人工合并的清单、测试记录和选定证据。**由接收方自行合并；本包没有改动整体项目稳定版。**

## 阅读顺序

1. [合并说明](P4_MERGE_GUIDE.md)：哪些代码是第四环节、在目标项目中从哪里接入、合并后如何回归。
2. [功能清单](P4_FUNCTION_MATRIX.md)：当前可用能力、条件能力及未证明的质量边界。
3. [完整功能核对表](P4_COMPLETE_FUNCTION_INVENTORY.md)：逐页面、逐能力核对文件及启用条件。
4. `PACKAGE_COVERAGE.json`：关键文件在包内的 SHA-256 覆盖记录。
5. [测试与方法比较](P4_TEST_AND_METHODS.md)：真实浏览器、后端、模型试验及失败案例。
6. `evidence/`：选择性保留的原始评测记录和真实页面截图。

R4 补入默认 PDF 解析、Qwen3-VL 4B 与 PaddleOCR-VL 1.6 的本机 OCR 对比记录。两视觉模型尚未接入正式上传流程；本次补充不改变默认 OCR 设置或整体诊断资格。

包根目录的 `PACKAGE_MANIFEST.json` 记录每个交付文件的 SHA-256。旁边的同名 `.zip.sha256` 记录压缩包哈希。包内未提供 Python 环境或模型权重；运行依赖以 `source/pyproject.toml` 为准。

本包是**可供第四环节人工合并的稳定源码快照**，不延伸为整个四环节项目在新设备上已通过验收或故障诊断准确率已被独立证明。源码中的 `0.7.1.post49`、`release.json` 属于原仓库版本语境，不能直接覆盖目标项目的版本和发布资质。
