# Qwen3-VL 4B 与 PaddleOCR-VL 1.6 本机 OCR 试验

日期：2026-11-07。范围：直接调用视觉模型识别文档图片；没有改动 AstroGraph 正式 OCR 流程，也没有测试完整 PaddleOCR-VL 文档解析流水线。

## 与项目默认解析链路对照

项目设置中的 OCR 默认值为 `disabled`；PDF 上传先走 `ASTROGRAPH_PDF_LAYOUT`（Poppler `pdftotext -layout`）。只有整份 PDF 无可提取文字且显式选择 `tesseract` 或 `docling` 时，才进入未审核 OCR 候选流程。因此“默认 OCR”实际上是**原生 PDF 文字层解析，并无默认视觉 OCR 模型**。

用相同来源做了默认解析实测：NASA 44 页 PDF 解析成功，44 页都有文字层，产生 365 个文档元素；第 11 页的栅格图片被标记为需要视觉复核。将同一合成中文表格 PNG 保存成无文字层 PDF 后，默认解析返回 `ValueError: PDF contains no extractable text; OCR output must be supplied as a separate reviewed representation`。复现脚本及原始摘要为 `probe_default_parser.py`、`default_parser_summary.json`。

| 项目场景 | 默认 PDF 解析 | Qwen3-VL 4B 直调 | PaddleOCR-VL 1.6 直调 |
| --- | --- | --- | --- |
| NASA 第 7 页，带文字层 | 成功读取原生文字层；可保留 PDF 页/块来源 | 图片转写与 PDF 的 306 个英文数字词序列一致 | 图片转写与 PDF 的 306 个英文数字词序列一致 |
| NASA 第 11 页，图文混排 | 读取文字层并标记栅格图待复核；原生文字排列不保证图示可读顺序 | 图中文字基本完整，按任务阶段分组，便于阅读 | 图中文字基本完整，但左右图文交错串接 |
| 合成中文表格，无文字层 | 解析失败，需显式 OCR 候选和人工复核 | 表格三行、六列及下方两句均读出 | 表格三行、六列读出；整页漏下方两句，裁切后可恢复 |

这不是三种 OCR 引擎的等条件竞赛：默认链路直接读取 PDF 内嵌文字，另两者读取渲染图像；中文表格是合成图，不能代表真实扫描手册。对于已有可靠文字层的 PDF，保留默认解析的来源定位优势；扫描页和复杂图示需要额外视觉候选和独立复核。

## 环境与调用

- Qwen3-VL 4B：用户运行的 LM Studio `http://127.0.0.1:1234/v1`，模型 ID `qwen/qwen3-vl-4b`。
- PaddleOCR-VL 1.6：LM Studio 中已加载主 GGUF，但 `/api/v1/models` 报告其 `vision:false`，直接传图片返回 HTTP 400，提示模型不支持图像输入。磁盘上有单独的 `PaddleOCR-VL-1.6-GGUF-mmproj.gguf`。为隔离验证，使用 LM Studio 自带 llama.cpp 运行时在 `127.0.0.1:1235` 显式指定主 GGUF 与 mmproj，接口报告支持图像，测试结束后已关闭 1235 临时进程。LM Studio 原 1234 服务未停止。
- 两模型在同一图片上使用相同的通用逐字识别提示、`temperature=0`、`max_tokens=2600`。NASA 页面来自前一轮 Qwen3-VL 试验，其提示和图片 SHA256 与本轮 PaddleOCR-VL 相同；调用并非同时进行。合成中文表格在本轮顺序调用。
- 直接调用主模型只能衡量模型读取图片的表现。官方 PaddleOCR-VL 流程还包括版面检测、裁切和结果合并；本试验不能代表该完整流程的效果。[PaddleOCR-VL 官方使用说明](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.md)

## 样本与结果

| 图片 | Qwen3-VL 4B | PaddleOCR-VL 1.6 直调 |
| --- | --- | --- |
| NASA Mars Climate Orbiter PDF 第 7 页，正文 | 与 PDF 内嵌文本的 306 个英文词序列一致 | 与 PDF 内嵌文本的 306 个英文词序列一致 |
| 同一 PDF 第 11 页，任务阶段图示 | 读取了所检查的 7 个关键短语，并按 Cruise、Launch、Mapping/Relay、Mars Orbit Insertion and Aerobraking 分组 | 通用提示也读取了关键内容，但将图示左右两组的文字按扫描行交错串接，句意不易直接阅读；原生 `ocr` 提示另将 `[biprop]` 识为 `[bipro]` |
| 合成中文反作用轮表格，已知原文 | 三行数据、列标题、时间、数值、单位及表格下两句核查提示均保留，输出为 Markdown 表格 | 三行数据及数值保留，单位 `°C` 输出为等价的 `℃`；表格使用 `<fcel>`/`<nl>` 标记，整页调用漏掉表格下方两句。单独裁切下方区域再调用，两个句子均被识别 |

中文合成表格的数值是测试造数，不是实际卫星遥测，也不能用于验证故障诊断正确性。第 11 页的 PDF 内嵌文字顺序与图示阅读顺序不同，不能只用词序列相似度判定图示 OCR 优劣。以上没有受控重复实验、统一硬件配置或完整文档解析评测，调用耗时不作性能排名。

## 对 AstroGraph 的判断

1. 对需要给维护人员直接阅读的图文混排材料，Qwen3-VL 4B 的这一轮输出更易读；中文合成表格也更完整。可先作为视觉识别候选，但需要在真实中文扫描件、表格和公式上扩大评测。
2. PaddleOCR-VL 的表格行列标记适合机器解析，但必须增加标记解析、版面分区和页脚/表外文字覆盖检查。下方文字裁切后可恢复，说明整页漏识不能简单归因于模型不认识文字。
3. 现有 LM Studio 1234 中的 PaddleOCR-VL 主模型尚未作为可接收图像的模型暴露。正式接入前需确保主 GGUF 与 mmproj 正确配对，并用一张带图文字的图片做接口预检。单看“READY”不足以确认视觉功能可用。
4. 暂不建议用本试验直接替换项目 OCR 默认路径。下一步应比较完整 PaddleOCR-VL 流程与 Qwen3-VL：同一批真实资料，核对全文覆盖、表格行列、关键数值及原文页码，并记录人工复核成本。

## 可复核文件

- `synthetic_rw_table.png`、`synthetic_rw_table_truth.json`：合成图及已知原文。
- `synthetic_qwen3_vl_4b_content.txt`、`synthetic_paddleocr_vl_1_6_content.txt`：同图同提示输出；对应 `*_raw.json` 为完整接口响应。
- `synthetic_rw_footer_crop.png`、`synthetic_paddle_footer_content.txt`：PaddleOCR-VL 区域裁切复测。
- `mco_page_07_generic_content.txt`、`mco_page_11_generic_content.txt`、`mco_page_11_native_ocr_content.txt`：PaddleOCR-VL NASA 页面输出。Qwen3-VL 对应输出位于 `../mineru-local-20261107/compare_mco_page_07_qwen_qwen3-vl-4b_content.txt` 与 `../mineru-local-20261107/compare_qwen_qwen3-vl-4b_content.txt`。
- `run_paddle_direct.py`、`run_chinese_compare.py`、`probe_paddle_footer.py`：复验调用脚本；1235 需要先显式加载主 GGUF 和 mmproj。`summary.json` 与 `synthetic_summary.json` 记录接口状态和原图哈希。
