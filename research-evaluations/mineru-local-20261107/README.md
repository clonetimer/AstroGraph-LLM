# 本机 MinerU 模型解析实验

日期：2026-11-07。只在 `D:/Project/AstoGraph-LLM/AstroGraph-LLM/research-evaluations/mineru-local-20261107/` 写入实验文件，未修改平台配置或远端服务。

## 目的与环境

核验用户提供的 `http://127.0.0.1:1234` 中自报为 `mineru2.5-pro-2605-1.2b` 的模型，对航天公开文献的文字、版面和插图能否提供可用解析。`/v1/models` 确实返回该模型 ID；这只是服务自报名称，未核对权重文件。接口为 OpenAI 兼容 `/v1/chat/completions`，**不是完整 MinerU CLI/HTTP 文档解析服务**。本机未发现可执行的 `mineru`/`magic-pdf` 命令。

资料为本地已有 NASA Mars Climate Orbiter Phase I 报告 `D:/Project/AstoGraph-LLM/knowledge-sources/batch2-20261024/public/originals/PUB2-MCO-PHASE1-1999.pdf`，SHA-256 为 `ed4c0c5a23e4e30df7279924ad0978cf00984fd4ebe58a1f7297ba157b0cdbc6`。选择 PDF 第 7 页（长段落与清单）和第 11 页（任务阶段插图），Poppler 按 130 dpi 渲染为 PNG。另裁出第 11 页插图区。`pdftotext -layout` 提取 PDF 自带文本作**机械对照**，不是人工金标准。

直接以图像和 MinerU 常见任务提示 `Layout Detection:`、`Text Recognition:`、`Image Analysis:` 调用模型；保存完整响应、纯文本、图像、参考文本、耗时与[自动指标](metrics.json)。官方 [MinerU 代码](https://github.com/opendatalab/mineru-vl-utils/blob/main/mineru_vl_utils/mineru_client.py)将这些提示放在分阶段识别流程中；本实验仅测本机模型接口，未复现完整的分区、裁剪和结果组装。

## 观察结果

| 输入 | 请求 | 结果 | 主要限制 |
|---|---|---|---|
| 第 7 页整页 | 文字识别 | HTTP 200，约 3.1 秒；305 个模型英文数字词与 306 个 PDF 内嵌文本词，顺序相似度 0.9984 | 只验证这一页的文字；没有人工逐字校对 |
| 第 7 页整页 | 版面检测 | HTTP 200，约 3.5 秒；返回连续文字 | 没有 `<\|box_start\|>` 版面框；不能据此组装有来源坐标的页面结构 |
| 第 11 页整页 | 文字识别/版面检测/图像分析 | 均 HTTP 200，分别约 1.9/2.1/1.9 秒；只读出顶部段落和图号 | 正文内嵌文本 181 词，模型整页文字只有 76 词；图中 7 个预设关键文字锚点全部缺失；版面框仍为 0 |
| 第 11 页插图区裁剪 | 文字识别 | HTTP 200，约 1.3 秒；7/7 预设锚点均出现 | 需要外部可靠的分区或裁剪流程；仅捕获图中文字，未验证图义 |
| 第 11 页插图区裁剪 | 图像分析 | HTTP 200，约 1.7 秒；返回流程图样式内容 | 出现原图未标注的 `Aerial`、`Final Output` 节点；不能作为已审核机理或任务事实 |

## 同机其他视觉模型对照

在同一 `1234` 服务、同一第 11 页整页图片上，用统一的“逐字转写所有可见文字，包括图中标签，不推断不可见内容”提示调用。模型切换会影响加载等待，下面的单次耗时**不可作稳定速度排名**。

| 模型 | 第 11 页整页图中锚点 | 第 7 页密集文字 | 单次观察 |
|---|---:|---|---|
| MinerU2.5-Pro-2605-1.2B，专用 `Text Recognition:` 提示 | 0/7 | PDF 内嵌文本词序相似度 0.9984 | 3.1 秒；整页插图漏读，裁图后可恢复 7/7 |
| Qwen3-VL 4B，通用转写提示 | 7/7 | 词序相似度 1.0000 | 第 11 页 11.6 秒；第 7 页切换模型后 58.7 秒；图内文字基本齐全，但未返回可验证区域坐标 |
| Qwen2.5-VL 7B，通用转写提示 | 7/7 | 本轮未测 | 第 11 页 74.1 秒；漏掉一条 Mapping Phase 信息，并把 `PMIRR` 转成 `PMIR` |
| 本机 MiniCPM5 2B 服务 | 未产生输出 | 未测 | 接口 HTTP 400，明确称该模型配置不支持图像输入 |

Qwen3-VL 第 11 页输出保留了七个预设字面锚点，但 PDF 原文中的阅读顺序、标点和图文关系仍需人工复核。第 7 页 `1.0000` 是英数字 token 的单页比较，不等于完整 OCR 准确率。完整请求与原始响应在 `compare_*.json`、`compare_*_content.txt`；[自动对比指标](metrics.json)可重新运行 `evaluate_probe.py` 生成。

`word_sequence_similarity` 是 `difflib.SequenceMatcher` 对英数字 token 的比值，仅供本页对照；裁图与整页的词数不能直接用于通用 OCR 排名。模型输出中没有 PDF 页码或版面框，实验程序也没有将图像描述自动写入知识库或图谱。

## 对平台接入的判断

1. 本机模型可作为**候选文本识别组件**继续评估。第 7 页表现好，但第 11 页整页漏掉整个插图区域；只把模型 ID 配到网页 OCR 设置不足以形成可信的文档解析链。
2. 若要把 MinerU 用于网页“新建分析”，需接入官方支持的文档编排服务或独立实现并验证渲染、布局分区、裁剪、识别、顺序组装、页码/框坐标和来源哈希。当前 `agents/mission/documents.py` 只支持 MinerU CLI 的 `-p/-o`，不能直接调用这个 OpenAI 兼容端口。
3. 识别结果、图示说明和图谱候选保持 `UNREVIEWED`；尤其不能把生成的 `Aerial`、`Final Output` 当 NASA 原文。进入 RAG 前应保留原 PDF、页号、区域坐标和人工审核记录。
4. 下一轮用中文扫描手册、表格、公式及图片混排文档做盲法对照；另用真正的 MinerU 文档服务或 CLI 复验完整处理链，再判断是否替换当前 Docling/Tesseract 方案。

**现阶段选型建议：**保留原生 PDF/DOCX 解析为首选，在需要图像理解的页面增加“分区 + 局部视觉识别”候选路径；本机可先以 Qwen3-VL 4B 做受控原型。若要比较成熟的端到端文档工具，把 [PaddleOCR-VL-1.6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md) / [PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR) 与 [MinerU 完整流程](https://github.com/opendatalab/MinerU)、当前 Docling 放在相同的中文航天手册与表格/公式测试集上，比较原文准确性、页码/区域定位、漏检、错误新增、资源与时延。PaddleOCR 官方基准成绩只能作候选依据，不能直接代表本项目质量。

## 文件

- `run_model_probe.py`：本机 API 调用和原始响应保存；`evaluate_probe.py`：离线指标。
- `mco_page_07*`、`mco_page_11*`：渲染图片、PDF 内嵌文本、模型原始响应及提取内容。
- `metrics.json`：自动汇总；人工观察与边界以本文件为准。

实验未使用真实卫星遥测、私有文献或远程 vLLM。没有运行完整 MinerU CLI，也未证明该模型在航天领域的总体解析准确率。
