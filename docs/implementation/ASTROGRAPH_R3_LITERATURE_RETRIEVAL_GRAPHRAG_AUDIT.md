# AstroGraph R3 | 文献处理链路、检索增强、双向 GraphRAG 基础审查

基线：2026-10-14 的 R2 Candidate（`astrograph_source_foundation_r2_candidate_20261014.zip`）。本文件描述 R3 源码变更；不宣称已有现场准确率增益。

## 1. 文献处理链路核查

| 阶段 | 文件与入口 | 核查结论与边界 |
| --- | --- | --- |
| 解析 | `retrieval/structured_document.py`, `knowledge/document_ingestion.py` | Markdown/DOCX/PDF 结构化解析具备章节/行/资源定位与数据哈希。图片的 `alt` 不能代表人工独立确认；R3 改为待审核。Markdown 引用的资源路径仅在明确 `asset_root` 内解析。 |
| 双轨预处理 | `knowledge/construction.py`, `retrieval/document_graph_bridge.py` | 同一原文形成 RAG 片段和候选 KG 片段，使用 crosswalk 来源关联；候选图不自动发布。 |
| 审核/治理 | `retrieval/document_governance.py`, `integrations/material_case_memory.py` | 独立审批、型号范围、快照读取、来源哈希。需要持续防止把上传文档误标为“已独立审核”。 |
| 多路检索 | `integrations/material_hierarchical_rag.py` | 文档→章节→片段；BM25+字符 n-gram+可选本地 Embedding。R3 加型号绑定的扩展词审核、双候选文献对照、已审核否定关系检索。 |
| 全局/社区 GraphRAG | `retrieval/approved_community_search.py` | 关系社区提取、来源引用。R3 修复审批版本等于查询版本的错误，不能丢弃更早但仍有效的关系。条件性关系不作为无条件确认。 |
| 局部多跳 | `integrations/material_multihop.py` | 保留最大三跳、条件/否定/反向边非因果约束；R3 增加双向症状检索与跨候选歧义显示。 |
| 诊断入口 | `integrations/material_reasoning.py` | 新增结果通过原 `material_reasoning.candidates` 与 `bidirectional_graph_comparison` 输出，不调整上游候选或正式诊断名次。 |

## 2. 高风险问题与本轮修正

**F-01 审批版本语义错误（高风险）**。社区检索以前要求 `edge.snapshot == query.snapshot`。数据库实际把关系获批时版本写到 `edge.snapshot`，并把 `version <= query.snapshot` 且未撤销的关系交给检索器。旧判断会丢失已审核的早期关系，破坏长链路径完整性。R3 在只读检索中允许 `0 <= approved_revision <= requested_snapshot`；**撤销判定仍依赖可信的 snapshot-pinned 知识库读取结果**，不能把任意旧边列表当成当前有效状态。

**F-02 关系提示词污染（高风险）**。原 `_graph_hints` 只校验极性与字符串标签。R3 增加独立审核、真实引用 SHA 与定位、型号范围、版本、冲突极性和非条件性断言约束；其作用仅限查询扩展，不赋予文献诊断投票权限。

**F-03 Markdown 图片说明自动变成“已审核”证据（高风险）**。原代码按 `alt` 非空且图片文件存在，将 `representation_reviewed` 设为真。R3 一律保留为 `False`，写入待审核警告；图片路径禁止越过 `asset_root`，哈希使用流式读文件。

**F-04 没有独立反证检索**。R3 增加来源绑定的否定/冲突关系审查输出；条件性否定明确标记，不能作为无条件反证；没有匹配证据不是对候选的反驳。

**F-05 同症状多假设歧义**。R3 对固定的已有故障候选进行正向路径探索，并根据日志文本做反向症状线索检索；当多个候选共用症状时直接呈现，而不是伪造多个独立支持证据。

## 3. 合同与权限保持不变

- 文献、图谱与 LLM 不新增候选故障，不写入正式融合排名，不产生现场真值，不自动下发控制命令。
- 原文摘录必须保留来源 hash 与定位，人工批准关系保留审批证明。
- 检索到的遥测日志文本属于 *lexical cue*，不是验证后的数值/传感器观测。
- 反向遍历是 *associative retrieval*，不是可倒推的因果关系。
- 跨型号、未审核、未来获批以及条件性语句一律不得获得无条件正面证据地位。

## 4. 已知局限（不作为本轮阻断项）

- 当前对比检索是确定性多通道融合、共享来源显式标记，并非学习得到的 Cross-Encoder 相关性重排；需要独立信息检索金标准后再引入模型。
- GraphRAG 的候选匹配依赖预先审核的实体标识/精确别名；检索召回不是经过实机验证的根因证明。
- 文献抽取尚未具备覆盖所有复杂否定、长距离指代、反事实条件与跨文档逻辑冲突的语义验证；下一阶段是声明级逐条核验。
- 解析器支持的版式不等于复杂扫描论文图表已经可用作可信数字证据；OCR/图像解释继续走人工审核。
- 测试范围为定向/集成回归，不能代替实测诊断收益。用户已说明本轮不重复旧验收/上游时刻/联机模型专项。

## 5. 复现方式

```sh
python -m pytest -q tests/test_foundation_r3_literature_bidirectional.py tests/test_material_hierarchical_rag.py tests/test_material_multihop_reasoning.py tests/test_next_rag_community_vectors_multimodal.py tests/test_material_knowledge_fusion.py tests/test_structured_document_retrieval.py
```

长任务改用 README 中的 `scripts/launch_logged_tests.py`：将输出保留在 `validation_logs/<job>/output.log`，完成时原子生成 `done.json`；启动失败也应完整保留失败日志。请以同批交付的完成文件记录的实际退出码和通过数为准。