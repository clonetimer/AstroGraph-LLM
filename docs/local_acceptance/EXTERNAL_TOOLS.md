# 外部工具：真实运行与协议夹具分开

状态：`CURRENT_POST49_REFERENCE`

先复制 `configs/acceptance/external.example.json` 为自己的宿主配置，填写真实路径和端点。相对路径以项目根目录解析，建议使用绝对路径。不要把密钥写入共享回执；**底层 LLM JSON 加载器不展开环境变量。**可用新增 `llm_service.py configure --api-key-env` 在生成本地配置时解析；不要在JSON中直接写`${KEY}`。严格控制工作区分享范围，宿主路径/端点也可能敏感。

大模型服务的完整启动、模型ID、鉴权、配置和实际调用方法见 [LLM_SERVICE_SETUP.md](LLM_SERVICE_SETUP.md)。`doctor`不做连通测试；23/24强制确定性，真实LLM运行39。

## 1. 建立外部验收工作区

```bash
python scripts/acceptance/run.py init --workspace ../AstroGraph-local/external01 --settings /absolute/path/host.external.json --document /absolute/path/mechanism.md --nontext-document /absolute/path/manual.pdf --query "gyro bias sensor fault propagation"
```

该命令只初始化，不下载模型、不启动服务、不访问网络。不要把 `.pdf` 同时作为默认文本样例传给本地30阶段；PDF/图片交给33阶段，查看结果后再以实际 Markdown 作为建图/检索输入创建新工作区。

## 2. PaperSeek

设置 `allow_external_literature=true`，配置 `paperseek_url` 或 `paperseek_source_root`，沿用现有适配合同。运行：

```bash
python scripts/acceptance/run.py run 34-paperseek --workspace ../AstroGraph-local/external01
```

实际请求/响应和来源见 `literature.json`、任务step文件。没有命中时 PARTIAL；失败不伪造论文。未覆盖任意网站全文下载。

## 3. MinerU

在本地独立环境安装并确认实际 MinerU CLI 可运行，将其路径写入 `mineru_executable`，并设置 `allow_mineru=true`。脚本调用既有适配器使用的 `-p INPUT -o OUTPUT` 接口，不偷偷换成文本解析器。

```bash
python scripts/acceptance/run.py run 33-mineru --workspace ../AstroGraph-local/external01
```

看 `conversion.json`、`document-work/mineru-*/` 中的 Markdown 与内容列表。程序成功只证明服务运行与输出合同，版面、图像、表格、公式和页码对应仍需人工核查；图像描述是未审核资料。

## 4. 图谱构建：A2O 或结构化 LLM

A2O 路径：设置 `allow_a2o=true` 和 `a2o_source_root`，确保原四阶段源码 CLI及其模型可用。LLM路径：`allow_a2o=false`，设置 `planner=llm_assisted` 和 `llm_config`，实际抽取后检查精确原文引用。

```bash
python scripts/acceptance/run.py run 35-graph-build --workspace ../AstroGraph-local/external01
```

输出必须待审。检查候选节点、关系、原文、单位、运行条件、必要观测、冲突与遗漏。没有配置抽取器则 BLOCKED，不生成假的机理来凑验收。

## 5. 混合检索和重排

填写真实本地 `embedding_model_path` 与 `reranker_model_path`，在选定模型的环境中安装其实际依赖并保存版本。不得仅有一个目录就声明模型已加载。

```bash
python scripts/acceptance/run.py run 37-rag-hybrid --workspace ../AstroGraph-local/external01
```

仅 `retrieval_mode=BM25_DENSE_RRF_CROSS_ENCODER` 才完成该功能验收。若结果是 BM25 或缺重排，报告 PARTIAL 与具体警告。排序改进需要另做真实相关性标注评价。

## 6. 真实 LLM 思考与创造

设置 `planner=llm_assisted`、真实 `llm_config`。不要使用脚本夹具作为该阶段后端。

```bash
python scripts/acceptance/run.py run 39-llm-planning --workspace ../AstroGraph-local/external01
```

检查 `llm-acceptance.json` 中模式、实际调用、被接受的规划、回退、引用和隔离状态。确定性回退并不算真实LLM验收通过。此处只验证真实链路，不证明比确定性规划更高诊断准确率。

## 7. 不具备服务时仍能检查适配协议

```bash
python scripts/acceptance/run.py run 36-external-contracts --workspace ../AstroGraph-local/demo01
python scripts/acceptance/run.py run --suite external --workspace ../AstroGraph-local/external01
```

36 是显式本地 HTTP/CLI/模型结构夹具；external 套件才尝试真实配置。未提供服务的阶段有 BLOCKED 回执，总退出码2，不会把“不报错”当成全部通过。
