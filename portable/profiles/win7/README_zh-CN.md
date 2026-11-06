> 当前源码开发版：2.8.0-dev.1。下文保留 V2.7.1 操作基础；本轮新增执行记录和报告版本，准确范围、构建方式与验证状态见源码 `docs/PORTABLE_OBSERVABILITY.md` 和 `docs/PORTABLE_OBSERVABILITY_VERIFICATION.md`。本轮未发布新的 Windows 二进制包；不要把历史测试记录当作本轮 Win7/Win10 实机验收。

# AstroGraph-LLM Win7／Python 3.8 便携试用版

版本：`2.7.1-win7-py38-preview.1`。日期：2026-11-04。

本包基于 V2.7.1，保留六页导航及外部诊断材料 → LLM 分析 → 引用、报告、追问、历史的功能，针对 **Windows 7 SP1 64 位（x64）** 更换运行时和依赖。**这是兼容性试用包，不是已经完成 Win7 实机验收的 Stable。** 不能用于 32 位 Windows，也不支持 Internet Explorer。

## 1. 先检查系统条件

- Windows 7 SP1 x64；安装路径建议短且可写，例如 `D:\AstroGraphWin7`，不要放入 Program Files，也不要直接在 ZIP 内运行。
- 已安装 KB2533623 或包含它的后续系统更新，以及 Universal C Runtime。若系统太旧，Python 可能尚未启动就出现“缺少 api-ms-win-crt…”或“无法定位程序输入点”；本包不会自动改动系统或下载更新，需要管理员使用可信离线系统补丁解决。
- 已有能够在该 Win7 上运行的现代浏览器。以 Chromium 109 同级能力作为兼容目标；当前仅执行了现代 Chromium 回归，未完成旧浏览器或 Win7 实机测试。不要用 IE，不包含浏览器安装包。
- 本包内置 **官方 Python 3.8.10 Windows x64 嵌入式解释器**和所需依赖；无需另装 Python、Node、pip，也不会在启动时下载依赖。它不使用系统中另外安装的 Python。
- 本包不含大模型服务软件、模型权重或 GPU 驱动。模型的系统和硬件兼容性需要另行确认。

Python 3.8 已结束官方支持。本试用使用官方最后一版提供 Windows 常规二进制的 3.8.10，而不是自行宣称拥有 3.8.20 的 Windows 运行时。仅在受控环境中处理可信材料；联网模型使用 HTTPS，证书验证保持开启。新增根证书包不等于更新 Python/OpenSSL 的所有安全修复。

## 2. 三步启动

1. **完整解压到全新目录**。不要把 Win10 包的 runtime/packages 复制过来，也不要覆盖现有 Win10 安装。
2. 双击 **CHECK.bat**。依次执行文件清单校验、系统条件检查和应用离线自检。通过时出现 `Package integrity passed.`、环境报告中的 `"status": "PASS"` 和 `Offline self-test passed.`；结果保存为 **CHECK.log**。离线自检使用脚本化本机接口，不会访问真实模型或互联网，也不代表实际模型已经配置完成。
3. 双击 **START.bat**。在已有浏览器打开控制台显示的本机地址。程序只监听 `127.0.0.1`，保留控制台；退出时按 Ctrl+C。浏览器没有自动打开时手动复制地址，且不要改为 IE。

环境检查只能在 Python 已成功启动后运行。若弹出 Windows 加载器错误而日志为空，保留错误窗口截图，并检查 SP1、系统更新、UCRT、x64 架构与完整解压；不需要重新 pip 安装。

## 3. 模型连接

### 在线 API：本次优先试用路线

进入“模型设置”，选择在线 API，填写兼容 Chat Completions 的 **HTTPS** Base URL、API Key 和实际模型 ID。测试连接查询模型列表，真正执行一次分析才能验证生成接口与 JSON 输出兼容性。在线分析会把所选诊断材料、分析要求和追问发送给配置的提供商。

服务端 SSL 校验保持开启，并增加随包的 certifi 根证书。证书错误时检查系统时间、服务证书和网络代理；不要禁用证书或主机名校验。遇到机构私有 CA，应由管理员按组织规定部署到信任库，而不是关闭校验。

### 本机离线模型：服务需另外验证

仅连接该电脑已经运行的兼容模型 API；本地地址只允许 `localhost`、`127.0.0.1` 等 loopback。LM Studio／Ollama 选项只是端口地址预设，**不表示它们的现行 Windows 安装包支持 Win7**。没有已验证的 Win7 推理服务时先使用在线 API。

本次未新增局域网 HTTP 模式，也不允许以关闭 HTTPS／地址检查的方式接入任意内网地址。单位已有 HTTPS 模型网关可以按在线 API 模式配置；真正无互联网时仍需网络能够到达该内部 HTTPS 服务。

API Key 仅保存在本次会话/服务内存中，不写入任务、备份或日志，刷新或重启后按需重新输入。即使系统已经安装 Python3.8，也应使用包内解释器和依赖，不混用原生扩展。

## 4. 界面和业务功能

| 页面 | 功能 |
|---|---|
| 开始分析 | 粘贴/上传材料、角色与编码、分析目标；模型状态前置、底部操作按钮 |
| 分析结果 | 摘要、综合结论、原因假设、支持与反向依据、核查建议、原文引用及导出 |
| 历史任务 | 列表与筛选、打开任务、备份 ZIP 与恢复 |
| 证据问答 | 针对当前报告追问，引用跳回原文；任务之间保持隔离 |
| 模型设置 | 在线 API／本机兼容接口、模型列表、输出预算与超时 |
| 使用帮助 | 输入边界、离线部署、错误排查及迁移 |

输入支持 TXT、MD、JSON、CSV、LOG、可提取文字的 PDF、DOCX；至少一份外部诊断，日志和参考资料可选。文本可用 UTF-8/BOM/UTF-16，旧中文文本可选择 GB18030。扫描 PDF 不自动 OCR；复杂版式与表格先核对提取结果。PDF 库仍是 pypdf 5.9.0，未完成安全升级，只导入确认可信的文档。Python3.8.10 本身也是旧运行时，不宜解析不明来源文件。

业务容量沿用原版：总文件 4 MiB、解析正文 180,000 字符，最多 8 份诊断、12 份日志、16 份参考资料。60,000 字符以上分块分析（约 12,000 字符，重叠 600，最多 24 块），完整摘要合计超过 120,000 字符时停止并保留检查点；实际模型上下文可能更小。每任务最多 12 次追问尝试，每题 4,000 字符。原文不会静默截断，但 LLM 概括不保证无遗漏。

模型分析与输入诊断结论分开保存；引用校验只验证来源存在，不证明模型推断正确。报告保存成功不等于真实模型调用成功。失败/中断任务可重新填原模型模式、地址和模型 ID 后继续；已通过的长文分块复用。成功任务不通过失败重试入口改写。

## 5. 数据迁移

用户材料、报告、追问、分块检查点保存在解压目录 `data/`，首次运行自动创建。历史页面可备份、恢复，重复任务编号拒绝覆盖，模型运行期间不能备份恢复。建议先备份旧数据，再到新目录试用；不得直接覆盖正在运行的旧版。API Key 不进入备份，但备份包含诊断原文，应按材料密级保护。

## 6. 常见问题

| 现象 | 优先检查 |
|---|---|
| Python 未启动，提示缺 DLL／程序输入点 | Win7 SP1、KB2533623 或后续更新、UCRT、x64 与完整解压 |
| `PYTHON38_REQUIRED`／`X64_RUNTIME_REQUIRED` | 使用本包 `runtime/python.exe`，不要替换为系统 Python 或 Win10 运行时 |
| `KB2533623_OR_SUPERSEDING_UPDATE_REQUIRED` | 让管理员安装包含 AddDllDirectory 支持的可信系统更新 |
| `UNIVERSAL_C_RUNTIME_REQUIRED` | 按组织流程安装 UCRT；便携包不能替代全部系统补丁 |
| `_pydantic_core` 无法加载 | 检查系统组件和隔离文件；不要拿 cp314 扩展替换 cp38 文件 |
| 清单 `changed file`／`missing` | 完整重新解压，检查安全软件隔离；不要删清单跳过校验 |
| 页面空白或脚本报错 | 换用已验证支持 Win7 的现代浏览器，不使用 IE |
| HTTPS 证书错误 | 系统时间、服务证书、代理证书；不关闭 TLS 校验 |
| 列表接口成功但分析失败 | 核对模型 ID、生成接口、JSON格式与输出预算，查看结果页原因代码 |

## 7. 包内依赖与验证范围

内置 Python 3.8.10；Pydantic 2.10.6；pydantic-core 2.27.2（CP38 x64）；annotated-types 0.7.0；typing-extensions 4.13.2；eval-type-backport 0.2.2；pypdf 5.9.0；certifi 2026.10.23。

运行源码在 `app/`，独立最小源码和测试在 `source/`，运行时和依赖在 `runtime/`、`packages/`。每个文件由 MANIFEST.json 记录 SHA-256 与大小，材料下载来源在 provenance/materials.json。文件校验不是数字签名。

已经在真实 Python3.8 的 Linux 环境验证应用，在 Windows Server 2022 构建机实际执行官方嵌入式 Python3.8.10 和原生依赖加载、模型校验、TLS 根证书及 CMD 启动。**这些不是原生 Win7、不是本包完整 Windows 端到端验收，也不代表生产模型分析质量。** 详细计数与失败记录见 `source/docs/WIN7_VERIFICATION.md`。

## 8. 开发者重建

需要本源码、官方 runtime ZIP 和 lock 指定的七个 Windows wheel，放到本地后可以无网络重建。目标机日常使用不需要这些构建材料或命令。

```shell
python portable/build_win7.py --runtime python-3.8.10-embed-amd64.zip --wheels-dir windows --output-dir dist-new
```

输出目录必须尚不存在。构建只接受锁定的哈希/版本/ABI；不读取旧 data，不打包模型、浏览器或字体。原先 Win10 的 build.py 和 refresh_*.py 是历史参考，Win7 重建只使用 build_win7.py。源码是最小模块集合，不能整体覆盖完整 AstroGraph 仓库。

## 一手参考

- Python3.8 Windows部署与系统组件：https://docs.python.org/3.8/using/windows.html
- 官方嵌入式3.8.10：https://www.python.org/downloads/release/python-3810/
- 3.8.20与结束支持说明：https://www.python.org/downloads/release/python-3820/
- pydantic-core 2.27.2 Windows构建固定Rust1.77的源码：https://github.com/pydantic/pydantic-core/blob/v2.27.2/.github/workflows/ci.yml
