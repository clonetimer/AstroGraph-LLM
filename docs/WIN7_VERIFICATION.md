# Win7／Python 3.8 便携兼容试用：验证记录

日期：2026-11-04；版本 `2.7.1-win7-py38-preview.1`。作者自审，无独立审查者。本记录不声称原生 Windows 7 验收。

## 材料与来源

应用基于实际 V2.7.1 交付包（SHA-256 `ff12ff5c565ec5086e5a73c43b9a155f6c8988b829b90804aad997fbc9f364a1`）及其配套源码，不覆盖旧 Win10 包。

- 官方 Python3.8.10 Windows x64 embed：SHA-256 `abbe314e9b41603dde0a823b76f5bbbe17b3de3e5ac4ef06b759da5466711271`；下载时同时核对官方发行页 MD5。
- 七个 Windows 运行依赖轮子按 PyPI 返回 SHA-256 校验；每个文件 URL、版本、SHA-256 固定于 `portable/win7_dependencies.lock.json`。
- pydantic-core2.27.2 上游 Windows 构建固定 Rust1.77 以保留 Win7 支持。实际 CP38 x64 wheel 的 PE 静态检查未发现选定的新系统专属静态导入，**但这种筛查不是完整 Windows API 兼容性证明**。
- Windows 与 Linux 开发依赖分开；没有把 Linux .so、cp314 .pyd、开发环境、浏览器、模型或字体放入发行包。

## Python3.8 实际执行

使用本地从官方源码构建的 Linux **Python3.8.20** 与和 Windows 相同版本的运行依赖。Windows 发行包内为 **Python3.8.10**，二者明确区分。

```shell
PYTHONPATH=<Linux-CP38-dependencies> python3.8 -B -m unittest discover -s portable -p 'test*.py' -v
```

源码目录：61 项，60 通过、1 项按源码目录条件跳过、0 失败/错误。跳过项为发行目录 MANIFEST 检查。

在已经构建解压的 Win7 包中：同套 61 项全部通过、0 跳过、0 失败。测试涵盖 TXT/GB18030/UTF-16、PDF/DOCX、长报告分块和引用、模型失败、追问、备份/恢复、断点续跑、会话隔离、代理/重定向/来源检查，以及新增3.8兼容层与离线构建边界。

从包中执行 `app/workbench.py --self-test` 通过：最小依赖、中文路径、本机 HTTP 子进程、来源引用、追问锁与保存、Markdown 导出。无真实模型和互联网调用。不是随包 Windows python.exe 在 Linux 上执行。

## Windows 主机实际执行

GitHub Actions：`41821203263`，Windows Server 2022。

- 实际解压官方 Python3.8.10 embed 与此次锁定的 Windows wheels。
- 在含空格的目录中通过 CMD／BAT 启动，成功加载 `_pydantic_core.cp38-win_amd64.pyd`。
- 校验 `list[Annotated[str,...]]`、`str | None` 的 Pydantic 模型、model_validate/model_dump。
- SSL hostname／证书链校验仍开启，系统+certifi 根证书加载成功（该主机记录118个）。
- 实际记录见 `WINDOWS_HOST_RUNTIME_CHECK.json`。

**这只是 Windows 主机的运行时/依赖/小型模型合同测试，不是本包完整应用在 Windows 的端到端测试，更不是 Windows7。** 目录名字是英文加空格，不据此声称中文 Windows 路径通过。原生 Win7、Win7旧浏览器以及实际在线／离线模型仍待目标设备验证。

## 浏览器与静态验证

真实 Linux Chromium 执行现有 HTML/CSS/JS，API 使用明确的测试数据：原功能3组、六页导航9组、布局微调10组，共22组通过。六页导航、文本/文件/密钥会话保持、来源跳转、历史筛选、状态轮询、PDF/DOCX上传、按钮可见性、响应式布局保留。

Python3.8 实际解析26个应用/检查/构建文件语法通过；两份 JavaScript 语法检查通过。启动校验不再传末尾带反斜杠的目录参数，保留路径引号修复，并禁用延迟展开以避免 `!` 路径被解释。

29个 EXE/DLL/PYD 均为 AMD64；完整筛查在 `WIN7_PE_STATIC_SCREEN.json`。静态检查不包括所有动态加载符号，也不模拟系统补丁安装。

## 先失败后修复与已知环境问题

1. 原代码在3.8下失败：typing.Annotated、importlib.resources.files、未延迟求值的 list[str]、Path.is_relative_to、hashlib.file_digest；对应兼容修改和新增回归通过，不关闭原有路径/引用/输出校验。
2. 原测试使用 str.removesuffix 和 ReportLab 的新 md5 参数；前者替换等价操作，后者改成标准库构造真实PDF夹具，不降低应用解析断言。测试生成器没有额外字体资源。
3. 本地编译的 Linux3.8 不含 _ctypes，环境检测改为仅在 Windows 分支导入 ctypes。Windows原生解释器内有_ctypes；此项不包装为 Linux 运行时全面验证。
4. 离线构建3项回归先因构建器不存在失败，实施后通过：拒绝错误runtime、缺少锁定wheel、拒绝覆盖已有目录。
5. 源码BAT初次空白检查识别CRLF为尾随字符；源码统一LF，发行构建明确输出CRLF，重新检查通过。

## 业务与安全边界

- 没有新诊断算法、训练、图谱或任意局域网HTTP模式；在线HTTPS／本机loopback守卫不降低。
- TLS加入当前certifi根证书、保持证书链与主机名校验；这不等于修补旧Python3.8.10/OpenSSL1.1.1k的全部漏洞。
- Python3.8已结束支持，pypdf5.9.0亦未在本轮安全升级。只解析可信文档，不对不明来源材料宣称安全。
- Win7所需SP1／KB2533623或后续更新／UCRT不是本包的一部分，系统更新可能需管理员。
- 模型服务、模型权重、浏览器和驱动不在包中；预设端口不代表模型软件支持Win7。

## 远程与构建

远程分支 `feat/portable-win7-py38` 中提交了材料获取和Windows运行时测试工作流；应用兼容修改随本次 ZIP／源码交付，尚未同步完整远程应用源码或合并main。

新构建入口 `portable/build_win7.py`：仅使用固定哈希runtime和7个wheel，不访问网络；要求新目录；原始Win10包保持不变。最终包清单、源码对应性、ZIP CRC和重复构建的独立收据随交付提供，不将测试输出与源码混为一谈。
