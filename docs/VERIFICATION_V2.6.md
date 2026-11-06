# V2.6 验证记录（2026-11-03）

## 应用回归

环境：Linux / Python 3.13.5 / Pydantic 2.13.5 / pypdf 5.9.0；隔离环境没有 NumPy、Pandas 或完整诊断引擎。

命令：`python -m unittest discover -s portable -p "test*.py" -v`

本轮完整便携版测试：40 项通过，0 失败。旧版 28 项测试保留；新增长文摘要、引用映射、失败续跑、输入绑定、备份互斥、Windows 路径别名、恢复状态、扫描文档提示、解析超时、包刷新和清单检查回归。测试数是整个便携子项目，不是整个 AstroGraph 历史研究测试库。

新增修复先观察到相应失败再修复。新增严格输入校验后，两个旧测试的任意来源编号改为生产算法生成的合法编号，原有恢复/身份断言保留。

## 浏览器组件

命令：`python portable/browser_smoke.py --chromium /usr/bin/chromium --output artifacts/ui`

真实 Linux Chromium 执行原生产 HTML/CSS/JavaScript，3 组行为通过：PDF/DOCX 选择及参考资料角色进入上传请求；对照栏目只显示一次、点击引用展开原文且不把文字当 HTML；失败任务可以点击真实重试入口。

浏览器通过 set_content 装载本地资源，API 返回为固定测试数据；这不是生产服务 URL 导航、完整浏览器端到端或 Windows 测试。原包“已完整浏览器验收”的笼统描述已撤下。应用测试另外执行真实本机 HTTP 模型协议及报告/追问/备份接口。

## 源码复核

本轮为作者单独复核，没有可用独立子代理。检查了长报告阶段数据传递、引用 ID 映射、任务重试状态、备份锁、文档解析子进程和前端文件/报告渲染路径；发现并修复前端仍拒绝 PDF/DOCX、对照栏目重复两项遗漏。

三份将修改的原项目核心文件，其 V1 交付源码的 Git blob SHA 与 GitHub 基线 5389bfd 的读取结果一致；单一补丁仅修改这些文件、增加长文模块以及 portable/、文档、工作流，不覆盖原项目 __init__.py 或其他模块。远程未写入。

## 未完成的资格

Windows 10 内置 Python 执行、CHECK/START 批处理、真实模型质量仍未验证。pypdf 仍为 5.9.0，升级未完成；文档解析超时保护不能替代升级。当前仅供可信本地材料试用，不发布 Stable。工作流已准备，未远程运行。

## 包级验证

最终 ZIP 的 CRC、全文件 SHA-256、Windows PE 架构、两次重复构建哈希以及单独源码补丁应用结果见随交付的 PackageVerification.json。Windows 二进制架构检查不表示在 Windows 运行过。
