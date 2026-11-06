# 便携源码归档与本次共同基线

本次 P0 将已交付的 V2.7.1 与 Win7 Python3.8 源码按文件带回完整仓库；不把最小源码集合整体覆盖 src/astrograph。

源输入：
- AstroGraph-LLM-Win10-x64-V2.7.1-preview.zip：ff12ff5c565ec5086e5a73c43b9a155f6c8988b829b90804aad997fbc9f364a1（历史交付标识）。
- AstroGraph-LLM-Portable-V2.7.1-source.zip 与 AstroGraph-LLM-Win7-Python38-V2.7.1-source.zip：来自本会话实际交付附件。
- 完整仓库基线：187453966fd5e0a5ce488f81e27dc01376982a75。

V2.6/V2.7 的 RELEASE_NOTES/VERIFICATION/PROGRESS、WIN7_* 与 WINDOWS_HOST_RUNTIME_CHECK.json 是从既有交付恢复的历史记录；其中成功数字和 Windows Server 执行结果不代表本轮执行。新验证只看 PORTABLE_OBSERVABILITY_VERIFICATION.md。

本轮以 Win7 兼容导入/路径/TLS辅助适配为共同源码基线，保留仓库较新的线程接收快照、运行锁与恢复理由；前端保留 V2.7.1 六页导航和微调，版本和系统标识由运行入口决定。现有完整平台的 Mission/Legacy 入口继续保留，共用新增执行记录组件，不以便携UI覆盖它们。

本轮没有下载/更换用户原便携发行包依赖，也没有新 Win10/Win7 原生验收。
