# AstroGraph post49 当前工程与治理策略

状态：`CURRENT_POST49_ENGINEERING_POLICY`

本文件取代旧 `DEV_GOAL.md` 作为稳定版当前工程策略入口。旧开发目标已归档，仅保留历史意义。

## 1. 框架

遥测/时序 → 数据质量与输入合同 → 检测 → 定位 → 分类 → Data-driven ranking → KG forward/backward → BiRRF/Decision Governor → committed diagnosis → non-voting literature/report。

## 2. 诊断语义

- Detection：是否异常；
- Localization：受影响部件/分系统；不自动等于根因；
- Classification：故障类型；
- Root cause：需要独立因果/机理证据，不由分类概率或 KG 路径自动确认。

## 3. 数据与评价

- truth/label/split/case identity 不得泄漏到推理特征；
- grouped/trajectory-level 独立性高于随机行切分；
- used replay 不能重命名为新独立确认；
- prediction-before-truth 协议用于新的 confirmatory evaluation；
- 不通过重切旧组制造新 generalization claim。

## 4. 执行权限

Capability discovery ≠ execution authorization。模型执行必须满足：host binding、artifact hash、channel order、unit/shape contract、provider admission。

## 5. 文献与知识

- 普通 RAG / reviewed claim 默认 non-voting；
- Anything2Ontology / knowledge extraction 生成 proposal，不直接更新 Active KG；
- 文献进入正式诊断投票前，需要独立 decision truth + governed promotion；
- 文献库、memory candidate、knowledge proposal、Active KG 分层保存。

## 6. 专家反馈与进化

允许：feedback → failure case → improvement candidate → sealed challenger → offline evaluation → promotion review packet → rollback-covered preflight。

禁止：无治理在线自改生产源码、模型、Active KG、confirmed memory 或工具权限。

## 7. post49 外部资格依赖

以 `governance/implementation/post49/RELEASE_READINESS.json` 为准。稳定软件发布不建立飞行/安全/跨卫星/任意 profile 的新资格。
