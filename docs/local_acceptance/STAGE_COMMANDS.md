# post49 Acceptance 阶段命令

状态：`CURRENT_POST49_ACCEPTANCE_COMMANDS`

完整清单永远以命令输出为准：

```bash
python scripts/acceptance/run.py list
python scripts/acceptance/run.py commands --workspace ../AstroGraph-local/post49
```

## 常用

```bash
python scripts/acceptance/run.py init --workspace ../AstroGraph-local/post49
python scripts/acceptance/run.py run --suite core --workspace ../AstroGraph-local/post49
python scripts/acceptance/run.py report --workspace ../AstroGraph-local/post49
```

## post49 release-readiness

```bash
python scripts/acceptance/run.py run 298-post49-version-convergence --workspace ../AstroGraph-local/post49
python scripts/acceptance/run.py run 299-post49-rc-continuity --workspace ../AstroGraph-local/post49
python scripts/acceptance/run.py run 300-post49-scenario-boundary --workspace ../AstroGraph-local/post49
python scripts/acceptance/run.py run 301-post49-governance-distribution --workspace ../AstroGraph-local/post49
python scripts/acceptance/run.py run 302-post49-stable-readiness --workspace ../AstroGraph-local/post49
```

旧 stage 的名字可能包含 dev/post 标识，因为 ID/receipt 必须可复现；这不表示对应旧文档仍是当前结论。
