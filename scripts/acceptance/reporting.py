"""Static offline index: no service, script, telemetry upload or external assets."""
from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from .common import CATALOG_PATH, load, now, save


def build(workspace: Path) -> dict:
    catalog = load(CATALOG_PATH)
    config = load(workspace/'acceptance.json')
    rows = []
    for stage in catalog['stages']:
        p = workspace/'stages'/stage['id']/'receipt.json'
        r = load(p) if p.exists() else None
        rows.append({'id': stage['id'], 'title': stage['title'],
                     'status': r['status'] if r else 'NOT_RUN', 'scope': stage['scope'],
                     'summary': r.get('summary', {}) if r else {},
                     'error': r.get('error') if r else None,
                     'artifacts': r.get('artifacts', []) if r else [],
                     'receipt': str(p.relative_to(workspace)) if r else None})
    counts = {status: sum(x['status'] == status for x in rows) for status in ('PASSED', 'PARTIAL', 'BLOCKED', 'FAILED', 'NOT_RUN')}
    output = {'generated_at': now(), 'telemetry_scope': config['telemetry_scope'], 'counts': counts,
              'scope': 'LOCAL_STAGE_ACCEPTANCE_NOT_NEW_DIAGNOSTIC_PERFORMANCE', 'stages': rows}
    save(workspace/'summary.json', output)
    lines = ['# AstroGraph-Agent 本地分阶段验收', '', f'生成时间：{output["generated_at"]}', '',
             '**注意：PASSED 只表示该阶段验收条件满足。单窗口推理、合成训练、协议夹具和真实外部模型互不替代。**', '',
             '计数：' + json.dumps(counts, ensure_ascii=False), '', '| 阶段 | 状态 | 验证范围 | 结果 |', '|---|---|---|---|']
    for row in rows:
        link = f'[回执]({row["receipt"]})' if row['receipt'] else '未运行'
        lines.append(f'| {row["id"]} {row["title"]} | {row["status"]} | {row["scope"]} | {link} |')
    lines += ['', '## 怎么读结果', '',
              '11 / 12 / 13：检测阈值、定位分布、原始与约束后分类；不是 Macro-F1。',
              '20 / 21 / 22：图路径、融合贡献、初始与补证后诊断。',
              '32 / 35：带来源检索结果、待审核图谱；不等于实时故障证据。',
              '40 / 41：候选梯度更新和三头推理；不覆盖活动模型。',
              '50 / 51 / 55：有真值评价侧才能产生三任务指标，保留不可用和未决分母。',
              '同阶段重跑需 --retry，会归档旧尝试；改输入/配置使用新工作区。']
    (workspace/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    e = html.escape
    blocks = []
    for row in rows:
        files = ''.join(f'<li><a href="{quote(item["path"])}">{e(item["path"].split("/",2)[-1])}</a> '
                        f'({item["bytes"]:,} bytes)</li>' for item in row['artifacts'])
        details = json.dumps({'summary': row['summary'], 'error': row['error']}, ensure_ascii=False, indent=2)
        blocks.append(f'<section id="{e(row["id"])}"><h2>{e(row["id"])} · {e(row["title"])}</h2>'
                      f'<p><strong>{e(row["status"])}</strong> — {e(row["scope"])}</p>'
                      f'<details><summary>查看摘要与错误</summary><pre>{e(details)}</pre></details>'
                      f'<details><summary>结果文件（{len(row["artifacts"])}）</summary><ul>{files or "未生成文件"}</ul></details></section>')
    nav = ''.join(f'<tr><td><a href="#{e(r["id"])}">{e(r["id"])} {e(r["title"])}</a></td><td>{e(r["status"])}</td><td>{e(r["scope"])}</td></tr>' for r in rows)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>AstroGraph-Agent 本地验收</title><style>body{max-width:1120px;margin:40px auto;padding:0 20px;font:16px/1.65 system-ui}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #ccc;text-align:left;padding:9px}pre{white-space:pre-wrap;word-break:break-word}section{border-top:2px solid #ccc;margin-top:30px}summary{cursor:pointer}li{overflow-wrap:anywhere}</style>'
    page += '<h1>AstroGraph-Agent 本地分阶段验收</h1><p>离线页面；点击阶段，再打开独立 JSON / CSV / 日志。</p>'
    page += '<p><strong>通过不等于性能提高。单窗口推理、合成训练、协议夹具、真实服务和完整评价分开记录。</strong></p>'
    page += '<pre>' + e(json.dumps(counts, ensure_ascii=False, indent=2)) + '</pre><table><thead><tr><th>阶段</th><th>状态</th><th>范围</th></tr></thead><tbody>' + nav + '</tbody></table>'
    page += ''.join(blocks) + '</html>'
    (workspace/'index.html').write_text(page, encoding='utf-8')
    return output
