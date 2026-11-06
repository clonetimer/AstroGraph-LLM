/* Shared execution receipts for the full platform and both portable profiles.
 * No raw prompts, responses or credentials are rendered here. Model text is DATA.
 */
(function (global) {
  'use strict';
  const stages = { RUN_STARTED: '开始运行', INPUT_VALIDATED: '材料完整性检查通过',
    MODEL_ANALYSIS: '模型分析', FOLLOW_UP: '证据追问', SCHEMA_VALIDATED: '输出结构检查通过',
    CITATIONS_VALIDATED: '引用归属检查通过', INDEPENDENT_VERIFICATION: '独立证据审阅',
    CHUNK_ANALYSIS: '片段分析',
    CHUNK_RESTORED: '复用已完成片段', SYNTHESIS: '综合分析', REPORT_WRITE: '保存报告版本',
    RUN_FINISHED: '本次运行结束', INTERRUPTED: '运行中断' };
  const states = { RUNNING: '运行中', SUCCEEDED: '分析成功', FAILED: '未完成',
    INTERRUPTED: '已中断', NOT_CONFIGURED: '未配置模型', INPUT_TOO_LARGE: '材料超限',
    CONTEXT_TOO_LARGE: '上下文超限', STARTED: '已发起，结果尚未记录',
    RESPONSE_RECEIVED: '收到响应', REPLAYED: '读取已有结果' };
  const roles = { diagnosis_results: '诊断报告', log_reports: '补充日志', references: '参考资料' };
  const artifactNames = new Set(['report.json', 'report.md', 'conversation.json']);
  const numeric = value => Number.isSafeInteger(value) && value >= 0;
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined && text !== null) el.textContent = String(text);
    if (className) el.className = className;
    return el;
  }
  function detail(title) {
    const box = node('details'); box.append(node('summary', title)); return box;
  }
  function tokenText(usage, field) {
    const value = usage[field];
    return numeric(value) ? String(value) : '未提供完整统计（已知部分 ' + (usage['known_' + field] || 0) + '）';
  }
  function costText(cost, partial = false) {
    if (cost?.status === 'CALCULATED' && typeof cost.estimated_cost === 'string')
      return cost.currency + ' ' + cost.estimated_cost;
    if (cost?.status === 'PARTIALLY_KNOWN' && cost.known_estimated_cost)
      return '已知 ' + (cost.cost_currency || 'USD') + ' ' + cost.known_estimated_cost + ' · 总额不完整';
    const labels = {
      PRICING_NOT_CONFIGURED: '未配置计价', PRICING_SNAPSHOT_INVALID: '计价快照无效',
      PRICING_IDENTITY_INCOMPLETE: '模型身份不完整', PRICING_IDENTITY_MISMATCH: '计价模型不匹配',
      USAGE_NOT_REPORTED: '服务商未返回用量', USAGE_INCOMPLETE: '用量不完整',
      USAGE_UNKNOWN_AFTER_FAILURE: '失败调用费用未知', NO_MODEL_REQUESTS: '无模型请求',
      PARTIALLY_KNOWN: '仅部分费用可计算'
    };
    return labels[cost?.status] || '费用未知';
  }
  function requestCostText(request) {
    const value = costText(request?.cost || {});
    const version = request?.pricing?.pricing_version;
    return version && request?.cost?.status === 'CALCULATED' ? value + ' · ' + version : value;
  }
  function render(host, snapshot, options = {}) {
    host.classList.add('execution-view'); host.replaceChildren(); host.hidden = false;
    host.append(node('h2', '执行记录、材料与报告版本'));
    if (!snapshot || snapshot.legacy) {
      host.append(node('p', '旧任务或当前接口未记录执行明细；不推算历史调用次数、tokens 或费用。'));
      return;
    }
    const usage = snapshot.usage || {}, summary = node('div', null, 'execution-metrics');
    for (const [label, value] of [['已记录请求尝试', usage.http_requests || 0], ['传输失败', usage.failed_requests || 0],
      ['输入 tokens', tokenText(usage, 'input_tokens')], ['输出 tokens', tokenText(usage, 'output_tokens')],
      ['费用估算', costText(usage)],
      ['计价状态', usage.cost_status || '未知']]) {
      const cell = node('div'); cell.append(node('span', label), node('strong', value)); summary.append(cell);
    }
    host.append(summary, node('p', '费用按任务创建时冻结的计价版本和服务端实际返回的 Token 用量估算；不代表供应商最终账单。失败调用、缺失用量或未配置计价不会被当作零费用。', 'execution-note'));
    if (snapshot.unavailable_runs) host.append(node('p', '有 ' + snapshot.unavailable_runs + ' 次记录无法读取，汇总不完整。', 'execution-warning'));
    for (const run of [...(snapshot.runs || [])].reverse()) {
      const section = node('section', null, 'execution-run');
      section.append(node('h3', (run.kind === 'follow_up' ? '证据追问' : '诊断材料分析') + ' · ' + (states[run.status] || '状态未知')),
        node('p', (run.started_at || '') + ' · ' + (run.model || '未记录模型') + ' · 运行 ' + run.run_id, 'execution-note'));
      if (run.reason_code) section.append(node('p', '原因代码：' + run.reason_code, 'execution-note'));
      const materials = detail('本次导入材料 · ' + (run.materials || []).length + ' 份');
      materials.append(node('p', '这里列出本任务导入的材料。长文综合阶段使用片段摘要，原文件保持不变。', 'execution-note'));
      for (const material of run.materials || []) {
        const item = node('article', null, 'execution-material');
        item.append(node('strong', material.filename), node('p', (roles[material.role] || '材料') + ' · ' +
          material.characters + ' 字符 · ' + material.segments + ' 个片段 · ' + material.first_locator + ' — ' + material.last_locator));
        if (material.original_sha256) item.append(node('code', '原文件 SHA-256 ' + material.original_sha256));
        materials.append(item);
      }
      const timeline = detail('执行阶段 · ' + (run.events || []).length + ' 条记录'), events = node('ol');
      for (const event of run.events || []) {
        const c = event.counts || {};
        const progress = numeric(c.index) && numeric(c.total) ? ' ' + c.index + '/' + c.total : '';
        events.append(node('li', (stages[event.code] || '执行事件') + progress + ' · ' + (event.at || '')));
      }
      timeline.append(events);
      const requests = detail('模型请求 · ' + (run.requests || []).length + ' 次'), scroll = node('div', null, 'execution-table-scroll');
      const table = node('table'), head = node('tr');
      for (const title of ['序号', '阶段', '模型 / 服务商', '传输状态', 'HTTP', '耗时 ms', '输入 tokens', '输出 tokens', '费用估算']) head.append(node('th', title));
      const thead = node('thead'); thead.append(head); table.append(thead); const body = node('tbody');
      for (const request of run.requests || []) {
        const tr = node('tr');
        const values = [request.sequence, stages[request.stage] || '模型请求',
          [request.model, request.provider].filter(Boolean).join(' · ') || '未记录',
          states[request.status] || '未知', request.http_status ?? '未记录', request.latency_ms ?? '未记录',
          request.usage?.input_tokens ?? '未提供', request.usage?.output_tokens ?? '未提供', requestCostText(request)];
        for (const value of values) tr.append(node('td', value)); body.append(tr);
      }
      table.append(body); scroll.append(table); requests.append(scroll);
      const artifacts = node('div', null, 'execution-artifacts');
      const base = String(options.baseUrl || '');
      const validBase = /^\/(?:api|v1\/platform)\/jobs\/[a-zA-Z0-9_-]+$/.test(base);
      for (const artifact of run.artifacts || []) {
        if (!validBase || !/^[a-f0-9]{32}$/.test(run.run_id) || !artifactNames.has(artifact.name)) continue;
        const url = base + '/runs/' + run.run_id + '/' + artifact.name;
        const link = node('a', '下载本次版本 ' + artifact.name); link.href = url; link.download = artifact.name;
        link.title = 'SHA-256 ' + (artifact.sha256 || '') + ' · ' + artifact.bytes + ' bytes';
        if (options.download) link.onclick = async event => {
          event.preventDefault();
          try { await options.download(url, artifact.name); }
          catch (_) { artifacts.append(node('p', '下载失败：请检查登录状态、文件完整性或重新打开任务。')); }
        };
        artifacts.append(link);
      }
      section.append(artifacts, materials, timeline, requests); host.append(section);
    }
  }
  async function mount(host, options) {
    host.classList.add('execution-view'); host.replaceChildren(node('p', '正在读取执行记录…'));
    try {
      const data = await options.api(options.baseUrl + '/execution');
      if (!host.isConnected || (options.current && !options.current())) return;
      render(host, data, options);
    } catch (_) {
      if (host.isConnected && (!options.current || options.current()))
        host.replaceChildren(node('p', '执行记录暂不可用；原始报告仍可查看。'));
    }
  }
  global.AstroGraphExecution = { render, mount };
})(window);
