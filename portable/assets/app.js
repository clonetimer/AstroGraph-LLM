/* Focused portable client. All model/user content is inserted through text nodes. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const MAX_BYTES = 4 * 1024 * 1024;
  const ROLE_LIMITS = { diagnosis_results: 8, log_reports: 12, references: 16 };
  const ROLE_LABELS = { diagnosis_results: '诊断报告', log_reports: '补充日志', references: '参考资料' };
  const STORAGE_KEY = 'astrograph.portable.public-settings.v1';
  const SETTINGS = ['mode', 'preset', 'base-url', 'model', 'max-tokens', 'timeout', 'strategy'];
  const ACTIVE = new Set(['RUNNING', 'QUEUED', 'CANCEL_REQUESTED']);
  const STATES = { QUEUED: '等待队列中', CANCEL_REQUESTED: '正在取消', CANCELLED: '已取消', RUNNING: '分析进行中', SUCCEEDED: '报告已完成', FAILED: '任务失败', INTERRUPTED: '任务已中断' };
  const REASONS = {
    MODEL_GENERATION_FAILED: '模型调用失败。请检查模型是否已加载、接口配置及超时设置。',
    MODEL_OUTPUT_INVALID: '模型输出未通过结构检查，本次没有可展示的模型分析。',
    CITATION_REFERENCE_INVALID: '模型引用了本任务之外的来源，引用检查未通过。',
    LLM_NOT_CONFIGURED: '没有配置可用模型。请填写当前模型 ID 与连接信息。',
    MODEL_CONFIGURATION_UNAVAILABLE: '模型配置不可用，请重新核对当前连接信息。',
    INPUT_CHARACTER_LIMIT_EXCEEDED: '解析后的材料超过 180,000 字容量，请拆分材料后新建任务。',
    CONTEXT_CHARACTER_LIMIT_EXCEEDED: '对话上下文已超过容量，请新建分析任务。',
    FOLLOW_UP_BUSY: '本任务已有追问正在处理，请稍后重新读取记录。',
    FOLLOW_UP_REQUEST_CONFLICT: '请求编号与问题内容冲突，请重新读取记录。',
    FOLLOW_UP_INPUT_CHANGED: '输入或原始报告发生变化，请重新打开该任务。',
    FOLLOW_UP_LIMIT_REACHED: '本任务已达到 12 次追问记录上限。',
    FOLLOW_UP_HISTORY_INVALID: '对话历史完整性检查未通过，请重新打开任务。',
    FOLLOW_UP_REPORT_NOT_READY: '原始报告尚未准备好，请等待任务完成。',
    SYNTHESIS_CONTEXT_TOO_LARGE: '分块分析已保存，但综合摘要超出容量。请按事件拆分材料后新建任务；没有截断摘要。',
    CHUNK_ANALYSIS_FAILED: '部分片段分析未完成。检查模型连接后点击“继续 / 重试分析”，已完成的片段会被复用。',
    SYNTHESIS_FAILED: '最终综合分析未完成。检查模型连接后可重试，不必重新导入材料。',
    CONTEXT_TOO_LARGE: '对话上下文超出容量，请新建分析任务。'
  };
  const state = { token: null, session: null, files: [], busy: false, job: null, report: null,
    submitting: false, contexts: null, sources: new Map(), conversation: null, epoch: 0, timer: null, attempt: null, jobs: [] };

  const navigate = route => window.AstroGraphShell.navigate(route);

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined && text !== null) el.textContent = String(text);
    return el;
  }
  function say(text, kind = '') {
    const el = $('global-message'); el.textContent = text;
    el.className = 'notice' + (kind ? ' ' + kind : ''); el.hidden = !text;
  }
  function setBadge(id, text, kind = 'neutral') {
    const el = $(id); el.textContent = text; el.className = 'badge ' + kind;
    if (id === 'connection-badge') updateConnectionSummary();
  }
  function readableError(error) {
    return REASONS[error.code] || error.message || '操作未完成，请检查本机工作台状态。';
  }
  async function uploadBackup(file) {
    if (!state.token) throw new Error('本机工作台尚未就绪，请刷新页面。');
    if (!file || file.size < 1 || file.size > 64 * 1024 * 1024) throw new Error('备份文件为空或超过 64 MiB。');
    const response = await fetch('/api/restore', { method: 'POST', cache: 'no-store', credentials: 'same-origin',
      headers: { 'Content-Type': 'application/zip', 'X-AstroGraph-Token': state.token }, body: file });
    let data; try { data = await response.json(); } catch (_) { throw new Error('恢复接口没有返回有效结果。'); }
    if (!response.ok) { const error = new Error(data.error?.message || '备份恢复失败。'); error.code = data.error?.code; throw error; }
    return data;
  }
  async function api(path, body) {
    const options = { cache: 'no-store', credentials: 'same-origin', headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      if (!state.token) throw new Error('本机工作台尚未就绪，请刷新页面。');
      options.method = 'POST'; options.headers['Content-Type'] = 'application/json';
      options.headers['X-AstroGraph-Token'] = state.token; options.body = JSON.stringify(body);
    }
    let response;
    try { response = await fetch(path, options); }
    catch (_) { throw new Error('无法连接本机工作台。请确认启动窗口仍在运行，然后刷新页面。'); }
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error('本机工作台没有返回有效数据，请重新读取任务。'); }
    if (!response.ok) {
      const error = new Error(data.error?.message || '操作未完成，请检查配置与材料。');
      error.code = data.error?.code; throw error;
    }
    return data;
  }
  function setBusy(busy) {
    state.busy = busy; document.body.classList.toggle('busy', busy);
    for (const id of ['analyze', 'probe', 'history', 'refresh-history', 'send-question', 'reload-conversation', 'resume-job', 'restore-trigger']) {
      $(id).disabled = busy || !state.token;
    }
    const canQueue = busy && ACTIVE.has(state.job?.status) && $('queue-analysis').checked;
    $('analyze').disabled = !state.token || state.submitting || (busy && !canQueue);
    $('analyze').textContent = canQueue ? '加入等待队列 →' : busy ? '模型操作进行中…' : '开始分析 →';
    for (const button of document.querySelectorAll('.history-open')) button.disabled = busy || !state.token;
    $('active-operation').hidden = !busy;
    $('active-operation-text').textContent = state.job?.status === 'RUNNING' ? '当前报告正在分析，可切换页面查看。' : '模型或文件操作进行中，请勿重复提交。';
    updateTurnLimit();
  }
  function phaseSettings() {
    const result = {};
    for (const [stage, id] of [['chunk','phase-chunk'],['synthesis','phase-synthesis'],['follow_up','phase-follow-up']]) {
      const model = $(id).value.trim(); if (model) result[stage] = { model };
    }
    return Object.keys(result).length ? { phases: result } : {};
  }
  function contextSelection() {
    const selected = state.contexts?.selection();
    return selected ? JSON.parse(JSON.stringify(selected)) : null;
  }
  function publicSettings() { return Object.fromEntries(SETTINGS.map(id => [id, $(id).value])); }
  function updateConnectionSummary() {
    const model = $('model').value.trim();
    $('analysis-model-summary').textContent = (model || '尚未填写模型 ID') + ' · ' + ($('mode').value === 'local' ? '本地模型' : '在线 API');
    const status = $('analysis-model-status'), tested = $('connection-badge').textContent;
    status.textContent = !model ? '未配置模型' : tested === '已连通' ? '列表接口已连通'
      : tested === '测试中' ? '正在测试连接' : tested === '测试未通过' ? '连接测试未通过' : '已填写 · 未测试';
    status.className = 'badge ' + (tested === '测试未通过' && model ? 'error' : !model ? 'warning' : 'neutral');
  }
  function saveSettings() {
    updateConnectionSummary();
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(publicSettings())); } catch (_) { /* Memory-only if unavailable. */ }
  }
  function connection(requireModel = true) {
    const mode = $('mode').value, base = $('base-url').value.trim(), model = $('model').value.trim();
    let url;
    try { url = new URL(base); } catch (_) { throw new Error('请输入完整的 API 基础地址。'); }
    if (base.length > 2000 || /[\\\x00-\x1f]/.test(base) || url.username || url.password || url.search || url.hash) throw new Error('API 基础地址不能包含用户名、密码、查询参数、反斜杠或控制字符。');
    if (mode === 'local') {
      if (!['http:', 'https:'].includes(url.protocol) ||
          !/^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/|$)/i.test(base)) {
        throw new Error('本地模式仅支持本机回环地址，例如 http://127.0.0.1:1234/v1。');
      }
    } else if (url.protocol !== 'https:') throw new Error('在线 API 模式必须使用 HTTPS 地址。');
    if (requireModel && !model) throw new Error('请填写模型 ID；可先测试连接并读取模型列表。');
    if (model.length > 300 || $('api-key').value.length > 4096 || /[\r\n\x00]/.test($('api-key').value)) throw new Error('模型 ID 或 API Key 格式不正确，请重新输入。');
    const maxTokens = Number($('max-tokens').value), timeout = Number($('timeout').value);
    if (!Number.isInteger(maxTokens) || maxTokens < 128 || maxTokens > 16384) throw new Error('最大输出 tokens 请输入 128 至 16384 的整数。');
    if (!Number.isFinite(timeout) || timeout < 10 || timeout > 600) throw new Error('超时请输入 10 至 600 秒。');
    saveSettings();
    return { mode, base_url: base.replace(/\/+$/, ''), model, api_key: $('api-key').value,
      max_tokens: maxTokens, timeout_s: timeout, response_format_strategy: $('strategy').value };
  }
  function updateMode() {
    const local = $('mode').value === 'local';
    $('mode-help').textContent = local
      ? '需先在本机启动 LM Studio 或 Ollama，并加载模型。模型与服务软件需自行准备。'
      : '分析时会把选定材料与问题发送至下方 API。请确认服务地址及材料使用范围。';
    $('send-disclosure').textContent = local
      ? '材料将发送给所配置的本地模型服务。'
      : '开始分析将把选定材料发送至所配置的在线 API。';
    $('preset').disabled = !local;
    updateConnectionSummary();
  }
  function resetConnectionBadge() {
    setBadge('connection-badge', '未测试');
    $('probe-status').textContent = '连接测试只读取模型列表，不生成分析。';
  }
  function applyPreset() {
    if ($('preset').value === 'lmstudio') $('base-url').value = 'http://127.0.0.1:1234/v1';
    if ($('preset').value === 'ollama') $('base-url').value = 'http://127.0.0.1:11434/v1';
    resetConnectionBadge(); saveSettings();
  }
  function sizeText(bytes) {
    return bytes < 1024 ? bytes + ' B' : bytes < 1024 * 1024 ? (bytes / 1024).toFixed(1) + ' KiB' : (bytes / (1024 * 1024)).toFixed(2) + ' MiB';
  }
  function inputStats() {
    const paste = $('paste').value;
    const total = state.files.reduce((sum, item) => sum + item.file.size, 0) + new TextEncoder().encode(paste).length;
    $('paste-count').textContent = paste.length.toLocaleString() + ' 字';
    $('input-stats').textContent = (state.files.length + (paste.trim() ? 1 : 0)) + ' 份材料 · ' + sizeText(total) + ' / 4 MiB';
    $('input-stats').classList.toggle('over-limit', total > MAX_BYTES);
    return total;
  }
  function selectControl(options, value, onChange, label) {
    const select = node('select'); select.setAttribute('aria-label', label);
    for (const [id, title] of Object.entries(options)) {
      const option = node('option', '', title); option.value = id; option.selected = id === value; select.append(option);
    }
    select.addEventListener('change', () => onChange(select.value)); return select;
  }
  function renderFiles() {
    $('file-list').replaceChildren();
    for (const item of state.files) {
      const li = node('li', 'file-item'), row = node('div', 'file-row');
      const name = node('span', 'file-name', item.file.name); name.title = item.file.name;
      const remove = node('button', 'remove-file', '×'); remove.type = 'button'; remove.setAttribute('aria-label', '移除 ' + item.file.name);
      remove.onclick = () => { state.files = state.files.filter(current => current !== item); renderFiles(); };
      row.append(name, node('span', 'file-size', sizeText(item.file.size)), remove);
      const options = node('div', 'file-options');
      const allowedRoles = /\.log$/i.test(item.file.name) ? { log_reports: ROLE_LABELS.log_reports }
        : /\.(txt|md|pdf|docx)$/i.test(item.file.name) ? ROLE_LABELS
        : { diagnosis_results: ROLE_LABELS.diagnosis_results, log_reports: ROLE_LABELS.log_reports };
      options.append(selectControl(allowedRoles, item.role, value => { item.role = value; }, item.file.name + ' 材料角色'));
      if (!/\.(pdf|docx)$/i.test(item.file.name)) {
        const advanced = node('details', 'file-advanced'), summary = node('summary');
        const encodingNames = { auto: '自动', 'utf-8': 'UTF-8', gb18030: 'GB18030' };
        const updateLabel = () => { summary.textContent = '编码：' + encodingNames[item.encoding]; };
        updateLabel(); advanced.open = !!item.encodingAdvancedOpen;
        advanced.addEventListener('toggle', () => { item.encodingAdvancedOpen = advanced.open; });
        advanced.append(summary, selectControl({ auto: '自动 / UTF-8 / BOM', 'utf-8': 'UTF-8', gb18030: 'GB18030' },
          item.encoding, value => { item.encoding = value; updateLabel(); }, item.file.name + ' 文件编码'));
        options.append(advanced);
      } else {
        options.append(node('span', 'document-encoding-note', '文档自动解析'));
      }
      li.append(row, options); $('file-list').append(li);
    }
    inputStats();
  }
  function safeFilename(name) {
    return name.length <= 120 && name === name.trim() && !/[<>:"/\\|?*\x00-\x1f]/.test(name) && !/[. ]$/.test(name) &&
      !/^(con|prn|aux|nul|com[0-9¹²³]|lpt[0-9¹²³])(?:\.|$)/i.test(name) &&
      name.toLocaleLowerCase() !== 'analysis_input.json' && name !== '.' && name !== '..';
  }
  function addFiles(files) {
    const names = new Set(state.files.map(item => item.file.name.toLocaleLowerCase()));
    const errors = [];
    for (const file of files) {
      const key = file.name.toLocaleLowerCase();
      if (!/\.(txt|md|json|csv|log|pdf|docx)$/i.test(file.name)) { errors.push(file.name + '：仅支持 TXT、MD、JSON、CSV、LOG、PDF、DOCX。'); continue; }
      if (!safeFilename(file.name)) { errors.push(file.name + '：文件名不适用于 Windows，请重命名后重试。'); continue; }
      if (names.has(key)) { errors.push(file.name + '：名称重复，请重命名后重试。'); continue; }
      if (!file.size) { errors.push(file.name + '：文件为空。'); continue; }
      if (state.files.length >= 36) { errors.push('最多添加 36 份文件，请按材料角色上限整理。'); break; }
      state.files.push({ file, role: /\.log$/i.test(file.name) ? 'log_reports' : 'diagnosis_results', encoding: 'auto' }); names.add(key);
    }
    renderFiles(); say(errors.join('\n'), errors.length ? 'error' : '');
  }
  function base64(bytes) {
    let binary = '';
    for (let i = 0; i < bytes.length; i += 16384) binary += String.fromCharCode(...bytes.subarray(i, i + 16384));
    return btoa(binary);
  }
  async function materials() {
    if (inputStats() > MAX_BYTES) throw new Error('选定材料总大小超过 4 MiB，请拆分后分析；不会自动截断。');
    const entries = state.files.map(item => ({ ...item }));
    const paste = $('paste').value;
    if (paste.trim()) {
      const names = new Set(entries.map(item => item.file.name.toLocaleLowerCase()));
      let name = 'pasted-report.txt', n = 2;
      while (names.has(name.toLocaleLowerCase())) name = 'pasted-report-' + n++ + '.txt';
      entries.push({ file: new File([paste], name, { type: 'text/plain' }), role: $('paste-role').value, encoding: 'utf-8' });
    }
    const counts = { diagnosis_results: 0, log_reports: 0, references: 0 };
    for (const item of entries) counts[item.role]++;
    if (!counts.diagnosis_results) throw new Error('请提供至少一份角色为“诊断报告”的材料。');
    for (const role of Object.keys(counts)) if (counts[role] > ROLE_LIMITS[role]) {
      throw new Error(ROLE_LABELS[role] + '最多 ' + ROLE_LIMITS[role] + ' 份（含粘贴文本），请调整角色或拆分材料。');
    }
    return Promise.all(entries.map(async item => ({ filename: item.file.name, role: item.role, encoding: item.encoding,
      content_base64: base64(new Uint8Array(await item.file.arrayBuffer())) })));
  }
  function probeIdentity() {
    // Snapshot remains in memory only; it must never enter logs or saved settings.
    return JSON.stringify([$('mode').value, $('base-url').value.trim(), $('model').value.trim(), $('api-key').value]);
  }
  function probeSettingsChanged() {
    resetConnectionBadge();
    $('probe-status').textContent = '配置已改变，请重新测试当前连接。';
  }
  async function probe() {
    let identity;
    try {
      const config = connection(false); identity = probeIdentity(); setBusy(true); say(''); setBadge('connection-badge', '测试中', 'running');
      $('probe-status').textContent = '正在读取模型列表，本次不会生成分析…';
      const data = await api('/api/probe', config);
      if (identity !== probeIdentity()) { probeSettingsChanged(); return; }
      const models = (data.models || []).filter(model => typeof model.id === 'string');
      $('models').replaceChildren(...models.map(model => { const option = node('option'); option.value = model.id; return option; }));
      if (!$('model').value.trim() && models.length) { $('model').value = models[0].id; saveSettings(); }
      setBadge('connection-badge', '已连通', 'success');
      $('probe-status').textContent = models.length
        ? '连接测试成功，读取到 ' + models.length + ' 个模型 ID。请核对所选模型；生成能力需在分析时验证。'
        : '接口连接成功，模型列表为空。请在服务端加载模型并填写模型 ID。';
    } catch (error) {
      if (identity !== undefined && identity !== probeIdentity()) { probeSettingsChanged(); }
      else { setBadge('connection-badge', '测试未通过', 'error'); $('probe-status').textContent = readableError(error); say(readableError(error), 'error'); }
    } finally { setBusy(false); }
  }
  function formatDate(value) {
    const date = new Date(value); return Number.isNaN(date.getTime()) ? String(value || '') : date.toLocaleString('zh-CN', { hour12: false });
  }
  async function refreshHistory() {
    const data = await api('/api/jobs');
    const selected = state.job?.job_id || $('history').value;
    $('history').replaceChildren(node('option', '', '选择历史任务…')); $('history').firstChild.value = '';
    state.jobs = Array.isArray(data.jobs) ? data.jobs : [];
    for (const job of state.jobs) {
      const option = node('option', '', formatDate(job.created_at) + ' · ' + (STATES[job.status] || job.status) + ' · ' + String(job.task || job.job_id).slice(0, 45));
      option.value = job.job_id; $('history').append(option);
    }
    $('history').value = selected;
    renderHistory();
  }
  function renderHistory() {
    const query = $('history-search').value.trim().toLocaleLowerCase();
    const status = $('history-status-filter').value;
    const jobs = state.jobs.filter(job => (!status || job.status === status) &&
      [job.task, job.job_id, job.model].some(value => String(value || '').toLocaleLowerCase().includes(query)));
    $('history-items').replaceChildren();
    $('history-count').textContent = '显示 ' + jobs.length + ' / ' + state.jobs.length + ' 项本机任务';
    $('history-empty').hidden = jobs.length > 0;
    $('history-empty').querySelector('h3').textContent = state.jobs.length ? '没有匹配的任务' : '暂无历史任务';
    $('history-empty').querySelector('p').textContent = state.jobs.length ? '请调整关键词或状态筛选。' : '完成一次分析后，任务会保存在这里。';
    for (const job of jobs) {
      const item = node('article', 'card history-item'), summary = node('div', 'history-summary');
      summary.append(node('h3', '', job.task || '外部诊断分析'),
        paragraph(formatDate(job.created_at) + ' · ' + (job.model || '未配置模型'), 'small muted'),
        paragraph(job.job_id, 'history-id'));
      const badge = node('span', 'badge history-status ' + (job.status === 'RUNNING' ? 'running' : job.status === 'FAILED' ? 'error' : 'neutral'), STATES[job.status] || job.status);
      const open = node('button', 'button secondary history-open', '打开任务'); open.type = 'button';
      open.disabled = state.busy || !state.token;
      open.onclick = () => { if (!state.busy) openJob(job.job_id); };
      item.append(summary, badge, open); $('history-items').append(item);
    }
  }
  function clearPoll() { if (state.timer) clearTimeout(state.timer); state.timer = null; }
  function knowledge(report) { return report?.analysis?.knowledge_enrichment || {}; }
  function inputHash() { return knowledge(state.report).model_explanation?.input_sha256 || state.report?.input_sha256 || state.conversation?.input_sha256; }
  function citationRow(ids, label = '依据') {
    const row = node('div', 'citation-row');
    if (!Array.isArray(ids) || !ids.length) return row;
    row.append(node('span', 'citation-label', label));
    for (const id of ids) {
      const button = node('button', 'citation-button', id); button.type = 'button';
      const source = state.sources.get(id);
      button.title = source ? source.filename + ' · ' + source.locator : '此引用来源不可用'; button.disabled = !source;
      button.onclick = () => {
        const target = document.getElementById(source.domId); if (!target) return;
        navigate('result');
        target.open = true; target.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
        target.classList.add('highlight'); setTimeout(() => target.classList.remove('highlight'), 2800);
      };
      row.append(button);
    }
    return row;
  }
  function card(title) { const el = node('section', 'card'); el.append(node('h3', '', title)); return el; }
  function paragraph(text, className = 'body-text') { return node('p', className, text); }
  function list(items, emptyText = '模型未列出此项。') {
    if (!Array.isArray(items) || !items.length) return paragraph(emptyText, 'small muted');
    const ul = node('ul', 'plain-list'); for (const item of items) ul.append(node('li', '', item)); return ul;
  }
  function detail(title, value) {
    if (!value) return document.createDocumentFragment();
    const el = node('div', 'detail-field'); el.append(node('strong', '', title), document.createTextNode(String(value))); return el;
  }
  function renderSources(sources) {
    state.sources.clear(); $('source-list').replaceChildren();
    for (const [index, source] of (Array.isArray(sources) ? sources : []).entries()) {
      const domId = 'source-' + index;
      state.sources.set(source.source_id, { ...source, domId });
      const item = node('details', 'source'); item.id = domId;
      const summary = node('summary'); summary.append(node('strong', '', source.filename), document.createTextNode(' · ' + (source.locator || '')), node('span', 'source-id', source.source_id));
      item.append(summary, node('pre', '', source.text), node('p', 'source-meta', (ROLE_LABELS[source.role] || source.role || '') + ' · SHA-256 ' + (source.sha256 || '')));
      $('source-list').append(item);
    }
    $('source-count').textContent = state.sources.size + ' 条';
    if (!state.sources.size) $('source-list').append(node('p', 'source-empty', '当前报告未提供可用的原文来源。'));
  }
  function renderAnalysis(model) {
    const host = $('analysis-content'); host.replaceChildren();
    const summary = card('分析摘要'); summary.append(paragraph(model.summary, 'body-text summary-text')); host.append(summary);
    if (!model.integrated_conclusion) summary.classList.add('report-overview');
    const findings = card('对诊断材料的解读');
    for (const finding of model.findings || []) {
      const section = node('div', 'finding'); section.append(paragraph(finding.text), citationRow(finding.citation_ids)); findings.append(section);
    }
    if (!model.findings?.length) findings.append(paragraph('模型未列出解读条目。', 'small muted')); host.append(findings);
    const hypotheses = card('原因假设与反向证据');
    for (const [index, hypothesis] of (model.hypotheses || []).entries()) {
      const section = node('article', 'hypothesis');
      section.append(node('h4', '', String(index + 1).padStart(2, '0') + ' · ' + hypothesis.title), paragraph(hypothesis.analysis), citationRow(hypothesis.citation_ids), citationRow(hypothesis.opposing_citation_ids, '反向依据'));
      section.append(detail('反向解释与限制', hypothesis.counterpoints), detail('不确定性', hypothesis.uncertainty), detail('区分性核查', hypothesis.discriminating_check));
      if (hypothesis.assumptions?.length) section.append(node('h4', '', '前提假设'), list(hypothesis.assumptions)); hypotheses.append(section);
    }
    if (!model.hypotheses?.length) hypotheses.append(paragraph('模型未提出可据材料继续分析的原因假设。', 'small muted')); host.append(hypotheses);
    const gaps = card('仍缺少的信息'); gaps.append(list(model.information_gaps)); host.append(gaps);
    const common = model.common_findings || [], disagreements = model.report_disagreements || [];
    if (common.length || disagreements.length) {
      const comparisonGrid = node('div', 'comparison-grid');
      for (const [title, items] of [['跨报告共识', common], ['报告间分歧', disagreements]]) {
        const comparison = card(title);
        for (const item of items) {
          const section = node('div', 'finding'); section.append(paragraph(item.text), citationRow(item.citation_ids)); comparison.append(section);
        }
        if (!items.length) comparison.append(paragraph('本次材料未形成此类条目。', 'small muted'));
        comparisonGrid.append(comparison);
      }
      host.append(comparisonGrid);
    }
    for (const [title, field] of [['互补证据', 'complementary_evidence'], ['事件与时间对齐', 'event_alignment']]) {
      const items = model[field] || [];
      if (!items.length) continue;
      const comparison = card(title);
      for (const item of items) {
        const section = node('div', 'finding'); section.append(paragraph(item.text), citationRow(item.citation_ids)); comparison.append(section);
      }
      host.append(comparison);
    }
    if (model.integrated_conclusion) {
      const integrated = card('多报告综合结论'); integrated.classList.add('report-overview'); integrated.append(paragraph(model.integrated_conclusion, 'body-text summary-text')); host.prepend(integrated);
    }
    const actions = card('建议核查'); actions.append(list(model.recommended_review_actions));
    for (const action of model.review_actions || []) {
      const section = node('div', 'finding'); section.append(node('h4', '', action.check), detail('核查目的', action.rationale), detail('预期观察', action.expected_observations), citationRow(action.citation_ids)); actions.append(section);
    }
    actions.classList.add('report-next-steps');
    host.querySelector('.report-overview').after(actions);
  }
  function renderReport(job, report) {
    $('empty-state').hidden = true; $('result').hidden = false;
    window.AstroGraphContexts.renderSnapshot($('context-snapshot'), report?.reviewed_context);
    $('job-task').textContent = job.task || '外部诊断分析';
    $('job-meta').textContent = formatDate(job.created_at) + ' · ' + (job.mode === 'online' ? '在线 API' : '本地模型') + ' · ' + (job.model || '未配置模型') + ' · ' + job.job_id;
    const path = '/api/jobs/' + encodeURIComponent(job.job_id);
    $('download-md').href = path + '/report.md'; $('download-json').href = path + '/report.json'; $('download-conversation').href = path + '/conversation.json';
    const ready = job.status === 'SUCCEEDED' && report;
    $('download-md').hidden = !ready; $('download-json').hidden = !ready;
    $('resume-job').hidden = job.status !== 'INTERRUPTED' && !job.resumable;
    $('resume-job').textContent = '继续 / 重试分析';
    const k = knowledge(report), model = k.model_explanation || {}, generation = k.generation || {};
    const successful = ready && generation.status === 'SUCCEEDED' && model.status === 'AVAILABLE_REQUIRES_EXPERT_REVIEW' && model.citation_reference_validation === 'PASS';
    renderSources(model.sources || []);
    const status = $('result-status'); status.className = 'notice' + (successful ? '' : ' warning');
    if (successful) {
      status.textContent = '模型分析已生成 · 来源归属检查通过 · 需专家复核'; renderAnalysis(model);
    } else {
      $('analysis-content').replaceChildren();
      const message = job.status === 'CANCELLED' ? '任务已取消，原始材料和已完成片段仍保留。需要时可核对原配置后手动重试。' : job.status === 'INTERRUPTED' ? '任务因工作台停止而中断。核对原模型连接后，点击“继续 / 重试分析”；已完成片段将从检查点恢复。'
        : job.status === 'FAILED' ? '任务未完成。请核对材料与模型连接；可续跑时使用下方“继续 / 重试分析”。'
        : REASONS[generation.reason_code] || (generation.status === 'NOT_CONFIGURED' ? '原始材料已导入，但尚未配置模型，本次没有生成模型分析。'
          : generation.status === 'INPUT_TOO_LARGE' ? '原始材料已导入，但超过模型分析容量，本次未调用模型。请拆分材料。'
          : '报告已保存，但模型调用或输出校验未成功，本次没有可展示的模型分析。');
      status.textContent = message + (generation.reason_code ? '\n原因代码：' + generation.reason_code : '');
      const info = card('本次分析状态'); info.append(paragraph(message), paragraph('可查看右侧原始材料；请到“模型设置”检查配置，再继续 / 重试分析。原始材料不会被静默截断。', 'small muted')); $('analysis-content').append(info);
    }
    $('follow-up').hidden = !ready;
    $('chat-empty').hidden = !!ready;
    $('chat-context').hidden = !ready;
    $('question-job-title').textContent = job.task || job.job_id;
    setBadge('job-badge', successful ? '模型分析已生成' : STATES[job.status] || job.status, successful ? 'success' : 'warning');
  }
  async function loadConversation(jobId, epoch) {
    try {
      const data = await api('/api/jobs/' + encodeURIComponent(jobId) + '/conversation');
      if (epoch !== state.epoch || state.job?.job_id !== jobId) return;
      acceptConversation(data); $('follow-up-status').textContent = '使用“模型设置”中的连接信息；重启后需重新输入 API Key。Ctrl + Enter 发送。';
    } catch (error) {
      if (epoch === state.epoch) { $('follow-up-status').textContent = readableError(error); $('send-question').disabled = true; }
    }
  }
  async function fetchJob(id, epoch) {
    const data = await api('/api/jobs/' + encodeURIComponent(id));
    if (epoch !== state.epoch) return;
    state.job = data.job; state.report = data.report; $('history').value = id;
    window.AstroGraphExecution.render($('execution-panel'), data.execution, {baseUrl:'/api/jobs/'+encodeURIComponent(id)});
    $('cancel-job').hidden = !ACTIVE.has(data.job.status);
    $('cancel-job').disabled = data.job.status === 'CANCEL_REQUESTED';
    if (ACTIVE.has(data.job.status)) {
      $('empty-state').hidden = true; $('result').hidden = true; $('job-progress').hidden = false;
      const progress = data.job.progress || null;
      if (progress?.stage === 'CHUNK_ANALYSIS') {
        $('progress-title').textContent = '正在分块分析 ' + progress.completed + ' / ' + progress.total;
        $('progress-copy').textContent = '按原文位置逐段分析，保留来源引用；全部片段完成后会继续综合分析。';
      } else if (progress?.stage === 'SYNTHESIS') {
        $('progress-title').textContent = '正在综合多段分析';
        $('progress-copy').textContent = '正在对齐不同片段的共识、冲突和互补证据。';
      } else {
        $('progress-title').textContent = '正在分析材料';
        const progress = data.job.progress;
      $('progress-copy').textContent = progress?.stage === 'CHUNK_ANALYSIS'
        ? `长报告分块分析：已完成 ${progress.completed}/${progress.total} 个片段。完成后将进行综合分析。`
        : progress?.stage === 'CHUNK_RESTORED'
          ? `正在恢复分块检查点：已保存 ${progress.completed}/${progress.total} 个片段。`
          : '正在等待模型响应与结构检查。完成后会自动显示结果。';
      }
      if (data.job.status !== 'RUNNING') {
        $('progress-title').textContent = STATES[data.job.status];
        $('progress-copy').textContent = data.job.status === 'QUEUED' ? '任务已保存；前序任务结束后执行。可取消等待任务。' : '等待当前请求结束后停止；已发出的模型调用可能产生费用。';
      }
      setBadge('job-badge', STATES[data.job.status], 'running'); setBusy(true);
      state.timer = setTimeout(() => fetchJob(id, epoch).catch(error => pollFailed(error, epoch)), 1400);
    } else {
      $('job-progress').hidden = true; setBusy(false); renderReport(data.job, data.report);
      await refreshHistory();
      if (epoch === state.epoch && data.job.status === 'SUCCEEDED' && data.report) await loadConversation(id, epoch);
    }
  }
  function pollFailed(error, epoch) {
    if (epoch !== state.epoch) return;
    clearPoll(); setBusy(false); $('job-progress').hidden = true;
    say(readableError(error) + ' 当前任务状态尚未确认；可从历史列表重新打开任务。', 'error');
  }
  async function openJob(id) {
    if (!id) return;
    navigate('result');
    clearPoll(); const epoch = ++state.epoch;
    $('execution-panel').hidden = true; $('execution-panel').replaceChildren();
    state.conversation = null; state.attempt = null; state.report = null; state.sources.clear(); $('question').value = ''; $('turns').replaceChildren(); $('follow-up').hidden = true;
    $('chat-empty').hidden = false; $('chat-context').hidden = true;
    say(''); setBusy(true); $('result').hidden = true; $('empty-state').hidden = true;
    $('job-progress').hidden = false; $('progress-title').textContent = '正在读取报告'; $('progress-copy').textContent = '正在从本机载入任务与原始材料。';
    try { await fetchJob(id, epoch); } catch (error) { pollFailed(error, epoch); }
  }
  async function analyze() {
    const queued = $('queue-analysis').checked;
    if (state.submitting || (state.busy && !(queued && ACTIVE.has(state.job?.status)))) return;
    try {
      const config = connection(), task = $('task').value.trim(), context = contextSelection(), phases = phaseSettings();
      if (!task || task.length > 4000) throw new Error('分析目标请输入 1 至 4,000 字。');
      state.submitting = true; setBusy(true); say(''); const files = await materials();
      const data = await api('/api/jobs', { task, files, connection: config, ...phases,
        ...(context ? {context} : {}), ...(queued ? {queue: true} : {}) });
      await refreshHistory(); await openJob(data.job.job_id);
    } catch (error) { setBusy(ACTIVE.has(state.job?.status)); say(readableError(error), 'error'); }
    finally { state.submitting = false; setBusy(state.busy); }
  }
  function acceptConversation(data) {
    if (data.job_id !== state.job?.job_id || !Array.isArray(data.turns) || !Array.isArray(data.sources) ||
      (inputHash() && data.input_sha256 !== inputHash())) throw new Error('对话与当前原始材料不匹配，请重新打开报告。');
    state.conversation = data; renderTurns(); updateTurnLimit();
  }
  function updateTurnLimit() {
    const limit = state.conversation?.limits?.max_turns || 12;
    const count = state.conversation?.turns?.length || 0;
    $('turn-limit').textContent = count + ' / ' + limit + ' 次记录 · 每问最多 4,000 字';
    $('send-question').disabled = state.busy || !state.token || !state.conversation || count >= limit;
  }
  function renderTurns() {
    $('turns').replaceChildren();
    for (const [index, turn] of (state.conversation?.turns || []).entries()) {
      const section = node('article', 'turn');
      const question = node('div', 'turn-question'); question.append(node('p', 'turn-heading', '追问 ' + (index + 1) + ' · ' + formatDate(turn.created_at)), paragraph(turn.question)); section.append(question);
      if (turn.status === 'SUCCEEDED' && turn.citation_reference_validation === 'PASS') {
        const answer = node('div', 'turn-answer'); answer.append(node('p', 'turn-heading', '模型回答 · 需复核'));
        for (const item of turn.answer || []) { const p = node('div', 'finding'); p.append(paragraph(item.text), citationRow(item.citation_ids)); answer.append(p); }
        if (turn.information_gaps?.length) answer.append(node('h4', '', '仍缺少的信息'), list(turn.information_gaps)); section.append(answer);
      } else section.append(node('p', 'notice warning', '本次追问未完成 · ' + turn.status + '\n' + (REASONS[turn.reason_code] || '模型调用或输出校验失败，没有可展示的回答。')));
      $('turns').append(section);
    }
  }
  function uuid() {
    if (crypto.randomUUID) return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16)); bytes[6] = (bytes[6] & 15) | 64; bytes[8] = (bytes[8] & 63) | 128;
    const hex = Array.from(bytes, x => x.toString(16).padStart(2, '0')).join('');
    return hex.slice(0, 8) + '-' + hex.slice(8, 12) + '-' + hex.slice(12, 16) + '-' + hex.slice(16, 20) + '-' + hex.slice(20);
  }
  async function followUp(event) {
    event.preventDefault(); if (state.busy || !state.conversation) return;
    const question = $('question').value.trim(), jobId = state.job.job_id, hash = inputHash(), epoch = state.epoch;
    try {
      if (!question || question.length > 4000) throw new Error('追问请输入 1 至 4,000 字。');
      const config = connection();
      if (!hash) throw new Error('原始材料标识缺失，请重新打开任务。');
      if (!state.attempt || state.attempt.question !== question || state.attempt.jobId !== jobId || state.attempt.input_sha256 !== hash) {
        state.attempt = { jobId, question, input_sha256: hash, request_id: uuid() };
      }
      const attempt = state.attempt; setBusy(true);
      $('follow-up-status').textContent = '正在等待模型回答；请保持本页打开。问题与回答将保存在当前任务中。';
      const data = await api('/api/jobs/' + encodeURIComponent(jobId) + '/follow-up', {
        question: attempt.question, input_sha256: attempt.input_sha256, request_id: attempt.request_id, connection: config, ...phaseSettings()
      });
      if (epoch !== state.epoch) return;
      acceptConversation(data);
      const turn = data.turns.find(item => item.request_id === attempt.request_id);
      if (!turn) throw new Error('提交结果尚未确认，请重新读取记录；重试当前问题会复用请求编号。');
      state.attempt = null;
      window.AstroGraphExecution.mount($('execution-panel'), {
        baseUrl:'/api/jobs/'+encodeURIComponent(jobId), api,
        current:()=>epoch===state.epoch && state.job?.job_id===jobId
      });
      const ok = turn.status === 'SUCCEEDED' && turn.citation_reference_validation === 'PASS';
      if (ok && $('question').value.trim() === question) $('question').value = '';
      $('follow-up-status').textContent = ok ? '回答已保存；可点击引用核对原文。' : '本次追问未完成，失败记录已保存。' + (REASONS[turn.reason_code] || '');
    } catch (error) {
      if (epoch === state.epoch) $('follow-up-status').textContent = readableError(error) + (state.attempt ? ' 问题已保留；如提交结果未确认，重试相同问题会复用请求编号。' : '');
    } finally { if (epoch === state.epoch) setBusy(false); }
  }
  function bind() {
    $('queue-analysis').onchange = () => setBusy(state.busy);
    $('cancel-job').onclick = async () => {
      const jobId = state.job?.job_id, epoch = state.epoch;
      if (!jobId || !ACTIVE.has(state.job.status)) return;
      $('cancel-job').disabled = true;
      try { await api('/api/jobs/' + encodeURIComponent(jobId) + '/cancel', {});
        if (epoch === state.epoch) { clearPoll(); await fetchJob(jobId, epoch); }
      } catch (error) { say(readableError(error), 'error'); $('cancel-job').disabled = false; }
    };
    $('files').onchange = event => { addFiles(event.target.files); event.target.value = ''; };
    $('paste').oninput = inputStats;
    $('dropzone').ondragover = event => { event.preventDefault(); $('dropzone').classList.add('dragging'); };
    $('dropzone').ondragleave = () => $('dropzone').classList.remove('dragging');
    $('dropzone').ondrop = event => { event.preventDefault(); $('dropzone').classList.remove('dragging'); addFiles(event.dataTransfer.files); };
    $('probe').onclick = probe; $('analyze').onclick = analyze;
    $('history').onchange = () => { if (!state.busy) openJob($('history').value); };
    $('history-search').oninput = renderHistory;
    $('history-status-filter').onchange = renderHistory;
    for (const id of ['model', 'base-url']) $(id).addEventListener('input', resetConnectionBadge);
    $('resume-job').onclick = async () => {
      if (!state.job || (state.job.status !== 'INTERRUPTED' && !state.job.resumable)) return;
      try {
        const config = connection(); setBusy(true); say('正在从已保存的分块检查点继续任务。');
        await api('/api/jobs/' + encodeURIComponent(state.job.job_id) + '/resume', { connection: config, ...phaseSettings(), ...($('queue-analysis').checked ? {queue:true} : {}) });
        await openJob(state.job.job_id);
      } catch (error) { setBusy(false); say(readableError(error), 'error'); }
    };
    $('restore-trigger').onclick = () => $('restore-file').click();
    $('restore-file').onchange = async () => {
      const file = $('restore-file').files?.[0]; $('restore-file').value = '';
      if (!file) return;
      if (!confirm('恢复备份会将其中的任务添加到当前数据目录；同编号任务会拒绝覆盖。继续吗？')) return;
      setBusy(true);
      try { const result = await uploadBackup(file); await refreshHistory(); say('备份恢复完成：新增 ' + result.restored_jobs + ' 项历史任务。'); }
      catch (error) { say(readableError(error), 'error'); }
      finally { setBusy(false); }
    };
    $('refresh-history').onclick = async () => {
      try { await refreshHistory(); if (state.job?.status === 'RUNNING') await openJob(state.job.job_id); else say('任务列表已刷新。'); }
      catch (error) { say(readableError(error), 'error'); }
    };
    $('load-sample').onclick = () => {
      const sample = state.session?.sample;
      const text = typeof sample === 'string' ? sample : sample?.text || sample?.content || '';
      if (!text) { say('示例材料尚不可用，请粘贴自己的诊断文本。', 'warning'); return; }
      if ($('paste').value.trim()) {
        const names = new Set(state.files.map(item => item.file.name.toLocaleLowerCase()));
        let filename = 'sample-report.txt', n = 2;
        while (names.has(filename.toLocaleLowerCase())) filename = 'sample-report-' + n++ + '.txt';
        addFiles([new File([text], filename, { type: 'text/plain' })]);
      } else { $('paste').value = text; $('paste-role').value = 'diagnosis_results'; inputStats(); }
      say('已载入演示材料。示例不代表真实诊断结论，仍需配置模型后手动开始分析。');
    };
    for (const id of SETTINGS) $(id).addEventListener('change', () => { saveSettings(); resetConnectionBadge(); });
    $('api-key').addEventListener('input', resetConnectionBadge);
    $('mode').addEventListener('change', updateMode); $('preset').addEventListener('change', applyPreset);
    $('base-url').addEventListener('input', () => { $('preset').value = 'custom'; resetConnectionBadge(); });
    $('follow-up-form').onsubmit = followUp;
    $('question').onkeydown = event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing) { event.preventDefault(); if (!$('send-question').disabled) $('follow-up-form').requestSubmit(); } };
    $('reload-conversation').onclick = () => { if (state.job) loadConversation(state.job.job_id, state.epoch); };
  }
  async function init() {
    bind(); setBusy(false);
    try {
      const data = await api('/api/session'); state.session = data; state.token = data.token;
      document.querySelectorAll('.version-label').forEach(el => {el.textContent = typeof data.version === 'string' ? data.version : '源码开发版';});
      const defaults = data.defaults?.connection || data.defaults || {};
      const mapping = { mode: 'mode', base_url: 'base-url', model: 'model', max_tokens: 'max-tokens', timeout_s: 'timeout', response_format_strategy: 'strategy' };
      for (const [key, id] of Object.entries(mapping)) if (defaults[key] !== undefined) $(id).value = String(defaults[key]);
      try {
        const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
        for (const id of SETTINGS) if (typeof stored[id] === 'string') $(id).value = stored[id];
      } catch (_) { /* Keep defaults if browser storage is disabled or malformed. */ }
      if (!['local', 'online'].includes($('mode').value)) $('mode').value = 'local';
      if (!$('strategy').value) $('strategy').value = 'auto';
      if (!$('preset').value) $('preset').value = 'custom';
      updateMode(); inputStats();
      $('service-dot').classList.add('ready'); $('service-status').textContent = '本机工作台已就绪'; setBusy(false);
      state.contexts = window.AstroGraphContexts.mount($('context-picker'), {
        api: (path, options = {}) => api(path, options.method === 'POST' ? JSON.parse(options.body || '{}') : undefined)
      });
      await refreshHistory();
    } catch (error) { $('service-status').textContent = '本机工作台未连接'; say(readableError(error), 'error'); }
  }
  init();
})();
