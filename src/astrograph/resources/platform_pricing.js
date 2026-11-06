import { $, showToast } from './platform_core.js';

let adminToken = '';

async function pricingApi(path='', options={}) {
  const headers = {...(options.headers || {})};
  if (adminToken) headers.Authorization = 'Bearer ' + adminToken;
  const response = await fetch('/v1/platform/settings/model-pricing' + path, {
    ...options, headers, cache: 'no-store',
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || body.error?.message || detail;
    } catch {}
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response.json();
}

function localDateTimeValue() {
  const local = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

function renderPrices(prices=[]) {
  const host = $('#model-pricing-list');
  if (!host) return;
  host.replaceChildren();
  if (!prices.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = '尚未登记模型价格。未登记价格的请求只统计 Token，不估算费用。';
    host.append(empty);
    return;
  }
  const wrap = document.createElement('div');
  wrap.className = 'execution-table-scroll';
  const table = document.createElement('table');
  const header = document.createElement('tr');
  for (const title of ['服务商', '模型 ID', '输入 / 百万 Token', '输出 / 百万 Token', '币种', '生效时间', '价格来源', '登记人']) {
    const cell = document.createElement('th'); cell.textContent = title; header.append(cell);
  }
  const thead = document.createElement('thead'); thead.append(header); table.append(thead);
  const tbody = document.createElement('tbody');
  for (const price of prices) {
    const row = document.createElement('tr');
    const values = [
      price.provider, price.model,
      price.input_per_million_tokens, price.output_per_million_tokens,
      price.currency, price.effective_from,
      (price.source_type || '') + ' · ' + (price.source_ref || ''),
      price.created_by,
    ];
    for (const value of values) {
      const cell = document.createElement('td'); cell.textContent = String(value ?? '—'); row.append(cell);
    }
    tbody.append(row);
  }
  table.append(tbody); wrap.append(table); host.append(wrap);
}

async function loadModelPricing() {
  const status = $('#model-pricing-status');
  if (status) status.textContent = '正在读取版本化价格表…';
  try {
    const value = await pricingApi();
    renderPrices(value.prices || []);
    if (status) status.textContent = (value.prices || []).length + ' 条价格版本 · ' + value.cost_scope;
    return value;
  } catch (error) {
    if (status) status.textContent = '读取失败：' + error.message + '。Bearer 模式请填写管理员 Token。';
    renderPrices([]);
    return null;
  }
}

function setupModelPricing() {
  if (!$('#model-pricing-status')) return;
  const token = $('#model-pricing-admin-token');
  if (token) token.addEventListener('change', () => {
    adminToken = token.value;
    token.value = '';
    loadModelPricing();
  });
  $('#model-pricing-refresh')?.addEventListener('click', () => loadModelPricing());
  const date = $('#model-pricing-effective-from');
  if (date && !date.value) date.value = localDateTimeValue();
  $('#model-pricing-form')?.addEventListener('submit', async event => {
    event.preventDefault();
    const button = $('#model-pricing-save');
    button.disabled = true;
    try {
      const localDate = $('#model-pricing-effective-from').value;
      if (!localDate) throw new Error('请选择价格生效时间');
      const payload = {
        provider: $('#model-pricing-provider').value.trim(),
        model: $('#model-pricing-model').value.trim(),
        currency: $('#model-pricing-currency').value.trim().toUpperCase(),
        input_per_million_tokens: $('#model-pricing-input').value.trim(),
        output_per_million_tokens: $('#model-pricing-output').value.trim(),
        effective_from: new Date(localDate).toISOString(),
        source_type: $('#model-pricing-source-type').value,
        source_ref: $('#model-pricing-source-ref').value.trim(),
      };
      const saved = await pricingApi('', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      $('#model-pricing-model').value = '';
      $('#model-pricing-input').value = '';
      $('#model-pricing-output').value = '';
      $('#model-pricing-source-ref').value = '';
      await loadModelPricing();
      showToast('已新增价格版本 #' + saved.price.revision + '；历史请求价格快照不变');
    } catch (error) {
      showToast('价格登记失败：' + error.message);
    } finally {
      button.disabled = false;
    }
  });
  loadModelPricing();
}

export { setupModelPricing, loadModelPricing };
