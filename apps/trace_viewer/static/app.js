const $ = (id) => document.getElementById(id);
const state = { trace: null, traces: [], meta: null, step: 0, block: 3, head: -1, data: null, timer: null, request: 0 };
const number = (value, digits = 3) => value == null ? '—' : Number(value).toFixed(digits);
const piece = (token) => token?.text || token?.piece?.replaceAll('Ġ', ' ').replaceAll('▁', ' ') || String(token?.id ?? '—');

async function getJSON(url) {
  const response = await fetch(url);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}

function setText(id, value) { $(id).textContent = value; }
function empty(node) { node.replaceChildren(); }
function option(value, label) { const node = document.createElement('option'); node.value = value; node.textContent = label; return node; }

async function start() {
  try {
    const traces = await getJSON('/api/traces');
    state.traces = traces;
    renderTraceOptions();
    if (traces.length) await loadTrace(traces[0].id);
  } catch (error) { setText('availability', `Unable to load traces: ${error.message}`); }
}

function renderTraceOptions() {
  const select = $('trace'), query = $('filter').value.trim().toLowerCase();
  const matches = state.traces.filter((trace) => `${trace.id} ${trace.source}`.toLowerCase().includes(query));
  const visible = state.trace && !matches.some((trace) => trace.id === state.trace)
    ? [state.traces.find((trace) => trace.id === state.trace), ...matches] : matches;
  empty(select);
  for (const trace of visible) select.append(option(trace.id, `${trace.id}${trace.hidden ? ' · residuals' : ''}${trace.logprobs ? ' · logprobs' : ''}`));
  if (state.trace) select.value = state.trace;
  $('filter').title = `${matches.length} matching traces`;
}

async function loadTrace(id) {
  stopPlay();
  state.trace = id;
  state.meta = await getJSON(`/api/trace/${encodeURIComponent(id)}`);
  state.step = 0;
  state.block = state.meta.full_attention_layers[0] ?? 0;
  state.head = -1;
  $('trace').value = id;
  $('step').max = Math.max(0, state.meta.tokens.length - 1);
  $('step').value = 0;
  const picker = $('image'); empty(picker);
  for (const image of state.meta.images) picker.append(option(image.index, `Frame ${image.index + 1} · positions ${image.start}–${image.end - 1}`));
  const heads = $('head'); empty(heads); heads.append(option(-1, 'Mean of heads'));
  for (let h = 0; h < 16; h++) heads.append(option(h, `Head ${h}`));
  heads.value = '-1';
  renderTimeline();
  renderBlocks();
  setText('capture', state.meta.has_hidden ? 'Residual + attention' : 'Attention only');
  setText('captureDetail', `${state.meta.checkpoint || state.meta.model_id || state.meta.model_type || 'model unknown'} · ${state.meta.generated_tokens} generated tokens · ${state.meta.prompt_tokens} prompt positions · ${state.meta.device || 'unknown'} · ${state.meta.model_revision?.slice(0, 10) || 'revision unknown'}`);
  setText('availability', state.meta.has_hidden ? 'Residual vectors are captured only for the configured generation steps; empty layers mean that step was not saved.' : 'This trace has no residual vectors. The viewer can show full-attention routing and generated-token data, but cannot reconstruct missing activations.');
  await loadStep();
}

function renderTimeline() {
  const node = $('tokens'); empty(node);
  state.meta.tokens.forEach((token, index) => {
    const button = document.createElement('button'); button.className = `token${state.step === index ? ' active' : ''}`;
    const indexNode = document.createElement('small'); indexNode.textContent = String(index).padStart(2, '0');
    button.append(indexNode, document.createTextNode(piece(token).slice(0, 18)));
    button.title = `Token ${index} · ID ${token.id}${token.logprob == null ? '' : ` · log p ${number(token.logprob)} nats`}`;
    button.addEventListener('click', () => { state.step = index; $('step').value = index; loadStep(); });
    node.append(button);
  });
}

function renderBlocks() {
  const node = $('blocks'); empty(node);
  for (let block = 0; block < state.meta.blocks; block++) {
    const isFull = state.meta.full_attention_layers.includes(block);
    const button = document.createElement('button');
    button.className = `block ${isFull ? 'full' : 'linear'}${block === state.block ? ' active' : ''}`;
    const dot = document.createElement('span'); dot.className = 'fill'; dot.dataset.block = block;
    const title = document.createTextNode(String(block).padStart(2, '0'));
    const label = document.createElement('span'); label.className = 'mini'; label.textContent = isFull ? 'softmax' : 'linear';
    button.append(dot, title, label);
    button.title = `Block ${block} · ${isFull ? 'full attention' : 'linear attention'}`;
    button.addEventListener('click', () => { state.block = block; loadStep(); });
    node.append(button);
  }
}

async function loadStep() {
  if (!state.meta?.tokens.length) return;
  const request = ++state.request;
  try {
    const data = await getJSON(`/api/trace/${encodeURIComponent(state.trace)}/step/${state.step}?block=${state.block}&head=${state.head}`);
    if (request !== state.request) return;
    state.data = data;
    render();
  } catch (error) { if (request === state.request) setText('routingState', `Error: ${error.message}`); }
}

function render() {
  const { data, meta } = state;
  $('stepValue').textContent = `${state.step + 1} / ${meta.tokens.length}`;
  $('step').value = state.step;
  setText('currentToken', JSON.stringify(piece(data.token)));
  setText('tokenMeta', `ID ${data.token.id}${data.token.logprob == null ? '' : ` · log p = ${number(data.token.logprob)} nats`}`);
  setText('queryPosition', String(data.query_position));
  [...$('tokens').children].forEach((node, index) => node.classList.toggle('active', index === state.step));
  [...$('blocks').children].forEach((node, index) => node.classList.toggle('active', index === state.block));
  for (const marker of document.querySelectorAll('.fill')) marker.classList.toggle('on', Boolean(data.hidden[Number(marker.dataset.block) + 1]));
  const isFull = meta.full_attention_layers.includes(state.block);
  setText('blockTitle', `Block ${String(state.block).padStart(2, '0')}`);
  setText('blockKind', isFull ? 'Conventional full attention' : 'Linear attention / recurrent state');
  const summary = data.hidden[state.block + 1];
  setText('norm', number(summary?.norm, 2));
  setText('deltaNorm', number(data.delta_norm, 2));
  setText('meanabs', number(summary?.mean_abs, 4));
  setText('peak', number(summary?.max_abs, 3));
  drawFeatures($('vectorMode').value === 'delta' ? data.delta_vector : data.vector);
  const attention = data.attention;
  setText('routingState', attention ? `${attention.heads} heads · ${attention.key_count} available keys` : isFull ? 'Attention row was not captured for this step' : 'No softmax attention row for this block');
  setText('promptMass', attention ? `${number(attention.prompt_mass * 100, 1)}%` : '—');
  setText('outputMass', attention ? `${number(attention.generation_mass * 100, 1)}%` : '—');
  setText('imageMass', attention ? `${number(attention.images.reduce((sum, item) => sum + (item.mass || 0), 0) * 100, 1)}%` : '—');
  renderTopKeys(attention);
  drawImage();
}

function drawFeatures(values) {
  const canvas = $('features'), ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#101b2b'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!values?.length) { ctx.fillStyle = '#8ca4b4'; ctx.font = '15px sans-serif'; ctx.fillText('Residual vector unavailable at this step', 20, 125); return; }
  const maximum = Math.max(1e-8, ...values.map((value) => Math.abs(value || 0)));
  const columns = 64, width = canvas.width / columns, height = canvas.height / Math.ceil(values.length / columns);
  values.forEach((value, index) => {
    const magnitude = Math.min(1, Math.pow(Math.abs(value || 0) / maximum, 0.55));
    ctx.fillStyle = value >= 0 ? `rgba(255,147,102,${0.08 + 0.92 * magnitude})` : `rgba(83,170,255,${0.08 + 0.92 * magnitude})`;
    ctx.fillRect((index % columns) * width, Math.floor(index / columns) * height, width, height);
  });
}

function renderTopKeys(attention) {
  const node = $('topKeys'); empty(node);
  if (!attention) return;
  const maximum = attention.top[0]?.weight || 1;
  attention.top.forEach((item) => {
    const line = document.createElement('div'); line.className = 'key';
    const label = document.createElement('span');
    const token = item.generated_index == null ? null : state.meta.tokens[item.generated_index];
    label.textContent = item.image != null ? `image ${item.image + 1} · ${item.position}` : token ? `out ${item.generated_index} · ${piece(token).slice(0, 8)}` : `prompt ${item.position}`;
    const bar = document.createElement('span'); bar.className = 'bar'; const fill = document.createElement('i'); fill.style.width = `${100 * item.weight / maximum}%`; bar.append(fill);
    const value = document.createElement('em'); value.textContent = number(item.weight, 4);
    line.append(label, bar, value); node.append(line);
  });
}

function drawImage() {
  const canvas = $('imageCanvas'), emptyNode = $('imageEmpty'), attention = state.data?.attention;
  const selected = Number($('image').value || 0);
  const grid = attention?.images.find((item) => item.index === selected);
  const image = state.meta.images.find((item) => item.index === selected);
  if (!image?.file) { canvas.hidden = true; emptyNode.hidden = false; setText('imageCaption', ''); return; }
  canvas.hidden = false; emptyNode.hidden = true;
  const bitmap = new Image();
  const trace = state.trace, step = state.step, block = state.block, head = state.head;
  bitmap.onload = () => {
    if (state.trace !== trace || state.step !== step || state.block !== block || state.head !== head) return;
    const cap = 1100 / Math.max(bitmap.width, bitmap.height);
    canvas.width = Math.max(1, Math.round(bitmap.width * Math.min(1, cap)));
    canvas.height = Math.max(1, Math.round(bitmap.height * Math.min(1, cap)));
    const ctx = canvas.getContext('2d'); ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    if (grid?.rows && grid.cols && grid.values.length === grid.rows * grid.cols) {
      const max = Math.max(1e-9, ...grid.values.map((value) => value || 0));
      grid.values.forEach((value, index) => {
        const x = index % grid.cols, y = Math.floor(index / grid.cols);
        const intensity = Math.pow(Math.max(0, value || 0) / max, 0.55);
        ctx.fillStyle = `rgba(255,96,30,${Math.min(0.82, intensity * 0.75)})`;
        ctx.fillRect(x * canvas.width / grid.cols, y * canvas.height / grid.rows, canvas.width / grid.cols + 0.5, canvas.height / grid.rows + 0.5);
      });
    }
  };
  bitmap.src = `/api/trace/${encodeURIComponent(state.trace)}/image/${encodeURIComponent(image.file)}`;
  setText('imageCaption', grid?.rows ? `${grid.rows} × ${grid.cols} visual tokens · ${number((grid.mass || 0) * 100, 2)}% of this head's attention to frame ${selected + 1}. Heatmap scales within this frame.` : 'No spatial attention map for the selected block and token.');
}

function stopPlay() { if (state.timer) clearInterval(state.timer); state.timer = null; $('play').textContent = '▶ Play'; }
function togglePlay() {
  if (state.timer) return stopPlay();
  $('play').textContent = '■ Pause';
  state.timer = setInterval(() => {
    if (state.step >= state.meta.tokens.length - 1) return stopPlay();
    state.step++; loadStep();
  }, 750);
}

$('trace').addEventListener('change', (event) => loadTrace(event.target.value));
$('filter').addEventListener('input', renderTraceOptions);
$('step').addEventListener('input', (event) => { state.step = Number(event.target.value); loadStep(); });
$('head').addEventListener('change', (event) => { state.head = Number(event.target.value); loadStep(); });
$('image').addEventListener('change', drawImage);
$('vectorMode').addEventListener('change', () => drawFeatures($('vectorMode').value === 'delta' ? state.data?.delta_vector : state.data?.vector));
$('features').addEventListener('mousemove', (event) => {
  const values = $('vectorMode').value === 'delta' ? state.data?.delta_vector : state.data?.vector;
  if (!values?.length) return;
  const bounds = $('features').getBoundingClientRect();
  const column = Math.floor((event.clientX - bounds.left) / bounds.width * 64);
  const row = Math.floor((event.clientY - bounds.top) / bounds.height * Math.ceil(values.length / 64));
  const index = row * 64 + column;
  setText('hoverFeature', index < values.length && index >= 0 ? `Dimension ${index}: ${number(values[index], 5)}` : '');
});
$('features').addEventListener('mouseleave', () => setText('hoverFeature', ''));
$('play').addEventListener('click', togglePlay);
document.addEventListener('keydown', (event) => {
  if (event.target.matches('input, select, button')) return;
  if (event.key === 'ArrowRight' && state.step < state.meta.tokens.length - 1) { state.step++; loadStep(); }
  if (event.key === 'ArrowLeft' && state.step > 0) { state.step--; loadStep(); }
});
start();
