/* gguf-trainer web GUI (vanilla JS, structure shared with the ggk GUI).
 *
 * The local Python server owns the project folder; this file edits the
 * project's config, starts/stops the detached pipeline and polls
 * /api/project, /api/log, /api/metrics and /api/hardware while it runs.
 * Files are picked by path via /api/browse — nothing is uploaded.
 */

'use strict';

const API_ROOT = new URL('.', location.href).pathname;
function apiPath(p) { return p.startsWith('/') ? API_ROOT + p.slice(1) : p; }

const PROJECT_KEY = 'gguf-trainer-project';
const THEME_KEY = 'gguf-trainer-theme';

let serverInfo = {};
let project = null;      // last /api/project payload
let config = null;       // editable copy of project.config
let dirty = false;
let pollTimer = null;
let hwTimer = null;
let logNext = 0;
let logText = '';
let metricsAt = 0;
let lastRuntime = '';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }
async function api(path, body) {
  const opts = body === undefined ? {} :
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const res = await fetch(apiPath(path), opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}
function filename(p) { return p ? p.replace(/\\/g, '/').replace(/\/$/, '').split('/').pop() : ''; }
function humanSize(n) {
  if (n == null) return '—';
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(2) + ' GB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(1) + ' MB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KB';
  return n + ' B';
}
function fmtDur(s) {
  if (s == null || !isFinite(s) || s < 0) return '—';
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
}
function fmtNum(v, d = 4) { return (v == null || !isFinite(v)) ? '—' : Number(v).toFixed(d); }
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function showError(msg) { $('error-text').textContent = msg; $('error-banner').style.display = 'flex'; }
function clearError() { $('error-banner').style.display = 'none'; }
function showNotice(msg) { $('notice-text').textContent = msg; $('notice-banner').style.display = 'flex'; }
$('error-close').onclick = clearError;
$('notice-close').onclick = () => { $('notice-banner').style.display = 'none'; };

function getPath(obj, key) { return key.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj); }
function setPath(obj, key, val) {
  const ks = key.split('.');
  let o = obj;
  for (const k of ks.slice(0, -1)) { if (o[k] == null || typeof o[k] !== 'object') o[k] = {}; o = o[k]; }
  o[ks[ks.length - 1]] = val;
}

// ─── Browse modal ────────────────────────────────────────────────────────────

let browseResolve = null;
let browseKind = 'any';

function openBrowse(kind, title, startPath) {
  return new Promise((resolve) => {
    browseResolve = resolve;
    browseKind = kind;
    $('browse-title').textContent = title;
    $('browse-pick-dir-btn').style.display = kind === 'dir' ? '' : 'none';
    $('browse-overlay').classList.add('open');
    browseTo(startPath || serverInfo.home || null);
  });
}
function closeBrowse(result) {
  $('browse-overlay').classList.remove('open');
  if (browseResolve) { browseResolve(result); browseResolve = null; }
}
async function browseTo(path) {
  try {
    const data = await api('/api/browse', { path, kind: browseKind });
    $('browse-current').textContent = data.path;
    $('browse-path-input').value = data.path;
    const ul = $('browse-list');
    ul.innerHTML = '';
    if (data.parent) {
      const li = document.createElement('li');
      li.innerHTML = '<span class="icon">↩</span> ..';
      li.onclick = () => browseTo(data.parent);
      ul.appendChild(li);
    }
    for (const e of data.entries) {
      const li = document.createElement('li');
      li.innerHTML = `<span class="icon">${e.is_dir ? '📁' : '📄'}</span> ${escapeHTML(e.name)}` +
        (e.is_project ? '<span class="proj">project</span>' : '') +
        (e.is_dir ? '' : `<span class="size">${humanSize(e.size)}</span>`);
      li.onclick = () => { e.is_dir ? browseTo(e.path) : closeBrowse(e.path); };
      ul.appendChild(li);
    }
  } catch (err) { showError('Browse failed: ' + err.message); }
}
$('browse-cancel-btn').onclick = () => closeBrowse(null);
$('browse-pick-dir-btn').onclick = () => closeBrowse($('browse-current').textContent);
$('browse-go-btn').onclick = () => {
  const v = $('browse-path-input').value.trim();
  if (!v) return;
  if (browseKind !== 'dir' && /\.[A-Za-z0-9]+$/.test(v)) closeBrowse(v); else browseTo(v);
};
$('browse-path-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('browse-go-btn').onclick(); });
$('browse-overlay').addEventListener('mousedown', (e) => { if (e.target === $('browse-overlay')) closeBrowse(null); });

// ─── Confirm dialog ──────────────────────────────────────────────────────────
// askConfirm({title, text, items, note, ok, kind}) -> Promise<boolean>.
// kind: 'info' (blue) | 'warn' (amber) | 'danger' (red, destructive).
// With `fields` ([{id, type: 'radio'|'checkbox', label, options: [{value, label, desc}], value}])
// it resolves with {id: value, ...} on OK and null on cancel.

let confirmResolve = null;
let confirmFields = null;
function askConfirm(opts) {
  return new Promise((resolve) => {
    confirmResolve = resolve;
    confirmFields = opts.fields || null;
    const fields = $('confirm-fields');
    fields.innerHTML = '';
    for (const f of (opts.fields || [])) {
      const g = document.createElement('div');
      g.className = 'field-group';
      if (f.label) g.innerHTML = `<div class="group-label">${escapeHTML(f.label)}</div>`;
      if (f.type === 'checkbox') {
        g.innerHTML += `<label class="opt"><input type="checkbox" name="cf-${f.id}" ${f.value ? 'checked' : ''}>
          <span>${escapeHTML(f.text || '')}${f.desc ? ' <span class="desc">' + escapeHTML(f.desc) + '</span>' : ''}</span></label>`;
      } else {
        for (const o of f.options) {
          g.innerHTML += `<label class="opt"><input type="radio" name="cf-${f.id}" value="${escapeHTML(o.value)}" ${o.value === f.value ? 'checked' : ''} ${o.disabled ? 'disabled' : ''}>
            <span>${escapeHTML(o.label)}${o.desc ? ' <span class="desc">' + escapeHTML(o.desc) + '</span>' : ''}</span></label>`;
        }
      }
      fields.appendChild(g);
    }
    const kind = opts.kind || 'info';
    const icon = $('confirm-icon');
    icon.className = 'confirm-icon ' + (kind === 'info' ? '' : kind);
    icon.innerHTML = {
      info: '<svg viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>',
      warn: '<svg viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
      danger: '<svg viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>',
    }[kind] || '?';
    $('confirm-title').textContent = opts.title || 'Are you sure?';
    $('confirm-text').textContent = opts.text || '';
    const ul = $('confirm-list');
    ul.innerHTML = '';
    for (const it of (opts.items || [])) {
      const li = document.createElement('li');
      if (typeof it === 'string') li.innerHTML = `<span class="mono" title="${escapeHTML(it)}">${escapeHTML(it)}</span>`;
      else { li.innerHTML = `<span class="label">${escapeHTML(it.label)}</span><span class="mono" title="${escapeHTML(it.value)}">${escapeHTML(it.value)}</span>`; }
      ul.appendChild(li);
    }
    $('confirm-note').textContent = opts.note || '';
    const ok = $('confirm-ok');
    ok.textContent = opts.ok || 'Continue';
    ok.className = 'btn ' + (kind === 'danger' ? 'btn-danger' : 'btn-primary');
    $('confirm-overlay').classList.add('open');
    ok.focus();
  });
}
function closeConfirm(result) {
  $('confirm-overlay').classList.remove('open');
  if (confirmFields) {
    let values = null;
    if (result) {
      values = {};
      for (const f of confirmFields) {
        if (f.type === 'checkbox') values[f.id] = $('confirm-fields').querySelector(`input[name="cf-${f.id}"]`).checked;
        else { const el = $('confirm-fields').querySelector(`input[name="cf-${f.id}"]:checked`); values[f.id] = el ? el.value : f.value; }
      }
    }
    result = values;
    confirmFields = null;
  }
  if (confirmResolve) { confirmResolve(result); confirmResolve = null; }
}
$('confirm-ok').onclick = () => closeConfirm(true);
$('confirm-cancel').onclick = () => closeConfirm(false);
$('confirm-overlay').addEventListener('mousedown', (e) => { if (e.target === $('confirm-overlay')) closeConfirm(false); });
document.addEventListener('keydown', (e) => {
  if (!$('confirm-overlay').classList.contains('open')) return;
  if (e.key === 'Escape') { e.preventDefault(); closeConfirm(false); }
  else if (e.key === 'Enter') { e.preventDefault(); closeConfirm(true); }
});

// ─── Config binding (data-cfg / data-cfg-path) ───────────────────────────────

function markDirty() {
  dirty = true;
  $('save-note').textContent = 'unsaved changes';
  $('save-btn').classList.add('btn-primary');
}

function bindConfigInputs() {
  document.querySelectorAll('[data-cfg]').forEach(el => {
    const key = el.dataset.cfg;
    el.addEventListener('change', () => {
      if (!config) return;
      let v;
      if (el.type === 'checkbox') v = el.checked;
      else if (el.type === 'number') v = el.value === '' ? 0 : Number(el.value);
      else v = el.value;
      setPath(config, key, v);
      markDirty();
      renderDerived();
    });
  });
  document.querySelectorAll('.browse-btn').forEach(btn => {
    btn.onclick = async () => {
      const key = btn.dataset.for;
      const span = document.querySelector(`[data-cfg-path="${key}"]`);
      const kind = btn.dataset.kind || span.dataset.kind || 'any';
      const cur = getPath(config, key) || '';
      const start = !cur ? null : kind === 'dir' ? cur : cur.replace(/[/\\][^/\\]*$/, '');
      const p = await openBrowse(kind, 'Select ' + key.split('.').pop().replace(/_/g, ' '), start);
      if (p) { setPath(config, key, p); markDirty(); renderConfig(); }
    };
  });
  document.querySelectorAll('[data-clear]').forEach(btn => {
    btn.onclick = () => { setPath(config, btn.dataset.clear, ''); markDirty(); renderConfig(); };
  });
  document.querySelectorAll('[data-cfg-path]').forEach(span => { span.dataset.placeholder = span.textContent; });
}

function renderConfig() {
  if (!config) return;
  document.querySelectorAll('[data-cfg]').forEach(el => {
    const v = getPath(config, el.dataset.cfg);
    if (el.type === 'checkbox') el.checked = !!v;
    else el.value = v == null ? '' : v;
  });
  document.querySelectorAll('[data-cfg-path]').forEach(span => {
    const v = getPath(config, span.dataset.cfgPath) || '';
    span.textContent = v ? v : span.dataset.placeholder;
    span.title = v;
    span.classList.toggle('placeholder', !v);
    span.classList.toggle('mono', !!v);
  });
  renderPresets();
  renderCorpusFiles();
  renderImageFolders();
  renderDerived();
}

function packInfo() {
  return (project && project.pack) ? project.pack : { num_queries: 256, out_dim: 2560, in_dim: 1024, needs_images: false, hints: {}, eval_keys: [], teacher_control: 'placement' };
}

function renderDerived() {
  if (!config) return;
  const pk = packInfo();
  const images = !!pk.needs_images;
  const presets = $('corpus-presets-box'), files = $('corpus-files-box');
  const filesMode = !images && config.corpus.mode === 'files';
  presets.style.display = filesMode ? 'none' : '';
  files.style.display = filesMode ? '' : 'none';
  $('corpus-mode-row').style.display = images ? 'none' : '';
  $('corpus-images-box').style.display = images ? '' : 'none';
  document.querySelectorAll('.corpus-text-only').forEach(el => { el.style.display = images ? 'none' : ''; });
  document.querySelectorAll('.precompute-vision').forEach(el => { el.style.display = images ? '' : 'none'; });
  // which teacher placement control the pack's teacher understands
  const control = pk.teacher_control || (images ? 'teacher_mode' : 'placement');
  $('precompute-placement-field').style.display = control === 'placement' ? '' : 'none';
  $('precompute-teacher-mode-field').style.display = control === 'teacher_mode' ? '' : 'none';
  $('export-sigvq-row').style.display = pk.id === 'llada_image' ? '' : 'none';
  $('corpus-badge').textContent = images ? 'images + instructions' : 'text prompts';
  $('train-summary').textContent = `width ${config.train.width} · depth ${config.train.depth} · ${config.train.steps} steps · batch ${config.train.batch_size}`;
  $('precompute-summary').textContent = `shards ${config.precompute.shard_size} · teacher batch ${config.precompute.teacher_batch || 'auto'} · ${config.precompute.device}`;
  let nSamples, perSample;
  if (images) {
    const c = config.corpus;
    nSamples = Number(c.n_image) + Number(c.n_two) + Number(c.n_text) + Number(c.n_val_image) + Number(c.n_val_text);
    perSample = project && project.sample_bytes ? project.sample_bytes : 2.4e6;
  } else {
    nSamples = config.corpus.mode === 'files' ? null : (Number(config.corpus.n_train) + Number(config.corpus.n_val));
    perSample = (project && project.sample_bytes) ? project.sample_bytes
      : pk.num_queries * pk.out_dim * 2 + 60 * pk.in_dim * 2;   // bf16 target rows + ~60 student tokens
  }
  $('shard-estimate').textContent = nSamples ? `Shards on disk: ≈ ${humanSize(nSamples * perSample)} for ${nSamples} samples (${humanSize(perSample)} each on average), under <project>/shards.` : '';
  $('output-name-preview').textContent = `→ ${config.name}-f16.gguf`;
}

function renderPresetList(boxId, list, key, onChange) {
  const box = $(boxId);
  box.innerHTML = '';
  for (const p of list || []) {
    const lab = document.createElement('label');
    lab.className = 'toggle-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = (config.corpus[key] || []).includes(p.id);
    cb.onchange = () => {
      const set = new Set(config.corpus[key] || []);
      cb.checked ? set.add(p.id) : set.delete(p.id);
      config.corpus[key] = [...set];
      markDirty();
      if (onChange) onChange();
    };
    lab.appendChild(cb);
    lab.insertAdjacentHTML('beforeend', ` <span>${escapeHTML(p.title)} <small>${escapeHTML(p.repo)}</small></span>`);
    box.appendChild(lab);
  }
}
function renderPresets() {
  renderPresetList('corpus-presets', serverInfo.corpus_presets, 'presets');
  // an image preset is a material: save so the Materials list shows its download right away
  renderPresetList('corpus-image-presets', serverInfo.image_presets, 'image_presets',
    () => saveConfig().then(() => refreshProject(true)).catch(e => showError(e.message)));
}

function renderImageFolders() {
  const box = $('corpus-image-folders');
  box.innerHTML = '';
  (config.corpus.image_folders || []).forEach((f, i) => {
    const row = document.createElement('div');
    row.className = 'item-row';
    row.innerHTML = `<div class="path-box"><span class="path-text" title="${escapeHTML(f)}">${escapeHTML(f)}</span></div>
      <button class="remove-btn" title="Remove">✕</button>`;
    row.querySelector('.remove-btn').onclick = () => { config.corpus.image_folders.splice(i, 1); markDirty(); renderImageFolders(); };
    box.appendChild(row);
  });
}
$('corpus-add-folder').onclick = async () => {
  const p = await openBrowse('dir', 'Select a folder of images', null);
  if (p) { (config.corpus.image_folders = config.corpus.image_folders || []).push(p); markDirty(); renderImageFolders(); }
};

function renderCorpusFiles() {
  const box = $('corpus-files');
  box.innerHTML = '';
  (config.corpus.extra_files || []).forEach((f, i) => {
    const row = document.createElement('div');
    row.className = 'item-row';
    row.innerHTML = `<div class="path-box"><span class="path-text" title="${escapeHTML(f)}">${escapeHTML(f)}</span></div>
      <button class="remove-btn" title="Remove">✕</button>`;
    row.querySelector('.remove-btn').onclick = () => { config.corpus.extra_files.splice(i, 1); markDirty(); renderCorpusFiles(); };
    box.appendChild(row);
  });
}
$('corpus-add-file').onclick = async () => {
  const p = await openBrowse('text', 'Select a prompt file (.txt / .jsonl)', null);
  if (p) { (config.corpus.extra_files = config.corpus.extra_files || []).push(p); markDirty(); renderCorpusFiles(); }
};

async function saveConfig() {
  if (!project || !config) return;
  const data = await api('/api/project/save', { path: project.path, config });
  applyProject(data);
  dirty = false;
  $('save-note').textContent = 'saved';
  $('save-btn').classList.remove('btn-primary');
}
$('save-btn').onclick = () => saveConfig().catch(e => showError(e.message));

// ─── Project ─────────────────────────────────────────────────────────────────

function applyProject(data) {
  project = data;
  if (!dirty) config = JSON.parse(JSON.stringify(data.config));
  localStorage.setItem(PROJECT_KEY, data.path);
  $('project-path-text').textContent = data.path;
  $('project-path-text').classList.remove('placeholder');
  $('project-path-text').classList.add('mono');
  $('setup-body').style.display = '';
  $('sb-project').textContent = data.config.name + ' · ' + filename(data.path);
  $('train-project-badge').textContent = data.config.name;
  $('pack-summary').textContent = data.pack.title + ' — ' + data.pack.description;
  const hints = data.pack.hints || {};
  for (const k of ['corpus', 'precompute', 'train', 'eval', 'output']) {
    const el = $(k + '-hint');
    if (el) { el.textContent = hints[k] || ''; el.style.display = hints[k] ? '' : 'none'; }
  }
  $('log-path').textContent = data.path + '/pipeline.log';
  $('output-dir-note').textContent = data.output_dir || data.config.export.output_dir || '';
  renderConfig();
  renderMaterials();
  renderPipeline();
  renderOutput();
  const r = data.runtime_status;
  if (r !== lastRuntime) {
    lastRuntime = r;
    if (r === 'running') { logNext = 0; logText = ''; }
  }
  schedulePolling();
}

async function openProject(path) {
  clearError();
  const data = await api('/api/project/open', { path });
  dirty = false;
  applyProject(data);
  if (data.runtime_status === 'interrupted') {
    showNotice('This project was running when the process died (reboot?). Press Start / Resume to continue from the last checkpoint or shard.');
  } else if (data.adopted && data.adopted.length) {
    showNotice('Materials already on this machine were found and linked: ' + data.adopted.join(', ') + '. Nothing to download for them.');
  }
}

$('project-open').onclick = async () => {
  const start = project ? project.path : serverInfo.projects_root;
  const p = await openBrowse('dir', 'Open a project folder', start);
  if (p) openProject(p).catch(e => showError(e.message));
};
$('project-new-toggle').onclick = () => {
  const box = $('project-new');
  box.style.display = box.style.display === 'none' ? '' : 'none';
};
$('new-root-browse').onclick = async () => {
  const p = await openBrowse('dir', 'Projects root folder', $('new-root-text').textContent);
  if (p) $('new-root-text').textContent = p;
};
$('new-pack').onchange = () => {
  const pk = (serverInfo.packs || []).find(p => p.id === $('new-pack').value);
  $('pack-desc').textContent = pk ? pk.description : '';
  $('new-name').placeholder = pk && pk.default_name ? pk.default_name : 'adapter';
};
$('project-create').onclick = async () => {
  clearError();
  try {
    const pk = (serverInfo.packs || []).find(p => p.id === $('new-pack').value);
    const data = await api('/api/project/create', {
      name: $('new-name').value.trim() || (pk && pk.default_name) || 'adapter', dir: $('new-root-text').textContent,
      pack: $('new-pack').value,
    });
    dirty = false;
    $('project-new').style.display = 'none';
    applyProject(data);
  } catch (e) { showError(e.message); }
};

// ─── Materials ───────────────────────────────────────────────────────────────

function renderMaterials() {
  const box = $('materials-list');
  box.innerHTML = '';
  for (const m of project.materials) {
    const row = document.createElement('div');
    row.className = 'mat-row';
    const job = m.job;
    const status = job && job.status === 'running' ? 'downloading' : m.status;
    const chipCls = { ready: 'chip-ready', partial: 'chip-partial', downloading: 'chip-running', missing: 'chip-missing', unknown: '' }[status] || '';
    let sizeTxt = '';
    if (m.kind === 'hf_snapshot') {
      sizeTxt = m.bytes_total ? `${humanSize(m.bytes_done)} / ${humanSize(m.bytes_total)}` : humanSize(m.bytes_done);
      if (m.n_files) sizeTxt += ` · ${m.n_present}/${m.n_files} files`;
    } else if (m.status === 'ready') sizeTxt = humanSize(m.bytes_done);
    row.innerHTML = `
      <div class="mat-head">
        <span class="mat-title">${escapeHTML(m.title)}</span>
        <span class="badge">${m.required ? 'required' : 'optional'}</span>
        <span class="chip ${chipCls}">${escapeHTML(status)}</span>
        <span style="flex:1"></span>
        <span class="hint-inline">${escapeHTML(sizeTxt)}</span>
      </div>
      <div class="mat-hint">${escapeHTML(m.hint)}${m.kind === 'hf_snapshot' ? ' <code>' + escapeHTML(m.repo) + '</code>' : ''}</div>`;
    if (m.kind === 'local_file') {
      const fr = document.createElement('div');
      fr.className = 'field-row';
      fr.innerHTML = `<div class="path-box"><span class="path-text ${m.path ? 'mono' : 'placeholder'}">${escapeHTML(m.path || 'Select the file…')}</span></div>
        <button class="btn">Browse</button>`;
      fr.querySelector('button').onclick = async () => {
        const p = await openBrowse('model', 'Select ' + m.title, m.path ? m.path.replace(/[/\\][^/\\]*$/, '') : null);
        if (p) {
          config.materials = config.materials || {};
          config.materials[m.id] = { path: p };
          markDirty();
          await saveConfig().catch(e => showError(e.message));
          await refreshProject(true);
        }
      };
      row.appendChild(fr);
    } else {
      const pr = document.createElement('div');
      pr.className = 'mat-progress';
      const frac = m.bytes_total ? Math.min(1, (job ? job.bytes_done : m.bytes_done) / m.bytes_total) : 0;
      const rate = job && job.rate ? ` · ${humanSize(job.rate)}/s` : '';
      const eta = job && job.rate && m.bytes_total ? ` · ETA ${fmtDur((m.bytes_total - job.bytes_done) / job.rate)}` : '';
      pr.innerHTML = `<div class="progress-wrap"><div class="progress-bar" style="width:${(frac * 100).toFixed(1)}%"></div></div>
        <span>${(frac * 100).toFixed(0)}%${rate}${eta}</span>
        <button class="btn btn-small">${status === 'ready' ? 'Downloaded ✓' : status === 'downloading' ? 'Downloading…' : m.status === 'partial' ? 'Continue download' : 'Download'}</button>`;
      const btn = pr.querySelector('button');
      // present on disk -> nothing to fetch: the button stays disabled so a
      // click can never start a duplicate download (Refresh re-checks the disk)
      btn.disabled = status === 'ready' || status === 'downloading';
      btn.title = status === 'ready' ? 'All files of this material are on disk' : '';
      btn.onclick = async () => {
        try {
          await api('/api/materials/download', { path: project.path, id: m.id });
          await refreshProject(true);
          schedulePolling();
        } catch (e) { showError(e.message); }
      };
      row.appendChild(pr);
      const pp = document.createElement('div');
      pp.className = 'mat-path';
      pp.textContent = (m.linked && status === 'ready' ? 'found outside the project, linked: ' : '') + m.path;
      row.appendChild(pp);
      if (job && job.status === 'failed') {
        const er = document.createElement('div');
        er.className = 'hint';
        er.style.color = 'var(--danger)';
        er.textContent = 'download failed: ' + (job.error || '').split('\n')[0];
        row.appendChild(er);
      }
    }
    box.appendChild(row);
  }
  renderMaterialsSummary();
  $('start-btn').disabled = false;
  $('start-btn').textContent = project.runtime_status === 'running' ? 'Pipeline running…' :
    (project.state.status === 'done' ? 'Run again (skips finished stages)' :
      (project.runtime_status === 'interrupted' || project.state.status === 'stopped' || project.state.status === 'failed') ? 'Resume pipeline' : 'Start pipeline');
  $('start-btn').disabled = project.runtime_status === 'running';
}
function renderMaterialsSummary() {
  const mats = project.materials || [];
  const hf = mats.filter(m => m.kind === 'hf_snapshot');
  const running = hf.filter(m => m.status === 'downloading' || (m.job && m.job.status === 'running'));
  const missing = hf.filter(m => m.wanted && (m.status === 'missing' || m.status === 'partial'));
  const optionalMissing = hf.filter(m => !m.wanted && m.status !== 'ready');
  const localMissing = mats.filter(m => m.kind === 'local_file' && m.required && m.status !== 'ready');
  const btn = $('materials-download-all');
  const sum = $('materials-summary');
  let remaining = 0, known = true;
  for (const m of missing) { if (m.bytes_total) remaining += Math.max(0, m.bytes_total - m.bytes_done); else known = false; }
  if (running.length) {
    btn.disabled = true;
    btn.textContent = 'Downloading…';
    sum.textContent = `Downloading ${running.map(m => m.title).join(', ')} — runs detached, survives closing this page, resumes after a reboot.`;
  } else if (missing.length) {
    btn.disabled = false;
    btn.textContent = `Download missing (${missing.length}${known && remaining ? ', ~' + humanSize(remaining) : ''})`;
    sum.textContent = `Not on disk: ${missing.map(m => m.title).join(', ')}. Press Download missing to fetch them into the project folder`
      + (localMissing.length ? `; the student GGUF must be selected with Browse.` : '.');
  } else {
    btn.disabled = true;
    btn.textContent = 'All materials present';
    const linked = hf.filter(m => m.linked && m.status === 'ready');
    sum.textContent = (localMissing.length
      ? 'Downloads complete. Still needed: ' + localMissing.map(m => m.title).join(', ') + ' — pick the file with Browse.'
      : 'Every material is on disk — nothing to download.')
      + (linked.length ? ` (${linked.map(m => m.title).join(', ')} found outside the project and linked.)` : '')
      + (optionalMissing.length ? ` Optional, not fetched: ${optionalMissing.map(m => m.title).join(', ')}.` : '');
  }
  sum.style.display = sum.textContent ? '' : 'none';
}
$('materials-download-all').onclick = async () => {
  clearError();
  const btn = $('materials-download-all');
  btn.disabled = true;
  btn.textContent = 'Starting…';
  try {
    if (dirty) await saveConfig();
    const data = await api('/api/materials/download_all', { path: project.path });
    applyProject(data);
    if (data.started && data.started.length) showNotice('Download started for: ' + data.started.join(', ') + '. Progress is shown per material; the pipeline can be started once everything is ready.');
    schedulePolling();
  } catch (e) { showError(e.message); renderMaterials(); }
};
$('materials-refresh').onclick = () => refreshProject(true).then(() => {
  if (project && project.adopted && project.adopted.length) showNotice('Found and linked: ' + project.adopted.join(', '));
});

// ─── Pipeline control ────────────────────────────────────────────────────────

async function startPipeline(only, force) {
  clearError();
  try {
    if (dirty) await saveConfig();
    if (!only) {
      const missing = project.materials.filter(m => m.required && m.status !== 'ready').map(m => m.title);
      if (missing.length && !await askConfirm({
        kind: 'warn', title: 'Required materials are not ready', ok: 'Start anyway',
        text: 'The pipeline will fail at the first stage that needs one of these:',
        items: missing, note: 'Use Download missing / Browse in the Materials section first, or start now if you only want to build the corpus.',
      })) return;
    }
    await api('/api/pipeline/start', { path: project.path, only, force: !!force, mock_teacher: $('mock-teacher').checked });
    await refreshProject(false);
    switchTab('train');
  } catch (e) { showError(e.message); }
}
$('start-btn').onclick = () => startPipeline(null);
$('train-start').onclick = () => startPipeline(null);
$('corpus-build').onclick = () => startPipeline(['corpus']);
// Re-run export (+ eval) or eval alone from checkpoints/best.pt — the
// artifacts are rebuilt even when the pipeline would consider them current,
// so a GGUF deleted by mistake comes back with one click.
function canReExport() {
  return !!(project && project.artifacts.train.has_best && project.runtime_status !== 'running');
}
async function reExport(stages) {
  if (!canReExport()) return;
  const art = project.artifacts;
  const exporting = stages.includes('export');
  // paths inside the project folder shown relative to it
  const rel = p => (p && p.replace(/\\/g, '/').startsWith(project.path.replace(/\\/g, '/') + '/')) ? p.replace(/\\/g, '/').slice(project.path.length + 1) : p;
  const items = [{ label: 'checkpoint', value: 'checkpoints/best.pt' + (art.train.step != null ? ' (step ' + art.train.step + ')' : '') }];
  if (exporting) items.push({ label: art.export.done ? 'overwrites' : 'writes', value: rel(art.export.path) });
  items.push({ label: exporting ? 'evaluation' : 'writes', value: rel(art.eval.path) });
  const ok = await askConfirm({
    kind: 'info',
    title: exporting ? (art.export.done ? 'Re-export the adapter GGUF' : 'Export the adapter GGUF') : 'Re-run the evaluation',
    text: exporting
      ? 'Rebuilds the f16 GGUF from the best checkpoint and evaluates it on the validation shards. Training is not touched.'
      : 'Evaluates the exported GGUF on the validation shards again. The GGUF is not touched.',
    items,
    note: 'Runs as the detached pipeline (about a minute); progress shows in the stage strip.',
    ok: exporting ? (art.export.done ? 'Re-export' : 'Export') : 'Evaluate',
  });
  if (!ok) return;
  await startPipeline(stages, true);
}
$('output-export').onclick = () => reExport(['export', 'eval']);
async function stopPipeline() {
  try { await api('/api/pipeline/stop', { path: project.path }); showNotice('Stop requested — the pipeline saves its checkpoint / finishes the current shard and exits.'); }
  catch (e) { showError(e.message); }
}
$('stop-btn').onclick = stopPipeline;
$('train-stop').onclick = stopPipeline;

// Snapshot export: the adapter at its current step -> <name>-step<N>-f16.gguf,
// while training runs (the trainer saves the step first) or from the
// checkpoint on disk after a stop. Training is never interrupted.
function canSnapshot() {
  return !!(project && !project.snapshot_blocker && !(project.snapshot_job && project.snapshot_job.status === 'running'));
}
async function exportSnapshot() {
  if (!canSnapshot()) return;
  const art = project.artifacts.train, tr = project.state.stages.train || {};
  const running = project.runtime_status === 'running' && project.state.stage === 'train';
  const liveStep = running && tr.step != null ? tr.step : art.step;
  const fmtCk = (step, cos) => step != null ? `step ${step}` + (cos != null ? `, val cos ${fmtNum(cos)}` : '') : 'not saved yet';
  const values = await askConfirm({
    kind: 'info',
    title: 'Export a snapshot of the adapter',
    text: running
      ? 'The trainer validates and saves the current step first, then the GGUF is written from it; training keeps running.'
      : 'Writes the GGUF from the checkpoint on disk. Resume training later and take more snapshots to compare.',
    fields: [
      { id: 'checkpoint', type: 'radio', label: 'Checkpoint', value: 'last', options: [
        { value: 'last', label: 'Latest step', desc: running ? `(around step ${liveStep != null ? liveStep : '?'} — saved on request)` : `(${fmtCk(art.step, art.last_val_cos)})`, disabled: !running && !art.has_last },
        { value: 'best', label: 'Best validation score so far', desc: `(${fmtCk(art.best_step, art.best_val_cos)})`, disabled: !art.has_best && !running },
      ] },
      { id: 'eval', type: 'checkbox', text: 'Also evaluate the GGUF on the validation shards', value: false,
        desc: running ? '(on the CPU while training owns the GPU — a few minutes)' : '(about a minute)' },
    ],
    items: [{ label: 'writes', value: `${project.config.name}-step<N>-f16.gguf in ${project.output_dir}` }],
    note: 'Runs detached like a download; the file appears under Output › Snapshots with the step and its validation cosine. The final export at the end of the run is not affected.',
    ok: 'Export snapshot',
  });
  if (!values) return;
  try {
    await api('/api/snapshot', { path: project.path, checkpoint: values.checkpoint, eval: !!values.eval });
    showNotice(running ? 'Snapshot requested — the trainer saves the current step, then the GGUF is exported.' : 'Snapshot export started.');
    await refreshProject(false);
  } catch (e) { showError(e.message); }
}
$('train-snapshot').onclick = () => exportSnapshot().catch(e => showError(e.message));
$('output-snapshot').onclick = () => exportSnapshot().catch(e => showError(e.message));
$('reset-select').onchange = async () => {
  const what = $('reset-select').value;
  $('reset-select').value = '';
  if (!what || !project) return;
  const spec = {
    train: { title: 'Reset training', text: 'Deletes the checkpoints; training restarts from step 0.', items: ['checkpoints/'], kind: 'danger', ok: 'Delete' },
    shards: { title: 'Reset shards + training', text: 'Deletes the precomputed teacher shards and the checkpoints; precompute and training run again.', items: ['shards/', 'checkpoints/'], kind: 'danger', ok: 'Delete' },
    corpus: { title: 'Reset everything after materials', text: 'Deletes the corpus, the shards and the checkpoints. Downloaded materials and exported files stay.', items: ['data/', 'shards/', 'checkpoints/'], kind: 'danger', ok: 'Delete' },
    state: { title: 'Clear the recorded status', text: 'Forgets which stages ran. No file is deleted; the next run re-detects progress from disk.', items: ['state.json'], kind: 'warn', ok: 'Clear' },
  }[what];
  if (!await askConfirm(spec)) return;
  try { applyProject(await api('/api/project/reset', { path: project.path, what })); }
  catch (e) { showError(e.message); }
};

function setStatus(runtime, stage) {
  const labels = { idle: 'Idle', running: 'Running', interrupted: 'Interrupted', stopped: 'Stopped', failed: 'Failed', done: 'Done' };
  const dot = { running: 'running', done: 'completed', failed: 'error', interrupted: 'error' }[runtime] || '';
  $('status-label').textContent = labels[runtime] || runtime;
  $('status-dot').className = 'dot' + (dot ? ' ' + dot : '');
  $('sb-status').textContent = labels[runtime] || runtime;
  $('stop-btn').style.display = runtime === 'running' ? '' : 'none';
  $('train-stop').style.display = runtime === 'running' ? '' : 'none';
  $('train-start').style.display = runtime === 'running' ? 'none' : '';
  $('sb-stage').textContent = stage ? (project.stage_titles[stage] || stage) : '–';
}

function stageProgress(stage, info, art) {
  // -> {frac (0..1 or null), text}
  if (!info) info = {};
  if (stage === 'corpus') {
    const ready = filename(art.train_file) + ' + ' + filename(art.val_file) + ' ready';
    return { frac: art.done ? 1 : null, text: info.status === 'running' && info.detail ? info.detail : (art.done ? ready : (info.detail || 'not built')) };
  }
  if (stage === 'precompute_val' || stage === 'precompute_train') {
    const total = info.total_shards || art.total_shards, done = info.done_shards != null ? info.done_shards : art.done_shards;
    if (!total) return { frac: null, text: info.detail || 'waiting for the corpus' };
    const frac = Math.min(1, (done + (info.shard_progress || 0)) / total);
    let t = `${done}/${total} shards`;
    if (info.prompts_per_s) t += ` · ${info.prompts_per_s.toFixed(2)} prompts/s`;
    if (info.eta_s != null && info.status === 'running') t += ` · ETA ${fmtDur(info.eta_s)}`;
    if (info.detail && info.status === 'running' && !done) t += ` · ${info.detail}`;
    return { frac, text: t };
  }
  if (stage === 'train') {
    const steps = info.steps || art.steps, step = info.step != null ? info.step : (art.step || 0);
    let t = `step ${step}/${steps}`;
    if (info.best_val_cos != null) t += ` · best val cos ${fmtNum(info.best_val_cos)}`;
    if (info.eta_s != null && info.status === 'running') t += ` · ETA ${fmtDur(info.eta_s)}`;
    return { frac: steps ? step / steps : null, text: t };
  }
  if (stage === 'export') return { frac: art.done ? 1 : null, text: art.done ? filename(art.path) : 'not exported' };
  if (stage === 'eval') return { frac: art.done ? 1 : null, text: art.done ? 'eval.json written' : (info.status || 'pending') };
  return { frac: null, text: '' };
}

function renderPipeline() {
  const st = project.state, art = project.artifacts, runtime = project.runtime_status;
  setStatus(runtime, st.stage);
  const strip = $('stage-strip');
  strip.innerHTML = '';
  let current = null;
  for (const s of project.stages) {
    const info = st.stages[s] || {};
    let status = info.status || (art[s].done ? 'done' : 'pending');
    if (art[s].done && status !== 'running') status = 'done';
    if (status === 'running' && runtime !== 'running') status = 'interrupted';
    // state.json remembers "done", but the file is gone (deleted / stale vs best.pt)
    if ((s === 'export' || s === 'eval') && status === 'done' && !art[s].done) status = 'missing';
    if (status === 'running' || (!current && status !== 'done' && status !== 'skipped')) current = current || s;
    const pr = stageProgress(s, info, art[s]);
    const div = document.createElement('div');
    div.className = 'stage ' + status;
    div.innerHTML = `<div class="name">${escapeHTML(project.stage_titles[s])}</div>
      <div class="sub" title="${escapeHTML(pr.text)}">${escapeHTML(status === 'pending' ? pr.text || 'pending' : status + (pr.text ? ' · ' + pr.text : ''))}</div>`;
    if ((s === 'export' || s === 'eval') && canReExport()) {
      div.classList.add('clickable');
      div.title = (s === 'export' ? 'Regenerate the GGUF (+ eval)' : 'Re-run the evaluation') + ' from checkpoints/best.pt';
      div.onclick = () => reExport(s === 'export' ? ['export', 'eval'] : ['eval']).catch(e => showError(e.message));
    }
    strip.appendChild(div);
  }
  const re = canReExport();
  $('output-export').style.display = re ? '' : 'none';
  $('output-export').textContent = art.export.done ? 'Re-export GGUF' : 'Export GGUF';
  $('stage-hint').style.display = re ? '' : 'none';
  const snap = canSnapshot();
  const snapPossible = snap || !!(art.train.has_last || art.train.has_best) || (runtime === 'running' && st.stage === 'train');
  $('train-snapshot').style.display = snapPossible ? '' : 'none';
  $('train-snapshot').disabled = !snap;
  $('output-snapshot').style.display = snapPossible ? '' : 'none';
  $('output-snapshot').disabled = !snap;
  $('snapshot-hint').style.display = snapPossible ? '' : 'none';
  renderSnapshotJob();
  const focus = st.stage && runtime === 'running' ? st.stage : (current || st.stage || 'corpus');
  const pr = stageProgress(focus, st.stages[focus], art[focus]);
  const bar = $('stage-bar');
  if (runtime === 'running' && pr.frac == null) { bar.className = 'progress-bar indeterminate'; }
  else { bar.className = 'progress-bar'; bar.style.width = ((pr.frac || 0) * 100).toFixed(1) + '%'; }
  $('stage-text').textContent = `${project.stage_titles[focus]}: ${pr.text}` +
    (st.started && runtime === 'running' ? ` · running for ${fmtDur(Date.now() / 1000 - st.started)}` : '');
  $('sb-progress').textContent = pr.text;
  const err = st.error && runtime !== 'running';
  $('run-error').style.display = err ? '' : 'none';
  $('run-error').textContent = err ? 'Last error: ' + st.error + ' (see Logs)' : '';
  const tr = st.stages.train || {};
  const cells = [
    ['step', tr.step != null ? `${tr.step} / ${tr.steps}` : (art.train.step != null ? `${art.train.step} / ${art.train.steps}` : '—')],
    ['loss', fmtNum(tr.loss)], ['rel_mse', fmtNum(tr.rel_mse)], ['cos', fmtNum(tr.cos)],
    ['val cos', fmtNum(tr.val_cos)], ['best val cos', fmtNum(tr.best_val_cos)],
    ...(tr.val_cos_vis != null ? [['val cos (vision)', fmtNum(tr.val_cos_vis)], ['val cos (text)', fmtNum(tr.val_cos_txt)]] : []),
    ['lr', tr.lr != null ? Number(tr.lr).toExponential(2) : '—'],
    ['prompts/s', tr.prompts_per_s != null ? Number(tr.prompts_per_s).toFixed(1) : '—'],
    ['ETA', tr.status === 'running' ? fmtDur(tr.eta_s) : '—'],
  ];
  $('train-metrics').innerHTML = cells.map(([k, v]) => `<div class="cell"><div class="k">${k}</div><div class="v">${escapeHTML(v)}</div></div>`).join('');
  $('metrics-note').textContent = art.train.has_last ? `checkpoints: last${art.train.has_best ? ' + best' : ''}` : 'no checkpoints yet';
  renderStepsControl();
}

// ─── Planned steps (Train tab) ───────────────────────────────────────────────
// The count lives in project.json (train.steps).  The trainer re-reads it at
// every log interval, so Apply works before, during and after a run.

function renderStepsControl() {
  const inp = $('train-steps');
  const saved = Number(project.config.train.steps);
  if (document.activeElement !== inp) inp.value = saved;
  const art = project.artifacts.train, st = project.state, running = project.runtime_status === 'running';
  const tr = st.stages.train || {};
  const live = running && st.stage === 'train' ? (tr.steps || saved) : null;
  const ckStep = art.step;
  let note = '';
  if (live != null && live !== saved) note = `saved ${saved}, trainer still on ${live} — picked up at the next log interval`;
  else if (!running && ckStep != null && ckStep >= saved) note = `checkpoint at step ${ckStep} — raise the count and press Start / Resume to train further`;
  else if (!running && ckStep != null) note = `checkpoint at step ${ckStep} of ${saved}`;
  $('train-steps-note').textContent = note;
  $('train-steps-apply').disabled = Number(inp.value) === saved;
}
$('train-steps').addEventListener('input', () => { $('train-steps-apply').disabled = Number($('train-steps').value) === Number(project.config.train.steps); });
$('train-steps').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); applySteps(); } });
$('train-steps-apply').onclick = () => applySteps();

async function applySteps() {
  if (!project) return;
  const v = Number($('train-steps').value);
  if (!Number.isInteger(v) || v < 1) { showError('Steps must be a positive whole number.'); return; }
  const before = Number(project.config.train.steps);
  if (v === before) return;
  clearError();
  try {
    const data = await api('/api/project/steps', { path: project.path, steps: v });
    if (config) config.train.steps = data.steps;      // keep the Setup form in sync without marking it dirty
    applyProject(data);
    const art = project.artifacts.train, running = project.runtime_status === 'running';
    const ckStep = art.step;
    if (running && project.state.stage === 'train') {
      showNotice(v > before ? `Planned steps ${before} → ${v}: the trainer continues to ${v} (picked up within a log interval).`
        : `Planned steps ${before} → ${v}: the trainer finishes at its next log interval (validate + save), then exports.`);
    } else if (!running && ckStep != null && ckStep < v && (ckStep >= before || project.state.status === 'done')) {
      showNotice(`Planned steps ${before} → ${v}. Press Start / Resume to continue training from step ${ckStep}.`);
    } else if (!running && ckStep != null && ckStep >= v) {
      showNotice(`Planned steps ${before} → ${v}: the checkpoint at step ${ckStep} already covers it — the training stage counts as done; export runs from best.pt.`);
    } else {
      showNotice(`Planned steps ${before} → ${v} saved.`);
    }
  } catch (e) { showError(e.message); }
}

// ─── Metrics chart ───────────────────────────────────────────────────────────

let chartRows = [];
function drawChart() {
  const c = $('chart');
  const W = c.clientWidth || 700, H = c.height;
  const dpr = window.devicePixelRatio || 1;
  c.width = W * dpr; c.style.height = H + 'px';
  const ctx = c.getContext('2d');
  ctx.scale(dpr, 1);
  ctx.clearRect(0, 0, W, H);
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  ctx.fillStyle = dark ? '#888' : '#666';
  ctx.font = '11px system-ui';
  if (!chartRows.length) { ctx.fillText('no training log yet', 12, 20); return; }
  const pad = { l: 44, r: 44, t: 10, b: 22 };
  const x0 = chartRows[0].step, x1 = Math.max(chartRows[chartRows.length - 1].step, x0 + 1);
  const X = s => pad.l + (s - x0) / (x1 - x0) * (W - pad.l - pad.r);
  const lossMax = Math.max(...chartRows.map(r => r.loss)) || 1;
  const YL = v => pad.t + (1 - v / lossMax) * (H - pad.t - pad.b);
  const cosVals = chartRows.map(r => r.cos).concat(chartRows.filter(r => r.val_cos != null).map(r => r.val_cos));
  const cMin = Math.max(-1, Math.min(...cosVals) - 0.02), cMax = 1;
  const YC = v => pad.t + (1 - (v - cMin) / (cMax - cMin)) * (H - pad.t - pad.b);
  ctx.strokeStyle = dark ? '#333' : '#e4e4e4';
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + i * (H - pad.t - pad.b) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText((lossMax * (1 - i / 4)).toFixed(2), 4, y + 4);
    ctx.fillText((cMax - (cMax - cMin) * i / 4).toFixed(3), W - pad.r + 4, y + 4);
  }
  for (let i = 0; i <= 4; i++) {
    const s = Math.round(x0 + (x1 - x0) * i / 4);
    ctx.fillText(String(s), X(s) - 10, H - 6);
  }
  const line = (color, pts) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    let first = true;
    for (const [x, y] of pts) { if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y); }
    ctx.stroke();
  };
  line('#4f6ef7', chartRows.map(r => [X(r.step), YL(r.loss)]));
  line('#2f9e63', chartRows.map(r => [X(r.step), YC(r.cos)]));
  const val = chartRows.filter(r => r.val_cos != null && r.val_cos > -1);
  if (val.length) line('#e0a52a', val.map(r => [X(r.step), YC(r.val_cos)]));
}
async function refreshMetrics() {
  if (!project) return;
  try { chartRows = (await api('/api/metrics?path=' + encodeURIComponent(project.path))).rows || []; drawChart(); }
  catch { /* ignore */ }
}
window.addEventListener('resize', drawChart);

// ─── Hardware ────────────────────────────────────────────────────────────────

function hwRows(hw) {
  const rows = [['OS', hw.os], ['Host', hw.hostname], ['CPU threads', hw.cpu_count ?? '—'],
    ['CPU load', hw.cpu_percent != null ? hw.cpu_percent + ' %' : '—'],
    ['RAM', hw.total_ram_mib ? `${((hw.total_ram_mib - (hw.available_ram_mib || 0)) / 1024).toFixed(1)} / ${(hw.total_ram_mib / 1024).toFixed(1)} GiB used` : '—']];
  for (const g of hw.gpus || []) {
    rows.push([`GPU ${g.index}`, g.name]);
    rows.push(['  VRAM', `${g.vram_used_mib ?? '?'} / ${g.vram_total_mib} MiB`]);
    rows.push(['  utilization', g.utilization_percent != null ? g.utilization_percent + ' %' : '—']);
    rows.push(['  temperature', g.temperature_c != null ? g.temperature_c + ' °C' + (g.power_w != null ? ` · ${g.power_w} W` : '') : '—']);
  }
  if (!(hw.gpus || []).length) rows.push(['GPU', 'none detected (nvidia-smi not found)']);
  if (hw.disk_free != null) rows.push(['Disk free (project)', humanSize(hw.disk_free)]);
  if (hw.process) rows.push(['Pipeline process', `pid ${hw.process.pid} · CPU ${hw.process.cpu_percent} % · RSS ${(hw.process.rss_mib / 1024).toFixed(1)} GiB · ${hw.process.threads} threads`]);
  return rows;
}
async function refreshHardware(liveOnly) {
  try {
    const hw = await api('/api/hardware' + (project ? '?path=' + encodeURIComponent(project.path) : ''));
    const rows = hwRows(hw);
    $('hw-live').innerHTML = rows.filter(([k]) => !/^(OS|Host)$/.test(k))
      .map(([k, v]) => `<div class="cell"><div class="k">${escapeHTML(k.trim())}</div><div class="v" style="font-size:12px">${escapeHTML(String(v))}</div></div>`).join('');
    $('hw-live-note').textContent = new Date().toLocaleTimeString();
    if (!liveOnly) {
      $('hw-info').innerHTML = rows.map(([k, v]) => `<span class="k">${escapeHTML(k)}</span><span>${escapeHTML(String(v))}</span>`).join('');
      const e = hw.env || {};
      const env = [['Python', `${e.python} (${e.executable})`], ['torch', `${e.torch} · CUDA ${e.cuda_version || 'n/a'} · available: ${e.cuda_available}`],
        ['transformers', e.transformers], ['accelerate', e.accelerate], ['gguf-connector', e['gguf-connector']], ['huggingface_hub', e.huggingface_hub],
        ['datasets', e.datasets], ['safetensors', e.safetensors]];
      for (const d of e.torch_devices || []) env.push([`cuda:${d.index}`, `${d.name} · ${(d.total_mib / 1024).toFixed(1)} GiB`]);
      $('env-info').innerHTML = env.map(([k, v]) => `<span class="k">${escapeHTML(k)}</span><span>${escapeHTML(String(v ?? '—'))}</span>`).join('');
      fillDevices(e.torch_devices || []);
    }
  } catch (err) {
    $('hw-info').innerHTML = `<span class="k">error</span><span>${escapeHTML(err.message)}</span>`;
  }
}
$('hw-refresh').onclick = () => refreshHardware(false);
function fillDevices(devs) {
  for (const id of ['precompute-device', 'train-device']) {
    const sel = $(id), cur = sel.value;
    sel.innerHTML = '<option value="auto">auto</option>' + devs.map(d => `<option value="cuda:${d.index}">cuda:${d.index} ${escapeHTML(d.name)}</option>`).join('') + '<option value="cpu">cpu</option>';
    sel.value = cur || 'auto';
  }
}

// ─── Logs ────────────────────────────────────────────────────────────────────

async function fetchLog() {
  if (!project) return;
  const data = await api(`/api/log?path=${encodeURIComponent(project.path)}&after=${logNext}`);
  if (data.next < logNext) { logText = ''; }
  if (data.text) {
    logText += data.text;
    if (logText.length > 400000) logText = logText.slice(-300000);
    const view = $('log-view');
    view.textContent = logText;
    if ($('log-follow').checked) view.scrollTop = view.scrollHeight;
  }
  logNext = data.next;
}
$('logs-refresh').onclick = () => fetchLog().catch(e => showError(e.message));
$('logs-clear').onclick = () => { logText = ''; $('log-view').textContent = ''; };

// ─── Output ──────────────────────────────────────────────────────────────────

function renderOutput() {
  const box = $('output-files');
  box.innerHTML = '';
  if (!project.output_files.length) box.innerHTML = '<p class="hint">Nothing exported yet.</p>';
  for (const f of project.output_files) {
    const row = document.createElement('div');
    row.className = 'file-row';
    row.innerHTML = `<span class="name" title="${escapeHTML(f.path)}">${escapeHTML(f.name)}</span>
      <span class="meta">${humanSize(f.size)} · ${new Date(f.mtime * 1000).toLocaleString()}</span>`;
    box.appendChild(row);
  }
  const ev = project.eval;
  if (ev) {
    let keys = (packInfo().eval_keys || []).filter(k => k in ev);
    if (!keys.length) keys = Object.keys(ev).filter(k => typeof ev[k] === 'number' && k !== 'time');
    $('eval-info').innerHTML = keys.map(k => `<span class="k">${k}</span><span>${typeof ev[k] === 'number' && !Number.isInteger(ev[k]) ? ev[k].toFixed(4) : escapeHTML(String(ev[k]))}</span>`).join('');
    $('eval-note').textContent = new Date(ev.time * 1000).toLocaleString() + ' · ' + filename(project.artifacts.eval.path);
  } else {
    $('eval-info').innerHTML = '<span class="k">—</span><span>no evaluation yet</span>';
    $('eval-note').textContent = '';
  }
  $('cmd-preview').textContent = project.engine_command || '';
  renderSnapshots();
}

function snapshotEvalKey(ev) {
  if (!ev) return null;
  const keys = (packInfo().eval_keys || []).filter(k => k in ev);
  return keys[0] || Object.keys(ev).find(k => typeof ev[k] === 'number' && k.startsWith('cos')) || null;
}
function renderSnapshotJob() {
  const job = project.snapshot_job || { status: 'none' };
  const el = $('snapshot-job');
  let text = '';
  if (job.status === 'running') text = 'Snapshot export running: ' + (job.phase || 'starting') + ' (see snapshot.log)';
  else if (job.status === 'failed') text = 'Last snapshot export failed: ' + (job.error || 'see snapshot.log');
  else if (job.status === 'done' && job.path) text = `Last snapshot: ${filename(job.path)} (step ${job.step}${job.val_cos != null ? ', val cos ' + fmtNum(job.val_cos) : ''})`;
  el.textContent = text;
  el.style.display = text ? '' : 'none';
  el.style.color = job.status === 'failed' ? 'var(--danger)' : '';
}
function renderSnapshots() {
  const box = $('snapshot-table');
  const snaps = project.snapshots || [];
  const job = project.snapshot_job || { status: 'none' };
  $('snapshots-note').textContent = job.status === 'running' ? 'exporting… ' + (job.phase || '') : (snaps.length ? `${snaps.length} snapshot${snaps.length > 1 ? 's' : ''}` : '');
  if (!snaps.length) {
    box.innerHTML = '<p class="hint">' + (project.snapshot_blocker && !(project.artifacts.train.has_last) ? 'No checkpoint yet — snapshots become available once the train stage has started.' : 'No snapshots yet. Press <b>Export snapshot</b> at any point of the run.') + '</p>';
    return;
  }
  const planned = snaps[snaps.length - 1].planned_steps;
  const rows = snaps.map(e => {
    const key = snapshotEvalKey(e.eval);
    const pct = e.planned_steps ? Math.round(100 * e.step / e.planned_steps) : null;
    return `<tr>
      <td class="num">${e.step}${pct != null ? ` <span class="tag">${pct}%</span>` : ''}${e.checkpoint === 'best' ? '<span class="tag best">best</span>' : ''}</td>
      <td class="num">${e.val_cos != null ? fmtNum(e.val_cos) : '—'}</td>
      <td class="num">${key ? `${fmtNum(e.eval[key])} <span class="tag">${escapeHTML(key)}</span>` : '—'}</td>
      <td class="file" title="${escapeHTML(e.path)}">${escapeHTML(e.name || filename(e.path))}</td>
      <td class="num">${humanSize(e.size || 0)}</td>
      <td class="num">${new Date(e.time * 1000).toLocaleString()}</td>
    </tr>`;
  });
  box.innerHTML = `<div style="overflow-x:auto"><table class="snap-table">
    <thead><tr><th>step${planned ? ' / ' + planned : ''}</th><th>val cos</th><th>eval</th><th>file</th><th>size</th><th>exported</th></tr></thead>
    <tbody>${rows.join('')}</tbody></table></div>`;
}
$('copy-cmd').onclick = () => {
  navigator.clipboard.writeText($('cmd-preview').textContent);
  $('copy-cmd').textContent = 'Copied!';
  setTimeout(() => { $('copy-cmd').textContent = 'Copy'; }, 1200);
};

// ─── Polling ─────────────────────────────────────────────────────────────────

async function refreshProject(online) {
  if (!project) return;
  try {
    const data = await api(`/api/project?path=${encodeURIComponent(project.path)}&online=${online ? 1 : 0}`);
    applyProject(data);
  } catch (e) { showError(e.message); }
}

function schedulePolling() {
  clearTimeout(pollTimer);
  const downloading = (project.materials || []).some(m => m.job && m.job.status === 'running')
    || (project.snapshot_job && project.snapshot_job.status === 'running');
  const running = project.runtime_status === 'running';
  if (!running && !downloading) { clearInterval(hwTimer); hwTimer = null; return; }
  pollTimer = setTimeout(async () => {
    await refreshProject(false);
    if (running) {
      fetchLog().catch(() => {});
      if (Date.now() - metricsAt > 8000 && (project.state.stage === 'train' || project.state.stage === 'export')) { metricsAt = Date.now(); refreshMetrics(); }
    }
  }, downloading && !running ? 2000 : 1500);
  if (!hwTimer) hwTimer = setInterval(() => refreshHardware(true), 3000);
}

// ─── Tabs / theme / init ─────────────────────────────────────────────────────

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-page').forEach(p => p.style.display = p.id === 'page-' + name ? '' : 'none');
  if (name === 'hardware') refreshHardware(false);
  if (name === 'train') { refreshMetrics(); refreshHardware(true); }
  if (name === 'logs') fetchLog().catch(() => {});
}
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => switchTab(t.dataset.tab)));

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  $('theme-icon-sun').style.display = theme === 'dark' ? '' : 'none';
  $('theme-icon-moon').style.display = theme === 'dark' ? 'none' : '';
  localStorage.setItem(THEME_KEY, theme);
  drawChart();
}
$('theme-btn').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
};
applyTheme(localStorage.getItem(THEME_KEY) || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));

async function init() {
  bindConfigInputs();
  try {
    serverInfo = await api('/api/status');
    $('app-version').textContent = 'v' + serverInfo.version;
    $('new-root-text').textContent = serverInfo.projects_root;
    $('new-pack').innerHTML = (serverInfo.packs || []).map(p => `<option value="${p.id}">${escapeHTML(p.title)}</option>`).join('');
    $('new-pack').onchange();
    if (serverInfo.mock_teacher) $('mock-teacher').checked = true;
  } catch (err) { showError('Could not reach the local server: ' + err.message); return; }
  refreshHardware(false);
  const last = new URLSearchParams(location.search).get('project') || localStorage.getItem(PROJECT_KEY) || serverInfo.last_project;
  if (last) {
    try { await openProject(last); }
    catch { $('project-new').style.display = ''; }
  } else {
    $('project-new').style.display = '';
  }
  const tab = location.hash.replace('#', '');
  if (tab && $('page-' + tab)) switchTab(tab);
}
init();
