const $ = (selector) => document.querySelector(selector);

let studio = null;
let drawer = null;
let button = null;
let loading = false;

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function boot() {
  studio = window.RealSaSStudio;
  const modebar = document.querySelector('.modebar');
  if (!studio || !modebar) {
    requestAnimationFrame(boot);
    return;
  }
  installButton(modebar);
  installDrawer();
  window.addEventListener('realsas:studio-saved', () => {
    if (drawer?.classList.contains('open')) void refresh();
  });
  window.addEventListener('realsas:studio-session', () => {
    if (drawer?.classList.contains('open')) void refresh();
  });
}

function installButton(modebar) {
  if ($('#studioHistoryToggle')) {
    button = $('#studioHistoryToggle');
    return;
  }
  button = document.createElement('button');
  button.id = 'studioHistoryToggle';
  button.textContent = 'HISTORY';
  button.title = 'Open immutable Studio authoring revision history';
  const save = $('#saveLayer');
  if (save) modebar.insertBefore(button, save);
  else modebar.append(button);
  button.onclick = () => toggle();
}

function installDrawer() {
  if ($('#studioHistoryDrawer')) {
    drawer = $('#studioHistoryDrawer');
    return;
  }
  drawer = document.createElement('aside');
  drawer.id = 'studioHistoryDrawer';
  drawer.className = 'studio-history-drawer';
  drawer.innerHTML = `
    <div class="history-head">
      <div><strong>STUDIO HISTORY</strong><span>Immutable authoring revisions</span></div>
      <button id="studioHistoryClose">×</button>
    </div>
    <div class="history-toolbar"><button id="studioHistoryRefresh">REFRESH</button><span id="studioHistoryState">—</span></div>
    <div id="studioHistoryList" class="history-list"><div class="history-empty">Open a persisted Studio project.</div></div>
    <div class="history-foot">History is append-only. Requalification state belongs to each authoring head; ProductRevision authority is never rewritten here.</div>`;
  document.body.append(drawer);
  $('#studioHistoryClose').onclick = () => close();
  $('#studioHistoryRefresh').onclick = () => void refresh();
}

function toggle() {
  if (!drawer) return;
  if (drawer.classList.contains('open')) close();
  else open();
}

function open() {
  drawer.classList.add('open');
  button?.classList.add('active');
  void refresh();
}

function close() {
  drawer?.classList.remove('open');
  button?.classList.remove('active');
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString(undefined, {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

async function refresh() {
  if (loading || !drawer?.classList.contains('open')) return;
  loading = true;
  const state = $('#studioHistoryState');
  const list = $('#studioHistoryList');
  if (state) state.textContent = 'LOADING…';
  try {
    const persistence = window.RealSaSStudioPersistence;
    const bundle = studio?.state?.bundle;
    if (!persistence || !bundle) {
      if (list) list.innerHTML = '<div class="history-empty">Open a character bundle first.</div>';
      if (state) state.textContent = 'NO SESSION';
      return;
    }
    const session = await persistence.ensureSession(bundle);
    if (!session?.project?.project_id) {
      if (list) list.innerHTML = '<div class="history-empty">This bundle is local/presentation-only and has no Platform Studio revision history.</div>';
      if (state) state.textContent = 'LOCAL ONLY';
      return;
    }
    const revisions = await persistence.history(100);
    if (state) state.textContent = `${revisions.length} REVISION${revisions.length === 1 ? '' : 'S'}`;
    if (!list) return;
    if (!revisions.length) {
      list.innerHTML = '<div class="history-empty">No user authoring revisions yet. The proof-qualified base is unchanged.</div>';
      return;
    }
    list.innerHTML = revisions.map((row) => {
      const sha = String(row.payload_sha256 || '');
      const head = Boolean(row.is_head);
      const requal = Boolean(row.requires_requalification);
      return `<article class="history-row${head ? ' head' : ''}">
        <div class="history-rev"><strong>REV ${Number(row.revision_number || 0)}</strong>${head ? '<span>HEAD</span>' : ''}</div>
        <div class="history-meta"><span>${escapeHTML(row.created_by || '—')}</span><span>${escapeHTML(formatTime(row.created_at))}</span></div>
        <div class="history-sha">${escapeHTML(sha.slice(0, 16) || '—')}</div>
        <div class="history-state ${requal ? 'requal' : 'qualified'}">${requal ? 'REQUALIFICATION REQUIRED' : 'QUALIFIED'}</div>
      </article>`;
    }).join('');
  } catch (error) {
    if (list) list.innerHTML = `<div class="history-error">${escapeHTML(error?.message || error)}</div>`;
    if (state) state.textContent = 'ERROR';
  } finally {
    loading = false;
  }
}

boot();
