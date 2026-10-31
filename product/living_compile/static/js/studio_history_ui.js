const $ = (selector) => document.querySelector(selector);

let studio = null;
let drawer = null;
let button = null;
let loading = false;
let restoring = false;

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
    <div class="history-foot">RESTORE is append-only: it copies the selected historical payload into a new revision. ProductRevision authority is never rewritten and the restored authoring head still requires compiler requalification.</div>`;
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

function applyRestoredHead(head) {
  if (!head?.authoring_payload || typeof head.authoring_payload !== 'object') {
    throw new Error('Restored Studio head does not contain an authoring payload');
  }
  studio.document.loadAuthoringLayer(head.authoring_payload);
  studio.journal.clear();
  studio.state.previewPose?.clear?.();
  studio.state.frame = null;
  studio.state.play = false;
  studio.dirty.hydrate(head);
  document.querySelector('.mode.active')?.click();
  window.dispatchEvent(new CustomEvent('realsas:studio-authoring-restored', {detail: head}));
}

async function restoreRevision(revisionNumber) {
  if (restoring) return;
  if (studio?.dirty?.status !== 'clean') {
    window.alert('Save or resolve current Studio edits before restoring a historical revision.');
    return;
  }
  const target = Number(revisionNumber);
  if (!Number.isInteger(target) || target <= 0) return;
  const accepted = window.confirm(
    `Restore REV ${target}?\n\nThis does not rewind history. RealSaS will create a new immutable revision from REV ${target}, and the new head will require compiler requalification.`,
  );
  if (!accepted) return;

  restoring = true;
  const state = $('#studioHistoryState');
  if (state) state.textContent = `RESTORING REV ${target}…`;
  try {
    const persistence = window.RealSaSStudioPersistence;
    if (!persistence?.restore) throw new Error('Studio restore capability unavailable');
    const payload = await persistence.restore(target);
    applyRestoredHead(payload.head);
    if (state) state.textContent = `RESTORED AS REV ${Number(payload.head.revision_number || 0)}`;
    await refresh(true);
  } catch (error) {
    if (state) state.textContent = 'RESTORE ERROR';
    window.alert(error?.message || String(error));
  } finally {
    restoring = false;
  }
}

async function refresh(force = false) {
  if ((!force && loading) || !drawer?.classList.contains('open')) return;
  loading = true;
  const state = $('#studioHistoryState');
  const list = $('#studioHistoryList');
  if (state && !restoring) state.textContent = 'LOADING…';
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
    if (state && !restoring) state.textContent = `${revisions.length} REVISION${revisions.length === 1 ? '' : 'S'}`;
    if (!list) return;
    if (!revisions.length) {
      list.innerHTML = '<div class="history-empty">No user authoring revisions yet. The proof-qualified base is unchanged.</div>';
      return;
    }
    list.innerHTML = revisions.map((row) => {
      const revision = Number(row.revision_number || 0);
      const sha = String(row.payload_sha256 || '');
      const head = Boolean(row.is_head);
      const requal = Boolean(row.requires_requalification);
      return `<article class="history-row${head ? ' head' : ''}">
        <div class="history-rev"><strong>REV ${revision}</strong>${head ? '<span>HEAD</span>' : ''}</div>
        <div class="history-meta"><span>${escapeHTML(row.created_by || '—')}</span><span>${escapeHTML(formatTime(row.created_at))}</span></div>
        <div class="history-sha">${escapeHTML(sha.slice(0, 16) || '—')}</div>
        <div class="history-row-foot">
          <div class="history-state ${requal ? 'requal' : 'qualified'}">${requal ? 'REQUALIFICATION REQUIRED' : 'QUALIFIED'}</div>
          ${head ? '' : `<button class="history-restore" data-restore-revision="${revision}">RESTORE</button>`}
        </div>
      </article>`;
    }).join('');
    list.querySelectorAll('[data-restore-revision]').forEach((restoreButton) => {
      restoreButton.onclick = () => void restoreRevision(restoreButton.dataset.restoreRevision);
    });
  } catch (error) {
    if (list) list.innerHTML = `<div class="history-error">${escapeHTML(error?.message || error)}</div>`;
    if (state) state.textContent = 'ERROR';
  } finally {
    loading = false;
  }
}

boot();
