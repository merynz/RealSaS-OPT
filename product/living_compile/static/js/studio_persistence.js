const nativeFetch = window.fetch.bind(window);

const state = {
  bundle: null,
  session: null,
  pending: null,
};

function requestPath(input) {
  try {
    const value = typeof input === 'string' ? input : input?.url;
    return new URL(value, window.location.href).pathname;
  } catch {
    return '';
  }
}

async function readJson(response) {
  try {
    return await response.clone().json();
  } catch {
    return null;
  }
}

async function ensureSession(bundle) {
  const normalized = String(bundle || '').trim();
  if (!normalized) return null;
  if (state.session && state.bundle === normalized) return state.session;
  if (state.pending && state.bundle === normalized) return state.pending;

  state.bundle = normalized;
  state.pending = (async () => {
    const response = await nativeFetch(`/api/studio/session?bundle=${encodeURIComponent(normalized)}`, {
      cache: 'no-store',
      headers: {'Accept': 'application/json'},
    });
    if (!response.ok) {
      // Presentation/local bundles intentionally have no Platform Studio state.
      state.session = null;
      return null;
    }
    const session = await response.json();
    if (!session?.project?.project_id || !Number.isInteger(Number(session?.head?.lock_version))) {
      throw new Error('Studio persistence returned an invalid session');
    }
    state.session = session;
    window.dispatchEvent(new CustomEvent('realsas:studio-session', {detail: session}));
    return session;
  })();

  try {
    return await state.pending;
  } finally {
    state.pending = null;
  }
}

async function history(limit = 50) {
  const projectID = state.session?.project?.project_id;
  if (!projectID) return [];
  const safeLimit = Math.max(1, Math.min(200, Number(limit) || 50));
  const response = await nativeFetch(
    `/api/studio/history?id=${encodeURIComponent(projectID)}&limit=${encodeURIComponent(safeLimit)}`,
    {cache: 'no-store', headers: {'Accept': 'application/json'}},
  );
  if (!response.ok) throw new Error(`Studio history unavailable (${response.status})`);
  const payload = await response.json();
  return payload.revisions || [];
}

window.fetch = async function realsasStudioFetch(input, init = {}) {
  const path = requestPath(input);
  const method = String(init?.method || (typeof input !== 'string' ? input?.method : '') || 'GET').toUpperCase();
  if (path !== '/api/living/edit-layer' || method !== 'POST' || !init?.body) {
    return nativeFetch(input, init);
  }

  let payload;
  try {
    payload = JSON.parse(String(init.body));
  } catch {
    return nativeFetch(input, init);
  }

  const session = await ensureSession(payload.bundle);
  if (!session) return nativeFetch(input, init);

  payload.studio_project_id = session.project.project_id;
  payload.expected_lock_version = Number(session.head.lock_version);
  const response = await nativeFetch(input, {...init, body: JSON.stringify(payload)});
  const responsePayload = await readJson(response);

  if (response.ok && responsePayload?.studio_revision) {
    const revision = responsePayload.studio_revision;
    state.session.head = {
      ...state.session.head,
      revision_number: Number(revision.revision_number),
      lock_version: Number(revision.lock_version),
      payload_sha256: revision.payload_sha256 || state.session.head.payload_sha256,
      requires_requalification: Boolean(revision.requires_requalification),
    };
    window.dispatchEvent(new CustomEvent('realsas:studio-saved', {detail: revision}));
  } else if (response.status === 409) {
    window.dispatchEvent(new CustomEvent('realsas:studio-conflict', {
      detail: responsePayload || {error: 'STUDIO_REVISION_CONFLICT'},
    }));
  }
  return response;
};

window.RealSaSStudioPersistence = {
  state,
  ensureSession,
  history,
};

const initialBundle = new URLSearchParams(window.location.search).get('bundle');
if (initialBundle) {
  void ensureSession(initialBundle).catch((error) => {
    console.warn('Studio persistence session unavailable', error);
  });
}
