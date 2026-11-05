import {requalificationPlan} from './studio_requalification_core.js';

const $ = (selector) => document.querySelector(selector);
let studio = null;
let capability = null;
let busy = false;
let progress = null;
let lastSignature = '';

function boot() {
  studio = window.RealSaSStudio;
  if (!studio?.document || !studio?.dirty) {
    requestAnimationFrame(boot);
    return;
  }
  installPanel();
  studio.dirty.addEventListener('change', () => render(true));
  window.addEventListener('realsas:studio-conflict', () => render(true));
  window.addEventListener('realsas:studio-saved', () => render(true));
  window.addEventListener('realsas:studio-restored', () => render(true));
  window.addEventListener('realsas:studio-requalification', (event) => {
    progress = event.detail || null;
    render(true);
  });
  loadCapability();
  requestAnimationFrame(frame);
}

async function loadCapability() {
  try {
    const response = await fetch('/api/project', {cache: 'no-store'});
    const project = response.ok ? await response.json() : null;
    capability = project?.studio_requalification || null;
  } catch {
    capability = null;
  }
  render(true);
}

function installPanel() {
  if ($('#requalificationStrip')) return;
  const modebar = document.querySelector('.modebar');
  if (!modebar) return;
  const section = document.createElement('section');
  section.id = 'requalificationStrip';
  section.className = 'requalification-strip qualified';
  section.innerHTML = `
    <div class="requal-status">
      <span class="requal-dot"></span>
      <div><strong id="requalTitle">QUALIFIED BASE</strong><span id="requalDetail">No Studio authoring changes invalidate the proof-qualified base.</span></div>
    </div>
    <div id="requalDomains" class="requal-domains" aria-label="Invalidated product domains"></div>
    <button id="requalAction" disabled>REQUALIFY</button>`;
  modebar.insertAdjacentElement('afterend', section);
  $('#requalAction').onclick = () => void requestRequalification();
}

function currentPlan() {
  return requalificationPlan(studio?.document?.edits || {}, {
    executorBound: Boolean(capability?.available),
  });
}

function animationSliceSupport(plan) {
  if (!plan.required) return {supported: false, reason: 'NO_REQUALIFICATION_REQUIRED'};
  if (!capability?.available) return {supported: false, reason: 'EXECUTOR_UNAVAILABLE'};
  if (capability?.supported_slice !== 'ANIMATION_LINEAR_V1') {
    return {supported: false, reason: 'EXECUTOR_CONTRACT_UNRECOGNIZED'};
  }
  const domains = plan.invalidated_domains || [];
  if (domains.length !== 2 || domains[0] !== 'MOTION' || domains[1] !== 'PROOF') {
    return {supported: false, reason: 'EDIT_DOMAINS_NOT_YET_SUPPORTED'};
  }
  const counts = plan.edit_counts || {};
  const nonAnimation = [
    'rig', 'mesh', 'mesh_topology', 'weights', 'bindings', 'rig_topology', 'ik', 'presentation',
  ].some((field) => Number(counts[field] || 0) !== 0);
  if (nonAnimation || Number(counts.animation || 0) !== 1) {
    return {supported: false, reason: 'ANIMATION_ONLY_ONE_CLIP_REQUIRED'};
  }

  const clip = studio.document.edits?.animation?.[0];
  const tracks = Array.isArray(clip?.tracks) ? clip.tracks : [];
  if (!tracks.length || tracks.some((track) => !Array.isArray(track?.keys) || !track.keys.length)) {
    return {supported: false, reason: 'ANIMATION_TRACK_KEYS_REQUIRED'};
  }
  for (const track of tracks) {
    for (const key of track.keys) {
      const curve = String(key?.curve || 'linear').toLowerCase();
      if (curve !== 'linear') return {supported: false, reason: 'LINEAR_CURVES_ONLY_V1'};
    }
  }
  return {supported: true, reason: 'ANIMATION_LINEAR_V1'};
}

function frame() {
  render();
  requestAnimationFrame(frame);
}

function render(force = false) {
  if (!studio?.document) return;
  installPanel();
  const plan = currentPlan();
  const dirty = studio.dirty?.snapshot?.() || {};
  const support = animationSliceSupport(plan);
  const signature = JSON.stringify([
    plan,
    dirty.status,
    dirty.revision_number,
    dirty.lock_version,
    dirty.last_error,
    capability,
    busy,
    progress,
  ]);
  if (!force && signature === lastSignature) return;
  lastSignature = signature;

  const strip = $('#requalificationStrip');
  const title = $('#requalTitle');
  const detail = $('#requalDetail');
  const domains = $('#requalDomains');
  const action = $('#requalAction');
  const exportButton = $('#exportRuntime');
  if (!strip || !title || !detail || !domains || !action) return;

  if (!plan.required) {
    strip.className = 'requalification-strip qualified';
    title.textContent = 'QUALIFIED BASE';
    detail.textContent = 'No Studio authoring changes invalidate the proof-qualified base.';
    domains.innerHTML = '<span class="requal-domain good">PROOF PASS</span>';
    action.disabled = true;
    action.textContent = 'NO REQUALIFICATION NEEDED';
    if (exportButton) {
      exportButton.textContent = 'EXPORT';
      exportButton.title = 'Export the proof-qualified native runtime package.';
      exportButton.dataset.exportScope = 'qualified-base';
    }
    return;
  }

  domains.innerHTML = plan.invalidated_domains
    .map((domain) => `<span class="requal-domain invalid">${domain}</span>`)
    .join('');

  if (busy || (progress && !progress.terminal)) {
    strip.className = 'requalification-strip ready';
    const stage = String(progress?.stage || 'MOTION').toUpperCase();
    title.textContent = `REQUALIFYING · ${stage}`;
    detail.textContent = progress?.detail || 'Running edited Motion through compiler qualification and terminal Proof.';
    action.disabled = true;
    action.textContent = 'REQUALIFICATION RUNNING';
  } else if (!capability?.available) {
    strip.className = 'requalification-strip blocked';
    title.textContent = 'REQUALIFICATION REQUIRED · EXECUTOR UNAVAILABLE';
    detail.textContent = 'Studio edits remain sibling authoring data. No edited-runtime PASS can be minted until the shipping requalification lane is available.';
    action.disabled = true;
    action.textContent = 'EXECUTOR UNAVAILABLE';
  } else if (!support.supported) {
    strip.className = 'requalification-strip blocked';
    title.textContent = 'REQUALIFICATION REQUIRED · EDIT SLICE NOT YET SHIPPED';
    detail.textContent = supportReason(support.reason);
    action.disabled = true;
    action.textContent = 'REQUALIFICATION BLOCKED';
  } else if (dirty.status === 'conflict') {
    strip.className = 'requalification-strip blocked';
    title.textContent = 'REQUALIFICATION BLOCKED · REVISION CONFLICT';
    detail.textContent = 'Reload or restore the current Studio HEAD before requesting compiler requalification.';
    action.disabled = true;
    action.textContent = 'RESOLVE CONFLICT FIRST';
  } else if (dirty.status === 'saving') {
    strip.className = 'requalification-strip ready';
    title.textContent = 'SAVING AUTHORING REVISION';
    detail.textContent = 'Requalification starts only from an immutable persisted Studio HEAD.';
    action.disabled = true;
    action.textContent = 'SAVING…';
  } else {
    strip.className = 'requalification-strip ready';
    title.textContent = 'REQUALIFICATION REQUIRED · MOTION → PROOF';
    detail.textContent = dirty.status === 'dirty'
      ? 'The supported animation edit will be saved first, then recompiled through Motion and Proof into a new ProductRevision.'
      : 'This saved animation revision can be recompiled through Motion and Proof into a new ProductRevision.';
    action.disabled = false;
    action.textContent = dirty.status === 'dirty' ? 'SAVE & REQUALIFY' : 'REQUEST REQUALIFICATION';
    action.title = 'The base ProductRevision remains immutable. Proof PASS creates a new ProductRevision and clean workspace.';
  }

  if (exportButton) {
    exportButton.textContent = 'EXPORT QUALIFIED BASE';
    exportButton.title = 'Exports the last proof-qualified base runtime only. Current Studio edits are excluded until requalification passes.';
    exportButton.dataset.exportScope = 'qualified-base-excludes-studio-edits';
  }
}

function supportReason(reason) {
  const messages = {
    EDIT_DOMAINS_NOT_YET_SUPPORTED: 'The first shipping requalification slice is animation-only. Mesh, rig, skin, mechanics, graphics and presentation adapters remain fail-closed.',
    ANIMATION_ONLY_ONE_CLIP_REQUIRED: 'The first shipping requalification slice accepts exactly one user animation clip and no other edit domains.',
    ANIMATION_TRACK_KEYS_REQUIRED: 'The user animation clip must contain at least one keyed animator-control track before it can be qualified.',
    LINEAR_CURVES_ONLY_V1: 'Linear animation keys are qualified in the first slice. Step and Bezier tangent execution remain fail-closed until their runtime semantics are courted.',
    EXECUTOR_CONTRACT_UNRECOGNIZED: 'The deployed Studio requalification contract does not match this editor build.',
  };
  return messages[reason] || 'This Studio edit set is not admitted by the current shipping requalification executor.';
}

function waitForSavedHead(timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    const started = performance.now();
    const check = () => {
      const dirty = studio.dirty.snapshot();
      if (dirty.status === 'clean' && Number(dirty.revision_number) > 0) {
        cleanup();
        resolve(dirty);
      } else if (dirty.status === 'conflict') {
        cleanup();
        reject(new Error(dirty.last_error || 'Studio revision conflict'));
      } else if (dirty.status === 'dirty' && dirty.last_error) {
        cleanup();
        reject(new Error(dirty.last_error));
      } else if (performance.now() - started > timeoutMs) {
        cleanup();
        reject(new Error('Timed out waiting for Studio revision save'));
      }
    };
    const timer = setInterval(check, 100);
    const onChange = () => check();
    const cleanup = () => {
      clearInterval(timer);
      studio.dirty.removeEventListener('change', onChange);
    };
    studio.dirty.addEventListener('change', onChange);
    check();
  });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function requestRequalification() {
  if (busy) return;
  const plan = currentPlan();
  const support = animationSliceSupport(plan);
  if (!support.supported) return;
  const persistence = window.RealSaSStudioPersistence;
  if (!persistence?.requestRequalification || !persistence?.requalificationStatus) return;

  busy = true;
  progress = {state: 'PREPARING', stage: 'SAVE', terminal: false, detail: 'Preparing immutable Studio authoring HEAD.'};
  render(true);
  try {
    if (studio.dirty.status === 'dirty') {
      $('#saveLayer')?.click();
      await waitForSavedHead();
    } else if (studio.dirty.status === 'saving') {
      await waitForSavedHead();
    }
    if (studio.dirty.status !== 'clean' || studio.dirty.revisionNumber <= 0) {
      throw new Error('A clean persisted Studio HEAD is required before requalification');
    }

    progress = await persistence.requestRequalification();
    render(true);
    while (!progress.terminal) {
      await delay(700);
      progress = await persistence.requalificationStatus(progress.requalification_id);
      render(true);
    }
    if (String(progress.state || '').toUpperCase() !== 'PASS' || !progress.result_compile_id) {
      throw new Error(progress.detail || 'Studio requalification did not pass');
    }

    const compileID = progress.result_compile_id;
    progress = {...progress, terminal: false, stage: 'WORKSPACE', detail: 'ProductRevision sealed. Materializing the new clean Studio workspace.'};
    render(true);
    while (true) {
      const response = await fetch(`/api/compile/status?id=${encodeURIComponent(compileID)}`, {cache: 'no-store'});
      const compile = await response.json();
      if (!response.ok) throw new Error(compile?.message || compile?.error || `Workspace status HTTP ${response.status}`);
      const state = String(compile.status || '').toUpperCase();
      if (state === 'PASS') break;
      if (compile.terminal || ['FAILED', 'ERROR'].includes(state)) {
        throw new Error(compile.detail || compile.message || 'Requalified ProductRevision workspace failed');
      }
      await delay(700);
    }

    const response = await fetch(`/api/compile/workspace?id=${encodeURIComponent(compileID)}`, {cache: 'no-store'});
    const workspace = await response.json();
    if (!response.ok || !String(workspace?.workspace_url || '').startsWith('/studio?bundle=')) {
      throw new Error(workspace?.message || workspace?.error || 'New qualified workspace handoff failed');
    }
    progress = {...progress, terminal: true, state: 'PASS', stage: 'FINALIZE', detail: 'Proof PASS. Opening the new clean qualified Studio revision.'};
    render(true);
    window.location.assign(workspace.workspace_url);
  } catch (error) {
    progress = {
      ...(progress || {}),
      state: 'FAILED',
      terminal: true,
      stage: progress?.stage || 'PROOF',
      detail: error?.message || String(error),
    };
    render(true);
  } finally {
    busy = false;
    render(true);
  }
}

boot();
