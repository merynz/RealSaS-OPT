import {requalificationPlan} from './studio_requalification_core.js';

const $ = (selector) => document.querySelector(selector);
let studio = null;
let executorBound = false;
let lastSignature = '';

function boot() {
  studio = window.RealSaSStudio;
  if (!studio?.document || !studio?.dirty) {
    requestAnimationFrame(boot);
    return;
  }
  installPanel();
  studio.dirty.addEventListener('change', render);
  window.addEventListener('realsas:studio-conflict', render);
  window.addEventListener('realsas:requalification-capability', (event) => {
    executorBound = Boolean(event.detail?.executor_bound);
    render(true);
  });
  loadCapability();
  requestAnimationFrame(frame);
}

async function loadCapability() {
  try {
    const response = await fetch('/api/project');
    const project = response.ok ? await response.json() : null;
    executorBound = Boolean(project?.studio_requalification_executor);
  } catch {
    executorBound = false;
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
  $('#requalAction').onclick = () => {
    if (!executorBound) return;
    window.dispatchEvent(new CustomEvent('realsas:requalification-requested'));
  };
}

function currentPlan() {
  return requalificationPlan(studio?.document?.edits || {}, {executorBound});
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
  const signature = JSON.stringify([plan, dirty.status, dirty.requires_requalification]);
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

  strip.className = `requalification-strip ${executorBound ? 'ready' : 'blocked'}`;
  title.textContent = executorBound ? 'REQUALIFICATION REQUIRED' : 'REQUALIFICATION REQUIRED · EXECUTOR UNBOUND';
  detail.textContent = executorBound
    ? 'Studio edits invalidate the domains below. Qualification must pass before an edited runtime can become authoritative.'
    : 'Studio edits are saved as sibling authoring data. No shipping Studio requalification executor is bound yet; no edited-runtime PASS can be minted.';
  domains.innerHTML = plan.invalidated_domains
    .map((domain) => `<span class="requal-domain invalid">${domain}</span>`)
    .join('');
  action.disabled = !executorBound;
  action.textContent = executorBound ? 'REQUEST REQUALIFICATION' : 'EXECUTOR UNBOUND';
  action.title = executorBound
    ? 'Request compiler requalification for this Studio authoring revision.'
    : 'A real Studio-authoring requalification capability must be bound before this action can run.';

  if (exportButton) {
    exportButton.textContent = 'EXPORT QUALIFIED BASE';
    exportButton.title = 'Exports the last proof-qualified base runtime only. Current Studio edits are excluded until requalification passes.';
    exportButton.dataset.exportScope = 'qualified-base-excludes-studio-edits';
  }
}

boot();
