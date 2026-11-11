const $$ = (selector) => [...document.querySelectorAll(selector)];
const $ = (selector) => document.querySelector(selector);

const STAGES = {
  input: {
    mode: null,
    note: 'Input belongs to Awaken. Studio opens an already proof-qualified workspace and never starts a source compile from the editor shell.',
  },
  shape: {
    mode: 'mesh',
    note: 'Shape is learned scene/surface evidence. It is proposal evidence only; Compiler remains the canonical Mesh authority.',
  },
  mesh: {
    mode: 'mesh',
    note: 'Mesh shows the editable canonical topology shared by rigging, mechanics and graphics.',
  },
  rig: {
    mode: 'rig',
    note: 'Rig shows the editable control-ready deform graph, local frames, controls and constraints.',
  },
  skin: {
    mode: 'weights',
    note: 'Skin binds the deform structure to the same canonical mesh. User edits remain staged until Compiler requalification.',
  },
  mechanics: {
    mode: 'rig',
    note: 'Mechanics executes the rig control/constraint program and owns post-bind deformation validity. It does not own appearance or draw order.',
  },
  graphics: {
    mode: 'bindings',
    note: 'Graphics owns the complete visual/presentation domain: source-preserving appearance, visibility, occlusion, fragmentation, draw order and presentation attachments. CAA is its appearance sublayer.',
  },
  animation: {
    mode: 'animation',
    note: 'Animation drives animator controls. Mechanics resolves the resulting deform-joint pose; direct joint tracks are derived output, not editing authority.',
  },
  proof: {
    mode: null,
    note: 'Proof is the final product-pass authority. Mechanics, Graphics and Motion must be qualified and runtime consequences must remain valid before export.',
  },
};

function setStage(name) {
  const stage = STAGES[name];
  if (!stage) return;
  $$('.stage').forEach((button) => button.classList.toggle('active', button.dataset.stage === name));
  const active = $('#activeProductStage');
  if (active) active.textContent = name.toUpperCase();
  const note = $('#productStageNote');
  if (note) note.textContent = stage.note;
  if (stage.mode) {
    const target = $(`.mode[data-mode="${stage.mode}"]`);
    target?.click();
  }
}

function setGatewayCapability(project) {
  const state = $('#compileState');
  const detail = $('#compileDetail');
  const progress = $('#compileProgress');
  const available = Boolean(project?.compile_available);
  const blockers = Array.isArray(project?.compile_blockers) ? project.compile_blockers : [];
  if (state) state.textContent = available ? 'PRODUCT GATEWAY READY' : 'PRODUCT SHELL READY';
  if (detail) detail.textContent = available
    ? 'Source compile starts in Awaken. This Studio session edits an already materialized proof-qualified character.'
    : 'Open an existing qualified puppet or return to Awaken. New compiles require the production gateway.';
  if (progress) progress.style.width = available ? '2%' : '0%';
}

async function loadProductCapabilities() {
  try {
    const response = await fetch('/api/project');
    const project = await response.json();
    setGatewayCapability(response.ok ? project : null);
  } catch {
    setGatewayCapability(null);
  }
}

function syncProofState() {
  const badge = $('#proofBadge');
  const state = $('#compileState');
  const detail = $('#compileDetail');
  if (!badge || !state || !detail) return;
  const text = badge.textContent.trim().toUpperCase();
  if (text.includes('PASS')) {
    state.textContent = 'COMPILED · PROOF PASS';
    detail.textContent = 'Character is available for inspection, editing, animation preview and qualified-base export.';
    $$('.stage').forEach((button) => button.classList.add('available'));
  } else if (text !== 'NO CHARACTER' && text !== 'NO BUNDLE') {
    state.textContent = 'COMPILED · REVIEW PROOF';
    detail.textContent = 'A compiled character is open, but export remains governed by the proof result.';
  }
}

$$('.stage').forEach((button) => button.addEventListener('click', () => setStage(button.dataset.stage)));

const proofBadge = $('#proofBadge');
if (proofBadge) new MutationObserver(syncProofState).observe(proofBadge, {childList: true, subtree: true, characterData: true});

$('#openBundle')?.addEventListener('click', () => {
  setTimeout(() => {
    syncProofState();
    const badge = $('#proofBadge')?.textContent?.trim().toUpperCase() || '';
    if (badge && badge !== 'NO CHARACTER' && badge !== 'NO BUNDLE') setStage('mesh');
  }, 250);
});

setStage('input');
loadProductCapabilities();
syncProofState();
