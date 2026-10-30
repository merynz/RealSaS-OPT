const $$ = (selector) => [...document.querySelectorAll(selector)];
const $ = (selector) => document.querySelector(selector);

const STAGES = {
  input: {
    mode: null,
    note: 'Input owns the source character and compile request. Production compile stays fail-closed until exact production model and proof bindings are ready.',
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

function setCompileCapability(project) {
  const button = $('#compileCharacter');
  const state = $('#compileState');
  const detail = $('#compileDetail');
  const progress = $('#compileProgress');
  const available = Boolean(project?.compile_available);
  const blockers = Array.isArray(project?.compile_blockers) ? project.compile_blockers : [];
  if (button) {
    button.disabled = !available;
    button.title = available ? 'Compile character' : 'Production compile is intentionally fail-closed.';
  }
  if (state) state.textContent = available ? 'READY TO COMPILE' : 'PRODUCT SHELL READY';
  if (detail) detail.textContent = available
    ? 'Input will run the exact production model releases, canonical product graph and Proof authority.'
    : blockers.length
      ? `Waiting on: ${blockers.join(' · ')}`
      : 'Compile remains fail-closed until production model/proof/revision bindings are complete.';
  if (progress) progress.style.width = available ? '2%' : '0%';
}

async function loadProductCapabilities() {
  try {
    const response = await fetch('/api/project');
    const project = await response.json();
    setCompileCapability(response.ok ? project : null);
  } catch {
    setCompileCapability(null);
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
    detail.textContent = 'Character is available for inspection, editing, animation preview and export.';
    $$('.stage').forEach((button) => button.classList.add('available'));
  } else if (text !== 'NO CHARACTER' && text !== 'NO BUNDLE') {
    state.textContent = 'COMPILED · REVIEW PROOF';
    detail.textContent = 'A compiled character is open, but export remains governed by the proof result.';
  }
}

$$('.stage').forEach((button) => button.addEventListener('click', () => setStage(button.dataset.stage)));

$('#compileCharacter')?.addEventListener('click', async () => {
  const button = $('#compileCharacter');
  if (button?.disabled) return;
  const state = $('#compileState');
  const detail = $('#compileDetail');
  if (state) state.textContent = 'STARTING COMPILE';
  if (detail) detail.textContent = 'Submitting production compile request…';
  try {
    const response = await fetch('/api/compile/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.message || `HTTP ${response.status}`);
  } catch (error) {
    if (state) state.textContent = 'COMPILE UNAVAILABLE';
    if (detail) detail.textContent = error.message;
  }
});

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
