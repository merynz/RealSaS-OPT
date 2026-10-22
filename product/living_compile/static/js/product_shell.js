const $$ = (selector) => [...document.querySelectorAll(selector)];
const $ = (selector) => document.querySelector(selector);

const STAGES = {
  input: {
    mode: null,
    note: 'Input owns the source character and compile request. Existing compiled bundles can already be opened; full inference binding remains fail-closed until P3 is complete.',
  },
  shape: {
    mode: 'mesh',
    note: 'Shape is the qualified continuous character surface. This view currently inspects the compiled carrier through the mesh editor; direct Shape editing is not yet promoted.',
  },
  mesh: {
    mode: 'mesh',
    note: 'Mesh shows the editable canonical topology used by mechanics, graphics and runtime.',
  },
  rig: {
    mode: 'rig',
    note: 'Rig shows the editable control hierarchy and pose controls.',
  },
  skin: {
    mode: 'weights',
    note: 'Skin shows deformation weights on the same canonical mesh. User edits remain staged until compiler requalification.',
  },
  graphics: {
    mode: 'bindings',
    note: 'Graphics owns source appearance and presentation relations such as visibility, draw order and attachment. Current UI exposes existing bindings while the new Graphics compiler is being connected.',
  },
  animation: {
    mode: 'animation',
    note: 'Animation previews proof-owned motion and lets the user stage editable clips and keys.',
  },
  proof: {
    mode: null,
    note: 'Proof is the final compiler verdict: geometry, mechanics, graphics and motion must all remain qualified before export.',
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
  if (button) {
    button.disabled = !available;
    button.title = available ? 'Compile character' : 'Full inference service is not bound to the product shell yet.';
  }
  if (state) state.textContent = available ? 'READY TO COMPILE' : 'EXISTING CHARACTER MODE';
  if (detail) detail.textContent = available
    ? 'Input will run the exact production model releases and compiler graph.'
    : 'Compile remains fail-closed until ProductRevision → ModelRelease → inference binding is complete.';
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
