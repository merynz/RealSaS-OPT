import {
  effectiveWeight,
  effectiveWeights,
  falloffWeight,
  meshMirrorAxisX,
  mirroredControlID,
  mirroredVertexIndex,
  normalizeVertexWeights,
  paintVertex,
  vertexAdjacency,
} from './studio_weight_core.js';

const $ = (selector) => document.querySelector(selector);

const settings = {
  mode: 'add',
  radiusPx: 42,
  strength: 0.12,
  falloff: 'smooth',
  replaceValue: 1,
  normalize: true,
  mirror: false,
  locked: new Set(),
  axisByMesh: new Map(),
};

const adjacencyCache = new Map();
let studio = null;
let S = null;
let documentModel = null;
let journal = null;
let dirty = null;
let panel = null;
let viewport = null;
let inspectorObserver = null;
let interaction = null;
let saveTimer = null;
let cursor = null;

function boot() {
  studio = window.RealSaSStudio;
  panel = document.querySelector('.viewport-panel');
  viewport = $('#viewport');
  if (!studio || !panel || !viewport) {
    requestAnimationFrame(boot);
    return;
  }
  S = studio.state;
  documentModel = studio.document;
  journal = studio.journal;
  dirty = studio.dirty;

  installCursor();
  panel.addEventListener('pointerdown', pointerDown, true);
  panel.addEventListener('pointermove', pointerMove, true);
  panel.addEventListener('pointerup', pointerUp, true);
  panel.addEventListener('pointercancel', pointerCancel, true);
  panel.addEventListener('pointerleave', pointerLeave, true);
  window.addEventListener('keydown', keyDown, true);

  const inspectorBody = $('#inspectorBody');
  if (inspectorBody) {
    inspectorObserver = new MutationObserver(() => ensureInspector());
    inspectorObserver.observe(inspectorBody, {childList: true, subtree: true});
  }
  ensureInspector();
  requestAnimationFrame(frame);
}

function frame() {
  if (S?.mode === 'weights') ensureInspector();
  if (cursor) cursor.hidden = S?.mode !== 'weights';
  requestAnimationFrame(frame);
}

function currentViewID() {
  return `V${S?.view ?? 0}`;
}

function deformControls() {
  const rig = S?.scene?.puppet?.rig || {};
  const rows = rig.deform_controls || rig.controls || [];
  return rows.filter((row) => row.view_id === currentViewID());
}

function meshes() {
  return (S?.scene?.puppet?.meshes || []).filter((row) => row.view_id === currentViewID());
}

function activeMesh() {
  const rows = meshes();
  return rows.find((row) => row.mesh_id === S?.sel?.mesh) || rows[0] || null;
}

function activeJointID() {
  const controls = deformControls();
  const current = S?.sel?.control;
  if (controls.some((row) => row.control_id === current)) return current;
  const fallback = controls[0]?.control_id || null;
  if (fallback && S?.sel) S.sel.control = fallback;
  return fallback;
}

function jointIDs() {
  return deformControls().map((row) => row.control_id);
}

function axisX(mesh) {
  if (!mesh) return 0;
  if (!settings.axisByMesh.has(mesh.mesh_id)) settings.axisByMesh.set(mesh.mesh_id, meshMirrorAxisX(mesh));
  return Number(settings.axisByMesh.get(mesh.mesh_id));
}

function adjacency(mesh) {
  if (!mesh) return [];
  const key = `${mesh.mesh_id}:${mesh.vertices?.length || 0}:${mesh.triangles?.length || 0}`;
  if (!adjacencyCache.has(key)) adjacencyCache.set(key, vertexAdjacency(mesh));
  return adjacencyCache.get(key);
}

function meshDelta(meshID, vertexIndex) {
  const edit = (S?.ed?.mesh || []).find((row) => row.mesh_id === meshID && row.vertex_index === vertexIndex);
  return edit?.translation_xy || [0, 0];
}

function vertexWorld(mesh, vertexIndex) {
  const point = mesh?.vertices?.[vertexIndex];
  if (!point) return null;
  const delta = meshDelta(mesh.mesh_id, vertexIndex);
  return [Number(point[0]) + Number(delta[0] || 0), Number(point[1]) + Number(delta[1] || 0)];
}

function intrinsicFromEvent(event) {
  const bounds = viewport.getBoundingClientRect();
  return [
    (event.clientX - bounds.left) * viewport.width / Math.max(1, bounds.width),
    (event.clientY - bounds.top) * viewport.height / Math.max(1, bounds.height),
  ];
}

function clientFromWorld(point) {
  const bounds = viewport.getBoundingClientRect();
  const intrinsic = [point[0] * S.cam.s + S.cam.x, point[1] * S.cam.s + S.cam.y];
  return [
    bounds.left + intrinsic[0] * bounds.width / viewport.width,
    bounds.top + intrinsic[1] * bounds.height / viewport.height,
  ];
}

function eventInsideViewport(event) {
  const bounds = viewport.getBoundingClientRect();
  return event.clientX >= bounds.left && event.clientX <= bounds.right && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
}

function consume(event) {
  event.preventDefault();
  event.stopPropagation();
  if (typeof event.stopImmediatePropagation === 'function') event.stopImmediatePropagation();
}

function pointerDown(event) {
  if (S?.mode !== 'weights' || event.button !== 0 || !eventInsideViewport(event) || !S.scene) return;
  const mesh = activeMesh();
  const jointID = activeJointID();
  if (!mesh || !jointID) return;
  consume(event);
  try { panel.setPointerCapture(event.pointerId); } catch {}
  interaction = {
    pointerID: event.pointerId,
    before: documentModel.snapshot(),
    changed: false,
  };
  applyBrush(event);
}

function pointerMove(event) {
  if (S?.mode !== 'weights') return;
  updateCursor(event);
  if (!interaction || interaction.pointerID !== event.pointerId) return;
  consume(event);
  applyBrush(event);
}

function pointerUp(event) {
  if (!interaction || interaction.pointerID !== event.pointerId) return;
  consume(event);
  finishStroke();
  try { panel.releasePointerCapture(event.pointerId); } catch {}
}

function pointerCancel(event) {
  if (!interaction || interaction.pointerID !== event.pointerId) return;
  consume(event);
  if (interaction.before) documentModel.replace(interaction.before);
  interaction = null;
  ensureInspector(true);
}

function pointerLeave() {
  if (cursor && !interaction) cursor.hidden = true;
}

function applyBrush(event) {
  const mesh = activeMesh();
  const jointID = activeJointID();
  const ids = jointIDs();
  if (!mesh || !jointID || !ids.length) return;
  const pointer = [event.clientX, event.clientY];
  const rows = [];
  let closest = {index: -1, distance: Infinity};

  mesh.vertices.forEach((_, vertexIndex) => {
    const world = vertexWorld(mesh, vertexIndex);
    if (!world) return;
    const client = clientFromWorld(world);
    const distance = Math.hypot(client[0] - pointer[0], client[1] - pointer[1]);
    if (distance < closest.distance) closest = {index: vertexIndex, distance};
    const falloff = falloffWeight(distance, settings.radiusPx, settings.falloff);
    if (falloff > 0) rows.push({vertexIndex, amount: settings.strength * falloff});
  });

  if (!rows.length) return;
  if (closest.index >= 0) {
    S.sel.mesh = mesh.mesh_id;
    S.sel.vertex = closest.index;
  }

  const locked = [...settings.locked];
  const adj = adjacency(mesh);
  const mirrorAxis = axisX(mesh);
  const controls = deformControls();
  const mirrorJointID = settings.mirror ? mirroredControlID(controls, jointID, mirrorAxis) : jointID;
  let changed = false;

  for (const row of rows) {
    const result = paintVertex({
      mesh,
      vertexIndex: row.vertexIndex,
      jointID,
      jointIDs: ids,
      edits: S.ed.weights,
      amount: row.amount,
      mode: settings.mode,
      replaceValue: settings.replaceValue,
      normalize: settings.normalize,
      lockedJointIDs: locked,
      adjacency: adj,
    });
    changed = changed || result.changed;

    if (settings.mirror) {
      const mirrorIndex = mirroredVertexIndex(mesh, row.vertexIndex, mirrorAxis);
      if (mirrorIndex >= 0 && (mirrorIndex !== row.vertexIndex || mirrorJointID !== jointID)) {
        const mirrored = paintVertex({
          mesh,
          vertexIndex: mirrorIndex,
          jointID: mirrorJointID,
          jointIDs: ids,
          edits: S.ed.weights,
          amount: row.amount,
          mode: settings.mode,
          replaceValue: settings.replaceValue,
          normalize: settings.normalize,
          lockedJointIDs: locked,
          adjacency: adj,
        });
        changed = changed || mirrored.changed;
      }
    }
  }

  if (changed) {
    interaction.changed = true;
    dirty.markDirty();
    updateStats();
  }
}

function finishStroke() {
  const stroke = interaction;
  interaction = null;
  if (!stroke?.changed) return;
  journal.commit(`Weight ${settings.mode} brush stroke`, stroke.before, documentModel.snapshot());
  dirty.markDirty();
  scheduleSave();
  ensureInspector(true);
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (dirty?.status === 'dirty') $('#saveLayer')?.click();
  }, 1200);
}

function operate(label, operation) {
  const before = documentModel.snapshot();
  operation();
  const after = documentModel.snapshot();
  if (!journal.commit(label, before, after)) return false;
  dirty.markDirty();
  scheduleSave();
  ensureInspector(true);
  return true;
}

function normalizeSelectedVertex() {
  const mesh = activeMesh();
  const vertexIndex = Number(S?.sel?.vertex ?? -1);
  if (!mesh || vertexIndex < 0 || vertexIndex >= mesh.vertices.length) return;
  operate('Normalize selected skin vertex', () => {
    normalizeVertexWeights({
      mesh,
      vertexIndex,
      jointIDs: jointIDs(),
      edits: S.ed.weights,
      lockedJointIDs: [...settings.locked],
      preferredJointID: activeJointID(),
    });
  });
}

function normalizeMesh() {
  const mesh = activeMesh();
  if (!mesh) return;
  operate('Normalize skin weights for mesh', () => {
    const ids = jointIDs();
    for (let vertexIndex = 0; vertexIndex < mesh.vertices.length; vertexIndex += 1) {
      normalizeVertexWeights({
        mesh,
        vertexIndex,
        jointIDs: ids,
        edits: S.ed.weights,
        lockedJointIDs: [...settings.locked],
        preferredJointID: activeJointID(),
      });
    }
  });
}

function resetMeshWeights() {
  const mesh = activeMesh();
  if (!mesh) return;
  operate('Reset user skin edits for mesh', () => {
    S.ed.weights = S.ed.weights.filter((row) => row.mesh_id !== mesh.mesh_id);
  });
}

function keyDown(event) {
  if (S?.mode !== 'weights') return;
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) return;
  if (event.key === '[') {
    settings.radiusPx = Math.max(4, settings.radiusPx - 4);
    consume(event);
    ensureInspector(true);
  } else if (event.key === ']') {
    settings.radiusPx = Math.min(240, settings.radiusPx + 4);
    consume(event);
    ensureInspector(true);
  }
}

function installCursor() {
  cursor = document.createElement('div');
  cursor.id = 'weightBrushCursor';
  cursor.className = 'weight-brush-cursor';
  cursor.hidden = true;
  panel.append(cursor);
}

function updateCursor(event) {
  if (!cursor || !eventInsideViewport(event)) {
    if (cursor && !interaction) cursor.hidden = true;
    return;
  }
  const bounds = panel.getBoundingClientRect();
  cursor.hidden = false;
  cursor.style.width = `${settings.radiusPx * 2}px`;
  cursor.style.height = `${settings.radiusPx * 2}px`;
  cursor.style.left = `${event.clientX - bounds.left}px`;
  cursor.style.top = `${event.clientY - bounds.top}px`;
}

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function ensureInspector(force = false) {
  if (S?.mode !== 'weights' || !S.scene) return;
  const body = $('#inspectorBody');
  if (!body) return;
  const mesh = activeMesh();
  const controls = deformControls();
  if (!mesh || !controls.length) return;
  const existing = $('#weightPainterPro');
  if (existing && !force && existing.dataset.mesh === mesh.mesh_id && existing.dataset.joint === activeJointID()) {
    updateStats();
    return;
  }

  const jointID = activeJointID();
  const vertexIndex = Number(S?.sel?.vertex ?? -1);
  const jointRows = controls.map((row) => {
    const active = row.control_id === jointID ? ' active' : '';
    const locked = settings.locked.has(row.control_id);
    return `<div class="weight-joint-row${active}" data-joint="${escapeHTML(row.control_id)}"><button class="weight-joint-select" data-joint-select="${escapeHTML(row.control_id)}">${escapeHTML(row.control_id)}</button><button class="weight-joint-lock${locked ? ' locked' : ''}" data-joint-lock="${escapeHTML(row.control_id)}" title="Lock weight">${locked ? 'LOCKED' : 'LOCK'}</button></div>`;
  }).join('');

  body.innerHTML = `<div id="weightPainterPro" class="inspector-section weight-painter" data-mesh="${escapeHTML(mesh.mesh_id)}" data-joint="${escapeHTML(jointID)}">
    <h3>WEIGHT PAINT · ${escapeHTML(mesh.mesh_id)}</h3>
    <div class="weight-grid">
      <label>MODE<select id="wpMode"><option value="add">ADD</option><option value="subtract">SUBTRACT</option><option value="replace">REPLACE</option><option value="smooth">SMOOTH</option></select></label>
      <label>FALLOFF<select id="wpFalloff"><option value="smooth">SMOOTH</option><option value="linear">LINEAR</option><option value="hard">HARD</option></select></label>
      <label>RADIUS<input id="wpRadius" type="number" min="4" max="240" step="2" value="${settings.radiusPx}"></label>
      <label>STRENGTH<input id="wpStrength" type="number" min="0.001" max="1" step="0.01" value="${settings.strength}"></label>
      <label>REPLACE<input id="wpReplace" type="number" min="0" max="1" step="0.05" value="${settings.replaceValue}"></label>
      <label>MIRROR X<input id="wpAxis" type="number" step="0.1" value="${axisX(mesh)}"></label>
    </div>
    <div class="weight-toggles">
      <label><input id="wpNormalize" type="checkbox"${settings.normalize ? ' checked' : ''}> AUTO NORMALIZE</label>
      <label><input id="wpMirror" type="checkbox"${settings.mirror ? ' checked' : ''}> MIRROR X</label>
    </div>
    <div id="weightStats" class="weight-stats mono"></div>
    <div class="weight-actions"><button id="wpNormalizeVertex"${vertexIndex < 0 ? ' disabled' : ''}>NORMALIZE VERTEX</button><button id="wpNormalizeMesh">NORMALIZE MESH</button><button id="wpResetMesh" class="danger">RESET USER WEIGHTS</button></div>
    <div class="weight-help">Paint directly in the viewport. [ / ] changes brush radius. Locks are preserved by normalization. Mirror is a Studio operator only; saved authority remains per-vertex deform-joint deltas.</div>
    <div class="weight-joint-list">${jointRows}</div>
  </div>`;

  $('#wpMode').value = settings.mode;
  $('#wpFalloff').value = settings.falloff;
  $('#wpMode').onchange = (event) => { settings.mode = event.target.value; };
  $('#wpFalloff').onchange = (event) => { settings.falloff = event.target.value; };
  $('#wpRadius').onchange = (event) => { settings.radiusPx = Math.max(4, Math.min(240, Number(event.target.value) || 42)); };
  $('#wpStrength').onchange = (event) => { settings.strength = Math.max(0.001, Math.min(1, Number(event.target.value) || 0.12)); };
  $('#wpReplace').onchange = (event) => { settings.replaceValue = Math.max(0, Math.min(1, Number(event.target.value) || 0)); };
  $('#wpAxis').onchange = (event) => { settings.axisByMesh.set(mesh.mesh_id, Number(event.target.value) || 0); };
  $('#wpNormalize').onchange = (event) => { settings.normalize = Boolean(event.target.checked); };
  $('#wpMirror').onchange = (event) => { settings.mirror = Boolean(event.target.checked); };
  $('#wpNormalizeVertex').onclick = normalizeSelectedVertex;
  $('#wpNormalizeMesh').onclick = normalizeMesh;
  $('#wpResetMesh').onclick = resetMeshWeights;

  body.querySelectorAll('[data-joint-select]').forEach((button) => {
    button.onclick = () => {
      S.sel.control = button.dataset.jointSelect;
      ensureInspector(true);
    };
  });
  body.querySelectorAll('[data-joint-lock]').forEach((button) => {
    button.onclick = () => {
      const id = button.dataset.jointLock;
      if (settings.locked.has(id)) settings.locked.delete(id); else settings.locked.add(id);
      ensureInspector(true);
    };
  });
  updateStats();
}

function updateStats() {
  const target = $('#weightStats');
  const mesh = activeMesh();
  const jointID = activeJointID();
  const vertexIndex = Number(S?.sel?.vertex ?? -1);
  if (!target || !mesh || !jointID) return;
  if (vertexIndex < 0 || vertexIndex >= mesh.vertices.length) {
    target.textContent = `joint ${jointID} · select/paint a vertex`;
    return;
  }
  const ids = jointIDs();
  const weights = effectiveWeights(mesh, vertexIndex, ids, S.ed.weights);
  const total = ids.reduce((sum, id) => sum + (weights.get(id) || 0), 0);
  target.textContent = `vertex ${vertexIndex} · ${jointID} ${(effectiveWeight(mesh, vertexIndex, jointID, S.ed.weights)).toFixed(4)} · Σ ${total.toFixed(4)} · edits ${S.ed.weights.filter((row) => row.mesh_id === mesh.mesh_id).length}`;
}

boot();
