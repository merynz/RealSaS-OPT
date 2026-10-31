import {
  boneGeometry,
  buildRigModel,
  hierarchyRows,
  mirroredBoneID,
  nextUserBoneID,
  rigMirrorAxisX,
  setBoneAnchor,
  setBoneGeometry,
  upsertReparent,
} from './studio_rig_core.js';

const $ = (selector) => document.querySelector(selector);
const SAFE_ID = /^[A-Za-z0-9_.:/-]+$/;

let studio = null;
let S = null;
let documentModel = null;
let journal = null;
let dirty = null;
let panel = null;
let viewport = null;
let overlay = null;
let saveTimer = null;
let drag = null;
let selectedIK = 0;
const mirrorAxisByView = new Map();

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
  installOverlay();

  const body = $('#inspectorBody');
  if (body) new MutationObserver(() => ensureInspector()).observe(body, {childList: true, subtree: true});
  panel.addEventListener('pointerdown', pointerDown, true);
  panel.addEventListener('pointermove', pointerMove, true);
  panel.addEventListener('pointerup', pointerUp, true);
  panel.addEventListener('pointercancel', pointerCancel, true);
  requestAnimationFrame(frame);
}

function currentViewID() {
  return `V${S?.view ?? 0}`;
}

function deformControls() {
  const rig = S?.scene?.puppet?.rig || {};
  return (rig.deform_controls || rig.controls || []).filter((row) => row.view_id === currentViewID());
}

function skeletonEdges() {
  return (S?.scene?.puppet?.rig?.edges || []).filter((row) => row.relation_type === 'skeleton_hierarchy');
}

function rigModel() {
  return buildRigModel({
    controls: deformControls(),
    edges: skeletonEdges(),
    rigEdits: S?.ed?.rig || [],
    topologyEdits: S?.ed?.topology || [],
  });
}

function selectedBone(model) {
  const id = S?.sel?.control;
  if (id && model.nodes.has(id)) return model.nodes.get(id);
  const first = hierarchyRows(model)[0] || null;
  if (first && S?.sel) S.sel.control = first.bone_id;
  return first;
}

function axisX(model) {
  const key = currentViewID();
  if (!mirrorAxisByView.has(key)) mirrorAxisByView.set(key, rigMirrorAxisX(model));
  return Number(mirrorAxisByView.get(key));
}

function operate(label, operation) {
  const before = documentModel.snapshot();
  operation();
  if (!journal.commit(label, before, documentModel.snapshot())) return false;
  dirty.markDirty();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (dirty.status === 'dirty') $('#saveLayer')?.click();
  }, 1200);
  ensureInspector(true);
  return true;
}

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function boneOptions(model, selectedID = '', allowRoot = true) {
  const rows = hierarchyRows(model);
  return `${allowRoot ? '<option value="">ROOT</option>' : ''}${rows.map((row) => `<option value="${escapeHTML(row.bone_id)}"${row.bone_id === selectedID ? ' selected' : ''}>${'· '.repeat(row.depth)}${escapeHTML(row.bone_id)}</option>`).join('')}`;
}

function ensureInspector(force = false) {
  if (!S?.scene || !['topology', 'ik'].includes(S.mode)) return;
  const body = $('#inspectorBody');
  if (!body) return;
  const model = rigModel();
  const bone = selectedBone(model);
  const signature = `${S.mode}:${currentViewID()}:${bone?.bone_id || ''}:${S.ed.topology.length}:${S.ed.ik.length}:${S.ed.rig.length}`;
  const existing = $('#rigEditorPro');
  if (existing && !force && existing.dataset.signature === signature) return;
  if (S.mode === 'topology') renderTopology(body, model, bone, signature);
  else renderIK(body, model, bone, signature);
}

function renderTopology(body, model, bone, signature) {
  const rows = hierarchyRows(model);
  const geometry = bone ? boneGeometry(model, bone.bone_id) : null;
  const parentID = bone?.parent_id || '';
  const userID = nextUserBoneID(model);
  const mirrorID = bone ? `${bone.bone_id}:mirror`.replace(/[^A-Za-z0-9_.:/-]/g, '_').slice(0, 128) : 'user:mirror';
  const hierarchy = rows.map((row) => `<button class="rig-tree-row${row.bone_id === bone?.bone_id ? ' active' : ''}" data-rig-select="${escapeHTML(row.bone_id)}" style="padding-left:${8 + row.depth * 14}px"><span>${row.user ? '◇' : '◆'}</span><span>${escapeHTML(row.bone_id)}</span><small>${row.user ? 'USER' : 'CANON'}</small></button>`).join('');

  body.innerHTML = `<div id="rigEditorPro" class="inspector-section rig-editor" data-signature="${escapeHTML(signature)}">
    <h3>RIG HIERARCHY · ${escapeHTML(currentViewID())}</h3>
    <div class="rig-tree">${hierarchy || '<div class="rig-empty">No deform joints</div>'}</div>
    ${bone ? `<div class="rig-selected">
      <div class="rig-selected-head"><strong>${escapeHTML(bone.bone_id)}</strong><span>${bone.user ? 'USER BONE' : 'CANONICAL DEFORM JOINT'}</span></div>
      <label>PARENT<select id="rigParent">${boneOptions(model, parentID, true)}</select></label>
      <div class="rig-grid"><label>X<input id="rigX" type="number" step="0.1" value="${bone.anchor[0]}"></label><label>Y<input id="rigY" type="number" step="0.1" value="${bone.anchor[1]}"></label><button id="rigApplyAnchor">APPLY POSITION</button></div>
      ${geometry?.parent_id ? `<div class="rig-grid"><label>LENGTH<input id="rigLength" type="number" min="0" step="0.1" value="${geometry.length}"></label><label>ANGLE°<input id="rigAngle" type="number" step="0.1" value="${geometry.angle_degrees}"></label><button id="rigApplyGeometry">APPLY BONE</button></div>` : '<div class="rig-note">Root bone has no parent-relative length/orientation.</div>'}
      <div class="rig-danger-row"><button id="rigDelete" class="danger">DELETE IN AUTHORING LAYER</button></div>
    </div>` : ''}
    <div class="rig-create"><h4>CREATE CHILD</h4><input id="rigNewID" value="${escapeHTML(userID)}"><button id="rigAdd">ADD CHILD</button></div>
    ${bone ? `<div class="rig-create"><h4>MIRROR DUPLICATE</h4><label>AXIS X<input id="rigMirrorAxis" type="number" step="0.1" value="${axisX(model)}"></label><input id="rigMirrorID" value="${escapeHTML(mirrorID)}"><button id="rigMirror">CREATE MIRROR</button></div>` : ''}
    <div class="rig-note">Canonical ProductRevision stays immutable. Position edits are rest/setup overlay transforms; user bones live in the authoring namespace and require requalification.</div>
  </div>`;

  body.querySelectorAll('[data-rig-select]').forEach((button) => {
    button.onclick = () => {
      S.sel.control = button.dataset.rigSelect;
      ensureInspector(true);
    };
  });
  if ($('#rigParent')) $('#rigParent').onchange = (event) => operate('Reparent rig bone', () => {
    const fresh = rigModel();
    if (!upsertReparent(S.ed.topology, fresh, bone.bone_id, event.target.value || null)) throw new Error('Invalid rig reparent');
  });
  if ($('#rigApplyAnchor')) $('#rigApplyAnchor').onclick = () => operate('Edit rig bone position', () => {
    setBoneAnchor({
      rigEdits: S.ed.rig,
      topologyEdits: S.ed.topology,
      model: rigModel(),
      boneID: bone.bone_id,
      anchor: [Number($('#rigX').value), Number($('#rigY').value)],
    });
  });
  if ($('#rigApplyGeometry')) $('#rigApplyGeometry').onclick = () => operate('Edit rig bone geometry', () => {
    setBoneGeometry({
      rigEdits: S.ed.rig,
      topologyEdits: S.ed.topology,
      model: rigModel(),
      boneID: bone.bone_id,
      length: Number($('#rigLength').value),
      angleDegrees: Number($('#rigAngle').value),
    });
  });
  if ($('#rigDelete')) $('#rigDelete').onclick = () => operate('Delete rig bone', () => {
    S.ed.topology.push({operation: 'delete_bone', bone_id: bone.bone_id});
    S.sel.control = bone.parent_id || null;
  });
  $('#rigAdd').onclick = () => {
    const id = $('#rigNewID').value.trim();
    if (!SAFE_ID.test(id) || id.length > 128) return;
    operate('Add rig bone', () => {
      const fresh = rigModel();
      if (fresh.nodes.has(id)) return;
      const parent = selectedBone(fresh);
      const anchor = parent ? [parent.anchor[0] + 24, parent.anchor[1]] : [0, 0];
      S.ed.topology.push({operation: 'add_bone', bone_id: id, parent_id: parent?.bone_id || null, anchor_xy: anchor, display_name: id});
      S.sel.control = id;
    });
  };
  if ($('#rigMirrorAxis')) $('#rigMirrorAxis').onchange = (event) => {
    mirrorAxisByView.set(currentViewID(), Number(event.target.value) || 0);
  };
  if ($('#rigMirror')) $('#rigMirror').onclick = () => {
    const id = $('#rigMirrorID').value.trim();
    if (!SAFE_ID.test(id) || id.length > 128) return;
    operate('Create mirrored rig bone', () => {
      const fresh = rigModel();
      const source = fresh.nodes.get(bone.bone_id);
      if (!source || fresh.nodes.has(id)) return;
      const axis = axisX(fresh);
      const parent = source.parent_id ? mirroredBoneID(fresh, source.parent_id, axis) : null;
      S.ed.topology.push({
        operation: 'add_bone',
        bone_id: id,
        parent_id: parent && fresh.nodes.has(parent) ? parent : source.parent_id,
        anchor_xy: [2 * axis - source.anchor[0], source.anchor[1]],
        display_name: id,
      });
      S.sel.control = id;
    });
  };
}

function renderIK(body, model, bone, signature) {
  const constraints = S.ed.ik || [];
  if (selectedIK >= constraints.length) selectedIK = Math.max(0, constraints.length - 1);
  const active = constraints[selectedIK] || null;
  const rows = hierarchyRows(model);
  const fallback = bone?.bone_id || rows[0]?.bone_id || '';
  const list = constraints.map((row, index) => `<button class="ik-list-row${index === selectedIK ? ' active' : ''}" data-ik-index="${index}"><span>${escapeHTML(row.constraint_id)}</span><small>${row.enabled === false ? 'OFF' : 'ON'}</small></button>`).join('');

  body.innerHTML = `<div id="rigEditorPro" class="inspector-section rig-editor" data-signature="${escapeHTML(signature)}">
    <h3>IK CONSTRAINTS</h3>
    <div class="ik-toolbar"><button id="ikNew">NEW IK</button>${active ? '<button id="ikDelete" class="danger">DELETE</button>' : ''}</div>
    <div class="ik-list">${list || '<div class="rig-empty">No user IK constraints yet.</div>'}</div>
    ${active ? `<div class="ik-editor-form">
      <label>ID<input id="ikEditID" value="${escapeHTML(active.constraint_id)}"></label>
      <label>START<select id="ikStart">${boneOptions(model, active.start_bone_id, false)}</select></label>
      <label>END<select id="ikEnd">${boneOptions(model, active.end_bone_id, false)}</select></label>
      <label>TARGET<select id="ikTarget">${boneOptions(model, active.target_bone_id, false)}</select></label>
      <div class="rig-grid"><label>MIX<input id="ikMix" type="number" min="0" max="1" step="0.05" value="${active.mix ?? 1}"></label><label>ITERATIONS<input id="ikIterations" type="number" min="1" max="128" step="1" value="${active.iterations ?? 12}"></label></div>
      <div class="weight-toggles"><label><input id="ikEnabled" type="checkbox"${active.enabled === false ? '' : ' checked'}> ENABLED</label><label><input id="ikBend" type="checkbox"${Number(active.bend_direction || 1) < 0 ? ' checked' : ''}> BEND NEGATIVE</label></div>
      <button id="ikApply">APPLY CONSTRAINT</button>
    </div>` : ''}
    <div class="rig-note">IK references deform bones only. True constraint execution remains owned by Mechanics; Studio edits are authoring data until requalified.</div>
  </div>`;

  body.querySelectorAll('[data-ik-index]').forEach((button) => {
    button.onclick = () => {
      selectedIK = Number(button.dataset.ikIndex);
      ensureInspector(true);
    };
  });
  $('#ikNew').onclick = () => operate('Add IK constraint', () => {
    const ids = hierarchyRows(rigModel()).map((row) => row.bone_id);
    const end = S.sel.control && ids.includes(S.sel.control) ? S.sel.control : ids[0] || fallback;
    const parent = rigModel().nodes.get(end)?.parent_id || ids[0] || end;
    let index = 1;
    const existing = new Set(S.ed.ik.map((row) => row.constraint_id));
    while (existing.has(`IK:USER:${index}`)) index += 1;
    S.ed.ik.push({constraint_id: `IK:USER:${index}`, start_bone_id: parent, end_bone_id: end, target_bone_id: end, mix: 1, iterations: 12, bend_direction: 1, enabled: true});
    selectedIK = S.ed.ik.length - 1;
  });
  if ($('#ikDelete')) $('#ikDelete').onclick = () => operate('Delete IK constraint', () => {
    S.ed.ik.splice(selectedIK, 1);
    selectedIK = Math.max(0, selectedIK - 1);
  });
  if ($('#ikApply')) $('#ikApply').onclick = () => {
    const id = $('#ikEditID').value.trim();
    if (!SAFE_ID.test(id) || id.length > 128) return;
    operate('Edit IK constraint', () => {
      const row = S.ed.ik[selectedIK];
      Object.assign(row, {
        constraint_id: id,
        start_bone_id: $('#ikStart').value,
        end_bone_id: $('#ikEnd').value,
        target_bone_id: $('#ikTarget').value,
        mix: Math.max(0, Math.min(1, Number($('#ikMix').value) || 0)),
        iterations: Math.max(1, Math.min(128, Math.round(Number($('#ikIterations').value) || 12))),
        bend_direction: $('#ikBend').checked ? -1 : 1,
        enabled: $('#ikEnabled').checked,
      });
    });
  };
}

function installOverlay() {
  overlay = document.createElement('canvas');
  overlay.className = 'rig-authoring-overlay';
  overlay.width = viewport.width;
  overlay.height = viewport.height;
  panel.append(overlay);
}

function frame() {
  drawOverlay();
  if (['topology', 'ik'].includes(S?.mode)) ensureInspector();
  requestAnimationFrame(frame);
}

function drawOverlay() {
  if (!overlay || !viewport) return;
  const visible = ['topology', 'ik'].includes(S?.mode) && Boolean(S?.scene);
  overlay.hidden = !visible;
  if (!visible) return;
  overlay.width = viewport.width;
  overlay.height = viewport.height;
  overlay.style.left = `${viewport.offsetLeft}px`;
  overlay.style.top = `${viewport.offsetTop}px`;
  overlay.style.width = `${viewport.clientWidth}px`;
  overlay.style.height = `${viewport.clientHeight}px`;
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const model = rigModel();
  const selected = S.sel.control;
  const sc = (point) => [point[0] * S.cam.s + S.cam.x, point[1] * S.cam.s + S.cam.y];
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  for (const node of model.nodes.values()) {
    if (!node.parent_id || !model.nodes.has(node.parent_id)) continue;
    const parent = model.nodes.get(node.parent_id);
    ctx.strokeStyle = node.user || parent.user ? 'rgba(255,211,107,.75)' : 'rgba(112,145,170,.25)';
    ctx.beginPath();
    ctx.moveTo(...sc(parent.anchor));
    ctx.lineTo(...sc(node.anchor));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const node of model.nodes.values()) {
    if (!node.user) continue;
    const point = sc(node.anchor);
    ctx.fillStyle = node.bone_id === selected ? '#ffffff' : '#ffd36b';
    ctx.strokeStyle = '#3a2f16';
    ctx.beginPath();
    ctx.arc(point[0], point[1], node.bone_id === selected ? 7 : 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
}

function clientFromWorld(point) {
  const bounds = viewport.getBoundingClientRect();
  const intrinsic = [point[0] * S.cam.s + S.cam.x, point[1] * S.cam.s + S.cam.y];
  return [bounds.left + intrinsic[0] * bounds.width / viewport.width, bounds.top + intrinsic[1] * bounds.height / viewport.height];
}

function worldFromEvent(event) {
  const bounds = viewport.getBoundingClientRect();
  const intrinsic = [
    (event.clientX - bounds.left) * viewport.width / Math.max(1, bounds.width),
    (event.clientY - bounds.top) * viewport.height / Math.max(1, bounds.height),
  ];
  return [(intrinsic[0] - S.cam.x) / S.cam.s, (intrinsic[1] - S.cam.y) / S.cam.s];
}

function userBoneHit(event) {
  const model = rigModel();
  let best = null;
  let distance = 12;
  for (const node of model.nodes.values()) {
    if (!node.user) continue;
    const client = clientFromWorld(node.anchor);
    const candidate = Math.hypot(client[0] - event.clientX, client[1] - event.clientY);
    if (candidate < distance) {
      distance = candidate;
      best = node;
    }
  }
  return best;
}

function consume(event) {
  event.preventDefault();
  event.stopPropagation();
  if (typeof event.stopImmediatePropagation === 'function') event.stopImmediatePropagation();
}

function pointerDown(event) {
  if (!['topology', 'ik'].includes(S?.mode) || event.button !== 0) return;
  const node = userBoneHit(event);
  if (!node) return;
  consume(event);
  try { panel.setPointerCapture(event.pointerId); } catch {}
  S.sel.control = node.bone_id;
  drag = {pointerID: event.pointerId, boneID: node.bone_id, before: documentModel.snapshot(), changed: false};
  ensureInspector(true);
}

function pointerMove(event) {
  if (!drag || drag.pointerID !== event.pointerId) return;
  consume(event);
  const model = rigModel();
  if (setBoneAnchor({rigEdits: S.ed.rig, topologyEdits: S.ed.topology, model, boneID: drag.boneID, anchor: worldFromEvent(event)})) {
    drag.changed = true;
    dirty.markDirty();
  }
}

function pointerUp(event) {
  if (!drag || drag.pointerID !== event.pointerId) return;
  consume(event);
  const current = drag;
  drag = null;
  try { panel.releasePointerCapture(event.pointerId); } catch {}
  if (current.changed) {
    journal.commit('Move user rig bone', current.before, documentModel.snapshot());
    dirty.markDirty();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { if (dirty.status === 'dirty') $('#saveLayer')?.click(); }, 1200);
  }
  ensureInspector(true);
}

function pointerCancel(event) {
  if (!drag || drag.pointerID !== event.pointerId) return;
  consume(event);
  documentModel.replace(drag.before);
  drag = null;
  ensureInspector(true);
}

boot();
