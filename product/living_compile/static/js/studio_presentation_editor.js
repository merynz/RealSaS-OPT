import {
  bindingEditFor,
  canonicalSlotOrder,
  moveSlot,
  normalizePresentationOverride,
  resetPresentationOverride,
  setSlotHidden,
  upsertBindingEdit,
} from './studio_presentation_core.js';

const $ = (selector) => document.querySelector(selector);
let studio = null;
let S = null;
let documentModel = null;
let journal = null;
let dirty = null;
let panel = null;
let viewport = null;
let overlay = null;
let saveTimer = null;
let selectedBindingID = null;

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
  requestAnimationFrame(frame);
}

function currentViewID() {
  return `V${S?.view ?? 0}`;
}

function meshes() {
  return (S?.scene?.puppet?.meshes || []).filter((row) => row.view_id === currentViewID());
}

function bindings() {
  return (S?.scene?.binding?.bindings || []).filter((row) => row.view_id === currentViewID());
}

function controls() {
  const rig = S?.scene?.puppet?.rig || {};
  return (rig.deform_controls || rig.controls || []).filter((row) => row.view_id === currentViewID());
}

function canonicalOrder() {
  return canonicalSlotOrder(meshes());
}

function presentation() {
  return normalizePresentationOverride(S?.ed?.presentation, canonicalOrder());
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

function controlOptions(selected) {
  return controls().map((row) => `<option value="${escapeHTML(row.control_id)}"${row.control_id === selected ? ' selected' : ''}>${escapeHTML(row.control_id)}</option>`).join('');
}

function ensureInspector(force = false) {
  if (!S?.scene || S.mode !== 'bindings') return;
  const body = $('#inspectorBody');
  if (!body) return;
  const canonical = canonicalOrder();
  const state = presentation();
  const bindingRows = bindings();
  if (!selectedBindingID || !bindingRows.some((row) => row.binding_id === selectedBindingID)) {
    selectedBindingID = bindingRows[0]?.binding_id || null;
  }
  const signature = `${currentViewID()}:${state.slot_order_back_to_front.join(',')}:${state.hidden_mesh_ids.join(',')}:${S.ed.bindings.length}:${selectedBindingID || ''}`;
  const existing = $('#presentationEditor');
  if (existing && !force && existing.dataset.signature === signature) return;
  const selected = bindingRows.find((row) => row.binding_id === selectedBindingID) || null;
  const edit = selected ? bindingEditFor(S.ed.bindings, selected, false) : null;
  const sourceIndex = new Map(canonical.map((id, index) => [id, index]));
  const slots = state.slot_order_back_to_front.map((meshID, index) => {
    const hidden = state.hidden_mesh_ids.includes(meshID);
    const source = sourceIndex.get(meshID);
    const moved = source !== index;
    return `<div class="slot-row${moved || hidden ? ' departed' : ''}" data-slot="${escapeHTML(meshID)}">
      <div class="slot-main"><strong>${escapeHTML(meshID)}</strong><small>${hidden ? 'HIDDEN OVERRIDE' : moved ? `ORDER ${source + 1}→${index + 1}` : `QUALIFIED ${index + 1}`}</small></div>
      <button data-slot-up="${escapeHTML(meshID)}"${index === state.slot_order_back_to_front.length - 1 ? ' disabled' : ''}>FRONT</button>
      <button data-slot-down="${escapeHTML(meshID)}"${index === 0 ? ' disabled' : ''}>BACK</button>
      <button data-slot-hide="${escapeHTML(meshID)}" class="${hidden ? 'active' : ''}">${hidden ? 'SHOW' : 'HIDE'}</button>
    </div>`;
  }).join('');
  const bindingList = bindingRows.map((row) => `<button class="binding-row${row.binding_id === selectedBindingID ? ' active' : ''}" data-binding-select="${escapeHTML(row.binding_id)}"><span>${escapeHTML(row.binding_id)}</span><small>${escapeHTML(row.primary_final_control_id)}</small></button>`).join('');

  body.innerHTML = `<div id="presentationEditor" class="inspector-section presentation-editor" data-signature="${escapeHTML(signature)}">
    <div class="presentation-head"><h3>GRAPHICS PRESENTATION · ${escapeHTML(currentViewID())}</h3><span class="evidence-badge ${state.graphics_evidence_departure ? 'warn' : 'good'}">${state.graphics_evidence_departure ? 'DEPARTS FROM GRAPHICS PROOF' : 'QUALIFIED GRAPHICS ORDER'}</span></div>
    <div class="presentation-note">Back → front order comes from CanonicalGraphicsIR. Reorder/hide is allowed only as sibling authoring data and requires requalification.</div>
    <div class="slot-list">${slots || '<div class="rig-empty">No render slots in this view.</div>'}</div>
    <button id="presentationReset">RESET TO QUALIFIED GRAPHICS</button>
    <h3>ATTACHMENT / SOCKET BINDINGS</h3>
    <div class="binding-list">${bindingList || '<div class="rig-empty">No binding sockets in this view.</div>'}</div>
    ${selected && edit ? `<div class="binding-form">
      <div class="binding-selected"><strong>${escapeHTML(selected.binding_id)}</strong><small>BASE ${escapeHTML(selected.primary_final_control_id)}</small></div>
      <label>DEFORM JOINT<select id="bindingControl">${controlOptions(edit.primary_final_control_id)}</select></label>
      <div class="rig-grid"><label>OFFSET X<input id="bindingTX" type="number" step="0.1" value="${edit.translation_xy[0]}"></label><label>OFFSET Y<input id="bindingTY" type="number" step="0.1" value="${edit.translation_xy[1]}"></label></div>
      <div class="rig-grid"><label>ROTATION°<input id="bindingRot" type="number" step="0.1" value="${edit.rotation_degrees}"></label><label>VISIBLE<input id="bindingVisible" type="checkbox"${edit.visible ? ' checked' : ''}></label></div>
      <div class="rig-grid"><label>SCALE X<input id="bindingSX" type="number" min="0.01" max="100" step="0.01" value="${edit.scale_xy[0]}"></label><label>SCALE Y<input id="bindingSY" type="number" min="0.01" max="100" step="0.01" value="${edit.scale_xy[1]}"></label></div>
      <button id="bindingApply">APPLY SOCKET OVERRIDE</button><button id="bindingReset">RESET SOCKET OVERRIDE</button>
    </div>` : ''}
    <div class="presentation-note">Socket transforms do not mutate canonical Rig/Mechanics. True attachment semantics remain compiler-owned until the edited revision is requalified.</div>
  </div>`;

  body.querySelectorAll('[data-slot-up]').forEach((button) => button.onclick = () => operate('Move slot toward front', () => {
    S.ed.presentation = moveSlot(S.ed.presentation, canonical, button.dataset.slotUp, 1);
  }));
  body.querySelectorAll('[data-slot-down]').forEach((button) => button.onclick = () => operate('Move slot toward back', () => {
    S.ed.presentation = moveSlot(S.ed.presentation, canonical, button.dataset.slotDown, -1);
  }));
  body.querySelectorAll('[data-slot-hide]').forEach((button) => button.onclick = () => operate('Toggle slot visibility', () => {
    const id = button.dataset.slotHide;
    const current = presentation();
    S.ed.presentation = setSlotHidden(current, canonical, id, !current.hidden_mesh_ids.includes(id));
  }));
  $('#presentationReset').onclick = () => operate('Reset Graphics presentation', () => {
    S.ed.presentation = resetPresentationOverride(canonical);
  });
  body.querySelectorAll('[data-binding-select]').forEach((button) => button.onclick = () => {
    selectedBindingID = button.dataset.bindingSelect;
    ensureInspector(true);
  });
  if ($('#bindingApply')) $('#bindingApply').onclick = () => operate('Edit attachment socket', () => {
    upsertBindingEdit(S.ed.bindings, selected, {
      primary_final_control_id: $('#bindingControl').value,
      translation_xy: [Number($('#bindingTX').value), Number($('#bindingTY').value)],
      rotation_degrees: Number($('#bindingRot').value),
      scale_xy: [Number($('#bindingSX').value), Number($('#bindingSY').value)],
      visible: $('#bindingVisible').checked,
    });
  });
  if ($('#bindingReset')) $('#bindingReset').onclick = () => operate('Reset attachment socket', () => {
    S.ed.bindings = S.ed.bindings.filter((row) => row.binding_id !== selected.binding_id);
  });
}

function installOverlay() {
  overlay = document.createElement('canvas');
  overlay.className = 'presentation-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  panel.append(overlay);
}

function resizeOverlay() {
  const rect = viewport.getBoundingClientRect();
  const panelRect = panel.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.style.left = `${rect.left - panelRect.left}px`;
  overlay.style.top = `${rect.top - panelRect.top}px`;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
  return {rect, dpr};
}

function bindingPoint(binding) {
  const edit = bindingEditFor(S.ed.bindings, binding, false);
  const base = binding.selected_socket_xy || binding.selected_anchor_xy || [0, 0];
  return [Number(base[0]) + edit.translation_xy[0], Number(base[1]) + edit.translation_xy[1]];
}

function drawOverlay() {
  if (!overlay) return;
  const {rect, dpr} = resizeOverlay();
  const ctx = overlay.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (S?.mode !== 'bindings') return;
  for (const binding of bindings()) {
    const edit = bindingEditFor(S.ed.bindings, binding, false);
    if (!edit.visible) continue;
    const world = bindingPoint(binding);
    const intrinsic = [world[0] * S.cam.s + S.cam.x, world[1] * S.cam.s + S.cam.y];
    const x = intrinsic[0] * rect.width / viewport.width;
    const y = intrinsic[1] * rect.height / viewport.height;
    const selected = binding.binding_id === selectedBindingID;
    const angle = edit.rotation_degrees * Math.PI / 180;
    const length = 16 * Math.max(0.5, Math.min(2, (edit.scale_xy[0] + edit.scale_xy[1]) / 2));
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.strokeStyle = selected ? '#ffffff' : '#62d0ff';
    ctx.fillStyle = selected ? '#ffffff' : '#62d0ff';
    ctx.lineWidth = selected ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(-6, 0); ctx.lineTo(length, 0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, -6); ctx.lineTo(0, 6); ctx.stroke();
    ctx.beginPath(); ctx.arc(0, 0, selected ? 4 : 3, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
  }
}

function frame() {
  if (S?.mode === 'bindings') ensureInspector();
  drawOverlay();
  requestAnimationFrame(frame);
}

boot();
