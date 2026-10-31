function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

export function canonicalSlotOrder(meshes = []) {
  return meshes.map((row) => String(row.mesh_id || '')).filter(Boolean);
}

export function normalizePresentationOverride(raw, canonicalOrder) {
  const canonical = [...canonicalOrder];
  if (!raw || typeof raw !== 'object') {
    return {
      view_id: null,
      source_graphics_order_back_to_front: canonical,
      slot_order_back_to_front: canonical,
      hidden_mesh_ids: [],
      graphics_evidence_departure: false,
    };
  }
  const order = Array.isArray(raw.slot_order_back_to_front)
    ? raw.slot_order_back_to_front.map(String)
    : canonical;
  const hidden = Array.isArray(raw.hidden_mesh_ids)
    ? [...new Set(raw.hidden_mesh_ids.map(String))]
    : [];
  const canonicalSet = new Set(canonical);
  if (order.length !== canonical.length || new Set(order).size !== canonical.length || order.some((id) => !canonicalSet.has(id))) {
    throw new Error('presentation order must be an exact canonical slot permutation');
  }
  if (hidden.some((id) => !canonicalSet.has(id))) throw new Error('hidden slot is not canonical');
  const departure = hidden.length > 0 || order.some((id, index) => id !== canonical[index]);
  return {
    view_id: raw.view_id ? String(raw.view_id) : null,
    source_graphics_order_back_to_front: canonical,
    slot_order_back_to_front: order,
    hidden_mesh_ids: hidden,
    graphics_evidence_departure: departure,
  };
}

export function moveSlot(raw, canonicalOrder, meshID, delta) {
  const state = normalizePresentationOverride(raw, canonicalOrder);
  const index = state.slot_order_back_to_front.indexOf(meshID);
  if (index < 0) throw new Error(`unknown slot: ${meshID}`);
  const next = Math.max(0, Math.min(state.slot_order_back_to_front.length - 1, index + Number(delta || 0)));
  if (next !== index) {
    const order = [...state.slot_order_back_to_front];
    const [item] = order.splice(index, 1);
    order.splice(next, 0, item);
    state.slot_order_back_to_front = order;
  }
  return normalizePresentationOverride(state, canonicalOrder);
}

export function setSlotHidden(raw, canonicalOrder, meshID, hidden) {
  const state = normalizePresentationOverride(raw, canonicalOrder);
  if (!canonicalOrder.includes(meshID)) throw new Error(`unknown slot: ${meshID}`);
  const set = new Set(state.hidden_mesh_ids);
  if (hidden) set.add(meshID);
  else set.delete(meshID);
  state.hidden_mesh_ids = [...set];
  return normalizePresentationOverride(state, canonicalOrder);
}

export function resetPresentationOverride(canonicalOrder, viewID = null) {
  return {...normalizePresentationOverride(null, canonicalOrder), view_id: viewID ? String(viewID) : null};
}

function pair(value, fallback) {
  if (!Array.isArray(value) || value.length !== 2) return [...fallback];
  const out = value.map(Number);
  return out.every(Number.isFinite) ? out : [...fallback];
}

function bindingViewID(binding) {
  return binding?.view_id ? String(binding.view_id) : null;
}

function sameBinding(row, binding) {
  const bindingID = String(binding?.binding_id || '');
  const viewID = bindingViewID(binding);
  if (String(row?.binding_id || '') !== bindingID) return false;
  if (!viewID) return true;
  return !row?.view_id || String(row.view_id) === viewID;
}

export function normalizeBindingEdit(raw, bindingID, defaultControlID, viewID = null) {
  const row = raw && typeof raw === 'object' ? raw : {};
  const rotation = Number(row.rotation_degrees ?? 0);
  const scale = pair(row.scale_xy, [1, 1]).map((value) => Math.max(0.01, Math.min(100, value)));
  return {
    view_id: row.view_id ? String(row.view_id) : (viewID ? String(viewID) : null),
    binding_id: String(bindingID),
    primary_final_control_id: String(row.primary_final_control_id || defaultControlID || ''),
    translation_xy: pair(row.translation_xy, [0, 0]),
    rotation_degrees: Number.isFinite(rotation) ? rotation : 0,
    scale_xy: scale,
    visible: row.visible !== false,
  };
}

export function bindingEditFor(edits, binding, create = false) {
  const bindingID = String(binding?.binding_id || '');
  const viewID = bindingViewID(binding);
  const rows = edits || [];
  let row = rows.find((item) => sameBinding(item, binding));
  if (!row && create) {
    row = normalizeBindingEdit(null, bindingID, binding?.primary_final_control_id, viewID);
    rows.push(row);
  }
  return normalizeBindingEdit(row, bindingID, binding?.primary_final_control_id, viewID);
}

export function upsertBindingEdit(edits, binding, patch = {}) {
  const bindingID = String(binding?.binding_id || '');
  const viewID = bindingViewID(binding) || (patch?.view_id ? String(patch.view_id) : null);
  if (!bindingID) throw new Error('binding id required');
  const identity = {...binding, view_id: viewID};
  let index = edits.findIndex((row) => sameBinding(row, identity));
  const current = normalizeBindingEdit(index >= 0 ? edits[index] : null, bindingID, binding?.primary_final_control_id, viewID);
  const next = normalizeBindingEdit({...current, ...clone(patch), view_id: viewID}, bindingID, binding?.primary_final_control_id, viewID);
  if (index < 0) edits.push(next);
  else edits[index] = next;
  return next;
}
