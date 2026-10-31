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

export function resetPresentationOverride(canonicalOrder) {
  return normalizePresentationOverride(null, canonicalOrder);
}

function pair(value, fallback) {
  if (!Array.isArray(value) || value.length !== 2) return [...fallback];
  const out = value.map(Number);
  return out.every(Number.isFinite) ? out : [...fallback];
}

export function normalizeBindingEdit(raw, bindingID, defaultControlID) {
  const row = raw && typeof raw === 'object' ? raw : {};
  const rotation = Number(row.rotation_degrees ?? 0);
  const scale = pair(row.scale_xy, [1, 1]).map((value) => Math.max(0.01, Math.min(100, value)));
  return {
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
  const rows = edits || [];
  let row = rows.find((item) => item.binding_id === bindingID);
  if (!row && create) {
    row = normalizeBindingEdit(null, bindingID, binding?.primary_final_control_id);
    rows.push(row);
  }
  return normalizeBindingEdit(row, bindingID, binding?.primary_final_control_id);
}

export function upsertBindingEdit(edits, binding, patch = {}) {
  const bindingID = String(binding?.binding_id || '');
  if (!bindingID) throw new Error('binding id required');
  let index = edits.findIndex((row) => row.binding_id === bindingID);
  const current = normalizeBindingEdit(index >= 0 ? edits[index] : null, bindingID, binding?.primary_final_control_id);
  const next = normalizeBindingEdit({...current, ...clone(patch)}, bindingID, binding?.primary_final_control_id);
  if (index < 0) edits.push(next);
  else edits[index] = next;
  return next;
}
