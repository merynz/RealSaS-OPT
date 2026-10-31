import assert from 'node:assert/strict';
import {authoringLayerToEdits, editsToAuthoringRequest} from '../../product/living_compile/static/js/studio_document.js';
import {
  bindingEditFor,
  canonicalSlotOrder,
  moveSlot,
  normalizePresentationOverride,
  resetPresentationOverride,
  setSlotHidden,
  upsertBindingEdit,
} from '../../product/living_compile/static/js/studio_presentation_core.js';

const meshes = [{mesh_id: 'BACK'}, {mesh_id: 'BODY'}, {mesh_id: 'FRONT'}];
const canonical = canonicalSlotOrder(meshes);
assert.deepEqual(canonical, ['BACK', 'BODY', 'FRONT']);

let state = normalizePresentationOverride(null, canonical);
assert.equal(state.graphics_evidence_departure, false);
state = moveSlot(state, canonical, 'BODY', 1);
assert.deepEqual(state.slot_order_back_to_front, ['BACK', 'FRONT', 'BODY']);
assert.equal(state.graphics_evidence_departure, true);
state = setSlotHidden(state, canonical, 'BACK', true);
assert.deepEqual(state.hidden_mesh_ids, ['BACK']);
assert.equal(state.graphics_evidence_departure, true);
const departedState = state;
state = resetPresentationOverride(canonical);
assert.equal(state.graphics_evidence_departure, false);
assert.throws(() => normalizePresentationOverride({slot_order_back_to_front: ['BACK', 'BODY'], hidden_mesh_ids: []}, canonical), /exact canonical slot permutation/);
assert.throws(() => setSlotHidden(state, canonical, 'UNKNOWN', true), /unknown slot/);

const edits = [];
const binding = {binding_id: 'B:V0:J:HAND', primary_final_control_id: 'J:HAND'};
let edit = bindingEditFor(edits, binding, false);
assert.deepEqual(edit.translation_xy, [0, 0]);
assert.deepEqual(edit.scale_xy, [1, 1]);
edit = upsertBindingEdit(edits, binding, {
  primary_final_control_id: 'J:ROOT',
  translation_xy: [12, -4],
  rotation_degrees: 25,
  scale_xy: [1.25, 0.75],
  visible: false,
});
assert.equal(edits.length, 1);
assert.equal(edit.primary_final_control_id, 'J:ROOT');
assert.deepEqual(edit.translation_xy, [12, -4]);
assert.deepEqual(edit.scale_xy, [1.25, 0.75]);
assert.equal(edit.visible, false);

const hydrated = authoringLayerToEdits({
  binding_edits: edits,
  presentation_override: departedState,
});
assert.deepEqual(hydrated.presentation.slot_order_back_to_front, ['BACK', 'FRONT', 'BODY']);
assert.equal(hydrated.bindings[0].rotation_degrees, 25);
const request = editsToAuthoringRequest('bundle', 'V0', hydrated);
assert.deepEqual(request.presentation_override.hidden_mesh_ids, ['BACK']);
assert.deepEqual(request.binding_edits[0].translation_xy, [12, -4]);
request.presentation_override.hidden_mesh_ids.push('FRONT');
assert.deepEqual(hydrated.presentation.hidden_mesh_ids, ['BACK']);

console.log('STUDIO_PRESENTATION_CORE_CONTRACT_PASS');
