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
const departedState = {...state, view_id: 'V0'};
state = resetPresentationOverride(canonical, 'V0');
assert.equal(state.graphics_evidence_departure, false);
assert.equal(state.view_id, 'V0');
assert.throws(() => normalizePresentationOverride({slot_order_back_to_front: ['BACK', 'BODY'], hidden_mesh_ids: []}, canonical), /exact canonical slot permutation/);
assert.throws(() => setSlotHidden(state, canonical, 'UNKNOWN', true), /unknown slot/);

const edits = [];
const binding = {binding_id: 'B:HAND', view_id: 'V0', primary_final_control_id: 'J:HAND'};
let edit = bindingEditFor(edits, binding, false);
assert.deepEqual(edit.translation_xy, [0, 0]);
assert.deepEqual(edit.scale_xy, [1, 1]);
edit = upsertBindingEdit(edits, binding, {
  view_id: 'V0',
  primary_final_control_id: 'J:ROOT',
  translation_xy: [12, -4],
  rotation_degrees: 25,
  scale_xy: [1.25, 0.75],
  visible: false,
});
assert.equal(edits.length, 1);
assert.equal(edit.view_id, 'V0');
assert.equal(edit.primary_final_control_id, 'J:ROOT');
assert.deepEqual(edit.translation_xy, [12, -4]);
assert.deepEqual(edit.scale_xy, [1.25, 0.75]);
assert.equal(edit.visible, false);

const sameIDOtherView = {binding_id: 'B:HAND', view_id: 'V1', primary_final_control_id: 'J:HAND'};
const otherEdit = upsertBindingEdit(edits, sameIDOtherView, {
  view_id: 'V1',
  primary_final_control_id: 'J:HAND',
  translation_xy: [-3, 2],
});
assert.equal(edits.length, 2);
assert.equal(otherEdit.view_id, 'V1');
assert.deepEqual(bindingEditFor(edits, binding).translation_xy, [12, -4]);
assert.deepEqual(bindingEditFor(edits, sameIDOtherView).translation_xy, [-3, 2]);

const hydrated = authoringLayerToEdits({
  view_id: 'V0',
  binding_edits: [edits[0]].map(({view_id, ...row}) => row),
  presentation_override: {...departedState, view_id: undefined},
});
assert.equal(hydrated.presentation.view_id, 'V0');
assert.equal(hydrated.bindings[0].view_id, 'V0');
assert.deepEqual(hydrated.presentation.slot_order_back_to_front, ['BACK', 'FRONT', 'BODY']);
assert.equal(hydrated.bindings[0].rotation_degrees, 25);
const requestV0 = editsToAuthoringRequest('bundle', 'V0', hydrated);
assert.deepEqual(requestV0.presentation_override.hidden_mesh_ids, ['BACK']);
assert.equal('view_id' in requestV0.presentation_override, false);
assert.deepEqual(requestV0.binding_edits[0].translation_xy, [12, -4]);
assert.equal('view_id' in requestV0.binding_edits[0], false);
requestV0.presentation_override.hidden_mesh_ids.push('FRONT');
assert.deepEqual(hydrated.presentation.hidden_mesh_ids, ['BACK']);

const requestV1 = editsToAuthoringRequest('bundle', 'V1', hydrated);
assert.equal(requestV1.presentation_override, null);
assert.deepEqual(requestV1.binding_edits, []);

console.log('STUDIO_PRESENTATION_CORE_CONTRACT_PASS');
