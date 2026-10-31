import assert from 'node:assert/strict';
import {
  editCounts,
  invalidatedDomains,
  requalificationPlan,
} from '../../product/living_compile/static/js/studio_requalification_core.js';

const empty = requalificationPlan({});
assert.equal(empty.required, false);
assert.equal(empty.state, 'QUALIFIED_BASE');
assert.deepEqual(empty.invalidated_domains, []);
assert.equal(empty.can_export_edited_runtime, false);
assert.equal(empty.qualified_base_export_remains_available, true);

assert.deepEqual(invalidatedDomains({animation: [{clip_id: 'USER'}]}), ['MOTION', 'PROOF']);
assert.deepEqual(invalidatedDomains({ik: [{constraint_id: 'IK:1'}]}), ['MECHANICS', 'MOTION', 'PROOF']);
assert.deepEqual(invalidatedDomains({weights: [{mesh_id: 'M'}]}), ['SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF']);
assert.deepEqual(invalidatedDomains({meshTopology: [{operation: 'split_edge'}]}), ['MESH', 'SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF']);
assert.deepEqual(invalidatedDomains({topology: [{operation: 'add_bone'}]}), ['RIG', 'SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF']);
assert.deepEqual(invalidatedDomains({
  rig: [{control_id: 'J:ROOT'}],
  animation: [{clip_id: 'USER'}],
}), ['RIG', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF']);

const presentationOnly = {
  presentation: {
    source_graphics_order_back_to_front: ['BACK', 'FRONT'],
    slot_order_back_to_front: ['FRONT', 'BACK'],
    hidden_mesh_ids: [],
  },
};
assert.deepEqual(invalidatedDomains(presentationOnly), ['GRAPHICS', 'PROOF']);
assert.equal(editCounts(presentationOnly).presentation, 1);

const blocked = requalificationPlan({bindings: [{binding_id: 'B'}]}, {executorBound: false});
assert.equal(blocked.required, true);
assert.equal(blocked.state, 'BLOCKED_EXECUTOR_UNBOUND');
assert.deepEqual(blocked.invalidated_domains, ['MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF']);

const ready = requalificationPlan({animation: [{clip_id: 'USER'}]}, {executorBound: true});
assert.equal(ready.state, 'READY_TO_REQUEST');
assert.equal(ready.executor_bound, true);
assert.deepEqual(ready.invalidated_domains, ['MOTION', 'PROOF']);
assert.equal(ready.canonical_product_mutated, false);

console.log('STUDIO_REQUALIFICATION_CORE_CONTRACT_PASS');
