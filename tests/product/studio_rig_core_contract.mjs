import assert from 'node:assert/strict';

import {
  boneGeometry,
  buildRigModel,
  canReparent,
  hierarchyRows,
  mirroredBoneID,
  nextUserBoneID,
  rigMirrorAxisX,
  setBoneAnchor,
  setBoneGeometry,
  upsertReparent,
} from '../../product/living_compile/static/js/studio_rig_core.js';

const controls = [
  {control_id: 'J:ROOT', anchor: [0, 0]},
  {control_id: 'J:L', anchor: [-1, 1]},
  {control_id: 'J:R', anchor: [1, 1]},
];
const edges = [
  {relation_type: 'skeleton_hierarchy', parent_control_id: 'J:ROOT', child_control_id: 'J:L'},
  {relation_type: 'skeleton_hierarchy', parent_control_id: 'J:ROOT', child_control_id: 'J:R'},
];
const rigEdits = [];
const topologyEdits = [];

let model = buildRigModel({controls, edges, rigEdits, topologyEdits});
assert.equal(hierarchyRows(model).length, 3);
assert.equal(rigMirrorAxisX(model), 0);
assert.equal(mirroredBoneID(model, 'J:L', 0), 'J:R');
assert.equal(nextUserBoneID(model), 'user:bone_1');
assert.equal(canReparent(model, 'J:ROOT', 'J:L'), false, 'cycle must be rejected');
assert.equal(canReparent(model, 'J:L', null), true);

setBoneAnchor({rigEdits, topologyEdits, model, boneID: 'J:L', anchor: [-2, 1]});
assert.deepEqual(rigEdits[0], {control_id: 'J:L', rotation_degrees: 0, translation_xy: [-1, 0]});
model = buildRigModel({controls, edges, rigEdits, topologyEdits});
assert.deepEqual(model.nodes.get('J:L').anchor, [-2, 1]);

const geometry = boneGeometry(model, 'J:L');
assert.ok(geometry.length > 2.2 && geometry.length < 2.3);
setBoneGeometry({rigEdits, topologyEdits, model, boneID: 'J:L', length: 2, angleDegrees: 90});
model = buildRigModel({controls, edges, rigEdits, topologyEdits});
assert.ok(Math.abs(model.nodes.get('J:L').anchor[0]) < 1e-8);
assert.ok(Math.abs(model.nodes.get('J:L').anchor[1] - 2) < 1e-8);

const id = nextUserBoneID(model);
topologyEdits.push({operation: 'add_bone', bone_id: id, parent_id: 'J:L', anchor_xy: [0, 3], display_name: id});
model = buildRigModel({controls, edges, rigEdits, topologyEdits});
assert.equal(model.nodes.get(id).user, true);
assert.equal(model.nodes.get(id).parent_id, 'J:L');
assert.equal(upsertReparent(topologyEdits, model, id, 'J:ROOT'), true);
model = buildRigModel({controls, edges, rigEdits, topologyEdits});
assert.equal(model.nodes.get(id).parent_id, 'J:ROOT');
setBoneAnchor({rigEdits, topologyEdits, model, boneID: id, anchor: [0.5, 2.5]});
model = buildRigModel({controls, edges, rigEdits, topologyEdits});
assert.deepEqual(model.nodes.get(id).anchor, [0.5, 2.5]);

topologyEdits.push({operation: 'delete_bone', bone_id: id});
model = buildRigModel({controls, edges, rigEdits, topologyEdits});
assert.equal(model.nodes.has(id), false);

console.log('STUDIO_RIG_CORE_CONTRACT_PASS');
