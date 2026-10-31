import assert from 'node:assert/strict';

import {
  effectiveWeight,
  falloffWeight,
  meshMirrorAxisX,
  mirroredControlID,
  mirroredVertexIndex,
  normalizeVertexWeights,
  paintVertex,
  setEffectiveWeight,
  vertexAdjacency,
} from '../../product/living_compile/static/js/studio_weight_core.js';

const mesh = {
  mesh_id: 'M:BODY',
  vertices: [[-1, 0], [1, 0], [-1, 1], [1, 1]],
  triangles: [[0, 1, 2], [1, 3, 2]],
  weight_solution: {
    per_vertex_weights: [
      [['J:L', 0.8], ['J:R', 0.2]],
      [['J:L', 0.2], ['J:R', 0.8]],
      [['J:L', 1.0]],
      [['J:R', 1.0]],
    ],
  },
};
const joints = ['J:L', 'J:R'];
const edits = [];

assert.equal(effectiveWeight(mesh, 0, 'J:L', edits), 0.8);
setEffectiveWeight(edits, mesh, 0, 'J:L', 0.5);
assert.equal(effectiveWeight(mesh, 0, 'J:L', edits), 0.5);
assert.equal(edits.length, 1);
setEffectiveWeight(edits, mesh, 0, 'J:L', 0.6);
assert.equal(edits.length, 1, 'same vertex/joint edit should be consolidated');
assert.equal(effectiveWeight(mesh, 0, 'J:L', edits), 0.6);

const normalized = normalizeVertexWeights({
  mesh,
  vertexIndex: 0,
  jointIDs: joints,
  edits,
  lockedJointIDs: ['J:R'],
  preferredJointID: 'J:L',
});
assert.equal(normalized.ok, true);
assert.ok(Math.abs(effectiveWeight(mesh, 0, 'J:R', edits) - 0.2) < 1e-8, 'locked joint must remain unchanged');
assert.ok(Math.abs(effectiveWeight(mesh, 0, 'J:L', edits) - 0.8) < 1e-8);

const adjacency = vertexAdjacency(mesh);
assert.deepEqual(adjacency[0].sort((a, b) => a - b), [1, 2]);
const painted = paintVertex({
  mesh,
  vertexIndex: 1,
  jointID: 'J:L',
  jointIDs: joints,
  edits,
  amount: 0.25,
  mode: 'add',
  normalize: true,
  lockedJointIDs: [],
  adjacency,
});
assert.equal(painted.changed, true);
const total = joints.reduce((sum, joint) => sum + effectiveWeight(mesh, 1, joint, edits), 0);
assert.ok(Math.abs(total - 1) < 1e-8, 'auto-normalize must keep vertex sum at one');

const beforeSmooth = effectiveWeight(mesh, 2, 'J:R', edits);
const smoothed = paintVertex({
  mesh,
  vertexIndex: 2,
  jointID: 'J:R',
  jointIDs: joints,
  edits,
  amount: 0.5,
  mode: 'smooth',
  normalize: false,
  adjacency,
});
assert.equal(smoothed.changed, true);
assert.ok(effectiveWeight(mesh, 2, 'J:R', edits) > beforeSmooth);

assert.equal(falloffWeight(0, 40, 'smooth'), 1);
assert.equal(falloffWeight(40, 40, 'smooth'), 0);
assert.ok(falloffWeight(20, 40, 'linear') > 0);
assert.equal(meshMirrorAxisX(mesh), 0);
assert.equal(mirroredVertexIndex(mesh, 0, 0), 1);
assert.equal(mirroredVertexIndex(mesh, 2, 0), 3);

const controls = [
  {control_id: 'J:L', anchor: [-1, 0]},
  {control_id: 'J:R', anchor: [1, 0]},
  {control_id: 'J:C', anchor: [0, 1]},
];
assert.equal(mirroredControlID(controls, 'J:L', 0), 'J:R');
assert.equal(mirroredControlID(controls, 'J:C', 0), 'J:C');

console.log('STUDIO_WEIGHT_CORE_CONTRACT_PASS');
