import assert from 'node:assert/strict';

import {
  baseMeshModel,
  edgeKey,
  meshEdges,
  modelArrays,
  nextUserFaceID,
  nextUserVertexID,
  replayMeshTopology,
  upsertMoveVertex,
} from '../../product/living_compile/static/js/studio_mesh_topology_core.js';

const mesh = {
  mesh_id: 'V0:BODY',
  vertices: [[0, 0], [2, 0], [2, 2], [0, 2]],
  triangles: [[0, 1, 2], [0, 2, 3]],
};

const base = baseMeshModel(mesh);
assert.equal(base.vertices.size, 4);
assert.equal(base.faces.size, 2);
assert.equal(meshEdges(base).length, 5);
assert.equal(nextUserVertexID(base), 'user:v1');
assert.equal(nextUserFaceID(base), 'user:f1');

const split = replayMeshTopology(mesh, [{
  operation: 'split_edge',
  mesh_id: 'V0:BODY',
  vertex_a_id: 'v:0',
  vertex_b_id: 'v:2',
  new_vertex_id: 'user:v1',
  t: 0.5,
}]);
assert.deepEqual(split.vertices.get('user:v1').position_xy, [1, 1]);
assert.equal(split.faces.size, 4);
assert.equal(meshEdges(split).find((row) => row.edge_id === edgeKey('v:0', 'user:v1')).face_ids.length, 2);

const addDelete = replayMeshTopology(mesh, [
  {operation: 'add_vertex', mesh_id: 'V0:BODY', vertex_id: 'user:v1', position_xy: [1, 3]},
  {operation: 'add_face', mesh_id: 'V0:BODY', face_id: 'user:f1', vertex_ids: ['v:2', 'v:3', 'user:v1']},
  {operation: 'delete_face', mesh_id: 'V0:BODY', face_id: 'user:f1'},
  {operation: 'delete_vertex', mesh_id: 'V0:BODY', vertex_id: 'user:v1'},
]);
assert.equal(addDelete.vertices.size, 4);
assert.equal(addDelete.faces.size, 2);

const moved = [];
upsertMoveVertex(moved, 'V0:BODY', 'v:1', [3, 0]);
upsertMoveVertex(moved, 'V0:BODY', 'v:1', [4, 0]);
assert.equal(moved.length, 1);
assert.deepEqual(moved[0].position_xy, [4, 0]);
const movedModel = replayMeshTopology(mesh, moved);
assert.deepEqual(movedModel.vertices.get('v:1').position_xy, [4, 0]);

const welded = replayMeshTopology(mesh, [{
  operation: 'weld_vertices',
  mesh_id: 'V0:BODY',
  keep_vertex_id: 'v:0',
  remove_vertex_id: 'v:1',
}]);
assert.equal(welded.vertices.size, 3);
assert.equal(welded.faces.size, 1);
assert.equal(modelArrays(welded).triangles.length, 1);

assert.throws(() => replayMeshTopology(mesh, [{
  operation: 'add_face',
  mesh_id: 'V0:BODY',
  face_id: 'user:bad',
  vertex_ids: ['v:0', 'v:1', 'v:0'],
}]), /degenerate face/);

assert.throws(() => replayMeshTopology(mesh, [
  {operation: 'add_vertex', mesh_id: 'V0:BODY', vertex_id: 'user:v1', position_xy: [1, -1]},
  {operation: 'add_vertex', mesh_id: 'V0:BODY', vertex_id: 'user:v2', position_xy: [1, 1]},
  {operation: 'add_face', mesh_id: 'V0:BODY', face_id: 'user:f1', vertex_ids: ['v:0', 'v:1', 'user:v1']},
  {operation: 'add_face', mesh_id: 'V0:BODY', face_id: 'user:f2', vertex_ids: ['v:0', 'v:1', 'user:v2']},
]), /non-manifold edge/);

console.log('STUDIO_MESH_TOPOLOGY_CORE_CONTRACT_PASS');
