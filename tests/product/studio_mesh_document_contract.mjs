import assert from 'node:assert/strict';

import {
  StudioDocument,
  authoringLayerToEdits,
  editsToAuthoringRequest,
} from '../../product/living_compile/static/js/studio_document.js';

const layer = {
  schema_version: 'RealSaS.UserPuppetEditLayer.v3',
  mesh_topology_edits: [{
    operation: 'split_edge',
    mesh_id: 'V0:BODY',
    vertex_a_id: 'v:0',
    vertex_b_id: 'v:1',
    new_vertex_id: 'user:v1',
    t: 0.5,
  }],
};

const edits = authoringLayerToEdits(layer);
assert.equal(edits.meshTopology.length, 1);
assert.equal(edits.meshTopology[0].new_vertex_id, 'user:v1');

const documentModel = new StudioDocument();
documentModel.loadAuthoringLayer(layer);
assert.equal(documentModel.differsFromPersisted(), false);
documentModel.edits.meshTopology.push({
  operation: 'move_vertex',
  mesh_id: 'V0:BODY',
  vertex_id: 'user:v1',
  position_xy: [12, 13],
});
assert.equal(documentModel.differsFromPersisted(), true);

const request = editsToAuthoringRequest('bundle', 'V0', documentModel.edits);
assert.equal(request.mesh_topology_edits.length, 2);
assert.equal(request.mesh_topology_edits[1].vertex_id, 'user:v1');
request.mesh_topology_edits[1].position_xy[0] = 999;
assert.equal(documentModel.edits.meshTopology[1].position_xy[0], 12);

console.log('STUDIO_MESH_DOCUMENT_CONTRACT_PASS');
