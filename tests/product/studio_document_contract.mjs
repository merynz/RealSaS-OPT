import assert from 'node:assert/strict';

import {
  StudioDocument,
  authoringLayerToEdits,
  editsToAuthoringRequest,
} from '../../product/living_compile/static/js/studio_document.js';
import {StudioOperationJournal} from '../../product/living_compile/static/js/studio_operation_journal.js';

const persistedLayer = {
  schema_version: 'RealSaS.UserPuppetEditLayer.v3',
  rig_pose_edits: [{control_id: 'ROOT', rotation_degrees: 0, translation_xy: [2, 3]}],
  mesh_vertex_edits: [],
  weight_edits: [],
  binding_edits: [],
  rig_topology_edits: [],
  ik_constraints: [],
  animation_edits: [{clip_id: 'USER_1', duration_seconds: 1, loop: true, tracks: []}],
};

const mapped = authoringLayerToEdits(persistedLayer);
assert.equal(mapped.rig[0].control_id, 'ROOT');
assert.equal(mapped.animation[0].clip_id, 'USER_1');

const documentModel = new StudioDocument();
documentModel.loadAuthoringLayer(persistedLayer);
assert.equal(documentModel.differsFromPersisted(), false);

const journal = new StudioOperationJournal(10);
const before = documentModel.snapshot();
documentModel.edits.rig[0].translation_xy[0] = 9;
const after = documentModel.snapshot();
assert.equal(journal.commit('Move ROOT', before, after), true);
assert.equal(documentModel.differsFromPersisted(), true);
assert.equal(journal.canUndo(), true);

const undone = journal.undo(documentModel);
assert.equal(undone.label, 'Move ROOT');
assert.equal(documentModel.edits.rig[0].translation_xy[0], 2);
assert.equal(documentModel.differsFromPersisted(), false);
assert.equal(journal.canRedo(), true);

const redone = journal.redo(documentModel);
assert.equal(redone.label, 'Move ROOT');
assert.equal(documentModel.edits.rig[0].translation_xy[0], 9);
assert.equal(documentModel.differsFromPersisted(), true);

const request = editsToAuthoringRequest('public/cmp_x/hash', 'V0', documentModel.edits);
assert.equal(request.view_id, 'V0');
assert.equal(request.edits[0].translation_xy[0], 9);
assert.equal(request.animation_edits[0].clip_id, 'USER_1');

// Serialization boundaries must clone state. Mutating a request may never
// mutate the active editor document behind the user's back.
request.edits[0].translation_xy[0] = 1000;
assert.equal(documentModel.edits.rig[0].translation_xy[0], 9);

console.log('STUDIO_DOCUMENT_CONTRACT_PASS');
