import assert from 'node:assert/strict';

import {
  StudioDocument,
  authoringLayerToEdits,
  editsToAuthoringRequest,
} from '../../product/living_compile/static/js/studio_document.js';
import {StudioOperationJournal} from '../../product/living_compile/static/js/studio_operation_journal.js';
import {StudioSelection} from '../../product/living_compile/static/js/studio_selection.js';
import {StudioToolContext} from '../../product/living_compile/static/js/studio_tool_context.js';
import {StudioTimeline, keyID} from '../../product/living_compile/static/js/studio_timeline.js';

const persistedLayer = {
  schema_version: 'RealSaS.UserPuppetEditLayer.v3',
  rig_pose_edits: [{control_id: 'ROOT', rotation_degrees: 0, translation_xy: [2, 3]}],
  mesh_vertex_edits: [],
  weight_edits: [],
  binding_edits: [],
  rig_topology_edits: [],
  ik_constraints: [],
  animation_edits: [{clip_id: 'USER_1', display_name: 'User 1', duration_seconds: 1, fps: 30, loop: true, tracks: []}],
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
request.edits[0].translation_xy[0] = 1000;
assert.equal(documentModel.edits.rig[0].translation_xy[0], 9);

// StudioSelection preserves selections per view while exposing only the active
// view to viewport tools. Multi-select and primary selection are deterministic.
const selection = new StudioSelection('V0');
selection.add({kind: 'control', view_id: 'V0', control_id: 'ROOT'});
selection.add({kind: 'control', view_id: 'V0', control_id: 'HAND_R'}, {primary: false});
selection.add({kind: 'vertex', view_id: 'V1', mesh_id: 'BODY', vertex_index: 7}, {primary: false});
assert.equal(selection.size(), 2);
assert.equal(selection.primary().control_id, 'ROOT');
selection.toggle({kind: 'control', view_id: 'V0', control_id: 'HAND_R'});
assert.equal(selection.size(), 1);
selection.setView('V1');
assert.equal(selection.size(), 1);
assert.equal(selection.primary().vertex_index, 7);
selection.setView('V0');
assert.equal(selection.size(), 1);
assert.equal(selection.primary().control_id, 'ROOT');

// StudioToolContext owns editor interaction semantics independent of the
// document. Snapping can be bypassed by a modifier without changing settings.
const tools = new StudioToolContext();
tools.setTool('translate');
tools.configureSnapping({translate: 2, rotateDegrees: 15, scale: 0.25});
assert.deepEqual(tools.snapTranslation([3.1, -2.9]), [4, -2]);
assert.deepEqual(tools.snapTranslation([3.1, -2.9], true), [3.1, -2.9]);
assert.equal(Math.round(tools.snapAngleRadians(Math.PI / 5) * 180 / Math.PI), 30);
tools.setCustomPivot([10, 20]);
assert.equal(tools.pivotMode, 'custom');
assert.deepEqual(tools.customPivot, [10, 20]);

// StudioTimeline is a real clip-authoring core: clips/tracks/keys are editable,
// key selections are movable/copyable, interpolation is authored, and sampling
// produces a transform preview without mutating the canonical ProductRevision.
const animation = [];
const timeline = new StudioTimeline(animation);
const clip = timeline.createClip({clipID: 'WALK', displayName: 'Walk', durationSeconds: 2, loop: true, fps: 30});
assert.equal(clip.clip_id, 'WALK');
assert.equal(animation.length, 1);

timeline.setPlayhead(0);
timeline.upsertTransformKey('ROOT', {translation_xy: [0, 0], rotation_degrees: 0, scale_xy: [1, 1]}, {curve: 'linear'});
timeline.setPlayhead(1);
timeline.upsertTransformKey('ROOT', {translation_xy: [10, 4], rotation_degrees: 90, scale_xy: [1.2, 0.8]}, {curve: 'bezier'});
assert.equal(clip.tracks.length, 1);
assert.equal(clip.tracks[0].keys.length, 2);

const mid = timeline.sample('ROOT', 0.5);
assert.ok(mid.translation_xy[0] > 0 && mid.translation_xy[0] < 10);
assert.ok(mid.rotation_degrees > 0 && mid.rotation_degrees < 90);

timeline.selectKey('ROOT', 1);
assert.equal(timeline.selectedKeys.has(keyID('ROOT', 1)), true);
assert.equal(timeline.copySelectedKeys(), 1);
timeline.setPlayhead(1.5);
assert.equal(timeline.pasteKeys(), 1);
assert.equal(clip.tracks[0].keys.length, 3);
assert.equal(timeline.selectedKeys.has(keyID('ROOT', 1.5)), true);

timeline.moveSelectedKeys(-0.5);
assert.equal(timeline.selectedKeys.has(keyID('ROOT', 1)), true);
timeline.setSelectedCurve('step');
const moved = clip.tracks[0].keys.find((item) => Math.abs(item.time_seconds - 1) <= 1e-6);
assert.equal(moved.curve, 'step');

timeline.duplicateClip('WALK', 'WALK_COPY');
assert.equal(animation.length, 2);
assert.equal(timeline.clipID, 'WALK_COPY');
timeline.setDuration(4, {scaleKeys: true});
assert.equal(timeline.clip().duration_seconds, 4);
assert.equal(timeline.clip().tracks[0].keys.at(-1).time_seconds, 3);

console.log('STUDIO_DOCUMENT_CONTRACT_PASS');
