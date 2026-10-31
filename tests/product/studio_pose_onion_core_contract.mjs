import assert from 'node:assert/strict';

import {StudioTimeline} from '../../product/living_compile/static/js/studio_timeline.js';
import {
  capturePoseSnapshot,
  ghostSampleTimes,
  pastePoseSnapshot,
} from '../../product/living_compile/static/js/studio_pose_onion_core.js';

const edits = [{
  clip_id: 'USER_SOURCE',
  display_name: 'Source',
  duration_seconds: 1,
  fps: 30,
  loop: true,
  tracks: [
    {
      control_id: 'C1',
      keys: [
        {time_seconds: 0, translation_xy: [0, 0], rotation_degrees: 0, scale_xy: [1, 1], curve: 'linear'},
        {time_seconds: 1, translation_xy: [2, 4], rotation_degrees: 90, scale_xy: [2, 0.5], curve: 'linear'},
      ],
    },
    {
      control_id: 'C0',
      keys: [
        {time_seconds: 0, translation_xy: [1, 1], rotation_degrees: -10, scale_xy: [1, 1], curve: 'linear'},
        {time_seconds: 1, translation_xy: [3, 1], rotation_degrees: 10, scale_xy: [1, 1], curve: 'linear'},
      ],
    },
  ],
}];

const timeline = new StudioTimeline(edits);
timeline.selectClip('USER_SOURCE');
const pose = capturePoseSnapshot(timeline, 'USER_SOURCE', 0.5);
assert.equal(pose.length, 2);
assert.deepEqual(pose.map((row) => row.control_id), ['C0', 'C1']);
assert.deepEqual(pose[0].translation_xy, [2, 1]);
assert.equal(pose[0].rotation_degrees, 0);
assert.deepEqual(pose[1].translation_xy, [1, 2]);
assert.equal(pose[1].rotation_degrees, 45);
assert.deepEqual(pose[1].scale_xy, [1.5, 0.75]);

const target = timeline.createClip({clipID: 'USER_TARGET', durationSeconds: 2, fps: 30});
assert.equal(target.clip_id, 'USER_TARGET');
const pasted = pastePoseSnapshot(timeline, pose, 0.25, {curve: 'linear'});
assert.equal(pasted, 2);
assert.equal(target.tracks.length, 2);
for (const track of target.tracks) {
  assert.equal(track.keys.length, 1);
  assert.equal(track.keys[0].time_seconds, 0.25);
  assert.equal(track.keys[0].curve, 'linear');
}
assert.equal(timeline.selectedKeys.size, 2);
assert.deepEqual(capturePoseSnapshot(timeline, 'USER_TARGET', 0.25), pose);

// Re-pasting at the same time replaces keys instead of manufacturing duplicates.
pose[0].rotation_degrees = 25;
assert.equal(pastePoseSnapshot(timeline, pose, 0.25, {curve: 'bezier'}), 2);
const c0 = target.tracks.find((track) => track.control_id === 'C0');
assert.equal(c0.keys.length, 1);
assert.equal(c0.keys[0].rotation_degrees, 25);
assert.equal(c0.keys[0].curve, 'bezier');

assert.deepEqual(
  ghostSampleTimes(0, 1, 30, {loop: false, frameOffset: 1}),
  {previous: null, next: 1 / 30},
);
assert.deepEqual(
  ghostSampleTimes(0, 1, 10, {loop: true, frameOffset: 1}),
  {previous: 0.9, next: 0.1},
);
assert.deepEqual(
  ghostSampleTimes(0.5, 1, 10, {loop: false, frameOffset: 2}),
  {previous: 0.3, next: 0.7},
);
assert.deepEqual(ghostSampleTimes(0.5, 0, 30), {previous: null, next: null});
assert.deepEqual(ghostSampleTimes(0.5, 1, 0), {previous: null, next: null});

console.log('STUDIO_POSE_ONION_CORE_CONTRACT_PASS');
