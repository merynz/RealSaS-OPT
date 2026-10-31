import assert from 'node:assert/strict';

import {StudioTimelineView} from '../../product/living_compile/static/js/studio_timeline_view.js';

const view = new StudioTimelineView(10);
assert.equal(view.span(), 10);
assert.equal(view.start, 0);
assert.equal(view.end(), 10);

view.setZoom(2, 5);
assert.equal(view.zoom, 2);
assert.equal(view.span(), 5);
assert.equal(view.start, 2.5);
assert.equal(view.end(), 7.5);
assert.equal(view.timeToUnit(2.5), 0);
assert.equal(view.timeToUnit(7.5), 1);
assert.equal(view.unitToTime(0.5), 5);

view.panBy(2);
assert.equal(view.start, 4.5);
assert.equal(view.end(), 9.5);
view.panBy(100);
assert.equal(view.start, 5);
assert.equal(view.end(), 10);
view.panBy(-100);
assert.equal(view.start, 0);
assert.equal(view.end(), 5);

view.setLoopRange(1.25, 3.75);
assert.deepEqual(view.loopBounds(), [1.25, 3.75]);
view.setLoopIn(2);
assert.deepEqual(view.loopBounds(), [2, 3.75]);
view.setLoopOut(4);
assert.deepEqual(view.loopBounds(), [2, 4]);
view.clearLoopRange();
assert.deepEqual(view.loopBounds(), [0, 10]);

view.bindDuration(2);
assert.equal(view.duration, 2);
assert.ok(view.start <= view.maxStart());
view.reset();
assert.equal(view.zoom, 1);
assert.equal(view.start, 0);
assert.equal(view.end(), 2);

assert.throws(() => view.setLoopRange(1.5, 1.5), /loop range end/);

console.log('STUDIO_TIMELINE_VIEW_CONTRACT_PASS');
