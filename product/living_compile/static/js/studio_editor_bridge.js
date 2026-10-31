import {keyID} from './studio_timeline.js';

const $ = (selector) => document.querySelector(selector);
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

function boot() {
  const studio = window.RealSaSStudio;
  const dope = $('#dope');
  if (!studio?.selection || !studio?.timeline || !dope) {
    requestAnimationFrame(boot);
    return;
  }

  const S = studio.state;
  const documentModel = studio.document;
  const journal = studio.journal;
  const dirty = studio.dirty;
  const timeline = studio.timeline;
  let keyDrag = null;
  let saveTimer = null;

  studio.selection.addEventListener('change', () => {
    // app.js owns the Inspector renderer. Replaying the active mode action is a
    // cheap local bridge that refreshes it without coupling StudioSelection to
    // the legacy workspace module.
    document.querySelector('.mode.active')?.click();
  });

  dirty.addEventListener('change', () => {
    if (timeline.animationEdits !== S.ed.animation) timeline.bind(S.ed.animation);
  });

  for (const selector of ['#undoEdit', '#redoEdit']) {
    $(selector)?.addEventListener('click', () => queueMicrotask(() => {
      timeline.bind(S.ed.animation);
    }));
  }

  dope.addEventListener('pointerdown', onKeyPointerDown, true);
  dope.addEventListener('pointermove', onKeyPointerMove, true);
  dope.addEventListener('pointerup', onKeyPointerUp, true);
  dope.addEventListener('pointercancel', onKeyPointerCancel, true);

  function activeClip() {
    if (timeline.animationEdits !== S.ed.animation) timeline.bind(S.ed.animation);
    return timeline.clip(S.clip);
  }

  function onKeyPointerDown(event) {
    const clip = activeClip();
    if (!clip) return;
    const hit = hitKey(event, clip);
    if (!hit) return; // preserve animation_editor scrubbing on empty timeline space
    event.stopImmediatePropagation();
    event.preventDefault();

    timeline.selectKey(hit.controlID, hit.time, {
      add: event.shiftKey || event.ctrlKey || event.metaKey,
      toggle: event.ctrlKey || event.metaKey,
    });
    keyDrag = {
      pointerID: event.pointerId,
      startX: event.clientX,
      before: documentModel.snapshot(),
      selected: [...timeline.selectedKeys],
      changed: false,
    };
    dope.setPointerCapture(event.pointerId);
  }

  function onKeyPointerMove(event) {
    if (!keyDrag || event.pointerId !== keyDrag.pointerID) return;
    event.stopImmediatePropagation();
    event.preventDefault();
    const clip = activeClip();
    if (!clip) return;
    const bounds = dope.getBoundingClientRect();
    const delta = (event.clientX - keyDrag.startX) / Math.max(1, bounds.width) * clip.duration_seconds;
    if (Math.abs(delta) <= 1e-6) return;

    documentModel.replace(keyDrag.before);
    timeline.bind(S.ed.animation);
    timeline.selectedKeys = new Set(keyDrag.selected);
    timeline.moveSelectedKeys(delta, {snapToFrames: !event.altKey});
    dirty.markDirty();
    keyDrag.changed = true;
  }

  function onKeyPointerUp(event) {
    if (!keyDrag || event.pointerId !== keyDrag.pointerID) return;
    event.stopImmediatePropagation();
    event.preventDefault();
    try { dope.releasePointerCapture(event.pointerId); } catch {}
    if (keyDrag.changed) {
      journal.commit('Move animation keys', keyDrag.before, documentModel.snapshot());
      dirty.markDirty();
      scheduleSave();
    }
    keyDrag = null;
  }

  function onKeyPointerCancel(event) {
    if (!keyDrag || event.pointerId !== keyDrag.pointerID) return;
    event.stopImmediatePropagation();
    documentModel.replace(keyDrag.before);
    timeline.bind(S.ed.animation);
    timeline.selectedKeys = new Set(keyDrag.selected);
    keyDrag = null;
  }

  function hitKey(event, clip) {
    const bounds = dope.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / Math.max(1, bounds.width) * dope.width;
    const y = (event.clientY - bounds.top) / Math.max(1, bounds.height) * dope.height;
    const rowHeight = 24;
    let best = null;
    let distance = 13;
    (clip.tracks || []).forEach((track, row) => {
      const ky = 28 + row * rowHeight;
      for (const key of track.keys || []) {
        const kx = clamp(key.time_seconds / Math.max(1e-6, clip.duration_seconds), 0, 1) * dope.width;
        const d = Math.hypot(kx - x, ky - y);
        if (d < distance) {
          distance = d;
          best = {controlID: track.bone_id, time: key.time_seconds, id: keyID(track.bone_id, key.time_seconds)};
        }
      }
    });
    return best;
  }

  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      if (dirty.status === 'dirty') $('#saveLayer')?.click();
    }, 1200);
  }
}

boot();
