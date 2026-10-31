import {keyID, trackControlID} from './studio_timeline.js';

const CURVES = new Set(['step', 'linear', 'bezier']);

function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function pair(value, fallback) {
  if (!Array.isArray(value) || value.length !== 2) return [...fallback];
  return [finite(value[0], fallback[0]), finite(value[1], fallback[1])];
}

export function capturePoseSnapshot(timeline, clipID, timeSeconds) {
  const clip = timeline?.clip?.(clipID);
  if (!clip) return [];
  const time = finite(timeSeconds, 0);
  const rows = [];
  for (const track of clip.tracks || []) {
    const controlID = trackControlID(track);
    if (!controlID) continue;
    const sampled = timeline.sample(controlID, time);
    if (!sampled) continue;
    rows.push({
      control_id: controlID,
      translation_xy: pair(sampled.translation_xy, [0, 0]),
      rotation_degrees: finite(sampled.rotation_degrees, 0),
      scale_xy: pair(sampled.scale_xy, [1, 1]),
    });
  }
  rows.sort((a, b) => a.control_id.localeCompare(b.control_id));
  return rows;
}

export function pastePoseSnapshot(timeline, snapshot, atSeconds, {curve = 'linear'} = {}) {
  const clip = timeline?.clip?.();
  if (!clip) throw new RangeError('active user clip required');
  if (!Array.isArray(snapshot) || !snapshot.length) return 0;
  const normalizedCurve = CURVES.has(curve) ? curve : 'linear';
  const time = Math.max(0, Math.min(finite(atSeconds, 0), Number(clip.duration_seconds || 0)));
  timeline.selectedKeys.clear();
  let pasted = 0;
  for (const row of snapshot) {
    const controlID = String(row?.control_id || '').trim();
    if (!controlID) continue;
    const track = timeline.ensureTrack(controlID);
    const key = {
      time_seconds: time,
      translation_xy: pair(row.translation_xy, [0, 0]),
      rotation_degrees: finite(row.rotation_degrees, 0),
      scale_xy: pair(row.scale_xy, [1, 1]),
      curve: normalizedCurve,
    };
    const index = track.keys.findIndex((candidate) => Math.abs(candidate.time_seconds - time) <= 1e-6);
    if (index >= 0) track.keys[index] = clone(key);
    else track.keys.push(clone(key));
    timeline.selectedKeys.add(keyID(controlID, time));
    pasted += 1;
  }
  timeline.sortAll();
  if (pasted) timeline.emit();
  return pasted;
}

export function ghostSampleTimes(timeSeconds, durationSeconds, fps, {loop = false, frameOffset = 1} = {}) {
  const duration = finite(durationSeconds, 0);
  const rate = finite(fps, 0);
  const offset = Math.max(1, Math.round(finite(frameOffset, 1)));
  if (!(duration > 0) || !(rate > 0)) return {previous: null, next: null};
  const time = Math.max(0, Math.min(finite(timeSeconds, 0), duration));
  const delta = offset / rate;
  return {
    previous: shiftedGhostTime(time - delta, duration, Boolean(loop)),
    next: shiftedGhostTime(time + delta, duration, Boolean(loop)),
  };
}

function shiftedGhostTime(value, duration, loop) {
  if (!loop) {
    if (value < 0 || value > duration) return null;
    return value;
  }
  if (duration <= 0) return null;
  let wrapped = value % duration;
  if (wrapped < 0) wrapped += duration;
  return wrapped;
}
