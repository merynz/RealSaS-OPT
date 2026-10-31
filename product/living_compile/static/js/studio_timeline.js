function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

const CURVES = new Set(['step', 'linear', 'bezier']);

export class StudioTimeline extends EventTarget {
  constructor(animationEdits = []) {
    super();
    this.animationEdits = animationEdits;
    this.clipID = animationEdits[0]?.clip_id || null;
    this.playhead = 0;
    this.loopRange = null;
    this.selectedKeys = new Set();
    this.clipboard = [];
  }

  bind(animationEdits) {
    this.animationEdits = animationEdits;
    if (!this.animationEdits.some((clip) => clip.clip_id === this.clipID)) {
      this.clipID = this.animationEdits[0]?.clip_id || null;
      this.playhead = 0;
      this.selectedKeys.clear();
    }
    this.emit();
  }

  clips() {
    return this.animationEdits;
  }

  clip(id = this.clipID) {
    return this.animationEdits.find((item) => item.clip_id === id) || null;
  }

  selectClip(id) {
    if (id !== null && !this.clip(id)) throw new RangeError(`unknown clip: ${id}`);
    this.clipID = id;
    this.playhead = 0;
    this.selectedKeys.clear();
    this.emit();
  }

  createClip({clipID, displayName, durationSeconds = 1, loop = true, fps = 30} = {}) {
    const id = String(clipID || this.nextClipID()).trim();
    if (!id) throw new TypeError('clip id required');
    if (this.clip(id)) throw new RangeError(`clip already exists: ${id}`);
    const clip = {
      clip_id: id,
      display_name: String(displayName || id),
      duration_seconds: positive(durationSeconds, 'duration_seconds'),
      loop: Boolean(loop),
      fps: positive(fps, 'fps'),
      tracks: [],
    };
    this.animationEdits.push(clip);
    this.clipID = id;
    this.playhead = 0;
    this.selectedKeys.clear();
    this.emit();
    return clip;
  }

  duplicateClip(sourceID = this.clipID, newID = null) {
    const source = this.clip(sourceID);
    if (!source) throw new RangeError('source clip required');
    const id = String(newID || this.nextClipID(`${source.clip_id}_COPY`));
    if (this.clip(id)) throw new RangeError(`clip already exists: ${id}`);
    const duplicate = clone(source);
    duplicate.clip_id = id;
    duplicate.display_name = `${source.display_name || source.clip_id} Copy`;
    this.animationEdits.push(duplicate);
    this.clipID = id;
    this.playhead = 0;
    this.selectedKeys.clear();
    this.emit();
    return duplicate;
  }

  deleteClip(id = this.clipID) {
    const index = this.animationEdits.findIndex((clip) => clip.clip_id === id);
    if (index < 0) return false;
    this.animationEdits.splice(index, 1);
    this.clipID = this.animationEdits[Math.min(index, this.animationEdits.length - 1)]?.clip_id || null;
    this.playhead = 0;
    this.selectedKeys.clear();
    this.emit();
    return true;
  }

  renameClip(name, id = this.clipID) {
    const clip = requiredClip(this.clip(id));
    clip.display_name = String(name || clip.clip_id).trim() || clip.clip_id;
    this.emit();
  }

  setDuration(seconds, {scaleKeys = false} = {}) {
    const clip = requiredClip(this.clip());
    const next = positive(seconds, 'duration_seconds');
    const previous = positive(clip.duration_seconds || 1, 'duration_seconds');
    if (scaleKeys && previous > 0) {
      const factor = next / previous;
      for (const track of clip.tracks || []) {
        for (const key of track.keys || []) key.time_seconds = clamp(key.time_seconds * factor, 0, next);
      }
    } else {
      for (const track of clip.tracks || []) {
        track.keys = (track.keys || []).filter((key) => key.time_seconds <= next + 1e-9);
      }
    }
    clip.duration_seconds = next;
    this.playhead = clamp(this.playhead, 0, next);
    this.sortAll();
    this.emit();
  }

  setLoop(loop) {
    requiredClip(this.clip()).loop = Boolean(loop);
    this.emit();
  }

  setFPS(fps) {
    requiredClip(this.clip()).fps = positive(fps, 'fps');
    this.emit();
  }

  setPlayhead(seconds, {snapToFrame = false} = {}) {
    const clip = this.clip();
    const duration = clip?.duration_seconds || 0;
    let time = clamp(Number(seconds) || 0, 0, duration);
    if (snapToFrame && clip) {
      const fps = positive(clip.fps || 30, 'fps');
      time = Math.round(time * fps) / fps;
    }
    this.playhead = time;
    this.emit();
    return time;
  }

  ensureTrack(controlID, clipID = this.clipID) {
    const clip = requiredClip(this.clip(clipID));
    const id = String(controlID || '').trim();
    if (!id) throw new TypeError('control id required');
    let track = (clip.tracks || []).find((item) => item.bone_id === id);
    if (!track) {
      track = {bone_id: id, keys: []};
      clip.tracks = clip.tracks || [];
      clip.tracks.push(track);
    }
    return track;
  }

  upsertTransformKey(controlID, transform, {time = this.playhead, curve = 'linear'} = {}) {
    const clip = requiredClip(this.clip());
    const track = this.ensureTrack(controlID);
    const normalizedCurve = CURVES.has(curve) ? curve : 'linear';
    const t = clamp(Number(time) || 0, 0, clip.duration_seconds || 0);
    const key = {
      time_seconds: t,
      translation_xy: finitePair(transform?.translation_xy, [0, 0]),
      rotation_degrees: finite(transform?.rotation_degrees, 0),
      scale_xy: finitePair(transform?.scale_xy, [1, 1]),
      curve: normalizedCurve,
      bezier: normalizedCurve === 'bezier' ? normalizeBezier(transform?.bezier) : undefined,
    };
    const index = track.keys.findIndex((item) => Math.abs(item.time_seconds - t) <= 1e-6);
    if (index >= 0) track.keys[index] = key;
    else track.keys.push(key);
    track.keys.sort((a, b) => a.time_seconds - b.time_seconds);
    this.selectedKeys.clear();
    this.selectedKeys.add(keyID(controlID, t));
    this.emit();
    return key;
  }

  deleteKeys(keys = [...this.selectedKeys]) {
    const clip = this.clip();
    if (!clip) return 0;
    const wanted = new Set(keys);
    let deleted = 0;
    for (const track of clip.tracks || []) {
      const before = track.keys.length;
      track.keys = track.keys.filter((key) => !wanted.has(keyID(track.bone_id, key.time_seconds)));
      deleted += before - track.keys.length;
    }
    clip.tracks = (clip.tracks || []).filter((track) => track.keys.length > 0);
    if (deleted) {
      this.selectedKeys.clear();
      this.emit();
    }
    return deleted;
  }

  selectKey(controlID, time, {add = false, toggle = false} = {}) {
    const id = keyID(controlID, time);
    if (!add && !toggle) this.selectedKeys.clear();
    if (toggle && this.selectedKeys.has(id)) this.selectedKeys.delete(id);
    else this.selectedKeys.add(id);
    this.emit();
  }

  clearKeySelection() {
    if (!this.selectedKeys.size) return;
    this.selectedKeys.clear();
    this.emit();
  }

  selectedKeyObjects() {
    const clip = this.clip();
    if (!clip) return [];
    const rows = [];
    for (const track of clip.tracks || []) {
      for (const key of track.keys || []) {
        if (this.selectedKeys.has(keyID(track.bone_id, key.time_seconds))) rows.push({track, key});
      }
    }
    return rows;
  }

  moveSelectedKeys(deltaSeconds, {snapToFrames = true} = {}) {
    const clip = requiredClip(this.clip());
    const rows = this.selectedKeyObjects();
    if (!rows.length) return 0;
    const fps = positive(clip.fps || 30, 'fps');
    const original = rows.map(({track, key}) => ({track, key, oldID: keyID(track.bone_id, key.time_seconds)}));
    this.selectedKeys.clear();
    for (const row of original) {
      let next = clamp(row.key.time_seconds + Number(deltaSeconds || 0), 0, clip.duration_seconds);
      if (snapToFrames) next = Math.round(next * fps) / fps;
      row.key.time_seconds = next;
      this.selectedKeys.add(keyID(row.track.bone_id, next));
    }
    this.sortAll();
    this.emit();
    return rows.length;
  }

  scaleSelectedKeys(factor, pivot = this.playhead) {
    const clip = requiredClip(this.clip());
    const rows = this.selectedKeyObjects();
    if (!rows.length) return 0;
    const amount = positive(factor, 'scale factor');
    this.selectedKeys.clear();
    for (const {track, key} of rows) {
      key.time_seconds = clamp(pivot + (key.time_seconds - pivot) * amount, 0, clip.duration_seconds);
      this.selectedKeys.add(keyID(track.bone_id, key.time_seconds));
    }
    this.sortAll();
    this.emit();
    return rows.length;
  }

  setSelectedCurve(curve) {
    if (!CURVES.has(curve)) throw new RangeError(`unsupported curve: ${curve}`);
    const rows = this.selectedKeyObjects();
    for (const {key} of rows) {
      key.curve = curve;
      if (curve === 'bezier') key.bezier = normalizeBezier(key.bezier);
      else delete key.bezier;
    }
    if (rows.length) this.emit();
    return rows.length;
  }

  setSelectedBezier(bezier) {
    const value = normalizeBezier(bezier);
    const rows = this.selectedKeyObjects();
    for (const {key} of rows) {
      key.curve = 'bezier';
      key.bezier = [...value];
    }
    if (rows.length) this.emit();
    return rows.length;
  }

  copySelectedKeys() {
    const rows = this.selectedKeyObjects();
    if (!rows.length) return 0;
    const origin = Math.min(...rows.map(({key}) => key.time_seconds));
    this.clipboard = rows.map(({track, key}) => ({
      bone_id: track.bone_id,
      offset_seconds: key.time_seconds - origin,
      key: clone(key),
    }));
    return this.clipboard.length;
  }

  pasteKeys(at = this.playhead) {
    const clip = requiredClip(this.clip());
    if (!this.clipboard.length) return 0;
    this.selectedKeys.clear();
    for (const row of this.clipboard) {
      const track = this.ensureTrack(row.bone_id);
      const key = clone(row.key);
      key.time_seconds = clamp(Number(at || 0) + row.offset_seconds, 0, clip.duration_seconds);
      const index = track.keys.findIndex((candidate) => Math.abs(candidate.time_seconds - key.time_seconds) <= 1e-6);
      if (index >= 0) track.keys[index] = key;
      else track.keys.push(key);
      this.selectedKeys.add(keyID(row.bone_id, key.time_seconds));
    }
    this.sortAll();
    this.emit();
    return this.clipboard.length;
  }

  sample(controlID, time = this.playhead) {
    const clip = this.clip();
    const track = clip?.tracks?.find((item) => item.bone_id === controlID);
    if (!track?.keys?.length) return null;
    const keys = track.keys;
    let a = keys[0];
    let b = keys[keys.length - 1];
    for (const key of keys) {
      if (key.time_seconds <= time) a = key;
      if (key.time_seconds >= time) { b = key; break; }
    }
    if (a === b || Math.abs(b.time_seconds - a.time_seconds) <= 1e-9) return clone(a);
    const q = clamp((time - a.time_seconds) / (b.time_seconds - a.time_seconds), 0, 1);
    const t = curveFactor(a, q);
    return {
      time_seconds: time,
      translation_xy: mixPair(a.translation_xy, b.translation_xy, t),
      rotation_degrees: mix(finite(a.rotation_degrees, 0), finite(b.rotation_degrees, 0), t),
      scale_xy: mixPair(a.scale_xy || [1, 1], b.scale_xy || [1, 1], t),
      curve: a.curve || 'linear',
    };
  }

  sortAll() {
    const clip = this.clip();
    if (!clip) return;
    for (const track of clip.tracks || []) track.keys.sort((a, b) => a.time_seconds - b.time_seconds);
  }

  nextClipID(prefix = 'USER') {
    let index = 1;
    let id = `${prefix}_${index}`;
    while (this.clip(id)) id = `${prefix}_${++index}`;
    return id;
  }

  snapshot() {
    return {
      clip_id: this.clipID,
      playhead: this.playhead,
      selected_keys: [...this.selectedKeys],
      clipboard_size: this.clipboard.length,
    };
  }

  emit() {
    this.dispatchEvent(new CustomEvent('change', {detail: this.snapshot()}));
  }
}

export function keyID(controlID, time) {
  return `${controlID}@${Number(time).toFixed(6)}`;
}

function requiredClip(clip) {
  if (!clip) throw new RangeError('active user clip required');
  return clip;
}

function positive(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) throw new RangeError(`${label} must be > 0`);
  return number;
}

function finite(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function finitePair(value, fallback) {
  if (!Array.isArray(value) || value.length !== 2) return [...fallback];
  return [finite(value[0], fallback[0]), finite(value[1], fallback[1])];
}

function normalizeBezier(value) {
  if (!Array.isArray(value) || value.length !== 4) return [0.25, 0.1, 0.25, 1];
  return value.map((item, index) => clamp(finite(item, [0.25, 0.1, 0.25, 1][index]), 0, 1));
}

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function mix(a, b, t) {
  return a + (b - a) * t;
}

function mixPair(a, b, t) {
  const aa = finitePair(a, [0, 0]);
  const bb = finitePair(b, [0, 0]);
  return [mix(aa[0], bb[0], t), mix(aa[1], bb[1], t)];
}

function curveFactor(key, q) {
  if (key.curve === 'step') return 0;
  if (key.curve !== 'bezier') return q;
  const [x1, y1, x2, y2] = normalizeBezier(key.bezier);
  let lo = 0;
  let hi = 1;
  let u = q;
  for (let i = 0; i < 10; i += 1) {
    u = (lo + hi) / 2;
    const x = cubicBezier(u, 0, x1, x2, 1);
    if (x < q) lo = u; else hi = u;
  }
  return cubicBezier(u, 0, y1, y2, 1);
}

function cubicBezier(t, p0, p1, p2, p3) {
  const u = 1 - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}
