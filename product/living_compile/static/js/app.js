import {StudioDocument} from './studio_document.js';
import {StudioOperationJournal} from './studio_operation_journal.js';
import {StudioDirtyState} from './studio_dirty_state.js';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const enc = encodeURIComponent;
const LAB = ['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW'];

const documentModel = new StudioDocument();
const journal = new StudioOperationJournal(120);
const dirty = new StudioDirtyState();

const S = {
  bundle: '',
  scene: null,
  view: 0,
  mode: 'rig',
  sel: {control: null, mesh: null, vertex: -1},
  img: null,
  imgUrl: '',
  frame: null,
  clips: [],
  clip: null,
  t: 0,
  play: false,
  last: 0,
  cam: {s: 1, x: 0, y: 0},
  drag: null,
  previewPose: new Map(),
  autosaveTimer: null,
  saveInFlight: false,
  get ed() {
    return documentModel.edits;
  },
};

window.RealSaSStudio = {state: S, document: documentModel, journal, dirty};

function toast(message, bad = false) {
  const element = $('#toast');
  element.textContent = message;
  element.className = `toast show${bad ? ' error' : ''}`;
  clearTimeout(toast.handle);
  toast.handle = setTimeout(() => { element.className = 'toast'; }, 2400);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get('content-type') || '';
  const value = contentType.includes('json') ? await response.json() : await response.blob();
  if (!response.ok) {
    const error = new Error(value?.message || value?.error || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return value;
}

const V = () => `V${S.view}`;
const meshes = () => S.scene?.puppet?.meshes?.filter((item) => item.view_id === V()) || [];
const controls = () => S.scene?.puppet?.rig?.controls?.filter((item) => item.view_id === V()) || [];
const bindings = () => S.scene?.binding?.bindings?.filter((item) => item.view_id === V()) || [];
const mesh = () => meshes().find((item) => item.mesh_id === S.sel.mesh) || meshes()[0];
const control = () => controls().find((item) => item.control_id === S.sel.control);

function rigValue(id) {
  return S.ed.rig.find((item) => item.control_id === id) || {
    control_id: id,
    rotation_degrees: 0,
    translation_xy: [0, 0],
  };
}

function rigEdit(id) {
  let edit = S.ed.rig.find((item) => item.control_id === id);
  if (!edit) {
    edit = {control_id: id, rotation_degrees: 0, translation_xy: [0, 0]};
    S.ed.rig.push(edit);
  }
  return edit;
}

function md(meshID, index) {
  return S.ed.mesh.find((item) => item.mesh_id === meshID && item.vertex_index === index)?.translation_xy || [0, 0];
}

function pmesh(item, index, animated = true) {
  const point = animated && S.frame?.frame?.mesh_vertices_by_id?.[item.mesh_id]?.[index] || item.vertices[index];
  const delta = md(item.mesh_id, index);
  return [point[0] + delta[0], point[1] + delta[1]];
}

function pcontrol(item) {
  const preview = S.previewPose.get(item.control_id);
  const edit = preview || rigValue(item.control_id);
  return [
    item.anchor[0] + (edit.translation_xy?.[0] || 0),
    item.anchor[1] + (edit.translation_xy?.[1] || 0),
  ];
}

const sc = (point) => [point[0] * S.cam.s + S.cam.x, point[1] * S.cam.s + S.cam.y];
const wc = (point) => [(point[0] - S.cam.x) / S.cam.s, (point[1] - S.cam.y) / S.cam.s];

function fit() {
  const points = [];
  meshes().forEach((item) => item.vertices.forEach((_, index) => points.push(pmesh(item, index, false))));
  controls().forEach((item) => points.push(pcontrol(item)));
  const canvas = $('#viewport');
  if (!points.length) return;
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  S.cam.s = Math.min(
    (canvas.width - 100) / Math.max(1, x1 - x0),
    (canvas.height - 100) / Math.max(1, y1 - y0),
  );
  S.cam.x = canvas.width / 2 - ((x0 + x1) / 2) * S.cam.s;
  S.cam.y = canvas.height / 2 - ((y0 + y1) / 2) * S.cam.s;
}

async function image() {
  if (S.imgUrl) URL.revokeObjectURL(S.imgUrl);
  S.img = null;
  const source = S.scene?.index?.source_views?.find((item) => item.view_index === S.view);
  const relative = source?.assets?.clean_cutout?.path || source?.assets?.raw_image?.path;
  if (!relative) return;
  try {
    const blob = await api(`/api/blob?bundle=${enc(S.bundle)}&path=${enc(relative)}`);
    S.imgUrl = URL.createObjectURL(blob);
    S.img = new Image();
    await new Promise((resolve, reject) => {
      S.img.onload = resolve;
      S.img.onerror = reject;
      S.img.src = S.imgUrl;
    });
  } catch (error) {
    toast(error.message, true);
  }
}

function affine(dst, src) {
  const [p0, p1, p2] = dst;
  const [t0, t1, t2] = src;
  const denominator = t0[0] * (t1[1] - t2[1]) + t1[0] * (t2[1] - t0[1]) + t2[0] * (t0[1] - t1[1]);
  if (Math.abs(denominator) < 1e-9) return null;
  const f = (q, k) => (q[0][k] * (t1[1] - t2[1]) + q[1][k] * (t2[1] - t0[1]) + q[2][k] * (t0[1] - t1[1])) / denominator;
  const g = (q, k) => (q[0][k] * (t2[0] - t1[0]) + q[1][k] * (t0[0] - t2[0]) + q[2][k] * (t1[0] - t0[0])) / denominator;
  const h = (q, k) => (q[0][k] * (t1[0] * t2[1] - t2[0] * t1[1]) + q[1][k] * (t2[0] * t0[1] - t0[0] * t2[1]) + q[2][k] * (t0[0] * t1[1] - t1[0] * t0[1])) / denominator;
  return [f(dst, 0), f(dst, 1), g(dst, 0), g(dst, 1), h(dst, 0), h(dst, 1)];
}

function tex(ctx, dst, src) {
  if (!S.img) return;
  const transform = affine(dst, src);
  if (!transform) return;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(...dst[0]);
  ctx.lineTo(...dst[1]);
  ctx.lineTo(...dst[2]);
  ctx.closePath();
  ctx.clip();
  ctx.transform(...transform);
  ctx.drawImage(S.img, 0, 0);
  ctx.restore();
}

function weight(item, index, joint) {
  let value = item.weight_solution?.per_vertex_weights?.[index]?.find((row) => row[0] === joint)?.[1] || 0;
  S.ed.weights
    .filter((edit) => edit.mesh_id === item.mesh_id && edit.vertex_index === index && edit.control_id === joint)
    .forEach((edit) => { value += edit.weight_delta; });
  return Math.max(0, Math.min(1, value));
}

function draw() {
  const canvas = $('#viewport');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#080c12';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#17212d';
  for (let x = 0; x < canvas.width; x += 50) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += 50) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }
  for (const item of meshes()) {
    const points = item.vertices.map((_, index) => pmesh(item, index));
    for (const triangle of item.triangles) {
      const destination = triangle.map((index) => sc(points[index]));
      if ($('#showImage').checked && S.img) tex(ctx, destination, triangle.map((index) => item.vertices[index]));
      ctx.beginPath();
      ctx.moveTo(...destination[0]);
      ctx.lineTo(...destination[1]);
      ctx.lineTo(...destination[2]);
      ctx.closePath();
      ctx.fillStyle = 'rgba(70,100,130,.08)';
      ctx.fill();
      if ($('#showMesh').checked) {
        ctx.strokeStyle = item.mesh_id === S.sel.mesh ? '#8dd2ff' : '#36516a';
        ctx.stroke();
      }
    }
    if ($('#showMesh').checked || S.mode === 'mesh' || S.mode === 'weights') {
      points.forEach((point, index) => {
        const screen = sc(point);
        const w = weight(item, index, S.sel.control || '');
        ctx.fillStyle = S.mode === 'weights'
          ? `rgba(255,210,90,${0.2 + 0.8 * w})`
          : (item.mesh_id === S.sel.mesh && index === S.sel.vertex ? 'white' : '#76a5c9');
        ctx.beginPath();
        ctx.arc(...screen, item.mesh_id === S.sel.mesh && index === S.sel.vertex ? 5 : 2.5, 0, 7);
        ctx.fill();
      });
    }
  }
  if ($('#showRig').checked) {
    const map = new Map(controls().map((item) => [item.control_id, item]));
    for (const edge of S.scene?.puppet?.rig?.edges || []) {
      const a = map.get(edge.parent_control_id);
      const b = map.get(edge.child_control_id);
      if (a && b) {
        ctx.strokeStyle = '#93a4b5';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(...sc(pcontrol(a)));
        ctx.lineTo(...sc(pcontrol(b)));
        ctx.stroke();
      }
    }
    for (const item of map.values()) {
      ctx.fillStyle = item.control_id === S.sel.control ? 'white' : '#62d0ff';
      ctx.beginPath();
      ctx.arc(...sc(pcontrol(item)), item.control_id === S.sel.control ? 6 : 4, 0, 7);
      ctx.fill();
    }
  }
  requestAnimationFrame(draw);
}

function syncDirtyFromDocument() {
  if (documentModel.differsFromPersisted()) dirty.markDirty();
  else dirty.markClean();
}

function scheduleAutosave() {
  clearTimeout(S.autosaveTimer);
  if (!S.bundle || dirty.status === 'conflict') return;
  S.autosaveTimer = setTimeout(() => {
    if (dirty.status === 'dirty' && !S.saveInFlight) void save({silent: true});
  }, 1200);
}

function commitOperation(label, before, {autosave = true} = {}) {
  if (!journal.commit(label, before, documentModel.snapshot())) return false;
  syncDirtyFromDocument();
  if (autosave) scheduleAutosave();
  updateStudioState();
  return true;
}

function operate(label, operation) {
  const before = documentModel.snapshot();
  operation();
  commitOperation(label, before);
}

function nearest(event) {
  const bounds = $('#viewport').getBoundingClientRect();
  const world = wc([
    (event.clientX - bounds.left) * $('#viewport').width / bounds.width,
    (event.clientY - bounds.top) * $('#viewport').height / bounds.height,
  ]);
  let best = null;
  let distance = 18 / S.cam.s;
  if (['rig', 'topology', 'ik', 'animation'].includes(S.mode)) {
    for (const item of controls()) {
      const point = pcontrol(item);
      const candidate = Math.hypot(point[0] - world[0], point[1] - world[1]);
      if (candidate < distance) {
        distance = candidate;
        best = {kind: 'control', item};
      }
    }
  } else {
    for (const item of meshes()) {
      item.vertices.forEach((_, index) => {
        const point = pmesh(item, index, false);
        const candidate = Math.hypot(point[0] - world[0], point[1] - world[1]);
        if (candidate < distance) {
          distance = candidate;
          best = {kind: 'vertex', item, index};
        }
      });
    }
  }
  if (best?.kind === 'control') {
    S.sel.control = best.item.control_id;
    const edit = rigEdit(best.item.control_id);
    S.drag = {
      kind: 'control',
      id: best.item.control_id,
      world,
      base: [...edit.translation_xy],
      before: documentModel.snapshot(),
      changed: false,
    };
  } else if (best?.kind === 'vertex') {
    S.sel.mesh = best.item.mesh_id;
    S.sel.vertex = best.index;
    S.drag = {
      kind: 'vertex',
      mesh: best.item.mesh_id,
      index: best.index,
      world,
      base: [...md(best.item.mesh_id, best.index)],
      before: documentModel.snapshot(),
      changed: false,
    };
  } else {
    S.drag = null;
  }
  render();
}

function move(event) {
  if (!S.drag) return;
  const bounds = $('#viewport').getBoundingClientRect();
  const world = wc([
    (event.clientX - bounds.left) * $('#viewport').width / bounds.width,
    (event.clientY - bounds.top) * $('#viewport').height / bounds.height,
  ]);
  const delta = [world[0] - S.drag.world[0], world[1] - S.drag.world[1]];
  if (S.drag.kind === 'control') {
    const edit = rigEdit(S.drag.id);
    edit.translation_xy = [S.drag.base[0] + delta[0], S.drag.base[1] + delta[1]];
  } else {
    let edit = S.ed.mesh.find((item) => item.mesh_id === S.drag.mesh && item.vertex_index === S.drag.index);
    if (!edit) {
      edit = {mesh_id: S.drag.mesh, vertex_index: S.drag.index, translation_xy: [0, 0]};
      S.ed.mesh.push(edit);
    }
    edit.translation_xy = [S.drag.base[0] + delta[0], S.drag.base[1] + delta[1]];
  }
  S.drag.changed = true;
  dirty.markDirty();
  inspector();
}

function finishDrag() {
  if (!S.drag) return;
  if (S.drag.changed) commitOperation(S.drag.kind === 'control' ? 'Move control' : 'Move mesh vertex', S.drag.before);
  S.drag = null;
}

function views() {
  const element = $('#views');
  element.innerHTML = '';
  LAB.forEach((label, index) => {
    const button = document.createElement('button');
    button.textContent = `${index} ${label}`;
    button.className = index === S.view ? 'active' : '';
    button.onclick = async () => {
      S.view = index;
      S.sel = {control: null, mesh: null, vertex: -1};
      views();
      await image();
      fit();
      render();
    };
    element.append(button);
  });
  $('#viewLabel').textContent = `VIEW ${S.view} · ${LAB[S.view]}`;
}

function tree() {
  const element = $('#sceneTree');
  element.innerHTML = '';
  for (const [name, rows, key] of [['RIG', controls(), 'control_id'], ['MESH', meshes(), 'mesh_id']]) {
    element.insertAdjacentHTML('beforeend', `<div class="row mono">▾ ${name} (${rows.length})</div>`);
    rows.forEach((item) => {
      const row = document.createElement('div');
      row.className = 'row';
      row.textContent = `  ${item[key]}`;
      row.onclick = () => {
        if (key === 'control_id') S.sel.control = item[key];
        else S.sel.mesh = item[key];
        render();
      };
      element.append(row);
    });
  }
}

const sec = (heading, body) => `<div class="inspector-section"><h3>${heading}</h3>${body}</div>`;
const opts = () => controls().map((item) => `<option>${item.control_id}</option>`).join('');

function inspector() {
  const element = $('#inspectorBody');
  if (!S.scene) {
    element.innerHTML = sec('NO BUNDLE', 'Open a V4 bundle.');
    return;
  }
  const selectedControl = control();
  const selectedMesh = mesh();
  let html = '';
  if (S.mode === 'rig') {
    if (selectedControl) {
      const edit = rigValue(selectedControl.control_id);
      html = sec(selectedControl.control_id, `<label>Rotation <input id="rot" type="number" value="${edit.rotation_degrees}"></label><br><label>X <input id="tx" type="number" value="${edit.translation_xy[0]}"></label><br><label>Y <input id="ty" type="number" value="${edit.translation_xy[1]}"></label><br><button id="reset">RESET</button>`);
    } else html = sec('RIG', 'Select a control.');
  } else if (S.mode === 'mesh') {
    html = sec('MESH', selectedMesh ? `${selectedMesh.mesh_id}<br>vertex ${S.sel.vertex}<br>${selectedMesh.vertices.length} vertices · ${selectedMesh.triangles.length} triangles` : 'Select a mesh.');
  } else if (S.mode === 'weights') {
    html = sec('WEIGHTS', `<select id="wj">${opts()}</select><select id="wm"><option>ADD</option><option>SUBTRACT</option><option>SMOOTH</option></select><input id="ws" type="number" step=".05" value=".1"><button id="paint">APPLY SELECTED</button>`);
  } else if (S.mode === 'bindings') {
    html = sec('BINDINGS', `<select id="bid">${bindings().map((item) => `<option>${item.binding_id}</option>`).join('')}</select><select id="bc">${opts()}</select><button id="bind">REASSIGN</button>`);
  } else if (S.mode === 'topology') {
    html = sec('TOPOLOGY', `Selected ${selectedControl?.control_id || '—'}<input id="bone" placeholder="user:bone"><select id="parent"><option value="">ROOT</option>${opts()}</select><button id="addBone">ADD</button><button id="reparent">REPARENT</button><button id="delBone" class="danger">DELETE</button><pre>${JSON.stringify(S.ed.topology, null, 2)}</pre>`);
  } else if (S.mode === 'ik') {
    html = sec('IK', `<input id="ikid" value="IK:${S.ed.ik.length}"><select id="iks">${opts()}</select><select id="ike">${opts()}</select><select id="ikt">${opts()}</select><button id="addIK">ADD IK</button><pre>${JSON.stringify(S.ed.ik, null, 2)}</pre>`);
  } else {
    html = sec('ANIMATION', `Selected bone: ${selectedControl?.control_id || '—'}<br>Create clips and keys in the timeline. Preview is non-authoritative until requalified.`);
  }
  element.innerHTML = html;
  wireInspector();
}

function liveNumber(selector, label, read, write) {
  const element = $(selector);
  if (!element) return;
  const before = documentModel.snapshot();
  element.oninput = () => {
    write(Number(element.value));
    dirty.markDirty();
  };
  element.onchange = () => commitOperation(label, before);
}

function wireInspector() {
  const selectedControl = control();
  const selectedMesh = mesh();
  if (selectedControl) {
    liveNumber('#rot', 'Rotate control', () => rigValue(selectedControl.control_id).rotation_degrees, (value) => { rigEdit(selectedControl.control_id).rotation_degrees = value; });
    liveNumber('#tx', 'Move control X', () => rigValue(selectedControl.control_id).translation_xy[0], (value) => { rigEdit(selectedControl.control_id).translation_xy[0] = value; });
    liveNumber('#ty', 'Move control Y', () => rigValue(selectedControl.control_id).translation_xy[1], (value) => { rigEdit(selectedControl.control_id).translation_xy[1] = value; });
  }
  if ($('#reset')) $('#reset').onclick = () => operate('Reset control', () => {
    const edit = rigEdit(selectedControl.control_id);
    edit.rotation_degrees = 0;
    edit.translation_xy = [0, 0];
    inspector();
  });
  if ($('#paint')) $('#paint').onclick = () => {
    if (!selectedMesh || S.sel.vertex < 0) return toast('Select a vertex', true);
    operate('Paint weight', () => {
      const joint = $('#wj').value;
      const strength = Math.abs(Number($('#ws').value) || 0.1);
      const mode = $('#wm').value;
      const delta = mode === 'ADD' ? strength : mode === 'SUBTRACT' ? -strength : 0;
      let edit = S.ed.weights.find((item) => item.mesh_id === selectedMesh.mesh_id && item.vertex_index === S.sel.vertex && item.control_id === joint);
      if (!edit) {
        edit = {mesh_id: selectedMesh.mesh_id, vertex_index: S.sel.vertex, control_id: joint, weight_delta: 0};
        S.ed.weights.push(edit);
      }
      edit.weight_delta = Math.max(-1, Math.min(1, edit.weight_delta + delta));
      S.sel.control = joint;
    });
    toast('Weight edit staged');
  };
  if ($('#bind')) $('#bind').onclick = () => operate('Reassign attachment', () => {
    S.ed.bindings.push({binding_id: $('#bid').value, primary_final_control_id: $('#bc').value});
    toast('Binding staged');
  });
  if ($('#addBone')) $('#addBone').onclick = () => {
    const id = $('#bone').value.trim();
    if (!id) return toast('Bone ID required', true);
    operate('Add bone', () => {
      S.ed.topology.push({operation: 'add_bone', bone_id: id, parent_id: $('#parent').value || null, anchor_xy: selectedControl?.anchor || [0, 0], display_name: id});
      inspector();
    });
  };
  if ($('#reparent')) $('#reparent').onclick = () => {
    if (!selectedControl) return;
    operate('Reparent bone', () => {
      S.ed.topology.push({operation: 'reparent', bone_id: selectedControl.control_id, parent_id: $('#parent').value || null});
      inspector();
    });
  };
  if ($('#delBone')) $('#delBone').onclick = () => {
    if (!selectedControl) return;
    operate('Delete bone', () => {
      S.ed.topology.push({operation: 'delete_bone', bone_id: selectedControl.control_id});
      inspector();
    });
  };
  if ($('#addIK')) $('#addIK').onclick = () => operate('Add IK constraint', () => {
    S.ed.ik.push({
      constraint_id: $('#ikid').value,
      start_bone_id: $('#iks').value,
      end_bone_id: $('#ike').value,
      target_bone_id: $('#ikt').value,
      mix: 1,
      iterations: 12,
      bend_direction: 1,
      enabled: true,
    });
    inspector();
  });
}

function userClip() {
  return S.ed.animation.find((item) => item.clip_id === S.clip);
}

function clips() {
  const element = $('#clipSelect');
  const user = S.ed.animation.map((item) => ({...item, user: true}));
  const all = [...S.clips, ...user];
  element.innerHTML = all.map((item) => `<option value="${item.clip_id}">${item.user ? '✎' : '▶'} ${item.display_name || item.clip_id}</option>`).join('');
  if (!S.clip && all.length) S.clip = all[0].clip_id;
  element.value = S.clip || '';
  tracks();
  dope();
}

function tracks() {
  const element = $('#trackList');
  const clip = userClip();
  element.innerHTML = clip
    ? clip.tracks.map((track) => `<div class="track">${track.bone_id}</div>`).join('')
    : '<div class="track mono">Proof-owned runtime clip</div>';
}

function dope() {
  const canvas = $('#dope');
  const ctx = canvas.getContext('2d');
  const user = userClip();
  const base = S.clips.find((item) => item.clip_id === S.clip);
  const duration = user?.duration_seconds || base?.duration_seconds || 1;
  ctx.fillStyle = '#090d12';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#243140';
  for (let index = 0; index <= 10; index += 1) {
    const x = index * canvas.width / 10;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  if (user) {
    user.tracks.forEach((track, row) => track.keys.forEach((key) => {
      const x = key.time_seconds / duration * canvas.width;
      const y = 35 + row * 22;
      ctx.fillStyle = '#ffd36b';
      ctx.fillRect(x - 4, y - 4, 8, 8);
    }));
  }
  ctx.strokeStyle = '#ff6c77';
  ctx.beginPath();
  ctx.moveTo(S.t / duration * canvas.width, 0);
  ctx.lineTo(S.t / duration * canvas.width, canvas.height);
  ctx.stroke();
  $('#timeReadout').textContent = `${S.t.toFixed(3)} / ${duration.toFixed(3)}`;
}

function newClip() {
  operate('Create animation clip', () => {
    const id = `USER_${S.ed.animation.length + 1}`;
    S.ed.animation.push({clip_id: id, display_name: id, duration_seconds: 1, loop: true, tracks: []});
    S.clip = id;
    S.t = 0;
    clips();
  });
}

function addKey() {
  const user = userClip();
  const selectedControl = control();
  if (!user || !selectedControl) return toast('Select user clip and bone', true);
  operate('Add animation key', () => {
    let track = user.tracks.find((item) => item.bone_id === selectedControl.control_id);
    if (!track) {
      track = {bone_id: selectedControl.control_id, keys: []};
      user.tracks.push(track);
    }
    const edit = rigValue(selectedControl.control_id);
    const key = {
      time_seconds: S.t,
      translation_xy: [...edit.translation_xy],
      rotation_degrees: edit.rotation_degrees,
      scale_xy: [1, 1],
      curve: $('#curveSelect').value,
    };
    track.keys = track.keys.filter((item) => Math.abs(item.time_seconds - S.t) > 1e-6);
    track.keys.push(key);
    track.keys.sort((a, b) => a.time_seconds - b.time_seconds);
    tracks();
    dope();
  });
}

function sampleUser() {
  const user = userClip();
  S.previewPose.clear();
  if (!user) return;
  for (const track of user.tracks) {
    const keys = track.keys;
    if (!keys.length) continue;
    let a = keys[0];
    let b = keys.at(-1);
    for (const key of keys) {
      if (key.time_seconds <= S.t) a = key;
      if (key.time_seconds >= S.t) { b = key; break; }
    }
    const q = b.time_seconds === a.time_seconds ? 0 : (S.t - a.time_seconds) / (b.time_seconds - a.time_seconds);
    const factor = a.curve === 'step' ? 0 : a.curve === 'cubic' ? q * q * (3 - 2 * q) : q;
    S.previewPose.set(track.bone_id, {
      rotation_degrees: a.rotation_degrees + (b.rotation_degrees - a.rotation_degrees) * factor,
      translation_xy: [0, 1].map((index) => a.translation_xy[index] + (b.translation_xy[index] - a.translation_xy[index]) * factor),
    });
  }
}

async function runtime() {
  S.previewPose.clear();
  if (!S.clip || userClip()) return;
  try {
    S.frame = await api(`/api/runtime/frame?bundle=${enc(S.bundle)}&clip=${enc(S.clip)}&time=${S.t}`);
  } catch (error) {
    S.play = false;
    toast(error.message, true);
  }
}

async function tick(time) {
  if (S.play) {
    const delta = (time - S.last) / 1000;
    S.last = time;
    const user = userClip();
    const base = S.clips.find((item) => item.clip_id === S.clip);
    const duration = user?.duration_seconds || base?.duration_seconds || 1;
    const loop = user?.loop ?? base?.loop;
    S.t += delta;
    if (S.t > duration) S.t = loop ? S.t % duration : duration;
    if (user) sampleUser(); else await runtime();
    dope();
  }
  requestAnimationFrame(tick);
}

async function hydratePersistence() {
  const persistence = window.RealSaSStudioPersistence;
  if (!persistence) {
    documentModel.reset();
    journal.clear();
    dirty.hydrate();
    return null;
  }
  const session = await persistence.ensureSession(S.bundle);
  journal.clear();
  if (session?.head?.authoring_payload && typeof session.head.authoring_payload === 'object') {
    documentModel.loadAuthoringLayer(session.head.authoring_payload);
  } else {
    documentModel.reset();
  }
  dirty.hydrate(session?.head || {});
  return session;
}

async function open() {
  try {
    clearTimeout(S.autosaveTimer);
    S.bundle = $('#bundlePath').value.trim();
    S.previewPose.clear();
    S.scene = await api(`/api/living/scene?bundle=${enc(S.bundle)}`);
    const runtimeClips = await api(`/api/runtime/clips?bundle=${enc(S.bundle)}`);
    S.clips = runtimeClips.clips || [];
    S.clip = S.clips[0]?.clip_id || null;
    const session = await hydratePersistence();
    $('#proofBadge').textContent = S.scene.proof.passed ? 'PROOF PASS' : 'PROOF BLOCK';
    $('#proofBadge').className = `badge ${S.scene.proof.passed ? 'good' : 'bad'}`;
    $('#stateHash').textContent = (S.scene.product.product_state_hash || '').slice(0, 16);
    $('#authorityText').textContent = 'Canonical V4 is read-only. User edits are sibling revisions and require requalification.';
    await image();
    fit();
    render();
    updateStudioState();
    const revision = Number(session?.head?.revision_number || 0);
    toast(revision ? `Studio reopened · revision ${revision}` : 'Compiled character opened');
  } catch (error) {
    toast(error.message, true);
  }
}

async function save({silent = false} = {}) {
  if (!S.bundle || S.saveInFlight || dirty.status === 'conflict') return;
  S.saveInFlight = true;
  clearTimeout(S.autosaveTimer);
  dirty.markSaving();
  try {
    const payload = documentModel.request(S.bundle, V());
    const response = await api('/api/living/edit-layer', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify(payload),
    });
    documentModel.markPersisted();
    dirty.markSaved(response.studio_revision || {
      requires_requalification: Boolean(response.requires_dynamic_revalidation),
    });
    if (!silent) toast(`Saved${response.requires_dynamic_revalidation ? ' · requalification required' : ''}`);
  } catch (error) {
    if (error.status === 409) {
      dirty.markConflict(error.message);
      toast('Studio conflict · reload latest revision before saving', true);
    } else {
      dirty.markError(error.message);
      if (!silent) toast(error.message, true);
    }
  } finally {
    S.saveInFlight = false;
    updateStudioState();
  }
}

function undoEdit() {
  const operation = journal.undo(documentModel);
  if (!operation) return;
  syncDirtyFromDocument();
  S.previewPose.clear();
  render();
  scheduleAutosave();
  toast(`Undo · ${operation.label}`);
}

function redoEdit() {
  const operation = journal.redo(documentModel);
  if (!operation) return;
  syncDirtyFromDocument();
  S.previewPose.clear();
  render();
  scheduleAutosave();
  toast(`Redo · ${operation.label}`);
}

function updateStudioState() {
  const element = $('#studioState');
  if (element) {
    const revision = dirty.revisionNumber ? `REV ${dirty.revisionNumber}` : 'UNSAVED';
    element.textContent = `${revision} · ${dirty.status.toUpperCase()}${dirty.requiresRequalification ? ' · REQUAL' : ''}`;
    element.dataset.state = dirty.status;
  }
  if ($('#undoEdit')) $('#undoEdit').disabled = !journal.canUndo();
  if ($('#redoEdit')) $('#redoEdit').disabled = !journal.canRedo();
}

function render() {
  views();
  tree();
  inspector();
  clips();
  $('#hint').textContent = S.mode === 'mesh'
    ? 'Drag vertices'
    : S.mode === 'rig'
      ? 'Drag controls'
      : 'User edits remain non-authoritative until requalified.';
  updateStudioState();
}

$$('.mode').forEach((button) => {
  button.onclick = () => {
    $$('.mode').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    S.mode = button.dataset.mode;
    $('#showWeights').checked = S.mode === 'weights';
    inspector();
  };
});

$('#openBundle').onclick = open;
$('#fitView').onclick = fit;
$('#saveLayer').onclick = () => void save();
if ($('#undoEdit')) $('#undoEdit').onclick = undoEdit;
if ($('#redoEdit')) $('#redoEdit').onclick = redoEdit;
$('#exportRuntime').onclick = async () => {
  try {
    const response = await api('/api/export', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({bundle: S.bundle}),
    });
    toast(response.package_path);
  } catch (error) {
    toast(error.message, true);
  }
};
$('#newClip').onclick = newClip;
$('#addKey').onclick = addKey;
$('#deleteKey').onclick = () => toast('Select/rewrite the key at the same time to replace it');
$('#clipSelect').onchange = async (event) => {
  S.clip = event.target.value;
  S.t = 0;
  S.frame = null;
  S.previewPose.clear();
  if (!userClip()) await runtime(); else sampleUser();
  tracks();
  dope();
};
$('#play').onclick = async () => {
  S.play = !S.play;
  S.last = performance.now();
  if (!userClip()) await runtime(); else sampleUser();
};
$('#stop').onclick = () => {
  S.play = false;
  S.t = 0;
  S.frame = null;
  S.previewPose.clear();
  dope();
};

const viewport = $('#viewport');
viewport.onpointerdown = (event) => {
  viewport.setPointerCapture(event.pointerId);
  nearest(event);
};
viewport.onpointermove = move;
viewport.onpointerup = finishDrag;
viewport.onpointercancel = finishDrag;
viewport.onwheel = (event) => {
  event.preventDefault();
  S.cam.s = Math.max(0.05, Math.min(20, S.cam.s * (event.deltaY > 0 ? 0.9 : 1.1)));
};

$('#dope').onclick = (event) => {
  const user = userClip();
  const base = S.clips.find((item) => item.clip_id === S.clip);
  const duration = user?.duration_seconds || base?.duration_seconds || 1;
  const bounds = $('#dope').getBoundingClientRect();
  S.t = Math.max(0, Math.min(duration, (event.clientX - bounds.left) / bounds.width * duration));
  if (user) sampleUser(); else void runtime();
  dope();
};

dirty.addEventListener('change', updateStudioState);
window.addEventListener('realsas:studio-conflict', (event) => {
  dirty.markConflict(event.detail?.message || event.detail?.error || 'Studio revision conflict');
});
window.addEventListener('keydown', (event) => {
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;
  const command = event.ctrlKey || event.metaKey;
  if (!command) return;
  if (event.key.toLowerCase() === 'z') {
    event.preventDefault();
    if (event.shiftKey) redoEdit(); else undoEdit();
  } else if (event.key.toLowerCase() === 'y') {
    event.preventDefault();
    redoEdit();
  } else if (event.key.toLowerCase() === 's') {
    event.preventDefault();
    void save();
  }
});

render();
requestAnimationFrame(draw);
requestAnimationFrame(tick);
