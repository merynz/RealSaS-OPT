import {
  meshEdges,
  modelArrays,
  nextUserFaceID,
  nextUserVertexID,
  replayMeshTopology,
  upsertMoveVertex,
} from './studio_mesh_topology_core.js';

const $ = (selector) => document.querySelector(selector);

let studio = null;
let S = null;
let documentModel = null;
let journal = null;
let dirty = null;
let panel = null;
let viewport = null;
let overlay = null;
let saveTimer = null;
let drag = null;
let structural = false;
let selectionMode = 'vertex';
let selectedVertices = [];
let selectedEdge = null;
let selectedFace = null;
let activeMeshID = null;
let lastError = '';

function boot() {
  studio = window.RealSaSStudio;
  panel = document.querySelector('.viewport-panel');
  viewport = $('#viewport');
  if (!studio || !panel || !viewport) {
    requestAnimationFrame(boot);
    return;
  }
  S = studio.state;
  documentModel = studio.document;
  journal = studio.journal;
  dirty = studio.dirty;
  installOverlay();

  const inspector = $('#inspectorBody');
  if (inspector) new MutationObserver(() => ensureInspector()).observe(inspector, {childList: true, subtree: true});
  panel.addEventListener('pointerdown', pointerDown, true);
  panel.addEventListener('pointermove', pointerMove, true);
  panel.addEventListener('pointerup', pointerUp, true);
  panel.addEventListener('pointercancel', pointerCancel, true);
  requestAnimationFrame(frame);
}

function currentViewID() {
  return `V${S?.view ?? 0}`;
}

function meshes() {
  return (S?.scene?.puppet?.meshes || []).filter((mesh) => mesh.view_id === currentViewID());
}

function activeMesh() {
  const rows = meshes();
  if (!rows.some((mesh) => mesh.mesh_id === activeMeshID)) activeMeshID = S?.sel?.mesh || rows[0]?.mesh_id || null;
  return rows.find((mesh) => mesh.mesh_id === activeMeshID) || rows[0] || null;
}

function topologyRows(meshID) {
  return (S?.ed?.meshTopology || []).filter((row) => row.mesh_id === meshID);
}

function meshDelta(meshID, vertexIndex) {
  const edit = (S?.ed?.mesh || []).find((row) => row.mesh_id === meshID && row.vertex_index === vertexIndex);
  return edit?.translation_xy || [0, 0];
}

function derivedModel(mesh = activeMesh()) {
  if (!mesh) return null;
  const model = replayMeshTopology(mesh, S?.ed?.meshTopology || []);
  for (const vertex of model.vertices.values()) {
    if (!Number.isInteger(vertex.canonical_index)) continue;
    const delta = meshDelta(mesh.mesh_id, vertex.canonical_index);
    vertex.position_xy = [vertex.position_xy[0] + Number(delta[0] || 0), vertex.position_xy[1] + Number(delta[1] || 0)];
  }
  return model;
}

function operate(label, operation) {
  const before = documentModel.snapshot();
  try {
    operation();
    for (const mesh of meshes()) replayMeshTopology(mesh, S.ed.meshTopology || []);
  } catch (error) {
    documentModel.replace(before);
    lastError = String(error?.message || error);
    ensureInspector(true);
    return false;
  }
  if (!journal.commit(label, before, documentModel.snapshot())) return false;
  dirty.markDirty();
  lastError = '';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (dirty.status === 'dirty') $('#saveLayer')?.click();
  }, 1200);
  ensureInspector(true);
  return true;
}

function escapeHTML(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function ensureInspector(force = false) {
  if (!S?.scene || S.mode !== 'mesh') return;
  const body = $('#inspectorBody');
  if (!body) return;
  const mesh = activeMesh();
  if (!mesh) return;
  let model;
  try {
    model = derivedModel(mesh);
    lastError = '';
  } catch (error) {
    lastError = String(error?.message || error);
    model = null;
  }
  const signature = `${currentViewID()}:${mesh.mesh_id}:${S.ed.meshTopology?.length || 0}:${S.ed.mesh?.length || 0}:${structural}:${selectionMode}:${selectedVertices.join(',')}:${selectedEdge?.edge_id || ''}:${selectedFace || ''}:${lastError}`;
  const current = $('#meshTopologyPro');
  if (current && !force && current.dataset.signature === signature) return;
  renderInspector(body, mesh, model, signature);
}

function renderInspector(body, mesh, model, signature) {
  const arrays = model ? modelArrays(model) : {vertices: [], faces: []};
  const edges = model ? meshEdges(model) : [];
  const meshOptions = meshes().map((row) => `<option value="${escapeHTML(row.mesh_id)}"${row.mesh_id === mesh.mesh_id ? ' selected' : ''}>${escapeHTML(row.mesh_id)}</option>`).join('');
  const selectedVertexText = selectedVertices.length ? selectedVertices.join(', ') : '—';
  const selectedEdgeText = selectedEdge ? selectedEdge.vertex_ids.join(' ↔ ') : '—';
  const selectedFaceText = selectedFace || '—';
  const validClass = lastError ? 'bad' : 'good';

  body.innerHTML = `<div id="meshTopologyPro" class="inspector-section mesh-topology-editor" data-signature="${escapeHTML(signature)}">
    <h3>MESH TOPOLOGY · ${escapeHTML(currentViewID())}</h3>
    <label>MESH<select id="meshTopoMesh">${meshOptions}</select></label>
    <div class="mesh-topology-toggle"><button id="meshStructuralToggle" class="${structural ? 'active' : ''}">${structural ? 'STRUCTURAL TOOL · ON' : 'STRUCTURAL TOOL · OFF'}</button></div>
    <div class="mesh-topology-modes">
      <button data-mesh-mode="vertex" class="${selectionMode === 'vertex' ? 'active' : ''}">VERTEX</button>
      <button data-mesh-mode="edge" class="${selectionMode === 'edge' ? 'active' : ''}">EDGE</button>
      <button data-mesh-mode="face" class="${selectionMode === 'face' ? 'active' : ''}">FACE</button>
    </div>
    <div class="mesh-topology-stats"><span>${arrays.vertices.length} V</span><span>${edges.length} E</span><span>${arrays.faces.length} F</span><strong class="${validClass}">${lastError ? 'INVALID' : 'LIVE VALID'}</strong></div>
    ${lastError ? `<div class="mesh-topology-error">${escapeHTML(lastError)}</div>` : ''}
    <div class="mesh-topology-selection">
      <div><small>VERTICES</small><strong>${escapeHTML(selectedVertexText)}</strong></div>
      <div><small>EDGE</small><strong>${escapeHTML(selectedEdgeText)}</strong></div>
      <div><small>FACE</small><strong>${escapeHTML(selectedFaceText)}</strong></div>
    </div>
    <div class="mesh-topology-actions">
      <button id="meshAddVertex">ADD VERTEX @ VIEW CENTER</button>
      <button id="meshSplitEdge"${selectedEdge ? '' : ' disabled'}>SPLIT SELECTED EDGE</button>
      <button id="meshWeld"${selectedVertices.length === 2 ? '' : ' disabled'}>WELD 2 VERTICES</button>
      <button id="meshAddFace"${selectedVertices.length === 3 ? '' : ' disabled'}>CREATE FACE FROM 3</button>
      <button id="meshDeleteFace"${selectedFace ? '' : ' disabled'}>DELETE FACE</button>
      <button id="meshDeleteVertex"${selectedVertices.length === 1 ? '' : ' disabled'}>DELETE UNREFERENCED VERTEX</button>
      <button id="meshRepair">PRUNE ISOLATED USER VERTICES</button>
      <button id="meshResetTopology" class="danger">RESET STRUCTURAL EDITS</button>
    </div>
    <div class="mesh-topology-help">Structural OFF preserves the normal transform workflow. Structural ON enables stable-ID vertex/edge/face editing. Shift-click vertices builds the 2/3-vertex operation set. Every operation is replayed locally and server-validated again on save.</div>
  </div>`;

  $('#meshTopoMesh').onchange = (event) => {
    activeMeshID = event.target.value;
    if (S?.sel) S.sel.mesh = activeMeshID;
    clearSelection();
    ensureInspector(true);
  };
  $('#meshStructuralToggle').onclick = () => {
    structural = !structural;
    clearSelection();
    ensureInspector(true);
  };
  body.querySelectorAll('[data-mesh-mode]').forEach((button) => {
    button.onclick = () => {
      selectionMode = button.dataset.meshMode;
      clearSelection();
      ensureInspector(true);
    };
  });
  $('#meshAddVertex').onclick = () => {
    if (!model) return;
    const id = nextUserVertexID(model);
    const center = worldFromIntrinsic([viewport.width / 2, viewport.height / 2]);
    operate('Add mesh vertex', () => S.ed.meshTopology.push({operation: 'add_vertex', mesh_id: mesh.mesh_id, vertex_id: id, position_xy: center}));
    selectedVertices = [id];
  };
  $('#meshSplitEdge').onclick = () => {
    if (!selectedEdge || !model) return;
    const id = nextUserVertexID(model);
    const [a, b] = selectedEdge.vertex_ids;
    if (operate('Split mesh edge', () => S.ed.meshTopology.push({operation: 'split_edge', mesh_id: mesh.mesh_id, vertex_a_id: a, vertex_b_id: b, new_vertex_id: id, t: 0.5}))) {
      selectionMode = 'vertex';
      selectedVertices = [id];
      selectedEdge = null;
    }
  };
  $('#meshWeld').onclick = () => {
    if (selectedVertices.length !== 2) return;
    const [keep, remove] = selectedVertices;
    if (operate('Weld mesh vertices', () => S.ed.meshTopology.push({operation: 'weld_vertices', mesh_id: mesh.mesh_id, keep_vertex_id: keep, remove_vertex_id: remove}))) {
      selectedVertices = [keep];
    }
  };
  $('#meshAddFace').onclick = () => {
    if (selectedVertices.length !== 3 || !model) return;
    const faceID = nextUserFaceID(model);
    if (operate('Create mesh face', () => S.ed.meshTopology.push({operation: 'add_face', mesh_id: mesh.mesh_id, face_id: faceID, vertex_ids: [...selectedVertices]}))) {
      selectionMode = 'face';
      selectedFace = faceID;
      selectedVertices = [];
    }
  };
  $('#meshDeleteFace').onclick = () => {
    if (!selectedFace) return;
    if (operate('Delete mesh face', () => S.ed.meshTopology.push({operation: 'delete_face', mesh_id: mesh.mesh_id, face_id: selectedFace}))) selectedFace = null;
  };
  $('#meshDeleteVertex').onclick = () => {
    if (selectedVertices.length !== 1) return;
    const vertexID = selectedVertices[0];
    if (operate('Delete mesh vertex', () => S.ed.meshTopology.push({operation: 'delete_vertex', mesh_id: mesh.mesh_id, vertex_id: vertexID}))) selectedVertices = [];
  };
  $('#meshRepair').onclick = () => {
    if (!model) return;
    const used = new Set([...model.faces.values()].flatMap((face) => face.vertex_ids));
    const isolated = [...model.vertices.values()].filter((vertex) => vertex.user && !used.has(vertex.vertex_id));
    if (!isolated.length) return;
    operate('Prune isolated mesh vertices', () => isolated.forEach((vertex) => S.ed.meshTopology.push({operation: 'delete_vertex', mesh_id: mesh.mesh_id, vertex_id: vertex.vertex_id})));
    selectedVertices = selectedVertices.filter((id) => !isolated.some((vertex) => vertex.vertex_id === id));
  };
  $('#meshResetTopology').onclick = () => operate('Reset mesh topology edits', () => {
    S.ed.meshTopology = S.ed.meshTopology.filter((row) => row.mesh_id !== mesh.mesh_id);
    clearSelection();
  });
}

function clearSelection() {
  selectedVertices = [];
  selectedEdge = null;
  selectedFace = null;
}

function installOverlay() {
  overlay = document.createElement('canvas');
  overlay.className = 'mesh-topology-overlay';
  overlay.setAttribute('aria-hidden', 'true');
  panel.append(overlay);
}

function syncOverlay() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const viewportBounds = viewport.getBoundingClientRect();
  const panelBounds = panel.getBoundingClientRect();
  const width = Math.max(1, Math.round(viewportBounds.width * dpr));
  const height = Math.max(1, Math.round(viewportBounds.height * dpr));
  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
    overlay.style.width = `${viewportBounds.width}px`;
    overlay.style.height = `${viewportBounds.height}px`;
  }
  overlay.style.left = `${viewportBounds.left - panelBounds.left}px`;
  overlay.style.top = `${viewportBounds.top - panelBounds.top}px`;
  return {dpr, viewportBounds};
}

function intrinsicFromClient(clientX, clientY) {
  const bounds = viewport.getBoundingClientRect();
  return [
    (clientX - bounds.left) * viewport.width / Math.max(1, bounds.width),
    (clientY - bounds.top) * viewport.height / Math.max(1, bounds.height),
  ];
}

function worldFromIntrinsic(point) {
  return [(point[0] - S.cam.x) / S.cam.s, (point[1] - S.cam.y) / S.cam.s];
}

function screenFromWorld(point, dpr = 1) {
  const bounds = viewport.getBoundingClientRect();
  const intrinsic = [point[0] * S.cam.s + S.cam.x, point[1] * S.cam.s + S.cam.y];
  return [intrinsic[0] * bounds.width / viewport.width * dpr, intrinsic[1] * bounds.height / viewport.height * dpr];
}

function eventInsideViewport(event) {
  const bounds = viewport.getBoundingClientRect();
  return event.clientX >= bounds.left && event.clientX <= bounds.right && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
}

function consume(event) {
  event.preventDefault();
  event.stopPropagation();
  event.stopImmediatePropagation?.();
}

function distanceToSegment(point, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const length2 = dx * dx + dy * dy;
  if (length2 <= 1e-12) return Math.hypot(point[0] - a[0], point[1] - a[1]);
  const t = Math.max(0, Math.min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2));
  return Math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy));
}

function pointInTriangle(point, a, b, c) {
  const sign = (p1, p2, p3) => (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
  const d1 = sign(point, a, b);
  const d2 = sign(point, b, c);
  const d3 = sign(point, c, a);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}

function hitTest(event, model) {
  const pointer = intrinsicFromClient(event.clientX, event.clientY);
  const screenPoint = (world) => [world[0] * S.cam.s + S.cam.x, world[1] * S.cam.s + S.cam.y];
  if (selectionMode === 'vertex') {
    let best = null;
    let distance = 14 * viewport.width / Math.max(1, viewport.getBoundingClientRect().width);
    for (const vertex of model.vertices.values()) {
      const p = screenPoint(vertex.position_xy);
      const candidate = Math.hypot(pointer[0] - p[0], pointer[1] - p[1]);
      if (candidate < distance) {
        distance = candidate;
        best = vertex;
      }
    }
    return best ? {kind: 'vertex', value: best} : null;
  }
  if (selectionMode === 'edge') {
    let best = null;
    let distance = 10 * viewport.width / Math.max(1, viewport.getBoundingClientRect().width);
    for (const edge of meshEdges(model)) {
      const a = screenPoint(model.vertices.get(edge.vertex_ids[0]).position_xy);
      const b = screenPoint(model.vertices.get(edge.vertex_ids[1]).position_xy);
      const candidate = distanceToSegment(pointer, a, b);
      if (candidate < distance) {
        distance = candidate;
        best = edge;
      }
    }
    return best ? {kind: 'edge', value: best} : null;
  }
  const world = worldFromIntrinsic(pointer);
  const faces = [...model.faces.values()].reverse();
  for (const face of faces) {
    const [a, b, c] = face.vertex_ids.map((id) => model.vertices.get(id).position_xy);
    if (pointInTriangle(world, a, b, c)) return {kind: 'face', value: face};
  }
  return null;
}

function pointerDown(event) {
  if (S?.mode !== 'mesh' || !structural || event.button !== 0 || !eventInsideViewport(event)) return;
  const mesh = activeMesh();
  if (!mesh) return;
  let model;
  try { model = derivedModel(mesh); } catch { return; }
  consume(event);
  const hit = hitTest(event, model);
  if (!hit) {
    clearSelection();
    ensureInspector(true);
    return;
  }
  if (hit.kind === 'vertex') {
    const id = hit.value.vertex_id;
    if (event.shiftKey) {
      if (selectedVertices.includes(id)) selectedVertices = selectedVertices.filter((value) => value !== id);
      else selectedVertices = [...selectedVertices, id].slice(-3);
    } else {
      selectedVertices = [id];
    }
    selectedEdge = null;
    selectedFace = null;
    const point = intrinsicFromClient(event.clientX, event.clientY);
    drag = {
      pointerID: event.pointerId,
      before: documentModel.snapshot(),
      vertex: hit.value,
      startWorld: worldFromIntrinsic(point),
      startPosition: [...hit.value.position_xy],
      changed: false,
    };
    try { panel.setPointerCapture(event.pointerId); } catch {}
  } else if (hit.kind === 'edge') {
    selectedEdge = hit.value;
    selectedVertices = [];
    selectedFace = null;
  } else {
    selectedFace = hit.value.face_id;
    selectedVertices = [];
    selectedEdge = null;
  }
  ensureInspector(true);
}

function pointerMove(event) {
  if (!drag || drag.pointerID !== event.pointerId || S?.mode !== 'mesh' || !structural) return;
  consume(event);
  const world = worldFromIntrinsic(intrinsicFromClient(event.clientX, event.clientY));
  const delta = [world[0] - drag.startWorld[0], world[1] - drag.startWorld[1]];
  const next = [drag.startPosition[0] + delta[0], drag.startPosition[1] + delta[1]];
  documentModel.replace(drag.before);
  if (Number.isInteger(drag.vertex.canonical_index)) {
    const index = drag.vertex.canonical_index;
    let edit = S.ed.mesh.find((row) => row.mesh_id === activeMeshID && row.vertex_index === index);
    if (!edit) {
      edit = {mesh_id: activeMeshID, vertex_index: index, translation_xy: [0, 0]};
      S.ed.mesh.push(edit);
    }
    const base = activeMesh().vertices[index];
    edit.translation_xy = [next[0] - base[0], next[1] - base[1]];
  } else {
    upsertMoveVertex(S.ed.meshTopology, activeMeshID, drag.vertex.vertex_id, next);
  }
  drag.changed = true;
  dirty.markDirty();
}

function pointerUp(event) {
  if (!drag || drag.pointerID !== event.pointerId) return;
  consume(event);
  if (drag.changed) {
    try {
      derivedModel(activeMesh());
      journal.commit('Move mesh topology vertex', drag.before, documentModel.snapshot());
      dirty.markDirty();
      clearTimeout(saveTimer);
      saveTimer = setTimeout(() => {
        if (dirty.status === 'dirty') $('#saveLayer')?.click();
      }, 1200);
    } catch (error) {
      documentModel.replace(drag.before);
      lastError = String(error?.message || error);
    }
  }
  drag = null;
  try { panel.releasePointerCapture(event.pointerId); } catch {}
  ensureInspector(true);
}

function pointerCancel(event) {
  if (!drag || drag.pointerID !== event.pointerId) return;
  consume(event);
  documentModel.replace(drag.before);
  drag = null;
  ensureInspector(true);
}

function drawOverlay() {
  if (!overlay || !S) return;
  const {dpr} = syncOverlay();
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (S.mode !== 'mesh') return;
  const mesh = activeMesh();
  if (!mesh) return;
  let model;
  try { model = derivedModel(mesh); } catch { return; }

  ctx.lineWidth = Math.max(1, dpr);
  for (const face of model.faces.values()) {
    const points = face.vertex_ids.map((id) => screenFromWorld(model.vertices.get(id).position_xy, dpr));
    ctx.beginPath();
    ctx.moveTo(...points[0]);
    ctx.lineTo(...points[1]);
    ctx.lineTo(...points[2]);
    ctx.closePath();
    ctx.fillStyle = selectedFace === face.face_id ? 'rgba(255,211,107,.16)' : 'rgba(76,158,210,.035)';
    ctx.fill();
    ctx.strokeStyle = selectedFace === face.face_id ? 'rgba(255,211,107,.95)' : 'rgba(105,171,214,.46)';
    ctx.stroke();
  }

  if (selectedEdge) {
    const [a, b] = selectedEdge.vertex_ids.map((id) => screenFromWorld(model.vertices.get(id)?.position_xy || [0, 0], dpr));
    ctx.lineWidth = Math.max(2, 2 * dpr);
    ctx.strokeStyle = 'rgba(255,211,107,.98)';
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
  }

  for (const vertex of model.vertices.values()) {
    const point = screenFromWorld(vertex.position_xy, dpr);
    const selected = selectedVertices.includes(vertex.vertex_id);
    ctx.beginPath();
    ctx.arc(point[0], point[1], (selected ? 5 : vertex.user ? 3.5 : 2.7) * dpr, 0, Math.PI * 2);
    ctx.fillStyle = selected ? '#fff' : vertex.user ? '#ffd36b' : '#6fb7e6';
    ctx.fill();
    if (vertex.user) {
      ctx.strokeStyle = 'rgba(8,12,18,.9)';
      ctx.stroke();
    }
  }
}

function frame() {
  if (S?.mode === 'mesh') ensureInspector();
  if (overlay) overlay.hidden = S?.mode !== 'mesh';
  drawOverlay();
  requestAnimationFrame(frame);
}

boot();
