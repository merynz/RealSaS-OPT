const EPSILON = 1e-8;

const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;

export function clamp01(value) {
  return Math.max(0, Math.min(1, finite(value)));
}

export function baseWeight(mesh, vertexIndex, jointID) {
  const rows = mesh?.weight_solution?.per_vertex_weights?.[vertexIndex] || [];
  for (const row of rows) {
    if (Array.isArray(row) && row[0] === jointID) return clamp01(row[1]);
  }
  return 0;
}

export function effectiveWeight(mesh, vertexIndex, jointID, edits = []) {
  let value = baseWeight(mesh, vertexIndex, jointID);
  for (const edit of edits) {
    if (edit.mesh_id === mesh.mesh_id && edit.vertex_index === vertexIndex && edit.control_id === jointID) {
      value += finite(edit.weight_delta);
    }
  }
  return clamp01(value);
}

export function effectiveWeights(mesh, vertexIndex, jointIDs, edits = []) {
  const out = new Map();
  for (const jointID of jointIDs) out.set(jointID, effectiveWeight(mesh, vertexIndex, jointID, edits));
  return out;
}

export function setEffectiveWeight(edits, mesh, vertexIndex, jointID, targetWeight) {
  const target = clamp01(targetWeight);
  const base = baseWeight(mesh, vertexIndex, jointID);
  let changed = false;
  for (let index = edits.length - 1; index >= 0; index -= 1) {
    const edit = edits[index];
    if (edit.mesh_id === mesh.mesh_id && edit.vertex_index === vertexIndex && edit.control_id === jointID) {
      edits.splice(index, 1);
      changed = true;
    }
  }
  const delta = target - base;
  if (Math.abs(delta) > EPSILON) {
    edits.push({
      mesh_id: mesh.mesh_id,
      vertex_index: vertexIndex,
      control_id: jointID,
      weight_delta: Math.max(-1, Math.min(1, delta)),
    });
    changed = true;
  }
  return changed;
}

export function vertexAdjacency(mesh) {
  const count = Array.isArray(mesh?.vertices) ? mesh.vertices.length : 0;
  const rows = Array.from({length: count}, () => new Set());
  for (const triangle of mesh?.triangles || []) {
    if (!Array.isArray(triangle) || triangle.length !== 3) continue;
    for (let index = 0; index < 3; index += 1) {
      const a = Number(triangle[index]);
      const b = Number(triangle[(index + 1) % 3]);
      if (Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a < count && b < count && a !== b) {
        rows[a].add(b);
        rows[b].add(a);
      }
    }
  }
  return rows.map((row) => [...row]);
}

export function smoothWeight(mesh, vertexIndex, jointID, edits, adjacency) {
  const neighbors = adjacency?.[vertexIndex] || [];
  if (!neighbors.length) return effectiveWeight(mesh, vertexIndex, jointID, edits);
  let total = 0;
  for (const neighbor of neighbors) total += effectiveWeight(mesh, neighbor, jointID, edits);
  return clamp01(total / neighbors.length);
}

export function normalizeVertexWeights({
  mesh,
  vertexIndex,
  jointIDs,
  edits,
  lockedJointIDs = [],
  preferredJointID = null,
}) {
  const locked = new Set(lockedJointIDs);
  const weights = effectiveWeights(mesh, vertexIndex, jointIDs, edits);
  const lockedTotal = jointIDs.reduce((sum, jointID) => sum + (locked.has(jointID) ? weights.get(jointID) || 0 : 0), 0);
  if (lockedTotal > 1 + 1e-6) {
    return {ok: false, reason: 'LOCKED_WEIGHT_SUM_EXCEEDS_ONE', locked_total: lockedTotal};
  }

  const unlocked = jointIDs.filter((jointID) => !locked.has(jointID));
  if (!unlocked.length) {
    return {ok: Math.abs(lockedTotal - 1) <= 1e-6, reason: 'ALL_JOINTS_LOCKED', locked_total: lockedTotal};
  }

  const remaining = Math.max(0, 1 - lockedTotal);
  const unlockedTotal = unlocked.reduce((sum, jointID) => sum + (weights.get(jointID) || 0), 0);
  const targets = new Map();
  if (unlockedTotal > EPSILON) {
    const scale = remaining / unlockedTotal;
    for (const jointID of unlocked) targets.set(jointID, clamp01((weights.get(jointID) || 0) * scale));
  } else {
    const fallback = unlocked.includes(preferredJointID) ? preferredJointID : unlocked[0];
    for (const jointID of unlocked) targets.set(jointID, jointID === fallback ? remaining : 0);
  }

  for (const [jointID, value] of targets) setEffectiveWeight(edits, mesh, vertexIndex, jointID, value);
  const total = jointIDs.reduce((sum, jointID) => sum + effectiveWeight(mesh, vertexIndex, jointID, edits), 0);
  return {ok: true, total, locked_total: lockedTotal};
}

export function falloffWeight(distance, radius, mode = 'smooth') {
  const r = Math.max(EPSILON, finite(radius, 1));
  const d = Math.max(0, finite(distance));
  if (d >= r) return 0;
  const q = clamp01(1 - d / r);
  if (mode === 'hard') return 1;
  if (mode === 'linear') return q;
  return q * q * (3 - 2 * q);
}

export function paintVertex({
  mesh,
  vertexIndex,
  jointID,
  jointIDs,
  edits,
  amount,
  mode = 'add',
  replaceValue = 1,
  normalize = true,
  lockedJointIDs = [],
  adjacency = null,
}) {
  const locked = new Set(lockedJointIDs);
  if (locked.has(jointID)) return {changed: false, reason: 'ACTIVE_JOINT_LOCKED'};

  const current = effectiveWeight(mesh, vertexIndex, jointID, edits);
  const a = clamp01(amount);
  let target = current;
  if (mode === 'subtract') target = current - a;
  else if (mode === 'replace') target = current + (clamp01(replaceValue) - current) * a;
  else if (mode === 'smooth') {
    const smooth = smoothWeight(mesh, vertexIndex, jointID, edits, adjacency || vertexAdjacency(mesh));
    target = current + (smooth - current) * a;
  } else target = current + a;

  target = clamp01(target);
  if (Math.abs(target - current) <= EPSILON) return {changed: false, reason: 'NO_CHANGE'};
  setEffectiveWeight(edits, mesh, vertexIndex, jointID, target);

  let normalization = {ok: true};
  if (normalize) {
    normalization = normalizeVertexWeights({
      mesh,
      vertexIndex,
      jointIDs,
      edits,
      lockedJointIDs,
      preferredJointID: jointID,
    });
  }
  return {changed: true, target, normalization};
}

export function meshMirrorAxisX(mesh) {
  const vertices = mesh?.vertices || [];
  if (!vertices.length) return 0;
  const xs = vertices.map((point) => finite(point?.[0]));
  return (Math.min(...xs) + Math.max(...xs)) / 2;
}

function boundsDiagonal(points) {
  if (!points.length) return 0;
  const xs = points.map((point) => finite(point?.[0]));
  const ys = points.map((point) => finite(point?.[1]));
  return Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
}

export function mirroredVertexIndex(mesh, vertexIndex, axisX = meshMirrorAxisX(mesh), toleranceRatio = 0.05) {
  const vertices = mesh?.vertices || [];
  const source = vertices[vertexIndex];
  if (!source) return -1;
  const target = [2 * axisX - finite(source[0]), finite(source[1])];
  let best = -1;
  let distance = Infinity;
  vertices.forEach((point, index) => {
    const candidate = Math.hypot(finite(point?.[0]) - target[0], finite(point?.[1]) - target[1]);
    if (candidate < distance) {
      distance = candidate;
      best = index;
    }
  });
  const tolerance = Math.max(EPSILON, boundsDiagonal(vertices) * Math.max(0, finite(toleranceRatio, 0.05)));
  return distance <= tolerance ? best : -1;
}

export function mirroredControlID(controls, controlID, axisX, toleranceRatio = 0.08) {
  const source = (controls || []).find((row) => row.control_id === controlID);
  if (!source || !Array.isArray(source.anchor)) return controlID;
  const points = controls.filter((row) => Array.isArray(row.anchor)).map((row) => row.anchor);
  const target = [2 * axisX - finite(source.anchor[0]), finite(source.anchor[1])];
  let best = source;
  let distance = Math.hypot(finite(source.anchor[0]) - target[0], finite(source.anchor[1]) - target[1]);
  for (const row of controls || []) {
    if (!Array.isArray(row.anchor)) continue;
    const candidate = Math.hypot(finite(row.anchor[0]) - target[0], finite(row.anchor[1]) - target[1]);
    if (candidate < distance) {
      distance = candidate;
      best = row;
    }
  }
  const tolerance = Math.max(EPSILON, boundsDiagonal(points) * Math.max(0, finite(toleranceRatio, 0.08)));
  return distance <= tolerance ? best.control_id : controlID;
}
