const EPSILON_AREA = 1e-8;

function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

function finitePoint(value, field = 'position_xy') {
  if (!Array.isArray(value) || value.length !== 2) throw new TypeError(`${field} must be [x,y]`);
  const point = value.map(Number);
  if (!point.every(Number.isFinite)) throw new TypeError(`${field} must be finite`);
  return point;
}

export function canonicalVertexID(index) {
  return `v:${Number(index)}`;
}

export function canonicalFaceID(index) {
  return `f:${Number(index)}`;
}

export function edgeKey(a, b) {
  return [String(a), String(b)].sort().join('|');
}

export function triangleArea2(a, b, c) {
  return (Number(b[0]) - Number(a[0])) * (Number(c[1]) - Number(a[1]))
    - (Number(b[1]) - Number(a[1])) * (Number(c[0]) - Number(a[0]));
}

export function baseMeshModel(mesh) {
  if (!mesh || typeof mesh !== 'object') throw new TypeError('mesh required');
  const vertices = new Map();
  const faces = new Map();
  (mesh.vertices || []).forEach((point, index) => {
    vertices.set(canonicalVertexID(index), {
      vertex_id: canonicalVertexID(index),
      position_xy: finitePoint(point, `vertices[${index}]`),
      canonical_index: index,
      user: false,
    });
  });
  (mesh.triangles || []).forEach((triangle, index) => {
    if (!Array.isArray(triangle) || triangle.length !== 3) throw new TypeError('triangulated mesh required');
    faces.set(canonicalFaceID(index), {
      face_id: canonicalFaceID(index),
      vertex_ids: triangle.map((vertexIndex) => canonicalVertexID(vertexIndex)),
      canonical_index: index,
      user: false,
    });
  });
  return {mesh_id: String(mesh.mesh_id || ''), vertices, faces};
}

function splitFace(face, a, b, newVertexID, operationIndex) {
  const ids = face.vertex_ids;
  const pairs = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];
  for (const [i, j, k] of pairs) {
    if ((ids[i] === a && ids[j] === b) || (ids[i] === b && ids[j] === a)) {
      const left = [ids[i], newVertexID, ids[k]];
      const right = [newVertexID, ids[j], ids[k]];
      return [
        {face_id: `user:split:${operationIndex}:${face.face_id}:0`, vertex_ids: left, user: true},
        {face_id: `user:split:${operationIndex}:${face.face_id}:1`, vertex_ids: right, user: true},
      ];
    }
  }
  return null;
}

function validateModel(model) {
  const duplicateFaces = new Set();
  const edges = new Map();
  for (const face of model.faces.values()) {
    const ids = face.vertex_ids || [];
    if (ids.length !== 3 || new Set(ids).size !== 3) throw new Error(`degenerate face: ${face.face_id}`);
    const points = ids.map((id) => model.vertices.get(id)?.position_xy);
    if (points.some((point) => !point)) throw new Error(`face references missing vertex: ${face.face_id}`);
    if (Math.abs(triangleArea2(points[0], points[1], points[2])) <= EPSILON_AREA) {
      throw new Error(`zero-area face: ${face.face_id}`);
    }
    const signature = [...ids].sort().join('|');
    if (duplicateFaces.has(signature)) throw new Error(`duplicate face: ${face.face_id}`);
    duplicateFaces.add(signature);
    [[ids[0], ids[1]], [ids[1], ids[2]], [ids[2], ids[0]]].forEach(([a, b]) => {
      const key = edgeKey(a, b);
      const count = (edges.get(key) || 0) + 1;
      if (count > 2) throw new Error(`non-manifold edge: ${key}`);
      edges.set(key, count);
    });
  }
  return model;
}

export function replayMeshTopology(mesh, operations = []) {
  const model = baseMeshModel(mesh);
  operations.forEach((raw, operationIndex) => {
    if (!raw || String(raw.mesh_id || '') !== model.mesh_id) return;
    const op = String(raw.operation || '').toLowerCase();
    if (op === 'add_vertex') {
      const vertexID = String(raw.vertex_id || '');
      if (!vertexID || model.vertices.has(vertexID)) throw new Error(`duplicate mesh vertex id: ${vertexID}`);
      model.vertices.set(vertexID, {vertex_id: vertexID, position_xy: finitePoint(raw.position_xy), canonical_index: null, user: true});
    } else if (op === 'move_vertex') {
      const vertexID = String(raw.vertex_id || '');
      const vertex = model.vertices.get(vertexID);
      if (!vertex) throw new Error(`move_vertex target missing: ${vertexID}`);
      vertex.position_xy = finitePoint(raw.position_xy);
    } else if (op === 'delete_vertex') {
      const vertexID = String(raw.vertex_id || '');
      if (!model.vertices.has(vertexID)) throw new Error(`delete_vertex target missing: ${vertexID}`);
      if ([...model.faces.values()].some((face) => face.vertex_ids.includes(vertexID))) throw new Error(`delete_vertex still referenced: ${vertexID}`);
      model.vertices.delete(vertexID);
    } else if (op === 'add_face') {
      const faceID = String(raw.face_id || '');
      const vertexIDs = (raw.vertex_ids || []).map(String);
      if (!faceID || model.faces.has(faceID)) throw new Error(`duplicate mesh face id: ${faceID}`);
      model.faces.set(faceID, {face_id: faceID, vertex_ids: vertexIDs, canonical_index: null, user: true});
    } else if (op === 'delete_face') {
      const faceID = String(raw.face_id || '');
      if (!model.faces.delete(faceID)) throw new Error(`delete_face target missing: ${faceID}`);
    } else if (op === 'split_edge') {
      const a = String(raw.vertex_a_id || '');
      const b = String(raw.vertex_b_id || '');
      const vertexID = String(raw.new_vertex_id || '');
      const va = model.vertices.get(a);
      const vb = model.vertices.get(b);
      if (!va || !vb || a === b) throw new Error('split_edge requires two existing distinct vertices');
      if (!vertexID || model.vertices.has(vertexID)) throw new Error(`duplicate mesh vertex id: ${vertexID}`);
      const incident = [...model.faces.values()].filter((face) => face.vertex_ids.includes(a) && face.vertex_ids.includes(b));
      if (incident.length < 1 || incident.length > 2) throw new Error(`split_edge requires manifold edge: ${edgeKey(a, b)}`);
      const t = raw.t == null ? 0.5 : Number(raw.t);
      if (!Number.isFinite(t) || t <= 0 || t >= 1) throw new Error('split_edge t must be inside (0,1)');
      model.vertices.set(vertexID, {
        vertex_id: vertexID,
        position_xy: [va.position_xy[0] * (1 - t) + vb.position_xy[0] * t, va.position_xy[1] * (1 - t) + vb.position_xy[1] * t],
        canonical_index: null,
        user: true,
        derived_from_edge: [a, b],
        edge_t: t,
      });
      for (const face of incident) {
        const replacements = splitFace(face, a, b, vertexID, operationIndex);
        if (!replacements) throw new Error(`split_edge orientation failure: ${face.face_id}`);
        model.faces.delete(face.face_id);
        replacements.forEach((replacement) => model.faces.set(replacement.face_id, replacement));
      }
    } else if (op === 'weld_vertices') {
      const keep = String(raw.keep_vertex_id || '');
      const remove = String(raw.remove_vertex_id || '');
      if (!model.vertices.has(keep) || !model.vertices.has(remove) || keep === remove) throw new Error('weld_vertices requires two existing distinct vertices');
      for (const [faceID, face] of [...model.faces]) {
        const next = face.vertex_ids.map((id) => id === remove ? keep : id);
        if (new Set(next).size < 3) model.faces.delete(faceID);
        else face.vertex_ids = next;
      }
      model.vertices.delete(remove);
    } else {
      throw new Error(`unsupported mesh topology operation: ${op}`);
    }
    validateModel(model);
  });
  return validateModel(model);
}

export function modelArrays(model) {
  const vertices = [...model.vertices.values()];
  const indexByID = new Map(vertices.map((vertex, index) => [vertex.vertex_id, index]));
  const faces = [...model.faces.values()];
  return {
    vertices,
    faces,
    points: vertices.map((vertex) => [...vertex.position_xy]),
    triangles: faces.map((face) => face.vertex_ids.map((id) => indexByID.get(id))),
    indexByID,
  };
}

export function meshEdges(model) {
  const result = new Map();
  for (const face of model.faces.values()) {
    const ids = face.vertex_ids;
    [[ids[0], ids[1]], [ids[1], ids[2]], [ids[2], ids[0]]].forEach(([a, b]) => {
      const key = edgeKey(a, b);
      if (!result.has(key)) result.set(key, {edge_id: key, vertex_ids: [a, b], face_ids: []});
      result.get(key).face_ids.push(face.face_id);
    });
  }
  return [...result.values()];
}

export function nextUserVertexID(model, prefix = 'user:v') {
  let index = 1;
  while (model.vertices.has(`${prefix}${index}`)) index += 1;
  return `${prefix}${index}`;
}

export function nextUserFaceID(model, prefix = 'user:f') {
  let index = 1;
  while (model.faces.has(`${prefix}${index}`)) index += 1;
  return `${prefix}${index}`;
}

export function upsertMoveVertex(operations, meshID, vertexID, positionXY) {
  const point = finitePoint(positionXY);
  for (let index = operations.length - 1; index >= 0; index -= 1) {
    const row = operations[index];
    if (row.mesh_id === meshID && row.operation === 'move_vertex' && row.vertex_id === vertexID) {
      row.position_xy = point;
      return row;
    }
  }
  const row = {operation: 'move_vertex', mesh_id: meshID, vertex_id: vertexID, position_xy: point};
  operations.push(row);
  return row;
}

export function cloneMeshTopology(value) {
  return clone(value || []);
}
