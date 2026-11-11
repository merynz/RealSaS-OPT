const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const pair = (value, fallback = [0, 0]) => Array.isArray(value) && value.length === 2
  ? [finite(value[0]), finite(value[1])]
  : [...fallback];

export function buildRigModel({controls = [], edges = [], rigEdits = [], topologyEdits = []} = {}) {
  const nodes = new Map();
  for (const control of controls) {
    const id = String(control.control_id || '');
    if (!id) continue;
    const baseAnchor = pair(control.anchor);
    const edit = rigEdits.find((row) => row.control_id === id);
    const translation = pair(edit?.translation_xy);
    nodes.set(id, {
      bone_id: id,
      canonical: true,
      user: false,
      deleted: false,
      base_anchor: baseAnchor,
      anchor: [baseAnchor[0] + translation[0], baseAnchor[1] + translation[1]],
      parent_id: null,
      source: control,
    });
  }

  for (const edge of edges) {
    if (edge.relation_type !== 'skeleton_hierarchy') continue;
    const child = nodes.get(String(edge.child_control_id || ''));
    const parentID = String(edge.parent_control_id || '');
    if (child && nodes.has(parentID)) child.parent_id = parentID;
  }

  for (const edit of topologyEdits) {
    const operation = String(edit.operation || '').toLowerCase();
    const boneID = String(edit.bone_id || '');
    if (!boneID) continue;
    if (operation === 'add_bone') {
      nodes.set(boneID, {
        bone_id: boneID,
        canonical: false,
        user: true,
        deleted: false,
        base_anchor: pair(edit.anchor_xy),
        anchor: pair(edit.anchor_xy),
        parent_id: edit.parent_id || null,
        source: edit,
      });
    } else if (operation === 'reparent') {
      const node = nodes.get(boneID);
      if (node) node.parent_id = edit.parent_id || null;
    } else if (operation === 'delete_bone') {
      const node = nodes.get(boneID);
      if (node) node.deleted = true;
    }
  }

  for (const [id, node] of [...nodes]) {
    if (node.deleted) nodes.delete(id);
  }
  for (const node of nodes.values()) {
    if (node.parent_id && !nodes.has(node.parent_id)) node.parent_id = null;
  }

  return {nodes};
}

export function hierarchyRows(model) {
  const nodes = model?.nodes || new Map();
  const children = new Map();
  for (const id of nodes.keys()) children.set(id, []);
  const roots = [];
  for (const node of nodes.values()) {
    if (node.parent_id && nodes.has(node.parent_id)) children.get(node.parent_id).push(node.bone_id);
    else roots.push(node.bone_id);
  }
  const sortIDs = (rows) => rows.sort((a, b) => String(a).localeCompare(String(b)));
  sortIDs(roots);
  for (const rows of children.values()) sortIDs(rows);

  const out = [];
  const visited = new Set();
  function walk(id, depth) {
    if (visited.has(id) || !nodes.has(id)) return;
    visited.add(id);
    out.push({...nodes.get(id), depth});
    for (const child of children.get(id) || []) walk(child, depth + 1);
  }
  for (const root of roots) walk(root, 0);
  for (const id of nodes.keys()) if (!visited.has(id)) walk(id, 0);
  return out;
}

export function canReparent(model, boneID, parentID) {
  const nodes = model?.nodes || new Map();
  if (!nodes.has(boneID)) return false;
  if (!parentID) return true;
  if (!nodes.has(parentID) || parentID === boneID) return false;
  let cursor = parentID;
  const seen = new Set();
  while (cursor && nodes.has(cursor)) {
    if (cursor === boneID) return false;
    if (seen.has(cursor)) return false;
    seen.add(cursor);
    cursor = nodes.get(cursor).parent_id;
  }
  return true;
}

export function upsertReparent(topologyEdits, model, boneID, parentID) {
  const parent = parentID || null;
  if (!canReparent(model, boneID, parent)) return false;
  const add = topologyEdits.find((row) => row.operation === 'add_bone' && row.bone_id === boneID);
  if (add) {
    add.parent_id = parent;
    return true;
  }
  for (let index = topologyEdits.length - 1; index >= 0; index -= 1) {
    const edit = topologyEdits[index];
    if (edit.operation === 'reparent' && edit.bone_id === boneID) {
      edit.parent_id = parent;
      return true;
    }
  }
  topologyEdits.push({operation: 'reparent', bone_id: boneID, parent_id: parent});
  return true;
}

export function setBoneAnchor({rigEdits, topologyEdits, model, boneID, anchor}) {
  const node = model?.nodes?.get(boneID);
  if (!node) return false;
  const target = pair(anchor, node.anchor);
  if (node.user) {
    const add = topologyEdits.find((row) => row.operation === 'add_bone' && row.bone_id === boneID);
    if (!add) return false;
    add.anchor_xy = target;
    return true;
  }
  let edit = rigEdits.find((row) => row.control_id === boneID);
  if (!edit) {
    edit = {control_id: boneID, rotation_degrees: 0, translation_xy: [0, 0]};
    rigEdits.push(edit);
  }
  edit.translation_xy = [target[0] - node.base_anchor[0], target[1] - node.base_anchor[1]];
  return true;
}

export function boneGeometry(model, boneID) {
  const node = model?.nodes?.get(boneID);
  if (!node) return null;
  const parent = node.parent_id ? model.nodes.get(node.parent_id) : null;
  if (!parent) return {length: 0, angle_degrees: 0, parent_id: null};
  const dx = node.anchor[0] - parent.anchor[0];
  const dy = node.anchor[1] - parent.anchor[1];
  return {
    length: Math.hypot(dx, dy),
    angle_degrees: Math.atan2(dy, dx) * 180 / Math.PI,
    parent_id: parent.bone_id,
  };
}

export function setBoneGeometry({rigEdits, topologyEdits, model, boneID, length, angleDegrees}) {
  const node = model?.nodes?.get(boneID);
  const parent = node?.parent_id ? model.nodes.get(node.parent_id) : null;
  if (!node || !parent) return false;
  const radius = Math.max(0, finite(length));
  const angle = finite(angleDegrees) * Math.PI / 180;
  return setBoneAnchor({
    rigEdits,
    topologyEdits,
    model,
    boneID,
    anchor: [parent.anchor[0] + Math.cos(angle) * radius, parent.anchor[1] + Math.sin(angle) * radius],
  });
}

export function nextUserBoneID(model, prefix = 'user:bone') {
  const nodes = model?.nodes || new Map();
  let index = 1;
  while (nodes.has(`${prefix}_${index}`)) index += 1;
  return `${prefix}_${index}`;
}

export function rigMirrorAxisX(model) {
  const xs = [...(model?.nodes?.values() || [])].map((node) => finite(node.anchor?.[0]));
  if (!xs.length) return 0;
  return (Math.min(...xs) + Math.max(...xs)) / 2;
}

export function mirroredBoneID(model, boneID, axisX = rigMirrorAxisX(model), toleranceRatio = 0.08) {
  const nodes = [...(model?.nodes?.values() || [])];
  const source = model?.nodes?.get(boneID);
  if (!source || !nodes.length) return boneID;
  const xs = nodes.map((node) => finite(node.anchor?.[0]));
  const ys = nodes.map((node) => finite(node.anchor?.[1]));
  const diagonal = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const target = [2 * axisX - source.anchor[0], source.anchor[1]];
  let best = source;
  let distance = Math.hypot(source.anchor[0] - target[0], source.anchor[1] - target[1]);
  for (const node of nodes) {
    const candidate = Math.hypot(node.anchor[0] - target[0], node.anchor[1] - target[1]);
    if (candidate < distance) {
      best = node;
      distance = candidate;
    }
  }
  return distance <= Math.max(1e-8, diagonal * Math.max(0, finite(toleranceRatio, 0.08))) ? best.bone_id : boneID;
}
