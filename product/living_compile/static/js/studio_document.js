export function emptyStudioEdits() {
  return {
    rig: [],
    mesh: [],
    meshTopology: [],
    weights: [],
    bindings: [],
    presentation: null,
    topology: [],
    ik: [],
    animation: [],
  };
}

function clone(value) {
  return typeof structuredClone === 'function'
    ? structuredClone(value)
    : JSON.parse(JSON.stringify(value));
}

function scopedRows(rows, viewID) {
  return (rows || []).filter((row) => !row?.view_id || row.view_id === viewID);
}

export function authoringLayerToEdits(layer) {
  if (!layer || typeof layer !== 'object') return emptyStudioEdits();
  const layerViewID = layer.view_id ? String(layer.view_id) : null;
  const bindings = clone(layer.binding_edits || []).map((row) => ({
    ...row,
    view_id: row.view_id || layerViewID,
  }));
  const presentation = layer.presentation_override
    ? {...clone(layer.presentation_override), view_id: layer.presentation_override.view_id || layerViewID}
    : null;
  return {
    rig: clone(layer.rig_pose_edits || []),
    mesh: clone(layer.mesh_vertex_edits || []),
    meshTopology: clone(layer.mesh_topology_edits || []),
    weights: clone(layer.weight_edits || []),
    bindings,
    presentation,
    topology: clone(layer.rig_topology_edits || []),
    ik: clone(layer.ik_constraints || []),
    animation: clone(layer.animation_edits || []),
  };
}

export function editsToAuthoringRequest(bundle, viewID, edits) {
  const presentation = edits.presentation && (!edits.presentation.view_id || edits.presentation.view_id === viewID)
    ? clone(edits.presentation)
    : null;
  if (presentation) delete presentation.view_id;
  const bindings = clone(scopedRows(edits.bindings, viewID)).map((row) => {
    delete row.view_id;
    return row;
  });
  return {
    bundle,
    view_id: viewID,
    edits: clone(edits.rig || []),
    mesh_vertex_edits: clone(edits.mesh || []),
    mesh_topology_edits: clone(edits.meshTopology || []),
    weight_edits: clone(edits.weights || []),
    binding_edits: bindings,
    presentation_override: presentation,
    rig_topology_edits: clone(edits.topology || []),
    ik_constraints: clone(edits.ik || []),
    animation_edits: clone(edits.animation || []),
  };
}

export class StudioDocument {
  constructor() {
    this.edits = emptyStudioEdits();
    this.persistedSnapshot = this.snapshot();
  }

  snapshot() {
    return clone(this.edits);
  }

  replace(snapshot) {
    const next = {...emptyStudioEdits(), ...(snapshot || {})};
    this.edits = clone(next);
    return this.edits;
  }

  loadAuthoringLayer(layer) {
    this.replace(authoringLayerToEdits(layer));
    this.persistedSnapshot = this.snapshot();
    return this.edits;
  }

  reset() {
    this.replace(emptyStudioEdits());
    this.persistedSnapshot = this.snapshot();
    return this.edits;
  }

  markPersisted() {
    this.persistedSnapshot = this.snapshot();
  }

  differsFromPersisted() {
    return JSON.stringify(this.edits) !== JSON.stringify(this.persistedSnapshot);
  }

  request(bundle, viewID) {
    return editsToAuthoringRequest(bundle, viewID, this.edits);
  }
}
