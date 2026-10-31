export function emptyStudioEdits() {
  return {
    rig: [],
    mesh: [],
    meshTopology: [],
    weights: [],
    bindings: [],
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

export function authoringLayerToEdits(layer) {
  if (!layer || typeof layer !== 'object') return emptyStudioEdits();
  return {
    rig: clone(layer.rig_pose_edits || []),
    mesh: clone(layer.mesh_vertex_edits || []),
    meshTopology: clone(layer.mesh_topology_edits || []),
    weights: clone(layer.weight_edits || []),
    bindings: clone(layer.binding_edits || []),
    topology: clone(layer.rig_topology_edits || []),
    ik: clone(layer.ik_constraints || []),
    animation: clone(layer.animation_edits || []),
  };
}

export function editsToAuthoringRequest(bundle, viewID, edits) {
  return {
    bundle,
    view_id: viewID,
    edits: clone(edits.rig || []),
    mesh_vertex_edits: clone(edits.mesh || []),
    mesh_topology_edits: clone(edits.meshTopology || []),
    weight_edits: clone(edits.weights || []),
    binding_edits: clone(edits.bindings || []),
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
