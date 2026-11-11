const DOMAIN_ORDER = ['MESH', 'RIG', 'SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF'];

const INVALIDATION = {
  rig: ['RIG', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF'],
  mesh: ['MESH', 'SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF'],
  meshTopology: ['MESH', 'SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF'],
  weights: ['SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF'],
  bindings: ['MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF'],
  topology: ['RIG', 'SKIN', 'MECHANICS', 'GRAPHICS', 'MOTION', 'PROOF'],
  ik: ['MECHANICS', 'MOTION', 'PROOF'],
  animation: ['MOTION', 'PROOF'],
};

function nonEmpty(value) {
  return Array.isArray(value) && value.length > 0;
}

function presentationDeparted(value) {
  if (!value || typeof value !== 'object') return false;
  if (value.graphics_evidence_departure === true) return true;
  if (Array.isArray(value.hidden_mesh_ids) && value.hidden_mesh_ids.length) return true;
  const source = value.source_graphics_order_back_to_front;
  const order = value.slot_order_back_to_front;
  return Array.isArray(source) && Array.isArray(order)
    && (source.length !== order.length || source.some((id, index) => id !== order[index]));
}

export function invalidatedDomains(edits = {}) {
  const set = new Set();
  for (const [field, domains] of Object.entries(INVALIDATION)) {
    if (!nonEmpty(edits[field])) continue;
    domains.forEach((domain) => set.add(domain));
  }
  if (presentationDeparted(edits.presentation)) {
    set.add('GRAPHICS');
    set.add('PROOF');
  }
  return DOMAIN_ORDER.filter((domain) => set.has(domain));
}

export function editCounts(edits = {}) {
  return {
    rig: edits.rig?.length || 0,
    mesh: edits.mesh?.length || 0,
    mesh_topology: edits.meshTopology?.length || 0,
    weights: edits.weights?.length || 0,
    bindings: edits.bindings?.length || 0,
    rig_topology: edits.topology?.length || 0,
    ik: edits.ik?.length || 0,
    animation: edits.animation?.length || 0,
    presentation: presentationDeparted(edits.presentation) ? 1 : 0,
  };
}

export function requalificationPlan(edits = {}, {executorBound = false} = {}) {
  const domains = invalidatedDomains(edits);
  const required = domains.length > 0;
  return {
    schema_version: 'RealSaS.StudioRequalificationPlan.v1',
    required,
    invalidated_domains: domains,
    edit_counts: editCounts(edits),
    executor_bound: Boolean(executorBound),
    state: !required ? 'QUALIFIED_BASE' : executorBound ? 'READY_TO_REQUEST' : 'BLOCKED_EXECUTOR_UNBOUND',
    can_export_edited_runtime: false,
    qualified_base_export_remains_available: true,
    canonical_product_mutated: false,
  };
}
