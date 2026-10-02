from __future__ import annotations

"""Generic local cavity retriangulation for residual static mesh-quality defects.

The operator removes an interior vertex whose one-ring is a topological disk and
retriangulates the fixed one-ring boundary using only surviving vertices.  It:
- requires edge + vertex-link manifold input,
- preserves the cavity boundary and all surviving vertex positions/supports,
- forbids internal chords already used outside the cavity,
- chooses the lexicographically best triangulation under frozen G3,
- requires strict local quality improvement with no companion-metric regression,
- requires sampled symmetric G1 surface deviation within the frozen budget,
- batches only pairwise-disjoint cavities,
- rechecks full manifoldness and global quality after every batch.

No product provenance claim is minted here: a Stage14 mechanical-remesh authority
must explicitly bind any retriangulated faces to source-surface support before
promotion.
"""

import functools
from collections import defaultdict
from dataclasses import replace

import numpy as np

from .canonical_mesh_quality_repair_v1 import (
    _edge,_edge_incidence,_incident_faces_by_vertex,_metric,_violates,
    _local_scale,_sampled_symmetric_local_deviation,_report,
)
from .canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from .hashing import content_sha256
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,MeshQualificationPolicyIR,
    canonical_mesh_candidate_lineage_hash,
)
from .types import QualificationError


def _vertex_link_cycle(vertex_id: str, incident_faces):
    v=str(vertex_id)
    graph=defaultdict(set)
    for face in incident_faces:
        row=[str(x) for x in face]
        if v not in row:
            continue
        others=[x for x in row if x!=v]
        if len(others)!=2:
            return None
        a,b=others
        graph[a].add(b);graph[b].add(a)
    if len(graph)<3 or any(len(nbs)!=2 for nbs in graph.values()):
        return None
    start=min(graph)
    cycle=[start]
    prev=None
    cur=start
    while True:
        nbs=sorted(graph[cur])
        nxt=nbs[0] if nbs[0]!=prev else nbs[1]
        if nxt==start:
            break
        if nxt in cycle:
            return None
        cycle.append(nxt)
        prev,cur=cur,nxt
        if len(cycle)>len(graph):
            return None
    return tuple(cycle) if len(cycle)==len(graph) else None


def _patch_quality(faces,positions,policy):
    metrics=tuple(_metric(face,positions) for face in faces)
    if not metrics:
        return None
    return {
        "violation_count":sum(_violates(m,policy) for m in metrics),
        "min_angle_deg":min(float(m["min_angle_deg"]) for m in metrics),
        "max_aspect":max(float(m["aspect_longest_over_min_altitude"]) for m in metrics),