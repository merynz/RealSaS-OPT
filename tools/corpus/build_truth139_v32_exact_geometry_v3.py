#!/usr/bin/env python3
"""Fast V2 geometry entrypoint with component-wise closed-solid containment.

Semantic outputs remain RealSaS.Truth139V32ExactGeometry.v2. This runtime
replaces V2's exhaustive three-ray pass with component-wise trimesh containment,
while retaining a deterministic three-direction direct-parity probe per component.
The independent V2 court still validates sampled sign rows using solid angle.
"""
from __future__ import annotations
import numpy as np
import trimesh

from tools.corpus import build_truth139_v32_exact_geometry_v2 as impl


def component_contains_fields(points, topology, closest_face):
    points=np.asarray(points,dtype=np.float64)
    closest_component=topology.face_component_id[np.asarray(closest_face,dtype=np.int64)]
    nearest_inside=np.zeros(len(points),dtype=np.bool_)
    union_inside=np.zeros(len(points),dtype=np.bool_)
    nearest_valid=np.zeros(len(points),dtype=np.bool_)
    all_components_ok=True
    probe=np.unique(np.linspace(0,len(points)-1,min(192,len(points)),dtype=np.int64))
    directions=(np.asarray([1.,.173,.319]),np.asarray([.271,1.,.113]),np.asarray([.431,.719,1.]))
    for component in np.unique(topology.face_component_id):
        face_ids=np.flatnonzero(topology.face_component_id==component)
        component_faces=topology.topology_faces[face_ids]
        mesh=trimesh.Trimesh(vertices=topology.topology_vertices,faces=component_faces,process=False)
        inside=np.asarray(mesh.contains(points),dtype=np.bool_)
        union_inside |= inside
        mask=closest_component==component
        nearest_inside[mask]=inside[mask]
        votes=np.stack([impl.ray_parity(points[probe],topology.topology_vertices,component_faces,d) for d in directions],axis=1)
        three_ray_agreement=np.mean(np.all(votes==votes[:,:1],axis=1))
        contains_agreement=np.mean(votes[:,0]==inside[probe])
        probe_ok=bool(three_ray_agreement>=.99 and contains_agreement>=.99)
        nearest_valid[mask]=probe_ok
        all_components_ok=all_components_ok and probe_ok
    union_valid=np.full(len(points),all_components_ok,dtype=np.bool_)
    return closest_component.astype(np.int32),nearest_valid,nearest_inside,union_valid,union_inside


impl.component_parity_fields=component_contains_fields

if __name__=="__main__":
    impl.main()
