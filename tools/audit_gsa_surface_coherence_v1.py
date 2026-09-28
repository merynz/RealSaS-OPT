from __future__ import annotations

import json, subprocess
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    _adaptive_voxel_compact,
    _self_zbuffer_support,
)

ROOT=Path(__file__).resolve().parents[1]


def compact_same_component_fold():
    # Two locally parallel sheets belong to ONE connected component (connection
    # can be outside this local patch). Points are spatially close but represent
    # distinct surface loci. Precomputed component labels cannot distinguish them.
    p=np.asarray([
        [-0.01,-0.01,-0.010],
        [ 0.01,-0.01,-0.010],
        [ 0.00, 0.01,-0.010],
        [-0.01,-0.01, 0.010],
        [ 0.01,-0.01, 0.010],
        [ 0.00, 0.01, 0.010],
    ],dtype=np.float64)
    n=np.asarray([
        [0,0,-1],[0,0,-1],[0,0,-1],
        [0,0, 1],[0,0, 1],[0,0, 1],
    ],dtype=np.float64)
    faces=np.asarray([[0,1,2],[3,4,5]],dtype=np.int64)
    labels=np.zeros((len(p),),dtype=np.int64)  # same connected component
    cp,cn,edges,divisions,inverse=_adaptive_voxel_compact(
        p,faces,n,target_nodes=1,
        preserve_connected_components=True,
        precomputed_component_labels=labels,
    )
    return {
      "input_vertex_count":len(p),
      "compact_node_count":len(cp),
      "inverse":inverse.tolist(),
      "compact_points":cp.tolist(),
      "compact_normals":cn.tolist(),
      "divisions":int(divisions),
      "distinct_sheets_merged":bool(len(set(inverse[:3]))==1 and len(set(inverse[3:]))==1 and inverse[0]==inverse[3]),
      "mean_point_between_sheets":bool(abs(float(cp[0,2]))<1e-12) if len(cp)==1 else False,
    }


def vertex_splat_visibility():
    camera={
      "origin":(0.0,0.0,0.0),
      "right":(1.0,0.0,0.0),
      "screen_up":(0.0,1.0,0.0),
      "forward":(0.0,0.0,1.0),
      "half_extent":1.0,
      "resolution":128,
    }
    # Front triangle fully covers the center in the continuous surface, but only
    # its CORNERS enter the point z-buffer. Back point lies at the center.
    front=np.asarray([
      [-0.65,-0.65,1.0],
      [ 0.65,-0.65,1.0],
      [ 0.00, 0.65,1.0],
    ],dtype=np.float64)
    back=np.asarray([[0.0,0.0,2.0]],dtype=np.float64)
    dense=np.concatenate((front,back),axis=0)
    compact=back.copy()
    support,raster,counts=_self_zbuffer_support(
      dense,compact,(camera,)*8,depth_tolerance=1e-6
    )
    # Center is inside the front triangle: barycentric/geometric ground truth says occluded.
    return {
      "point_splat_marks_back_visible_all_views":bool(np.all(support[0])),
      "support":support[0].tolist(),
      "raster":raster[0].tolist(),
      "visible_counts":list(map(int,counts)),
      "continuous_front_triangle_contains_center":True,
      "continuous_surface_expected_back_visibility":False,
    }


fold=compact_same_component_fold()
vis=vertex_splat_visibility()
payload={
  "schema":"RealSaS.GSASurfaceCoherenceAdversaries.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "same_component_fold_compaction":fold,
  "vertex_splat_visibility":vis,
  "findings":[
    {
      "id":"GSA_COMPONENT_AWARE_VOXEL_COMPACTION_CAN_MERGE_GEODESICALLY_DISTINCT_SHEETS",
      "severity":"P0",
      "confirmed":bool(fold["distinct_sheets_merged"] and fold["mean_point_between_sheets"]),
      "class":"SURFACE_TOPOLOGY_COLLAPSE",
      "design_before_code":"Compare surface-sampling, cell-aware clustering, geodesic/topological clustering, and projection-back-to-surface variants under the fixed node budget. Connected-component labels alone are insufficient."
    },
    {
      "id":"GSA_VERTEX_SPLAT_ZBUFFER_CAN_MISCLASSIFY_OCCLUDED_SURFACE_AS_VISIBLE",
      "severity":"P1",
      "confirmed":bool(vis["point_splat_marks_back_visible_all_views"]),
      "class":"VISIBILITY_EVIDENCE_APPROXIMATION",
      "design_before_code":"Use a topology-aware surface visibility operator (e.g. exact admitted face raster/depth) or explicitly calibrate a conservative equivalent; do not infer source support from vertex splats."
    }
  ],
  "claim_boundary":"Synthetic adversaries isolate current GSA operators. They do not choose final point-budget, mesh, or visibility implementation."
}
out=ROOT/"canonical"/"GSA_SURFACE_COHERENCE_ADVERSARIES_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
if not all(f["confirmed"] for f in payload["findings"]):
    raise SystemExit(2)
