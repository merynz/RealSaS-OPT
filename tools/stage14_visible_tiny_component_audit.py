from __future__ import annotations
import json, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from compiler.realsas_compiler_core.substrate.adequacy_v1 import _metrics

OUT=Path(os.environ.get("REALSAS_STAGE14_COMPONENT_OUT","stage14_component_out")); OUT.mkdir(parents=True,exist_ok=True)

policy={
 "min_candidate_nodes":128,"max_candidate_nodes":8192,"candidate_growth_factor":2.0,"refinement_rounds":3,
 "max_dense_to_surface_p95_norm":0.03,"max_dense_to_surface_max_norm":0.08,
 "max_normal_p95_deg":30.0,"max_projected_p95_px":6.0,"max_projected_max_px":16.0,
 "component_min_dense_fraction":0.001,"min_nodes_per_component":8,"max_component_alias_nodes":0,
}

# Eight compact nodes cover a large component exactly.
node_positions=np.asarray([
 [-.03,-.03,0],[-.03,.03,0],[.03,-.03,0],[.03,.03,0],
 [-.015,-.015,0],[-.015,.015,0],[.015,-.015,0],[.015,.015,0],
],dtype=float)
nodes=[]
for i,p in enumerate(node_positions):
    px=512.0+p[0]*512.0; py=512.0+p[1]*512.0
    nodes.append(SimpleNamespace(
      P=tuple(p),derived_normal=(0,0,1),support_views=(0,),raster_bindings=((0,(px,py)),)
    ))
surface=SimpleNamespace(surface_nodes=tuple(nodes),local_relations=())

# Main component: 4000 dense samples exactly on those nodes.
dense_main=np.repeat(node_positions,500,axis=0)
# Tiny disconnected but visible component = 2/4002 = 0.00049975 < frozen 0.001 floor.
# It is only 0.02 normalized units (~10.24 px) from a represented main node, so
# global spatial/projected max tolerances do not force a dedicated node.
tiny=np.asarray([[.05,.03,0],[.05,.031,0]],dtype=float)
dense_world=np.concatenate((dense_main,tiny),axis=0)
dense_normals=np.tile(np.asarray([[0,0,1.0]]),(len(dense_world),1))
dense_labels=np.concatenate((np.zeros(len(dense_main),dtype=np.int64),np.ones(len(tiny),dtype=np.int64)))

dense_support=np.zeros((len(dense_world),8),dtype=bool); dense_support[:,0]=True
dense_raster=np.zeros((len(dense_world),8,2),dtype=float)
dense_raster[:len(dense_main),0,0]=512.0+dense_main[:,0]*512.0
dense_raster[:len(dense_main),0,1]=512.0+dense_main[:,1]*512.0
dense_raster[len(dense_main):,0,0]=512.0+tiny[:,0]*512.0
dense_raster[len(dense_main):,0,1]=512.0+tiny[:,1]*512.0

m=_metrics(
 surface=surface,dense_world=dense_world,dense_normals=dense_normals,dense_labels=dense_labels,
 dense_support=dense_support,dense_raster=dense_raster,normalization_half_extent=1.0,policy=policy,
)
out={
 "schema":"RealSaS.Stage14VisibleTinyComponentEligibilityAudit.v1",
 "status":"FAIL_CURRENT_POLICY_VISIBLE_COMPONENT_ESCAPE" if m["passed"] else "PASS_CURRENT_POLICY_REJECTS_ESCAPE",
 "subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,
 "current_policy":policy,
 "tiny_component_dense_fraction":float(len(tiny)/len(dense_world)),
 "tiny_component_visible_in_view0":True,
 "dedicated_surface_nodes_for_tiny_component":0,
 "metrics":m,
 "expected_product_semantics":"VISIBLE_DISCONNECTED_COMPONENT_MUST_NOT_BE_EXEMPT_BY_DENSE_VERTEX_FRACTION",
}
(OUT/"STAGE14_VISIBLE_TINY_COMPONENT_ELIGIBILITY_AUDIT.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
# The experiment succeeds when it demonstrates the loophole.
raise SystemExit(0 if out["status"]=="FAIL_CURRENT_POLICY_VISIBLE_COMPONENT_ESCAPE" else 2)
