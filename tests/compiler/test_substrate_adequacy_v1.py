import math
import numpy as np
import pytest

from compiler.realsas_compiler_core.substrate.adequacy_v1 import (
    select_adequate_rigging_surface_v1,
    substrate_adequacy_report_hash_v1,
)
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3


def _mesh():
    pytest.importorskip("skimage")
    a=np.linspace(-1.0,1.0,26,dtype=np.float32)
    z,y,x=np.meshgrid(a,a,a,indexing="ij")
    field=np.sqrt(x*x+y*y+z*z)-0.62
    return extract_zero_surface_mesh_v3(field)


def _cams(res=96):
    rows=[]
    for view in range(8):
        yaw=math.radians(45.0*view)
        forward=np.asarray([-math.sin(yaw),-math.cos(yaw),0.0])
        rows.append({
            "view_index":view,"origin":(-4.0*forward).tolist(),
            "right":[-math.cos(yaw),math.sin(yaw),0.0],
            "screen_up":[0.0,0.0,1.0],"forward":forward.tolist(),
            "half_extent":1.05,"resolution":res,
        })
    return rows


def _policy():
    return {
        "min_candidate_nodes":128,"max_candidate_nodes":512,"candidate_growth_factor":2.0,"refinement_rounds":2,
        "max_dense_to_surface_p95_norm":0.20,"max_dense_to_surface_max_norm":0.30,
        "max_normal_p95_deg":45.0,"max_projected_p95_px":12.0,"max_projected_max_px":24.0,
        "component_min_dense_fraction":0.01,"min_nodes_per_component":8,"max_component_alias_nodes":0,
    }


def test_substrate_adequacy_selects_deterministic_passing_n():
    mesh=_mesh()
    kwargs=dict(
        vertices_normalized=mesh.vertices_normalized,faces=mesh.faces,implicit_normals=mesh.normals,cameras=_cams(),
        normalization_center=(0,0,0),normalization_half_extent=1.0,authority_label="TEST",
        source_run_id="RUN",source_checkpoint_sha256="a"*64,source_zero_surface_sha256="b"*64,
        normal_k=24,visibility_depth_tolerance_norm=0.03,adequacy_policy=_policy(),
    )
    a,ra=select_adequate_rigging_surface_v1(**kwargs)
    b,rb=select_adequate_rigging_surface_v1(**kwargs)
    assert a is not None and b is not None
    assert a.geometry_lineage_hash==b.geometry_lineage_hash
    assert ra["adequacy_report_hash"]==substrate_adequacy_report_hash_v1(ra)
    assert ra["adequacy_report_hash"]==rb["adequacy_report_hash"]
    assert ra["selected_actual_node_count"]==len(a.surface_nodes)
    assert 64<=len(a.surface_nodes)<=512
    assert ra["status"]=="PASS"


def test_substrate_adequacy_fails_closed_when_no_candidate_meets_policy():
    mesh=_mesh()
    p=_policy()
    p["max_dense_to_surface_max_norm"]=0.0
    surface,report=select_adequate_rigging_surface_v1(
        mesh.vertices_normalized,mesh.faces,mesh.normals,_cams(),
        normalization_center=(0,0,0),normalization_half_extent=1.0,authority_label="TEST",
        source_run_id="RUN",source_checkpoint_sha256="a"*64,source_zero_surface_sha256="b"*64,
        normal_k=24,visibility_depth_tolerance_norm=0.03,adequacy_policy=p,
    )
    assert surface is None
    assert report["status"]=="FAIL"
    assert report["adequacy_report_hash"]==substrate_adequacy_report_hash_v1(report)
