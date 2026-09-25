import math
import numpy as np
import pytest

from compiler.realsas_compiler_core.substrate.adequacy_v1 import (
    _eligible_dense_components,
    select_adequate_rigging_surface_v1,
    substrate_adequacy_report_hash_v1,
)
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    _adaptive_voxel_compact,
    mesh_connected_component_labels_v1,
)


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


def test_visible_tiny_component_can_be_forced_eligible_independent_of_dense_fraction():
    labels=np.concatenate((np.zeros(4000,dtype=np.int64),np.ones(2,dtype=np.int64)))
    support=np.zeros((len(labels),8),dtype=bool)
    support[:,0]=True

    historical,by_fraction,visible=_eligible_dense_components(
        labels,support,component_min_dense_fraction=0.001,
        visible_component_always_eligible=False,
    )
    assert historical=={0}
    assert by_fraction=={0}
    assert visible=={0,1}

    hardened,by_fraction2,visible2=_eligible_dense_components(
        labels,support,component_min_dense_fraction=0.001,
        visible_component_always_eligible=True,
    )
    assert hardened=={0,1}
    assert by_fraction2=={0}
    assert visible2=={0,1}


def test_unobserved_tiny_component_still_respects_dense_fraction_floor():
    labels=np.concatenate((np.zeros(4000,dtype=np.int64),np.ones(2,dtype=np.int64)))
    support=np.zeros((len(labels),8),dtype=bool)
    support[:4000,0]=True

    eligible,by_fraction,visible=_eligible_dense_components(
        labels,support,component_min_dense_fraction=0.001,
        visible_component_always_eligible=True,
    )
    assert eligible=={0}
    assert by_fraction=={0}
    assert visible=={0}


def test_component_aware_voxel_compaction_prevents_cross_component_cluster_alias():
    # Two disconnected, spatially coincident rings intentionally stress the voxel key.
    count=40
    theta=np.linspace(0.0,2.0*math.pi,count,endpoint=False)
    ring=np.column_stack((0.25*np.cos(theta),0.25*np.sin(theta),np.zeros(count)))
    points=np.concatenate((ring,ring),axis=0)
    normals=np.tile(np.asarray([[0.0,0.0,1.0]]),(len(points),1))
    faces=[]
    for offset in (0,count):
        center=offset
        # fan-like local triangles over ring indices; enough to establish two disjoint components
        for i in range(1,count-1):
            faces.append((offset,offset+i,offset+i+1))
    faces=np.asarray(faces,dtype=np.int64)
    dense_component=np.concatenate((np.zeros(count,dtype=np.int64),np.ones(count,dtype=np.int64)))

    _,_,_,_,baseline_inverse=_adaptive_voxel_compact(
        points,faces,normals,target_nodes=64,preserve_connected_components=False,
    )
    _,_,_,_,aware_inverse=_adaptive_voxel_compact(
        points,faces,normals,target_nodes=64,preserve_connected_components=True,
    )

    def mixed_cluster_count(inverse):
        owners={}
        for i,cluster in enumerate(inverse):
            owners.setdefault(int(cluster),set()).add(int(dense_component[i]))
        return sum(len(v)>1 for v in owners.values())

    assert mixed_cluster_count(baseline_inverse)>0
    assert mixed_cluster_count(aware_inverse)==0


def test_component_aware_compaction_precomputed_labels_are_exactly_equivalent():
    count=48
    theta=np.linspace(0.0,2.0*math.pi,count,endpoint=False)
    ring=np.column_stack((0.3*np.cos(theta),0.3*np.sin(theta),np.zeros(count)))
    points=np.concatenate((ring,ring+np.asarray([0.0,0.0,0.4])),axis=0)
    normals=np.tile(np.asarray([[0.0,0.0,1.0]]),(len(points),1))
    faces=[]
    for offset in (0,count):
        for i in range(1,count-1):
            faces.append((offset,offset+i,offset+i+1))
    faces=np.asarray(faces,dtype=np.int64)
    labels=mesh_connected_component_labels_v1(len(points),faces,face_chunk_size=11)

    auto=_adaptive_voxel_compact(
        points,faces,normals,target_nodes=64,preserve_connected_components=True,
    )
    cached=_adaptive_voxel_compact(
        points,faces,normals,target_nodes=64,preserve_connected_components=True,
        precomputed_component_labels=labels,
    )
    for a,b in zip(auto[:4],cached[:4]):
        if isinstance(a,np.ndarray):
            assert np.array_equal(a,b)
        else:
            assert a==b
    assert np.array_equal(auto[4],cached[4])


def test_failed_selector_can_return_demo_evidence_without_mutating_product_status():
    mesh=_mesh()
    p=_policy()
    p["max_dense_to_surface_max_norm"]=0.0
    surface,report=select_adequate_rigging_surface_v1(
        mesh.vertices_normalized,mesh.faces,mesh.normals,_cams(),
        normalization_center=(0,0,0),normalization_half_extent=1.0,authority_label="TEST",
        source_run_id="RUN",source_checkpoint_sha256="a"*64,source_zero_surface_sha256="b"*64,
        normal_k=24,visibility_depth_tolerance_norm=0.03,adequacy_policy=p,
        return_closest_nonpassing_evidence=True,
    )
    assert surface is not None
    assert report["status"]=="FAIL"
    assert report["selected_target_node_cap"] is None
    assert report["selected_actual_node_count"] is None
    assert report["selected_surface_lineage_hash"]==""
    closest=report["diagnostic_closest_nonpassing_candidate"]
    assert closest is not None
    assert closest["surface_lineage_hash"]==surface.geometry_lineage_hash
    assert closest["actual_node_count"]==len(surface.surface_nodes)
    assert closest["violations"]["finite"] is True
    assert report["adequacy_report_hash"]==substrate_adequacy_report_hash_v1(report)
