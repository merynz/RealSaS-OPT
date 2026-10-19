import json
from pathlib import Path
import numpy as np
import torch

from models.iris.vnext.metric_geometry_v1 import IRISMetricGeometryAuxHeadV1, metric_geometry_objective_v1
from tools.iris_vnext.h_geo_p1_geometry_v1 import ExactTriangleBVHV1, _closest_points_to_triangles, recover_prepared_normalization_v1, sample_surface_exact_replay_v1


def test_metric_geometry_aux_shapes_and_gradients():
    model=IRISMetricGeometryAuxHeadV1(); h=torch.randn(19,384); out=model(h)
    assert out["distance_iris"].shape==(19,)
    assert out["offset_to_surface_iris"].shape==(19,3)
    assert out["normal_unit"].shape==(19,3)
    assert out["near_logit"].shape==(19,)
    d=torch.rand(19)*0.1; n=torch.nn.functional.normalize(torch.randn(19,3),dim=-1)
    loss=metric_geometry_objective_v1(out,distance_iris=d,offset_to_surface_iris=torch.randn(19,3)*0.02,normal_unit=n,normal_valid=torch.ones(19,dtype=torch.bool),near_label=d<=0.03)
    total=sum(loss.values()); assert torch.isfinite(total); total.backward(); assert any(p.grad is not None for p in model.parameters())


def test_exact_bvh_matches_all_triangle_bruteforce():
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1]],dtype=np.float64); faces=np.array([[0,1,2],[1,3,2],[0,1,4],[0,4,2]],dtype=np.int64); bvh=ExactTriangleBVHV1(vertices,faces,leaf_size=1); rng=np.random.default_rng(11)
    for p in rng.uniform(-.5,1.5,size=(50,3)):
        q,_,ds=bvh.nearest_one(p); allq=_closest_points_to_triangles(p,bvh.tri); alld=((allq-p)**2).sum(1); j=int(alld.argmin()); assert abs(ds-float(alld[j]))<1e-12; assert abs(float(((q-p)**2).sum())-float(alld[j]))<1e-12


def test_normalization_replay_recovers_exact_isotropic_transform():
    vertices=np.array([[0.,0.,0.],[2.,0.,0.],[0.,3.,0.],[0.,0.,4.]],dtype=np.float64); faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],dtype=np.int64); source,_,_,_=sample_surface_exact_replay_v1(vertices,faces,4096,seed_text="Snake"); scale=.17; translation=np.array([.12,-.23,.08]); stored=source*scale+translation; r=recover_prepared_normalization_v1(subject_id="Snake",rest_vertices_source_world=vertices,faces=faces,stored_surface_points_iris=stored,maximum_error_iris=1e-10); assert abs(r.scale-scale)<1e-12; assert np.max(np.abs(np.asarray(r.translation)-translation))<1e-12; assert r.maximum_replay_error_iris<1e-12


def test_prereg_is_minimal_existing_carrier_retraining_and_tessa_blocked():
    p=json.loads(Path("canonical/H_GEO_P1_PREREGISTRATION_V1.json").read_text()); assert p["architecture"]["architecture_expansion_allowed"] is False; assert p["architecture"]["trainable_existing_module"]=="scene_core.field"; assert p["architecture"]["training_only_metric_geometry_head"]=="RealSaS.IRISMetricGeometryAuxHead.v1"; assert p["architecture"]["field_reinterpreted_as_signed_distance"] is False; assert p["truth"]["point_cloud_proxy_used_for_primary_truth"] is False; assert p["truth"]["signed_distance_claimed"] is False; assert p["interpretation"]["pass_authorizes_surf_a2"] is False; assert p["interpretation"]["pass_authorizes_tessa"] is False; assert p["independent_court"]["thresholds_may_change_after_results"] is False
