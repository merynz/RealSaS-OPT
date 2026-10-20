import json
from pathlib import Path
import torch
import torch.nn.functional as F

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.explicit_geometry_carrier_v1 import GeometryHighResolutionRefinerV1, continuous_xyz_features_v1

ROOT=Path(__file__).resolve().parents[2]
def j(path): return json.loads((ROOT/path).read_text())

def test_p2_result_is_sealed_fail_and_earns_explicit_carrier_study():
    r=j("docs/research/IRIS_H_GEO_P2_RESULT_20261020.json")
    assert r["execution_source_sha"]=="3a572989ab379975ed7cda5f0dd8b6f33de246b3"
    assert r["status"]=="SCIENTIFIC_FAIL_H_GEO_P2_EXISTING_ARCHITECTURE_METRIC_GEOMETRY"
    assert r["apparatus_valid"] is True
    assert r["training"]["optimizer_steps"]==4800
    assert r["independent_court"]["gates"]["absolute"]=={"mean_distance_mae":True,"max_endpoint_p95":False,"min_normal":True,"min_near_ba":True}
    assert r["independent_court"]["gates"]["vs_p1"]=={"distance":True,"endpoint":True,"normal":True,"near":True}
    assert r["interpretation"]["explicit_geometry_carrier_preregistration_justified"] is True
    assert r["interpretation"]["tessa_authorized"] is False

def test_p3_binding_reuses_truth_and_p2_is_comparison_only():
    b=j("canonical/H_GEO_P3_INPUT_BINDING_V1.json")
    assert b["exact_triangle_dataset"]["receipt"]["sha256"]=="6c27974a03e9f202adeda1b25f642d95006dcca7cbbf0ceb660484ea91c78dc5"
    assert b["exact_triangle_dataset"]["reuse_bytes_exactly"] is True
    assert b["exact_triangle_dataset"]["rebuild_or_resample_allowed"] is False
    assert b["v8_initialization"]["checkpoint"]["sha256"]=="0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
    assert b["p2_comparison_only"]["initialization_allowed"] is False
    assert b["p2_comparison_only"]["checkpoint"]["sha256"]=="0bdcd24af776e322bfbb1a004d6034167770b7950ecccb9b04d6d689597854ee"

def test_p3_preregistration_locks_explicit_hires_boundary():
    p=j("canonical/H_GEO_P3_PREREGISTRATION_V1.json"); c=p["independent_court"]
    assert p["architecture"]["architecture_expansion_allowed"] is True
    assert p["architecture"]["p2_checkpoint_initialization_allowed"] is False
    assert p["architecture"]["base_geometry_plane_resolution"]==64
    assert p["architecture"]["high_resolution_geometry_plane_resolution"]==128
    assert p["training"]["carrier_steps"]==4800
    assert p["training"]["initialization_seed"]==27090001
    assert p["training"]["scientific_entrypoint"]=="tools.iris_vnext.train_fit8_h_geo_p3_seeded_v1"
    assert c["probe_initialization_seed"]==27099991
    assert c["maximum_p3_mean_distance_mae_iris"]==0.025
    assert c["maximum_p3_subject_endpoint_p95_iris"]==0.075
    assert c["minimum_endpoint_p95_reduction_vs_p2"]==0.10
    assert c["thresholds_may_change_after_results"] is False
    assert p["interpretation"]["pass_authorizes_tessa"] is False
    assert p["interpretation"]["pass_authorizes_surf_a2"] is False

def test_hires_refiner_starts_as_exact_bilinear_identity_and_xyz_is_21d():
    cfg=IRISVNextConfigV1(); refiner=GeometryHighResolutionRefinerV1(cfg); x=torch.randn(1,3,cfg.plane_channels,8,8); y=refiner(x); expected=F.interpolate(x.reshape(3,cfg.plane_channels,8,8),scale_factor=2.0,mode="bilinear",align_corners=False).reshape_as(y)
    assert y.shape[-2:]==(16,16)
    assert torch.equal(y,expected)
    assert continuous_xyz_features_v1(torch.zeros(2,7,3)).shape==(2,7,21)

def test_p3_training_has_no_p2_initialization_and_shared_v8_is_not_optimized():
    text=(ROOT/"tools/iris_vnext/train_fit8_h_geo_p3_v1.py").read_text()
    assert 'p.add_argument("--p2-checkpoint"' not in text
    assert "p2_checkpoint_used_for_initialization" in text
    assert "for p in v8.parameters(): p.requires_grad_(False)" in text
    assert '"carrier_state":carrier.state_dict()' in text
    assert '"scene_state"' not in text
    audit=(ROOT/"tools/iris_vnext/audit_fit8_h_geo_p3_v1.py").read_text()
    assert 'p.add_argument("--p2-checkpoint",required=True)' in audit

def test_p3_seeded_entrypoints_are_locked():
    train_seed=(ROOT/"tools/iris_vnext/train_fit8_h_geo_p3_seeded_v1.py").read_text(); court_seed=(ROOT/"tools/iris_vnext/audit_fit8_h_geo_p3_seeded_v1.py").read_text()
    assert "INITIALIZATION_SEED = 27090001" in train_seed
    assert "PROBE_INITIALIZATION_SEED = 27099991" in court_seed
