import json
from pathlib import Path
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.model_v1 import IRISVNextV1
from tools.iris_vnext.h_geo_p2_runtime_v1 import set_h_geo_p2_trainable_v1

ROOT=Path(__file__).resolve().parents[2]
def j(p): return json.loads((ROOT/p).read_text())

def test_p1_result_is_sealed_fail_without_tessa_authority():
    r=j("docs/research/IRIS_H_GEO_P1_RESULT_20261020.json")
    assert r["execution_source_sha"]=="4a4236db120ffec32dc33d278341e21f58d93529"
    assert r["status"]=="SCIENTIFIC_FAIL_H_GEO_P1_METRIC_CARRIER_RESCUE"
    assert r["apparatus_valid"] is True
    assert r["training"]["optimizer_steps"]==3600
    assert r["independent_court"]["gates"]["absolute"]=={"mean_distance_mae":False,"max_endpoint_p95":False,"min_normal":False,"min_near_ba":False}
    assert r["interpretation"]["surf_a2_authorized"] is False
    assert r["interpretation"]["tessa_authorized"] is False

def test_p2_reuses_exact_p1_truth_but_initializes_from_v8():
    b=j("canonical/H_GEO_P2_INPUT_BINDING_V1.json"); p=j("canonical/H_GEO_P2_PREREGISTRATION_V1.json")
    assert b["exact_triangle_dataset"]["receipt"]["sha256"]=="6c27974a03e9f202adeda1b25f642d95006dcca7cbbf0ceb660484ea91c78dc5"
    assert b["exact_triangle_dataset"]["reuse_bytes_exactly"] is True
    assert b["exact_triangle_dataset"]["rebuild_or_resample_allowed"] is False
    assert b["v8_initialization"]["checkpoint"]["sha256"]=="0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
    assert b["p1_comparison_only"]["initialization_allowed"] is False
    assert b["p1_comparison_only"]["checkpoint"]["sha256"]=="2f73a783391a16f09d12e50f240c0fbaaae40f0e044d1546bf12590eba0c5e5d"
    assert p["architecture"]["base_model"]=="IRIS_VNEXT_V1_EXACT_V8_INITIALIZATION"
    assert p["architecture"]["architecture_expansion_allowed"] is False
    assert p["architecture"]["trainable_existing_modules"]==["scene_core.scene","scene_core.field"]

def test_p2_trainable_census_is_scene_plus_field_body_only():
    model=IRISVNextV1(IRISVNextConfigV1()); scene,field,names=set_h_geo_p2_trainable_v1(model)
    assert scene and field
    assert any(n.startswith("scene_core.scene.") for n in names)
    assert any(n.startswith("scene_core.field.") for n in names)
    assert all(n.startswith(("scene_core.scene.","scene_core.field.input_norm.","scene_core.field.input_projection.","scene_core.field.blocks.","scene_core.field.feature_projection.","scene_core.field.final_norm.")) for n in names)
    assert not any("cube_extraction" in n or "view_bank" in n for n in names)

def test_p2_preregistered_court_and_determinism():
    p=j("canonical/H_GEO_P2_PREREGISTRATION_V1.json"); c=p["independent_court"]
    assert p["training"]["initialization_seed"]==27060001
    assert p["training"]["scientific_entrypoint"]=="tools.iris_vnext.train_fit8_h_geo_p2_seeded_v1"
    assert c["probe_initialization_seed"]==27079991
    assert c["scientific_entrypoint"]=="tools.iris_vnext.audit_fit8_h_geo_p2_seeded_v1"
    assert c["maximum_p2_mean_distance_mae_iris"]==0.025
    assert c["maximum_p2_subject_endpoint_p95_iris"]==0.075
    assert c["minimum_p2_subject_normal_alignment"]==0.70
    assert c["minimum_p2_subject_near_balanced_accuracy"]==0.80
    assert c["minimum_distance_mae_reduction_vs_p1"]==0.15
    assert c["minimum_endpoint_p95_reduction_vs_p1"]==0.15
    assert c["thresholds_may_change_after_results"] is False
    assert p["interpretation"]["pass_authorizes_surf_a2"] is False
    assert p["interpretation"]["pass_authorizes_tessa"] is False
    assert p["interpretation"]["fail_authorizes_unpreregistered_explicit_geometry_training"] is False
    train_seed=(ROOT/"tools/iris_vnext/train_fit8_h_geo_p2_seeded_v1.py").read_text()
    court_seed=(ROOT/"tools/iris_vnext/audit_fit8_h_geo_p2_seeded_v1.py").read_text()
    assert "INITIALIZATION_SEED = 27060001" in train_seed
    assert "PROBE_INITIALIZATION_SEED = 27079991" in court_seed

def test_p2_training_forward_recomputes_scene_planes_and_has_no_p1_init_input():
    text=(ROOT/"tools/iris_vnext/train_fit8_h_geo_p2_v1.py").read_text()
    assert 'planes=model.encode_scene(src["image_tokens"],src["token_xy"],src["camera_features"])' in text
    assert 'p.add_argument("--p1-checkpoint"' not in text
    assert 'torch.load(Path(args.p1_checkpoint)' not in text
    assert 'model.scene_core.scene.load_state_dict(ck["scene_state"]' in text
    assert '"p1_checkpoint_used_for_initialization":False' in text

def test_p2_runtime_does_not_create_live_sources_inside_inference_mode():
    text=(ROOT/"tools/iris_vnext/h_geo_p2_runtime_v1.py").read_text()
    assert "source=load_subject_source(shard,ex[\"sha256\"],device)" in text
    assert "with torch.inference_mode():\n                planes=model.encode_scene" in text
    assert "with torch.inference_mode():\n        for subject in SUBJECT_IDS" not in text
