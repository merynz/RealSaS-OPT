import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def J(path): return json.loads((ROOT/path).read_text())

def test_h_geo_p1_trigger_is_exact_h_rep_fail_and_v8_authority():
    b=J("canonical/H_GEO_P1_INPUT_BINDING_V1.json"); assert b["trigger_evidence"]["h_rep_p0_execution_source_sha"]=="479e8e53cf0d7df234a3e64b824bb704b861bf19"; assert b["trigger_evidence"]["h_rep_p0_status"]=="SCIENTIFIC_FAIL_H_REP_P0_LOCAL_GEOMETRY_DECODABILITY"; assert b["trigger_evidence"]["h_rep_p0_local_alignment_gates_all_passed"] is True; assert b["v8"]["checkpoint"]["sha256"]=="0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"; assert b["prepared"]["manifest"]["sha256"]=="6561bf2087f4fac3699c673b5c93fb3d8dbc2d1ff90a5065d976cae70688b4d8"; assert b["teachers"]["batch_receipt"]["sha256"]=="37b203f9c16e1a5dddbab1709ebb3806e486d66d1ed5b48a42817bc0044a7843"; assert len(b["teachers"]["subjects"])==8 and len(b["prepared"]["subjects"])==8; assert b["transport_policy"]["drive_tree_scan_forbidden"] is True; assert b["transport_policy"]["second_google_oauth_forbidden"] is True

def test_h_geo_p1_source_boundary_has_no_tessa_or_surface_extractor_dependency():
    paths=["models/iris/vnext/metric_geometry_v1.py","tools/iris_vnext/h_geo_p1_geometry_v1.py","tools/iris_vnext/h_geo_p1_runtime_v1.py","tools/iris_vnext/build_fit8_h_geo_p1_dataset_v1.py","tools/iris_vnext/preflight_h_geo_p1_a100_v1.py","tools/iris_vnext/train_fit8_h_geo_p1_v1.py","tools/iris_vnext/audit_fit8_h_geo_p1_v1.py"]; text="\n".join((ROOT/p).read_text() for p in paths)
    for token in ("models.tessa","PinnedFlexiCubesV1","FrozenHSurfaceRescueV1","decode_vnext_sparse_surface","materialize_fit8_direct_surface"): assert token not in text

def test_h_geo_p1_prereg_keeps_downstream_blocked():
    p=J("canonical/H_GEO_P1_PREREGISTRATION_V1.json"); assert p["scope"].endswith("NOT_SURF_A2__NOT_TESSA_AUTHORIZATION"); assert p["runtime_admission"]["optimizer_step_allowed_before_all_authority_gates"] is False; assert p["runtime_admission"]["drive_tree_discovery_allowed"] is False; assert p["interpretation"]["pass_authorizes_surf_a2"] is False; assert p["interpretation"]["pass_authorizes_tessa"] is False; assert p["interpretation"]["pass_makes_h_geo_p1_product_authority"] is False
