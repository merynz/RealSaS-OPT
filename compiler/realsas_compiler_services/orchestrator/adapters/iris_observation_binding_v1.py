"""Bind independent IRIS observation proof to actual Stage01/03/05/07 inputs."""
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import load_file_ref

SHARED_PROFILE="SHARED_IRIS_IMAGE_DERIVED_FSTAR_V1"
SOURCE_SCENE_PROFILE="SOURCE_SCENE_BIND_V2"

def iris_observation_binding(ctx,source_hashes,rasters,masks):
    cfg=dict(ctx["run_manifest"].get("source_audit") or {})
    iris_cfg=dict(ctx["run_manifest"].get("iris_fit") or {})
    if iris_cfg.get("training_profile")==SHARED_PROFILE and cfg.get("profile")!=SOURCE_SCENE_PROFILE:
        raise QualificationError("IRIS_SHARED_COHORT_SOURCE_SCENE_PROFILE_REQUIRED")
    ref=cfg.get("iris_observation_admission")
    if not ref:
        if cfg.get("iris_observation_admission_required"):
            raise QualificationError("IRIS_OBSERVATION_ADMISSION_REQUIRED")
        return None
    proof=load_file_ref(ref,expected_schema="RealSaS.IrisBindObservationAdmission.v1")
    if proof.get("status")!="PASS_IRIS_BIND_OBSERVATIONS":
        raise QualificationError("IRIS_OBSERVATION_ADMISSION_STATUS")
    if proof["source_sha256"] not in source_hashes:
        raise QualificationError("IRIS_OBSERVATION_RAW_SOURCE_NOT_STAGE1_BOUND")
    if proof["subject_id"] != cfg.get("subject_key"):
        raise QualificationError("IRIS_OBSERVATION_SUBJECT_BINDING")
    policy=proof["policy"]
    if (policy["resolution"]!=1024 or policy["view_count"]!=8
            or policy["maximum_relative_vertex_error"]!=1e-5
            or policy["maximum_mask_gap_px"]!=2.
            or proof["relative_vertex_error"]>policy["maximum_relative_vertex_error"]):
        raise QualificationError("IRIS_OBSERVATION_COURT_POLICY_DRIFT")
    cameras=ctx["run_manifest"]["observation"]["camera_bundle"]
    if cameras["sha256"]!=proof["camera_sha256"]:
        raise QualificationError("IRIS_OBSERVATION_CAMERA_BINDING")
    views=proof["views"]
    if [v["view_index"] for v in views]!=list(range(8)):
        raise QualificationError("IRIS_OBSERVATION_VIEW_SET")
    for i,view in enumerate(views):
        if (view["raster_sha256"]!=rasters[i][0]["image"]["sha256"]
                or view["mask_sha256"]!=masks[i][0]["mask"]["sha256"]
                or view["maximum_mask_gap_px"]>2. or view["border_margin_px"]<8):
            raise QualificationError("IRIS_OBSERVATION_RASTER_OR_MASK_BINDING")
    import hashlib,json
    binding={"proof_sha256":ref["sha256"],"source_sha256":proof["source_sha256"],
             "camera_sha256":proof["camera_sha256"],"views":proof["views"]}
    identity=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"proof_sha256":ref["sha256"],"source_sha256":proof["source_sha256"],
            "observation_binding":identity,
            "camera_sha256":proof["camera_sha256"],"subject_key":proof["subject_id"],
            "scope":"IRIS_BIND_OBSERVATIONS_ONLY","full_native_motion_truth_admitted":False}
