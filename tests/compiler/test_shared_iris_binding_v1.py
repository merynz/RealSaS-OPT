import copy
import pytest
import numpy as np
from types import SimpleNamespace
import json,hashlib
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.shared_iris_execution_v1 import validate_shared_iris_payload,value_hash
from tools.training.export_iris_cohort_surfaces_v1 import per_view_court

def test_shared_profile_is_explicit_and_policy_bound(tmp_path):
    from compiler.realsas_compiler_services.orchestrator.adapters.shared_iris_execution_v1 import shared_profile_policy,SHARED_PROFILE
    policy={"schema":"RealSaS.SharedIrisTrainingPolicy.v1","subject_count":8,
            "teacher_inputs_in_forward":False,
            "sampling":"ROUND_ROBIN__HALF_FRESH_UNIFORM_HALF_FROZEN_NEAR_ZERO",
            "fit8_pass_from_training_loss_allowed":False}
    path=tmp_path/"policy.json";path.write_text(json.dumps(policy))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    ctx={"run_manifest":{"iris_fit":{"training_profile":SHARED_PROFILE,
          "shared_training_policy":{"path":str(path),"sha256":digest}}}}
    prereg=SimpleNamespace(qualification_policy={"shared_cohort_required":True,
             "shared_training_policy_sha256":digest,"training_profile":SHARED_PROFILE})
    assert shared_profile_policy(ctx,prereg)["shared_cohort_required"]
    prereg.qualification_policy["shared_training_policy_sha256"]="0"*64
    with pytest.raises(QualificationError,match="PREREG_BINDING"):shared_profile_policy(ctx,prereg)
    ctx["run_manifest"]["iris_fit"]["training_profile"]="UNKNOWN"
    with pytest.raises(QualificationError,match="PROFILE_UNKNOWN"):shared_profile_policy(ctx)
    assert shared_profile_policy({"run_manifest":{"iris_fit":{}}}) is None

def data():
    ids=["subject_a","subject_b"]
    manifest={"subjects":[{"subject_id":s} for s in ids],"teacher_inputs_in_forward":False}
    policy={"subject_count":2,"steps":4}
    binding={"subject_ids":ids,"manifest_sha256":value_hash(manifest),"policy_sha256":value_hash(policy)}
    payload={"schema":"RealSaS.SharedIrisFitExecutionPayload.v1",
             "status":"TRAINING_COMPLETE__COMPILER_QUALIFICATION_PENDING",
             "shared_model_count":1,"shared_optimizer_count":1,"binding":binding,
             "binding_sha256":value_hash(binding),"selected_checkpoint_sha256":"a"*64,
             "subject_exposures":dict.fromkeys(ids,2),"actual_optimizer_steps":4,
             "per_subject_evaluations":[{"step":4,"subjects":[{"subject_id":s} for s in ids]}]}
    return payload,manifest,policy

def test_shared_checkpoint_binding_and_hidden_subjects_fail():
    payload,manifest,policy=data()
    result=validate_shared_iris_payload(payload,manifest,policy,checkpoint_sha256="a"*64)
    assert result["subject_ids"]==["subject_a","subject_b"]
    for key,value,error in [
        ("shared_model_count",2,"NOT_SHARED"),
        ("subject_exposures",{"subject_a":3,"subject_b":1},"IMBALANCE"),
        ("selected_checkpoint_sha256","b"*64,"CHECKPOINT"),
        ("fit8_pass_claimed",True,"UNAUTHORIZED")]:
        changed=copy.deepcopy(payload);changed[key]=value
        with pytest.raises(QualificationError,match=error):
            validate_shared_iris_payload(changed,manifest,policy,checkpoint_sha256="a"*64)
    changed=copy.deepcopy(payload);changed["per_subject_evaluations"][0]["subjects"].pop()
    with pytest.raises(QualificationError,match="HIDDEN_FAILED_SUBJECT"):
        validate_shared_iris_payload(changed,manifest,policy,checkpoint_sha256="a"*64)

def test_existing_per_view_compiler_court_reads_exact_raster_and_reports_loss():
    # Square aligns exactly with four pixel centers; use current Compiler raster.
    vertices=np.array([[-.5,-.5,0.],[.5,-.5,0.],[.5,.5,0.],[-.5,.5,0.]])
    faces=np.array([[0,1,2],[0,2,3]])
    camera={"origin":[0,0,-2],"right":[1,0,0],"screen_up":[0,1,0],"half_extent":1.,"resolution":8}
    mask=np.zeros((8,8),np.uint8);mask[2:6,2:6]=1
    result,pred,_=per_view_court(vertices,faces,camera,mask.tobytes())
    assert result["recall"]==1. and result["precision"]==1. and result["pass"]
    result,_,_=per_view_court(vertices*.4,faces,camera,mask.tobytes())
    assert result["recall"]<1. and not result["pass"]
