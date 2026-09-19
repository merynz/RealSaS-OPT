from dataclasses import replace
from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.product_authority_v1 import CarrierCoverageThresholdIR,MeshQualificationPolicyIR
from compiler.realsas_compiler_core.rest_preservation_policy_v1 import (
    build_rest_source_preservation_policy,qualify_rest_source_preservation,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import (
    RestPreservationMeasurementSetIR,RestPreservationViewMeasurementIR,
    rest_preservation_measurement_set_hash,REST_PRESERVATION_METRIC_CONTRACT_HASH,
)
from compiler.realsas_compiler_core.types import QualificationError


def _mesh_policy():
    return MeshQualificationPolicyIR(
        0.125,0.25,7.5,16.0,
        (
            CarrierCoverageThresholdIR("MESH",0.97,0.995,0.005,0.005),
            CarrierCoverageThresholdIR("PLANAR",0.99,0.995,0.0025,0.0025),
        ),
        "mesh-policy-hash",
    )


def _result():
    return {
        "schema":"RealSaS.RestSourcePreservationCalibrationResult.v1",
        "status":"PASS",
        "authority":"SUBJECT_FREE_SYNTHETIC_ONLY__NO_KNIGHT_OR_MAGE_INPUT",
        "source_commit":"abc",
        "workflow_run_id":1,
        "workflow_job_id":2,
        "measurement_implementation":{"metric_contract_hash":REST_PRESERVATION_METRIC_CONTRACT_HASH},
        "silhouette_selection":{
            "max_positive_identity_p95_px":0.0,
            "min_negative_one_pixel_shift_p95_px":1.0,
            "selected_p95_ceiling_px":0.5,
        },
        "sensitivity":{"a":True,"b":True},
        "exact_policy_candidates":{
            "max_overlap_rgba_mismatch_pixel_count":0,
            "max_overlap_rgba_max_abs_channel_error_u8":0,
            "max_cross_view_source_geometry_fraction":0.0,
        },
    }


def _doc():
    return {
        "schema":"RealSaS.RestSourcePreservationProductPolicy.v1",
        "status":"FROZEN_AFTER_SUBJECT_FREE_CALIBRATION_BEFORE_KNIGHT_STAGE32_RESULT",
        "authority":"NORMATIVE_STAGE32_PRODUCT_ADMISSION_POLICY__NOT_EMPIRICAL_KNIGHT_FIT",
        "calibration_result":{"source_commit":"abc","workflow_run_id":1,"workflow_job_id":2},
        "per_view_thresholds":{
            "min_alpha_recall":0.97,
            "min_alpha_precision":0.995,
            "max_largest_coherent_hole_fraction":0.005,
            "max_interior_uncovered_fraction":0.005,
            "max_silhouette_edge_p95_px":0.5,
            "max_overlap_rgba_mismatch_pixel_count":0,
            "max_overlap_rgba_max_abs_channel_error_u8":0,
            "min_direct_source_geometry_fraction":1.0,
            "max_cross_view_source_geometry_fraction":0.0,
        },
    }


def _measurements(*,p95=0.0,mismatch=0,cross=0.0,recall=1.0):
    rows=tuple(
        RestPreservationViewMeasurementIR(
            i,"s"*64,"f"*64,"r"*64,"m"*64,
            recall,1.0,recall,0.0,0.0,0.0,p95,p95,
            0.0,0.0,0.0,100,mismatch,
            0.0 if mismatch==0 else 0.01,
            0 if mismatch==0 else 1,
            1.0-cross,cross,
        )
        for i in range(8)
    )
    value=RestPreservationMeasurementSetIR(rows,"render","obs","")
    return replace(value,measurement_set_hash=rest_preservation_measurement_set_hash(value))


def test_policy_is_rederived_from_calibration_and_mesh_floor():
    policy=build_rest_source_preservation_policy(
        policy_document=_doc(),calibration_result=_result(),mesh_policy=_mesh_policy()
    )
    assert policy.max_silhouette_edge_p95_px==0.5
    assert policy.min_alpha_recall==0.97
    assert policy.max_overlap_rgba_mismatch_pixel_count==0
    assert policy.max_cross_view_source_geometry_fraction==0.0


def test_policy_rejects_threshold_laundering():
    doc=_doc()
    doc["per_view_thresholds"]["max_silhouette_edge_p95_px"]=0.75
    with pytest.raises(QualificationError,match="SILHOUETTE_THRESHOLD_DRIFT"):
        build_rest_source_preservation_policy(
            policy_document=doc,calibration_result=_result(),mesh_policy=_mesh_policy()
        )


def test_all_eight_views_must_pass_every_rule():
    policy=build_rest_source_preservation_policy(
        policy_document=_doc(),calibration_result=_result(),mesh_policy=_mesh_policy()
    )
    rest=SimpleNamespace(render_set_hash="render")
    obs=SimpleNamespace(observation_set_hash="obs")
    good=qualify_rest_source_preservation(
        _measurements(),policy=policy,rest_render_set=rest,observation_set=obs
    )
    assert good.qualification_report["every_view_passed_every_rule"] is True
    with pytest.raises(QualificationError,match="SILHOUETTE_P95"):
        qualify_rest_source_preservation(
            _measurements(p95=0.6),policy=policy,rest_render_set=rest,observation_set=obs
        )
    with pytest.raises(QualificationError,match="OVERLAP_RGBA_MISMATCH"):
        qualify_rest_source_preservation(
            _measurements(mismatch=1),policy=policy,rest_render_set=rest,observation_set=obs
        )
    with pytest.raises(QualificationError,match="CROSS_VIEW_SOURCE_FRACTION"):
        qualify_rest_source_preservation(
            _measurements(cross=0.1),policy=policy,rest_render_set=rest,observation_set=obs
        )
