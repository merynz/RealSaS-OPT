from __future__ import annotations

import json

import numpy as np
import pytest

from models.materialization_contract_v1 import sha256
from models.iris.v32 import training_input_contract_v1 as iris


def reference(path, schema):
    return {"path": str(path), "sha256": sha256(path), "schema": schema}


def iris_subject(tmp_path, optional_state):
    source = tmp_path / "source.glb"
    source.write_bytes(b"fixture-source")
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({"schema": iris.CAMERA_SCHEMA}))
    raster = tmp_path / "view.png"
    raster.write_bytes(b"fixture-raster")
    mask = tmp_path / "view.mask.bin"
    mask.write_bytes(b"\x01")
    geometry = tmp_path / "surface.npz"
    np.savez_compressed(geometry, surface_query_xyz_canonical=np.array([[0., 0., 0.], [0., 0., 1.]]), unsigned_surface_distance=np.array([0., 1.]))
    return {
        "schema": iris.SCHEMA,
        "subject_id": "partial-subject",
        "source_sha256": sha256(source),
        "source": reference(source, "RealSaS.SourceFile.v1"),
        "split": "TRAIN",
        "eligibility_computed_by_notebook": False,
        "training_authorized": False,
        "family_status": {**dict.fromkeys(iris.REQUIRED_PASS, "PASS"), **dict.fromkeys(iris.OPTIONAL, optional_state)},
        "observation": {
            "camera_bundle": reference(camera, iris.CAMERA_SCHEMA),
            "source_rasters": [{"view_index": i, "image": reference(raster, "RealSaS.SourceRasterRGBA.v1")} for i in range(8)],
            "source_foreground_masks": [{"view_index": i, "mask": reference(mask, "RealSaS.SourceForegroundMask.v1")} for i in range(8)],
        },
        "surface_geometry": reference(geometry, iris.GEOMETRY_SCHEMA),
        "authority": {
            "matrix_cell_status_copied_not_recomputed": True,
            "split_membership_copied_not_recomputed": True,
            "missing_family_targets_zero_filled": False,
            "component_local_sign_used_as_global_sdf": False,
            "fstar_used_as_global_sign_authority": False,
        },
    }


def validate(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return iris.validate_manifest(path)


@pytest.mark.parametrize("state", ["MASKED", "FAIL", "QUARANTINED"])
def test_optional_nonpass_preserves_surface_supervision_without_sign_targets(tmp_path, state):
    manifest = iris_subject(tmp_path, state)
    result = validate(tmp_path, manifest)
    assert result["family_status"]["GLOBAL_SOLID_SDF"] == state
    assert result["surface_query_count"] == 2
    assert result["global_sdf_valid_count"] == 0
    assert result["training_authorized"] is False
    manifest["global_solid"] = {"path": "forged-solid.npz"}
    with pytest.raises(ValueError, match="NONPASS_GLOBAL_SDF_PAYLOAD_FORBIDDEN"):
        validate(tmp_path, manifest)
    manifest.pop("global_solid")
    manifest["family_payloads"] = {"APPEARANCE_CAA": [{"path": "forged-appearance.npz"}]}
    with pytest.raises(ValueError, match="NONPASS_FAMILY_PAYLOAD_FORBIDDEN"):
        validate(tmp_path, manifest)


def test_unknown_or_failed_required_family_never_materializes(tmp_path):
    manifest = iris_subject(tmp_path, "MASKED")
    manifest["family_status"]["GLOBAL_SOLID_SDF"] = "UNKNOWN"
    with pytest.raises(ValueError, match="FAMILY_STATUS_VALUE"):
        validate(tmp_path, manifest)
    manifest["family_status"]["GLOBAL_SOLID_SDF"] = "FAIL"
    manifest["family_status"]["SURFACE_GEOMETRY_UDF"] = "FAIL"
    with pytest.raises(ValueError, match="REQUIRED_FAMILY_NOT_PASS"):
        validate(tmp_path, manifest)


def test_pass_solid_requires_actual_signed_two_class_targets(tmp_path):
    manifest = iris_subject(tmp_path, "MASKED")
    manifest["family_status"]["GLOBAL_SOLID_SDF"] = "PASS"
    solid = tmp_path / "solid.npz"
    arrays = dict(query_xyz_canonical=np.array([[0., 0., 0.], [2., 0., 0.]]), signed_distance=np.array([-.5, .5]), occupancy=np.array([True, False]), valid=np.array([True, True]))
    np.savez_compressed(solid, **arrays)
    manifest["global_solid"] = reference(solid, iris.GLOBAL_SDF_SCHEMA)
    assert validate(tmp_path, manifest)["global_sdf_valid_count"] == 2
    arrays["signed_distance"][0] = .5
    np.savez_compressed(solid, **arrays)
    manifest["global_solid"] = reference(solid, iris.GLOBAL_SDF_SCHEMA)
    with pytest.raises(ValueError, match="INSIDE_SIGN"):
        validate(tmp_path, manifest)


@pytest.mark.parametrize("source_schema", sorted(iris.UNMATERIALIZED_SOURCE_VISIBILITY_SCHEMAS))
def test_source_hit_visibility_diagnostic_cannot_be_laundered_to_materialized_pass(tmp_path, source_schema):
    manifest=iris_subject(tmp_path,"MASKED")
    manifest["family_status"]["VISIBILITY_OCCLUSION"]="PASS"
    candidate=tmp_path/"source-hit-candidate.json"
    candidate.write_text(json.dumps({
        "schema":source_schema,
        "status":"PRIVATE_SOURCE_HIT_VISIBILITY_CANDIDATES__GRAPHICS_OWNER_SEPARATE",
        "training_authorized":False,
        "graphics_family_authority_minted":False,
    }))
    manifest["family_payloads"]={
        "VISIBILITY_OCCLUSION":[reference(candidate,source_schema)],
    }
    with pytest.raises(ValueError,match="SOURCE_HIT_DIAGNOSTIC_NOT_TRAINING_TARGET"):
        validate(tmp_path,manifest)


def test_nonpass_visibility_diagnostic_payload_also_forbidden(tmp_path):
    manifest=iris_subject(tmp_path,"MASKED")
    candidate=tmp_path/"visibility.json"
    schema="RealSaS.Corpus370BoundedSourceHitVisibilityCandidateCourt.v1"
    candidate.write_text(json.dumps({"schema":schema,"training_authorized":False}))
    manifest["family_payloads"]={"VISIBILITY_OCCLUSION":[reference(candidate,schema)]}
    with pytest.raises(ValueError,match="NONPASS_FAMILY_PAYLOAD_FORBIDDEN"):
        validate(tmp_path,manifest)
