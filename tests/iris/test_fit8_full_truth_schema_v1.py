from __future__ import annotations

from tools.corpus.fit8_full_truth_schema_v1 import (
    DETERMINISTIC_DERIVED,
    DETERMINISTIC_RENDERED,
    DIAGNOSTIC_ONLY,
    FIT8FullTruthTeacherBundleManifestV1,
    HEAD_TARGETS_V1,
    OPTIONAL_DIAGNOSTIC,
    QUALIFIED_DERIVED,
    RAW_AUTHORED,
    REQUIRED_WITH_VALIDITY,
    TEACHER_CHANNELS_V1,
    TRUTH_AUTHORITY_CLASSES,
    validate_schema_v1,
)


def _by_name():
    return {spec.name: spec for spec in TEACHER_CHANNELS_V1}


def test_truth_authority_classes_are_explicit_and_closed():
    validate_schema_v1()
    assert TRUTH_AUTHORITY_CLASSES == {
        RAW_AUTHORED,
        DETERMINISTIC_DERIVED,
        QUALIFIED_DERIVED,
        DETERMINISTIC_RENDERED,
        DIAGNOSTIC_ONLY,
    }
    assert all(spec.authority_class in TRUTH_AUTHORITY_CLASSES for spec in TEACHER_CHANNELS_V1)
    assert all(
        spec.availability == OPTIONAL_DIAGNOSTIC
        for spec in TEACHER_CHANNELS_V1
        if spec.authority_class == DIAGNOSTIC_ONLY
    )


def test_raw_and_normalized_skin_are_not_silently_collapsed():
    specs = _by_name()
    assert specs["raw_skin_joint_weights"].authority_class == RAW_AUTHORED
    assert specs["normalized_skin_joint_weights"].authority_class == DETERMINISTIC_DERIVED
    assert specs["raw_skin_joint_weights"].name != specs["normalized_skin_joint_weights"].name
    assert "Kraw" in specs["raw_skin_joint_weights"].shape


def test_geometry_sign_and_glTF_equivalence_require_explicit_validity():
    specs = _by_name()
    assert specs["render_to_geometric_equivalence"].availability == REQUIRED_WITH_VALIDITY
    assert specs["component_sign_authority"].availability == REQUIRED_WITH_VALIDITY
    assert specs["field_signed_distance"].availability == REQUIRED_WITH_VALIDITY
    assert specs["field_unsigned_distance"].availability != OPTIONAL_DIAGNOSTIC
    assert specs["diagnostic_position_weld"].authority_class == DIAGNOSTIC_ONLY


def test_hidden_appearance_and_semantics_share_decomposition_states():
    specs = _by_name()
    required = {
        "decomposition_state_manifest",
        "decomposition_rgba",
        "decomposition_depth",
        "decomposition_normal",
        "decomposition_owner_id",
        "decomposition_material_region_id",
        "decomposition_surface_address",
    }
    assert required <= set(specs)
    assert required <= set(HEAD_TARGETS_V1["CAO"])
    assert {
        "decomposition_state_manifest",
        "decomposition_rgba",
        "decomposition_owner_id",
        "decomposition_material_region_id",
    } <= set(HEAD_TARGETS_V1["NERF_SCAFFOLD"])


def test_rigid_mechanical_ownership_is_not_attachment_authority():
    specs = _by_name()
    assert specs["deformation_mode"].authority_class == QUALIFIED_DERIVED
    assert specs["mechanical_owner_joint"].authority_class == QUALIFIED_DERIVED
    assert specs["attachment_relation"].authority_class == RAW_AUTHORED
    assert "PARENT_CONSTRAINT_SOCKET" in specs["attachment_relation"].producer_truth


def test_dynamic_context_is_explicit_for_T_and_CAO():
    assert "dynamic_pose_manifest" in HEAD_TARGETS_V1["T"]
    assert "dynamic_pose_manifest" in HEAD_TARGETS_V1["CAO"]
    assert "dynamic_reference_jacobian" in HEAD_TARGETS_V1["T"]
    assert "dynamic_canonical_rgba" in HEAD_TARGETS_V1["CAO"]
    assert "dynamic_ray_hit_stack" in HEAD_TARGETS_V1["CAO"]


def test_manifest_requires_artifacts_and_validity_fail_closed():
    artifacts = {
        spec.name: f"artifact://{spec.name}"
        for spec in TEACHER_CHANNELS_V1
        if spec.artifact_required
    }
    validity = {
        spec.name: f"validity://{spec.name}"
        for spec in TEACHER_CHANNELS_V1
        if spec.availability == REQUIRED_WITH_VALIDITY
    }
    value = FIT8FullTruthTeacherBundleManifestV1(
        subject_key="fit8_schema_test",
        source_sha256="a" * 64,
        canonical_camera_contract_sha256="b" * 64,
        normalization_contract_sha256="c" * 64,
        channel_artifacts=artifacts,
        channel_validity=validity,
    )
    value.validate()


def test_all_head_targets_reference_declared_channels():
    names = {spec.name for spec in TEACHER_CHANNELS_V1}
    for channels in HEAD_TARGETS_V1.values():
        assert set(channels) <= names
