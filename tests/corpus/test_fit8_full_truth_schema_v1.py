from tools.corpus.fit8_full_truth_schema_v1 import (
    HEAD_TARGETS_V1,
    REQUIRED_WITH_VALIDITY,
    TEACHER_CHANNELS_V1,
    validate_schema_v1,
)


def test_fit8_full_truth_schema_is_generic_and_complete_for_unified_heads():
    validate_schema_v1()
    names = {row.name for row in TEACHER_CHANNELS_V1}
    for required in (
        "authored_vertices",
        "authored_faces",
        "surface_xyz",
        "surface_rgba",
        "owner_instance_id",
        "material_region_id",
        "canonical_rgba",
        "canonical_depth",
        "canonical_normal",
        "amodal_owner_mask",
        "ray_hit_stack",
        "pairwise_overlap_relation",
        "dynamic_pose_vertex_xyz",
        "dynamic_edge_stretch",
        "dynamic_face_area_ratio",
        "dynamic_face_condition",
        "dynamic_contact_relation",
    ):
        assert required in names
    assert set(HEAD_TARGETS_V1) == {"G", "A", "S", "F", "T", "CAO"}


def test_missing_natural_asset_channels_require_explicit_validity_not_fake_zero_truth():
    policies = {row.name: row.availability for row in TEACHER_CHANNELS_V1}
    for channel in (
        "authored_uv",
        "face_material_id",
        "joint_hierarchy",
        "skin_joint_indices",
        "skin_joint_weights",
        "dynamic_pose_joint_matrices",
        "dynamic_pose_vertex_xyz",
    ):
        assert policies[channel] == REQUIRED_WITH_VALIDITY


def test_topology_head_is_supervised_by_dynamic_quality_not_only_static_geometry():
    targets = set(HEAD_TARGETS_V1["T"])
    assert "dynamic_pose_vertex_xyz" in targets
    assert "dynamic_edge_stretch" in targets
    assert "dynamic_face_area_ratio" in targets
    assert "dynamic_face_condition" in targets
    assert "dynamic_contact_relation" in targets


def test_cao_has_both_surface_identity_and_dynamic_presentation_truth():
    targets = set(HEAD_TARGETS_V1["CAO"])
    assert "source_barycentric" in targets
    assert "owner_instance_id" in targets
    assert "material_region_id" in targets
    assert "ray_hit_stack" in targets
    assert "dynamic_overlap_relation" in targets
    assert "dynamic_contact_relation" in targets
