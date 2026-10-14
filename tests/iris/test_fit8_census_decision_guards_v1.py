from __future__ import annotations

from tools.corpus.fit8_full_truth_schema_v1 import (
    DETERMINISTIC_DERIVED,
    OPTIONAL_DIAGNOSTIC,
    QUALIFIED_DERIVED,
    RAW_AUTHORED,
    REQUIRED_WITH_VALIDITY,
    TEACHER_CHANNELS_V1,
)


def _specs():
    return {spec.name: spec for spec in TEACHER_CHANNELS_V1}


def test_uv_is_validity_gated_not_universal_authority():
    specs = _specs()
    assert specs["source_uv"].authority_class == RAW_AUTHORED
    assert specs["source_uv"].availability == REQUIRED_WITH_VALIDITY
    assert specs["surface_rgba"].availability == REQUIRED_WITH_VALIDITY


def test_raw_skin_capacity_is_not_hard_coded_to_four():
    specs = _specs()
    raw_joints = specs["raw_skin_joint_indices"]
    raw_weights = specs["raw_skin_joint_weights"]
    assert "Kraw" in raw_joints.shape
    assert "Kraw" in raw_weights.shape
    assert "4" not in raw_joints.shape
    assert "4" not in raw_weights.shape
    assert raw_joints.authority_class == RAW_AUTHORED
    assert raw_weights.authority_class == RAW_AUTHORED


def test_raw_and_normalized_skin_keep_distinct_provenance():
    specs = _specs()
    assert specs["raw_skin_joint_weights"].authority_class == RAW_AUTHORED
    assert specs["normalized_skin_joint_weights"].authority_class == DETERMINISTIC_DERIVED
    assert specs["raw_skin_joint_indices"].authority_class == RAW_AUTHORED
    assert specs["normalized_skin_joint_indices"].authority_class == DETERMINISTIC_DERIVED


def test_render_topology_and_geometric_equivalence_are_not_collapsed():
    specs = _specs()
    assert specs["source_polygon_topology"].authority_class == RAW_AUTHORED
    assert specs["render_to_geometric_equivalence"].authority_class == QUALIFIED_DERIVED
    assert specs["render_to_geometric_equivalence"].availability == REQUIRED_WITH_VALIDITY
    assert specs["diagnostic_position_weld"].availability == OPTIONAL_DIAGNOSTIC


def test_unsigned_distance_survives_when_signed_distance_abstains():
    specs = _specs()
    assert specs["field_unsigned_distance"].authority_class == DETERMINISTIC_DERIVED
    assert specs["field_unsigned_distance"].availability != OPTIONAL_DIAGNOSTIC
    assert specs["field_signed_distance"].authority_class == QUALIFIED_DERIVED
    assert specs["field_signed_distance"].availability == REQUIRED_WITH_VALIDITY
    assert specs["component_sign_authority"].availability == REQUIRED_WITH_VALIDITY


def test_component_material_owner_attachment_axes_remain_distinct():
    specs = _specs()
    required_distinct = {
        "source_face_object_id",
        "surface_geometric_component_id",
        "surface_material_id",
        "mechanical_owner_joint",
        "deformation_mode",
        "attachment_relation",
        "surface_owner_region_id",
    }
    assert required_distinct <= set(specs)
    assert len(required_distinct) == len({specs[name].name for name in required_distinct})
    assert specs["attachment_relation"].authority_class == RAW_AUTHORED
    assert specs["mechanical_owner_joint"].authority_class == QUALIFIED_DERIVED
    assert specs["deformation_mode"].authority_class == QUALIFIED_DERIVED


def test_open_surface_support_cannot_be_reduced_to_signed_distance():
    specs = _specs()
    # The schema must retain both component topology/sign validity and broad
    # unsigned-distance truth. A future explicit surface-support grid/channel
    # may strengthen this guard, but these channels are the minimum invariant.
    assert "geometric_component_manifest" in specs
    assert "component_sign_authority" in specs
    assert "field_unsigned_distance" in specs
    assert "field_signed_distance" in specs
    assert specs["component_sign_authority"].availability == REQUIRED_WITH_VALIDITY


def test_appearance_authority_is_surface_addressed_not_uv_addressed():
    specs = _specs()
    assert "surface_source_address" in specs
    assert "canonical_surface_address" in specs
    assert specs["surface_source_address"].authority_class == DETERMINISTIC_DERIVED
    assert specs["source_uv"].availability == REQUIRED_WITH_VALIDITY
