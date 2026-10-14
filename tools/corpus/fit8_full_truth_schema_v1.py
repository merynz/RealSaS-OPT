from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "RealSaS.FIT8FullTruthTeacherBundle.v1"
CANONICAL_VIEW_COUNT = 8

REQUIRED = "REQUIRED"
REQUIRED_WITH_VALIDITY = "REQUIRED_WITH_VALIDITY"
OPTIONAL_DIAGNOSTIC = "OPTIONAL_DIAGNOSTIC"
AVAILABILITY_POLICIES = {REQUIRED, REQUIRED_WITH_VALIDITY, OPTIONAL_DIAGNOSTIC}

RAW_AUTHORED = "RAW_AUTHORED"
DETERMINISTIC_DERIVED = "DETERMINISTIC_DERIVED"
QUALIFIED_DERIVED = "QUALIFIED_DERIVED"
DETERMINISTIC_RENDERED = "DETERMINISTIC_RENDERED"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
TRUTH_AUTHORITY_CLASSES = {
    RAW_AUTHORED,
    DETERMINISTIC_DERIVED,
    QUALIFIED_DERIVED,
    DETERMINISTIC_RENDERED,
    DIAGNOSTIC_ONLY,
}


@dataclass(frozen=True)
class TeacherChannelSpecV1:
    name: str
    domain: str
    dtype: str
    shape: str
    producer_truth: str
    consumed_by: tuple[str, ...]
    authority_class: str
    availability: str = REQUIRED
    notes: str = ""

    def validate(self) -> None:
        if not self.name or not self.domain or not self.dtype or not self.shape:
            raise ValueError("FIT8_TEACHER_CHANNEL_IDENTITY_MISSING")
        if not self.producer_truth:
            raise ValueError("FIT8_TEACHER_CHANNEL_PRODUCER_MISSING")
        if not self.consumed_by:
            raise ValueError("FIT8_TEACHER_CHANNEL_CONSUMER_MISSING")
        if self.availability not in AVAILABILITY_POLICIES:
            raise ValueError("FIT8_TEACHER_CHANNEL_AVAILABILITY_INVALID")
        if self.authority_class not in TRUTH_AUTHORITY_CLASSES:
            raise ValueError("FIT8_TEACHER_CHANNEL_AUTHORITY_CLASS_INVALID")
        if self.authority_class == DIAGNOSTIC_ONLY and self.availability != OPTIONAL_DIAGNOSTIC:
            raise ValueError("FIT8_TEACHER_DIAGNOSTIC_CHANNEL_MUST_BE_OPTIONAL")

    @property
    def artifact_required(self) -> bool:
        return self.availability != OPTIONAL_DIAGNOSTIC


# The schema is intentionally factored. Exact-byte FIT8 census showed that
# object, material, geometric component, mechanical owner and deformation mode
# are different axes even inside one authored asset. REQUIRED_WITH_VALIDITY
# means an artifact must exist and must explicitly state which entries are valid;
# missing source information is never silently turned into zero/false truth.
TEACHER_CHANNELS_V1 = (
    # ------------------------------------------------------------------
    # 1. Exact source-format authorship. Preserve what the author/exporter
    # actually supplied before any welding, triangulation or normalization.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "source_format_manifest", "source", "json", "{}",
        "EXACT_SOURCE_BYTES_AND_IMPORT_METADATA", ("G", "A", "S", "F", "T", "CAO"),
        RAW_AUTHORED,
    ),
    TeacherChannelSpecV1(
        "source_vertices", "source_geometry", "float32", "[Vsrc,3]",
        "AUTHORED_OR_EXPORTED_VERTEX_POSITIONS", ("G", "F", "T", "CAO"),
        RAW_AUTHORED,
    ),
    TeacherChannelSpecV1(
        "source_polygon_topology", "source_geometry", "ragged", "[P,*]",
        "AUTHORED_SOURCE_POLYGONS_OR_EXACT_GLTF_TRIANGLES", ("F", "T", "CAO"),
        RAW_AUTHORED,
        notes="Preserve n-gons for FBX; glTF is already indexed triangle render topology.",
    ),
    TeacherChannelSpecV1(
        "source_vertex_normals", "source_geometry", "float32", "[Vsrc,3]",
        "AUTHORED_OR_EXPORTED_NORMALS", ("G", "F", "T", "CAO"),
        RAW_AUTHORED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "source_uv", "source_mesh_corner_or_vertex", "float32", "[* ,2]",
        "AUTHORED_OR_EXPORTED_UV", ("A", "S", "CAO"),
        RAW_AUTHORED, availability=REQUIRED_WITH_VALIDITY,
        notes="FIT8 contains both UV-textured and material-color-only assets.",
    ),
    TeacherChannelSpecV1(
        "source_face_object_id", "source_geometry", "int32", "[P]",
        "AUTHORED_OBJECT_HIERARCHY", ("S", "T", "CAO"),
        RAW_AUTHORED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "source_face_material_id", "source_geometry", "int32", "[P]",
        "AUTHORED_MATERIAL_SLOTS", ("A", "S", "CAO"),
        RAW_AUTHORED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "source_object_parent_graph", "source_object_graph", "json", "[C,E]",
        "AUTHORED_OBJECT_HIERARCHY", ("S", "CAO"),
        RAW_AUTHORED,
    ),
    TeacherChannelSpecV1(
        "source_material_manifest", "source_material_graph", "json", "[M]",
        "AUTHORED_MATERIAL_TEXTURE_PARAMETERS", ("A", "S", "CAO"),
        RAW_AUTHORED, availability=REQUIRED_WITH_VALIDITY,
    ),

    # ------------------------------------------------------------------
    # 2. Geometry equivalence/topology qualification. glTF render vertices may
    # be duplicated at normal/UV/material seams; a diagnostic weld is evidence,
    # never silent canonical topology.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "render_to_geometric_equivalence", "geometry_equivalence", "json", "[Vsrc]",
        "QUALIFIED_GEOMETRIC_EQUIVALENCE_FROM_SOURCE_ATTRIBUTES", ("G", "S", "F", "T", "CAO"),
        QUALIFIED_DERIVED, availability=REQUIRED_WITH_VALIDITY,
        notes="Maps source/render vertex identities to qualified geometric equivalence groups without erasing source identity.",
    ),
    TeacherChannelSpecV1(
        "geometric_component_manifest", "geometry_component", "json", "[Cgeo]",
        "QUALIFIED_CONNECTED_SURFACE_COMPONENTS", ("G", "S", "F", "T", "CAO"),
        QUALIFIED_DERIVED,
        notes="Includes closed/open/non-manifold/orientation diagnostics per component.",
    ),
    TeacherChannelSpecV1(
        "diagnostic_position_weld", "geometry_diagnostic", "json", "{}",
        "POSITION_WELD_DIAGNOSTIC_ONLY", ("G", "S", "F", "T", "CAO"),
        DIAGNOSTIC_ONLY, availability=OPTIONAL_DIAGNOSTIC,
        notes="Never promoted to teacher topology merely because seam duplicates disappear.",
    ),
    TeacherChannelSpecV1(
        "teacher_triangles", "teacher_surface", "int64", "[Ft,3]",
        "QUALIFIED_TRIANGULATION_BOUND_TO_SOURCE_POLYGONS", ("G", "A", "S", "F", "T", "CAO"),
        QUALIFIED_DERIVED,
    ),
    TeacherChannelSpecV1(
        "teacher_triangle_source_binding", "teacher_surface", "json", "[Ft]",
        "EXACT_BINDING_TO_SOURCE_POLYGON_RENDER_IDS", ("A", "S", "F", "T", "CAO"),
        DETERMINISTIC_DERIVED,
    ),
    TeacherChannelSpecV1(
        "component_sign_authority", "geometry_component", "json", "[Cgeo]",
        "QUALIFIED_CLOSEDNESS_ORIENTATION_NESTING_SIGN_POLICY", ("G", "F"),
        QUALIFIED_DERIVED, availability=REQUIRED_WITH_VALIDITY,
        notes="Closed edge-manifold is necessary but not by itself sufficient for global metric sign authority.",
    ),

    # ------------------------------------------------------------------
    # 3. Mechanical authorship. Raw authored/exported skin and normalized
    # mechanical teacher skin are separate artifacts with separate provenance.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "joint_hierarchy", "rig", "json", "[J,E]",
        "AUTHORED_RIG", ("S", "T"), RAW_AUTHORED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "inverse_bind_matrices", "rig", "float32", "[J,4,4]",
        "AUTHORED_OR_EXPORTED_SKIN_BIND", ("S", "T"), RAW_AUTHORED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "raw_skin_joint_indices", "source_mesh_vertex", "int32", "[Vsrc,Kraw]",
        "AUTHORED_OR_EXPORTED_SKIN", ("S", "T"), RAW_AUTHORED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Kraw is variable; FIT8 FBX reaches at least seven influences.",
    ),
    TeacherChannelSpecV1(
        "raw_skin_joint_weights", "source_mesh_vertex", "float32", "[Vsrc,Kraw]",
        "AUTHORED_OR_EXPORTED_SKIN", ("S", "T"), RAW_AUTHORED,
        availability=REQUIRED_WITH_VALIDITY,
        notes="Do not silently renormalize raw FBX cluster weights.",
    ),
    TeacherChannelSpecV1(
        "normalized_skin_joint_indices", "teacher_surface_vertex", "int32", "[Vt,K]",
        "DETERMINISTIC_NORMALIZATION_OF_QUALIFIED_RAW_SKIN", ("S", "T"),
        DETERMINISTIC_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "normalized_skin_joint_weights", "teacher_surface_vertex", "float32", "[Vt,K]",
        "DETERMINISTIC_NORMALIZATION_OF_QUALIFIED_RAW_SKIN", ("S", "T"),
        DETERMINISTIC_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "mechanical_owner_joint", "teacher_surface_vertex_or_component", "int32", "[*]",
        "DERIVED_FROM_NORMALIZED_SKIN_AND_COMPONENT_SUPPORT", ("S", "T", "CAO"),
        QUALIFIED_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "deformation_mode", "geometry_component", "enum", "[Cgeo]",
        "DERIVED_FROM_COMPONENT_SKIN_DISTRIBUTION_AND_RIG_HIERARCHY", ("S", "T", "CAO"),
        QUALIFIED_DERIVED, availability=REQUIRED_WITH_VALIDITY,
        notes="Examples include smooth-skinned versus rigid-bone-owned; never inferred from material name alone.",
    ),
    TeacherChannelSpecV1(
        "attachment_relation", "component_graph", "json", "[Eattach]",
        "EXPLICIT_AUTHORED_PARENT_CONSTRAINT_SOCKET_RELATION", ("S", "T", "CAO"),
        RAW_AUTHORED, availability=REQUIRED_WITH_VALIDITY,
        notes="Rigid bone ownership is not automatically an attachment relation.",
    ),

    # ------------------------------------------------------------------
    # 4. Qualified mechanical reference carrier used only to evaluate authored
    # motion consequences. T is not required to copy this exact topology.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "mechanical_reference_vertices", "mechanical_teacher_surface", "float32", "[Vm,3]",
        "QUALIFIED_MECHANICAL_REFERENCE_SURFACE", ("T",), QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "mechanical_reference_faces", "mechanical_teacher_surface", "int64", "[Fm,3]",
        "QUALIFIED_MECHANICAL_REFERENCE_SURFACE", ("T",), QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "mechanical_reference_source_binding", "mechanical_teacher_surface", "json", "[Vm,Fm]",
        "EXACT_OR_QUALIFIED_BINDING_TO_TEACHER_SURFACE", ("T",), QUALIFIED_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),

    # ------------------------------------------------------------------
    # 5. Continuous shared-surface samples. Geometry, appearance, semantics and
    # mechanics use one sample identity instead of post-hoc rematching.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "surface_xyz", "continuous_surface", "float32", "[N,3]",
        "SAMPLED_QUALIFIED_TEACHER_SURFACE", ("G", "A", "S", "F", "T", "CAO"),
        DETERMINISTIC_DERIVED,
    ),
    TeacherChannelSpecV1(
        "surface_normal", "continuous_surface", "float32", "[N,3]",
        "SAMPLED_QUALIFIED_TEACHER_SURFACE", ("G", "F", "T", "CAO"),
        DETERMINISTIC_DERIVED,
    ),
    TeacherChannelSpecV1(
        "surface_rgba", "continuous_surface", "float32", "[N,4]",
        "AUTHORED_MATERIAL_UV_OR_UNLIT_RENDER_EVALUATION", ("A", "CAO"),
        DETERMINISTIC_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "surface_source_address", "continuous_surface", "json", "[N]",
        "EXACT_SOURCE_POLYGON_FACE_BARYCENTRIC_ADDRESS", ("A", "S", "T", "CAO"),
        DETERMINISTIC_DERIVED,
    ),
    TeacherChannelSpecV1(
        "surface_source_object_id", "continuous_surface", "int32", "[N]",
        "AUTHORED_OBJECT_ID_AT_SURFACE_ADDRESS", ("S", "T", "CAO"),
        DETERMINISTIC_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "surface_geometric_component_id", "continuous_surface", "int32", "[N]",
        "QUALIFIED_GEOMETRIC_COMPONENT_AT_SURFACE_ADDRESS", ("G", "S", "F", "T", "CAO"),
        QUALIFIED_DERIVED,
    ),
    TeacherChannelSpecV1(
        "surface_material_id", "continuous_surface", "int32", "[N]",
        "AUTHORED_MATERIAL_AT_SURFACE_ADDRESS", ("A", "S", "CAO"),
        DETERMINISTIC_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "surface_normalized_skin", "continuous_surface", "ragged", "[N,*]",
        "BARYCENTRIC_INTERPOLATION_OF_NORMALIZED_SKIN", ("S", "T"),
        DETERMINISTIC_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),

    # ------------------------------------------------------------------
    # 6. Off-surface field truth. Preserve legacy IRIS direct 3D supervision
    # without fabricating global sign on open/non-manifold components.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "field_query_xyz", "continuous_field", "float32", "[Q,3]",
        "DETERMINISTIC_QUERY_SAMPLING_CONTRACT", ("G", "F"), DETERMINISTIC_DERIVED,
    ),
    TeacherChannelSpecV1(
        "field_unsigned_distance", "continuous_field", "float32", "[Q]",
        "DISTANCE_TO_QUALIFIED_TEACHER_SURFACE", ("G", "F"), DETERMINISTIC_DERIVED,
    ),
    TeacherChannelSpecV1(
        "field_signed_distance", "continuous_field", "float32", "[Q]",
        "SIGNED_DISTANCE_WHERE_COMPONENT_SIGN_AUTHORITY_EXISTS", ("G", "F"),
        QUALIFIED_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "field_nearest_surface_address", "continuous_field", "json", "[Q]",
        "NEAREST_QUALIFIED_SURFACE_ADDRESS", ("G", "F", "S"), DETERMINISTIC_DERIVED,
    ),

    # ------------------------------------------------------------------
    # 7. Subject-local semantic normalization. IDs are teacher-local targets;
    # the network should learn embeddings/affinity, not global owner numbers.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "owner_partition_manifest", "semantic_partition", "json", "[Cowner]",
        "QUALIFIED_GENERIC_OWNER_NORMALIZATION", ("S", "T", "CAO"), QUALIFIED_DERIVED,
        notes="May split/merge source objects/components only under explicit deterministic/qualified policy.",
    ),
    TeacherChannelSpecV1(
        "surface_owner_region_id", "continuous_surface", "int32", "[N]",
        "OWNER_PARTITION_AT_SURFACE_ADDRESS", ("S", "T", "CAO"), QUALIFIED_DERIVED,
    ),
    TeacherChannelSpecV1(
        "surface_material_region_id", "continuous_surface", "int32", "[N]",
        "QUALIFIED_MATERIAL_REGION_AT_SURFACE_ADDRESS", ("A", "S", "CAO"),
        QUALIFIED_DERIVED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "component_visibility_graph", "semantic_visibility_graph", "json", "[C,E]",
        "DERIVED_FROM_COMPONENT_SUPPORT_AUTHORED_RELATIONS_AND_CANONICAL_RAYS", ("A", "S", "CAO"),
        QUALIFIED_DERIVED,
        notes="Generic replacement for fixed human hair>cloth>body ordering.",
    ),

    # ------------------------------------------------------------------
    # 8. Canonical eight-view truth. Only canonical_rgba + camera are forward
    # RGB inputs in Phase A; privileged channels remain teacher-only.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "canonical_rgba", "canonical_view", "uint8", "[8,H,W,4]",
        "DETERMINISTIC_UNLIT_SOURCE_RENDER", ("NERF_SCAFFOLD", "G", "A", "S", "F", "T", "CAO"),
        DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "canonical_depth", "canonical_view", "float32", "[8,H,W]",
        "DETERMINISTIC_SOURCE_RENDER", ("G", "F", "T", "CAO"), DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "canonical_normal", "canonical_view", "float32", "[8,H,W,3]",
        "DETERMINISTIC_SOURCE_RENDER", ("G", "F", "T"), DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "canonical_owner_id", "canonical_view", "int32", "[8,H,W]",
        "RASTERIZED_OWNER_PARTITION", ("NERF_SCAFFOLD", "S", "CAO"), DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "canonical_material_region_id", "canonical_view", "int32", "[8,H,W]",
        "RASTERIZED_MATERIAL_PARTITION", ("NERF_SCAFFOLD", "A", "S", "CAO"),
        DETERMINISTIC_RENDERED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "canonical_surface_address", "canonical_view", "json", "[8,H,W]",
        "FIRST_HIT_FACE_BARYCENTRIC_ADDRESS", ("A", "S", "T", "CAO"), DETERMINISTIC_RENDERED,
    ),

    # ------------------------------------------------------------------
    # 9. Hidden/decomposition-state truth. This is StdGEN's semantic masking
    # lesson translated to a generic Component/Visibility Graph.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "decomposition_state_manifest", "decomposition_state", "json", "[D]",
        "GRAPH_VALID_COMPONENT_PEEL_ISOLATION_STATES", ("NERF_SCAFFOLD", "A", "S", "CAO"),
        QUALIFIED_DERIVED,
        notes="D is subject-local and variable; states record included/excluded regions and provenance.",
    ),
    TeacherChannelSpecV1(
        "decomposition_rgba", "decomposition_state_view", "archive", "{D}[8,H,W,4]",
        "DETERMINISTIC_RENDER_OF_DECOMPOSITION_STATES", ("NERF_SCAFFOLD", "A", "CAO"),
        DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "decomposition_depth", "decomposition_state_view", "archive", "{D}[8,H,W]",
        "DETERMINISTIC_RENDER_OF_DECOMPOSITION_STATES", ("G", "F", "CAO"), DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "decomposition_normal", "decomposition_state_view", "archive", "{D}[8,H,W,3]",
        "DETERMINISTIC_RENDER_OF_DECOMPOSITION_STATES", ("G", "F"), DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "decomposition_owner_id", "decomposition_state_view", "archive", "{D}[8,H,W]",
        "RASTERIZED_OWNER_PARTITION_IN_DECOMPOSITION_STATE", ("NERF_SCAFFOLD", "S", "CAO"),
        DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "decomposition_material_region_id", "decomposition_state_view", "archive", "{D}[8,H,W]",
        "RASTERIZED_MATERIAL_PARTITION_IN_DECOMPOSITION_STATE", ("NERF_SCAFFOLD", "A", "S", "CAO"),
        DETERMINISTIC_RENDERED, availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "decomposition_surface_address", "decomposition_state_view", "archive", "{D}[8,H,W]",
        "FIRST_HIT_SURFACE_ADDRESS_IN_DECOMPOSITION_STATE", ("A", "S", "CAO"), DETERMINISTIC_RENDERED,
    ),

    # ------------------------------------------------------------------
    # 10. Amodal and all-hit truth. Do not flatten physical visibility to one Z.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "amodal_owner_mask", "canonical_view_component", "bool", "[8,Cowner,H,W]",
        "COMPONENT_ISOLATION_RENDER", ("S", "CAO"), DETERMINISTIC_RENDERED,
    ),
    TeacherChannelSpecV1(
        "amodal_material_mask", "canonical_view_material", "bool", "[8,Cmat,H,W]",
        "MATERIAL_ISOLATION_RENDER", ("A", "S", "CAO"), DETERMINISTIC_RENDERED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "ray_hit_stack", "canonical_view_ray", "ragged", "[8,H,W,*]",
        "ALL_HIT_RAY_QUERY", ("S", "CAO"), DETERMINISTIC_RENDERED,
        notes="Ordered surface-address/owner/material/depth hits; never reduced to nearest-Z only.",
    ),
    TeacherChannelSpecV1(
        "pairwise_overlap_relation", "canonical_view_relation", "json", "[Eview]",
        "DERIVED_FROM_RAY_HIT_STACK", ("S", "CAO"), DETERMINISTIC_DERIVED,
    ),

    # ------------------------------------------------------------------
    # 11. Dynamic authored truth. Pose IDs are explicit context; contact inferred
    # from geometry/motion remains derived evidence unless source constraints say
    # otherwise.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "dynamic_pose_manifest", "dynamic_pose", "json", "[P]",
        "SELECTED_AUTHORED_CLIP_FRAME_TIMES", ("S", "T", "CAO"), DETERMINISTIC_DERIVED,
    ),
    TeacherChannelSpecV1(
        "dynamic_pose_joint_matrices", "dynamic_pose_rig", "float32", "[P,J,4,4]",
        "EVALUATED_AUTHORED_ANIMATION", ("S", "T"), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_reference_vertex_xyz", "dynamic_pose_mesh", "float32", "[P,Vm,3]",
        "MECHANICAL_REFERENCE_SURFACE_UNDER_AUTHORED_MOTION", ("T", "CAO"), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_reference_vertex_normal", "dynamic_pose_mesh", "float32", "[P,Vm,3]",
        "MECHANICAL_REFERENCE_SURFACE_UNDER_AUTHORED_MOTION", ("T", "CAO"), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_reference_jacobian", "dynamic_pose_quality", "float32", "[P,Fm,3,2]",
        "DERIVED_FROM_AUTHORED_REST_AND_POSED_REFERENCE_SURFACE", ("T",), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_edge_stretch", "dynamic_pose_quality", "float32", "[P,Em]",
        "DERIVED_FROM_AUTHORED_REST_AND_POSED_REFERENCE_SURFACE", ("T",), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_face_area_ratio", "dynamic_pose_quality", "float32", "[P,Fm]",
        "DERIVED_FROM_AUTHORED_REST_AND_POSED_REFERENCE_SURFACE", ("T",), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_face_condition", "dynamic_pose_quality", "float32", "[P,Fm]",
        "DERIVED_FROM_AUTHORED_REST_AND_POSED_REFERENCE_SURFACE", ("T",), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_self_intersection", "dynamic_pose_quality", "ragged", "[P,*]",
        "DERIVED_FROM_AUTHORED_POSED_REFERENCE_SURFACE", ("T",), DIAGNOSTIC_ONLY,
        availability=OPTIONAL_DIAGNOSTIC,
    ),
    TeacherChannelSpecV1(
        "dynamic_canonical_rgba", "dynamic_pose_view", "uint8", "[P,8,H,W,4]",
        "DETERMINISTIC_AUTHORED_MOTION_RENDER", ("A", "S", "CAO"), DETERMINISTIC_RENDERED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_canonical_depth", "dynamic_pose_view", "float32", "[P,8,H,W]",
        "DETERMINISTIC_AUTHORED_MOTION_RENDER", ("S", "CAO"), DETERMINISTIC_RENDERED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_canonical_owner_id", "dynamic_pose_view", "int32", "[P,8,H,W]",
        "RASTERIZED_OWNER_PARTITION_UNDER_AUTHORED_MOTION", ("S", "CAO"), DETERMINISTIC_RENDERED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_ray_hit_stack", "dynamic_pose_view_ray", "ragged", "[P,8,H,W,*]",
        "ALL_HIT_RAY_QUERY_UNDER_AUTHORED_MOTION", ("S", "CAO"), DETERMINISTIC_RENDERED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_overlap_relation", "dynamic_pose_relation", "json", "[P,Eview]",
        "DERIVED_FROM_DYNAMIC_RAY_HIT_STACK", ("S", "CAO"), DETERMINISTIC_DERIVED,
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_contact_evidence", "dynamic_pose_relation", "json", "[P,Econtact]",
        "EXPLICIT_SOURCE_CONTACT_OR_GEOMETRIC_MOTION_CONTACT_WITH_ROW_PROVENANCE", ("S", "T", "CAO"),
        QUALIFIED_DERIVED, availability=REQUIRED_WITH_VALIDITY,
        notes="Rows distinguish explicit authored relation from geometry/motion-derived persistent contact evidence.",
    ),
)


HEAD_TARGETS_V1: Mapping[str, tuple[str, ...]] = {
    "NERF_SCAFFOLD": (
        "canonical_rgba", "canonical_owner_id", "canonical_material_region_id",
        "decomposition_state_manifest", "decomposition_rgba", "decomposition_owner_id",
        "decomposition_material_region_id",
    ),
    "G": (
        "source_vertices", "source_vertex_normals", "teacher_triangles",
        "geometric_component_manifest", "component_sign_authority", "surface_xyz",
        "surface_normal", "field_query_xyz", "field_unsigned_distance",
        "field_signed_distance", "field_nearest_surface_address", "canonical_depth",
        "canonical_normal", "decomposition_depth", "decomposition_normal",
    ),
    "A": (
        "source_uv", "source_face_material_id", "source_material_manifest",
        "surface_rgba", "surface_material_id", "surface_material_region_id",
        "canonical_rgba", "canonical_material_region_id", "decomposition_state_manifest",
        "decomposition_rgba", "decomposition_material_region_id",
        "decomposition_surface_address", "amodal_material_mask", "dynamic_canonical_rgba",
    ),
    "S": (
        "source_face_object_id", "source_object_parent_graph", "geometric_component_manifest",
        "joint_hierarchy", "raw_skin_joint_indices", "raw_skin_joint_weights",
        "normalized_skin_joint_indices", "normalized_skin_joint_weights",
        "mechanical_owner_joint", "deformation_mode", "attachment_relation",
        "surface_source_object_id", "surface_geometric_component_id",
        "surface_normalized_skin", "owner_partition_manifest", "surface_owner_region_id",
        "surface_material_region_id", "component_visibility_graph", "canonical_owner_id",
        "canonical_material_region_id", "decomposition_state_manifest", "decomposition_owner_id",
        "decomposition_material_region_id", "amodal_owner_mask", "amodal_material_mask",
        "ray_hit_stack", "pairwise_overlap_relation", "dynamic_pose_manifest",
        "dynamic_pose_joint_matrices", "dynamic_canonical_owner_id", "dynamic_ray_hit_stack",
        "dynamic_overlap_relation", "dynamic_contact_evidence",
    ),
    "F": (
        "teacher_triangles", "teacher_triangle_source_binding", "component_sign_authority",
        "surface_xyz", "surface_normal", "field_query_xyz", "field_unsigned_distance",
        "field_signed_distance", "canonical_rgba", "canonical_depth", "canonical_normal",
        "decomposition_depth", "decomposition_normal",
    ),
    "T": (
        "teacher_triangles", "teacher_triangle_source_binding", "geometric_component_manifest",
        "joint_hierarchy", "inverse_bind_matrices", "normalized_skin_joint_indices",
        "normalized_skin_joint_weights", "mechanical_owner_joint", "deformation_mode",
        "attachment_relation", "mechanical_reference_vertices", "mechanical_reference_faces",
        "mechanical_reference_source_binding", "surface_xyz", "surface_normal",
        "surface_source_address", "surface_normalized_skin", "owner_partition_manifest",
        "surface_owner_region_id", "canonical_surface_address", "dynamic_pose_manifest",
        "dynamic_pose_joint_matrices", "dynamic_reference_vertex_xyz",
        "dynamic_reference_vertex_normal", "dynamic_reference_jacobian",
        "dynamic_edge_stretch", "dynamic_face_area_ratio", "dynamic_face_condition",
        "dynamic_contact_evidence",
    ),
    "CAO": (
        "source_format_manifest", "source_uv", "source_face_object_id",
        "source_face_material_id", "source_object_parent_graph", "source_material_manifest",
        "geometric_component_manifest", "mechanical_owner_joint", "deformation_mode",
        "attachment_relation", "surface_xyz", "surface_normal", "surface_rgba",
        "surface_source_address", "surface_source_object_id", "surface_geometric_component_id",
        "surface_material_id", "owner_partition_manifest", "surface_owner_region_id",
        "surface_material_region_id", "component_visibility_graph", "canonical_rgba",
        "canonical_depth", "canonical_owner_id", "canonical_material_region_id",
        "canonical_surface_address", "decomposition_state_manifest", "decomposition_rgba",
        "decomposition_depth", "decomposition_owner_id", "decomposition_material_region_id",
        "decomposition_surface_address", "amodal_owner_mask", "amodal_material_mask",
        "ray_hit_stack", "pairwise_overlap_relation", "dynamic_pose_manifest",
        "dynamic_reference_vertex_xyz", "dynamic_reference_vertex_normal",
        "dynamic_canonical_rgba", "dynamic_canonical_depth", "dynamic_canonical_owner_id",
        "dynamic_ray_hit_stack", "dynamic_overlap_relation", "dynamic_contact_evidence",
    ),
}


@dataclass(frozen=True)
class FIT8FullTruthTeacherBundleManifestV1:
    subject_key: str
    source_sha256: str
    canonical_camera_contract_sha256: str
    normalization_contract_sha256: str
    channel_artifacts: Mapping[str, str]
    channel_validity: Mapping[str, str] = field(default_factory=dict)
    dynamic_pose_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("FIT8_TEACHER_SCHEMA_VERSION_INVALID")
        if not self.subject_key:
            raise ValueError("FIT8_TEACHER_SUBJECT_KEY_MISSING")
        for name, value in (
            ("SOURCE", self.source_sha256),
            ("CAMERA", self.canonical_camera_contract_sha256),
            ("NORMALIZATION", self.normalization_contract_sha256),
        ):
            if len(str(value)) != 64:
                raise ValueError(f"FIT8_TEACHER_{name}_SHA_INVALID")

        known = {spec.name: spec for spec in TEACHER_CHANNELS_V1}
        unknown = set(self.channel_artifacts) - set(known)
        if unknown:
            raise ValueError(f"FIT8_TEACHER_UNKNOWN_CHANNEL:{sorted(unknown)}")
        unknown_validity = set(self.channel_validity) - set(known)
        if unknown_validity:
            raise ValueError(f"FIT8_TEACHER_UNKNOWN_VALIDITY_CHANNEL:{sorted(unknown_validity)}")

        missing = [
            spec.name for spec in TEACHER_CHANNELS_V1
            if spec.artifact_required and not self.channel_artifacts.get(spec.name)
        ]
        if missing:
            raise ValueError(f"FIT8_TEACHER_REQUIRED_CHANNEL_MISSING:{missing}")
        missing_validity = [
            spec.name for spec in TEACHER_CHANNELS_V1
            if spec.availability == REQUIRED_WITH_VALIDITY and not self.channel_validity.get(spec.name)
        ]
        if missing_validity:
            raise ValueError(f"FIT8_TEACHER_VALIDITY_ARTIFACT_MISSING:{missing_validity}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_schema_v1() -> None:
    names = []
    for spec in TEACHER_CHANNELS_V1:
        spec.validate()
        names.append(spec.name)
    if len(names) != len(set(names)):
        raise ValueError("FIT8_TEACHER_CHANNEL_DUPLICATE")
    for head, channels in HEAD_TARGETS_V1.items():
        missing = sorted(set(channels) - set(names))
        if missing:
            raise ValueError(f"FIT8_TEACHER_HEAD_TARGET_UNKNOWN:{head}:{missing}")


validate_schema_v1()


__all__ = [
    "AVAILABILITY_POLICIES",
    "CANONICAL_VIEW_COUNT",
    "DETERMINISTIC_DERIVED",
    "DETERMINISTIC_RENDERED",
    "DIAGNOSTIC_ONLY",
    "FIT8FullTruthTeacherBundleManifestV1",
    "HEAD_TARGETS_V1",
    "OPTIONAL_DIAGNOSTIC",
    "QUALIFIED_DERIVED",
    "RAW_AUTHORED",
    "REQUIRED",
    "REQUIRED_WITH_VALIDITY",
    "SCHEMA_VERSION",
    "TEACHER_CHANNELS_V1",
    "TRUTH_AUTHORITY_CLASSES",
    "TeacherChannelSpecV1",
    "validate_schema_v1",
]
