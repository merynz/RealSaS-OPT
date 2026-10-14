from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "RealSaS.FIT8FullTruthTeacherBundle.v1"
CANONICAL_VIEW_COUNT = 8

REQUIRED = "REQUIRED"
REQUIRED_WITH_VALIDITY = "REQUIRED_WITH_VALIDITY"
OPTIONAL_DIAGNOSTIC = "OPTIONAL_DIAGNOSTIC"
AVAILABILITY_POLICIES = {REQUIRED, REQUIRED_WITH_VALIDITY, OPTIONAL_DIAGNOSTIC}


@dataclass(frozen=True)
class TeacherChannelSpecV1:
    name: str
    domain: str
    dtype: str
    shape: str
    producer_truth: str
    consumed_by: tuple[str, ...]
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

    @property
    def artifact_required(self) -> bool:
        return self.availability != OPTIONAL_DIAGNOSTIC


# The schema is factored by truth type rather than one flat semantic label map.
# This lets Snake, Spider, Horse, Bat, Blob, Shark and Mech share one contract
# without forcing human anatomy or StdGEN's hair/cloth/body ontology onto every
# subject. REQUIRED_WITH_VALIDITY means the bundle must contain an artifact that
# explicitly says which entries are valid; absence is never silently imputed.
TEACHER_CHANNELS_V1 = (
    # ------------------------------------------------------------------
    # Exact authored carrier/topology domain.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "authored_vertices", "authored_mesh", "float32", "[V,3]",
        "AUTHORED_GEOMETRY", ("G", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "authored_faces", "authored_mesh", "int64", "[F,3]",
        "AUTHORED_TOPOLOGY", ("F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "authored_vertex_normals", "authored_mesh", "float32", "[V,3]",
        "AUTHORED_OR_DETERMINISTIC_GEOMETRY", ("G", "F", "T", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "authored_uv", "authored_mesh_corner", "float32", "[F,3,2]",
        "AUTHORED_UV", ("A", "S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "face_object_id", "authored_mesh", "int32", "[F]",
        "AUTHORED_OBJECT_HIERARCHY", ("S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "face_material_id", "authored_mesh", "int32", "[F]",
        "AUTHORED_MATERIAL_SLOTS", ("A", "S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "object_parent_graph", "component_graph", "json", "[C,E]",
        "AUTHORED_OBJECT_HIERARCHY", ("S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "material_manifest", "material_graph", "json", "[M]",
        "AUTHORED_MATERIAL_TEXTURES", ("A", "S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),

    # ------------------------------------------------------------------
    # Mechanical authorship and deformation domain.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "joint_hierarchy", "rig", "json", "[J,E]",
        "AUTHORED_RIG", ("S", "T"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "inverse_bind_matrices", "rig", "float32", "[J,4,4]",
        "AUTHORED_SKIN", ("S", "T"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "skin_joint_indices", "authored_mesh_vertex", "int32", "[V,K]",
        "AUTHORED_SKIN", ("S", "T"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "skin_joint_weights", "authored_mesh_vertex", "float32", "[V,K]",
        "AUTHORED_SKIN", ("S", "T"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dominant_joint_id", "authored_mesh_vertex", "int32", "[V]",
        "DERIVED_FROM_AUTHORED_SKIN", ("S",),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "deformation_mode", "component", "enum", "[C]",
        "AUTHORED_SKIN_PARENT_CONSTRAINT", ("S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "attachment_relation", "component_graph", "json", "[E]",
        "AUTHORED_PARENT_CONSTRAINT_RIG", ("S", "T", "CAO"),
    ),

    # ------------------------------------------------------------------
    # Continuous shared-surface samples: the same sample identity supervises
    # geometry, appearance and semantic ownership.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "surface_xyz", "continuous_surface", "float32", "[N,3]",
        "AUTHORED_GEOMETRY", ("G", "A", "S", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "surface_normal", "continuous_surface", "float32", "[N,3]",
        "AUTHORED_OR_DETERMINISTIC_GEOMETRY", ("G", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "surface_rgba", "continuous_surface", "float32", "[N,4]",
        "AUTHORED_MATERIAL_UV_OR_RENDER", ("A", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "source_face_id", "continuous_surface", "int64", "[N]",
        "AUTHORED_TOPOLOGY", ("T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "source_barycentric", "continuous_surface", "float32", "[N,3]",
        "AUTHORED_TOPOLOGY", ("A", "S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "source_object_id", "continuous_surface", "int64", "[N]",
        "AUTHORED_OBJECT_HIERARCHY", ("S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "source_material_id", "continuous_surface", "int64", "[N]",
        "AUTHORED_MATERIAL_SLOTS", ("A", "S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "owner_instance_id", "continuous_surface", "int32", "[N]",
        "AUTHORED_COMPONENT_IDENTITY", ("S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "material_region_id", "continuous_surface", "int32", "[N]",
        "AUTHORED_MATERIAL_IDENTITY", ("A", "S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),

    # ------------------------------------------------------------------
    # Canonical eight-view ground truth. These are teacher targets, not
    # additional non-RGB inference inputs.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "canonical_rgba", "canonical_view", "uint8", "[8,H,W,4]",
        "DETERMINISTIC_RENDER", ("G", "A", "S", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_depth", "canonical_view", "float32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("G", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_normal", "canonical_view", "float32", "[8,H,W,3]",
        "DETERMINISTIC_RENDER", ("G", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_owner_id", "canonical_view", "int32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_material_id", "canonical_view", "int32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("A", "S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "canonical_face_id", "canonical_view", "int32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_barycentric", "canonical_view", "float32", "[8,H,W,3]",
        "DETERMINISTIC_RENDER", ("A", "S", "T", "CAO"),
    ),

    # ------------------------------------------------------------------
    # Generic semantic decomposition / hidden truth.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "amodal_owner_mask", "canonical_view_component", "bool", "[8,C,H,W]",
        "COMPONENT_ISOLATION_RENDER", ("S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "amodal_material_mask", "canonical_view_material", "bool", "[8,M,H,W]",
        "MATERIAL_ISOLATION_RENDER", ("A", "S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "ray_hit_stack", "canonical_view_ray", "ragged", "[8,H,W,*]",
        "ALL_HIT_RAY_QUERY", ("S", "CAO"),
        notes="Ordered face/owner/material/depth/barycentric hits; never flattened to nearest-Z only.",
    ),
    TeacherChannelSpecV1(
        "pairwise_overlap_relation", "canonical_view_relation", "json", "[E_view]",
        "DERIVED_FROM_RAY_HIT_STACK", ("S", "CAO"),
    ),

    # ------------------------------------------------------------------
    # Dynamic authored truth. This is what allows topology training to reward
    # an animation-suitable carrier rather than only a pretty static surface.
    # ------------------------------------------------------------------
    TeacherChannelSpecV1(
        "dynamic_pose_joint_matrices", "dynamic_pose_rig", "float32", "[P,J,4,4]",
        "AUTHORED_ANIMATION", ("S", "T"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_pose_vertex_xyz", "dynamic_pose_mesh", "float32", "[P,V,3]",
        "AUTHORED_ANIMATION_PLUS_SKIN", ("T", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_pose_vertex_normal", "dynamic_pose_mesh", "float32", "[P,V,3]",
        "AUTHORED_ANIMATION_PLUS_SKIN", ("T", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_edge_stretch", "dynamic_pose_quality", "float32", "[P,E_mesh]",
        "DERIVED_FROM_AUTHORED_POSED_MESH", ("T",),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_face_area_ratio", "dynamic_pose_quality", "float32", "[P,F]",
        "DERIVED_FROM_AUTHORED_POSED_MESH", ("T",),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_face_condition", "dynamic_pose_quality", "float32", "[P,F]",
        "DERIVED_FROM_AUTHORED_POSED_MESH", ("T",),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_self_intersection", "dynamic_pose_quality", "ragged", "[P,*]",
        "DERIVED_FROM_AUTHORED_POSED_MESH", ("T",),
        availability=OPTIONAL_DIAGNOSTIC,
    ),
    TeacherChannelSpecV1(
        "dynamic_owner_depth", "dynamic_pose_view", "ragged", "[P,8,H,W]",
        "AUTHORED_ANIMATION_RENDER", ("S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
        notes="Representative motion poses; RGB is optional here, ownership/order truth is not.",
    ),
    TeacherChannelSpecV1(
        "dynamic_overlap_relation", "dynamic_pose_relation", "json", "[P,E_view]",
        "DERIVED_FROM_DYNAMIC_RAY_HITS", ("S", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
    TeacherChannelSpecV1(
        "dynamic_contact_relation", "dynamic_pose_relation", "json", "[P,E_contact]",
        "AUTHORED_ATTACHMENT_PLUS_GEOMETRIC_CONTACT", ("S", "T", "CAO"),
        availability=REQUIRED_WITH_VALIDITY,
    ),
)


HEAD_TARGETS_V1: Mapping[str, tuple[str, ...]] = {
    "G": (
        "authored_vertices", "authored_vertex_normals", "surface_xyz",
        "surface_normal", "canonical_depth", "canonical_normal",
    ),
    "A": (
        "authored_uv", "face_material_id", "material_manifest", "surface_rgba",
        "source_material_id", "material_region_id", "canonical_rgba",
        "canonical_material_id", "amodal_material_mask",
    ),
    "S": (
        "face_object_id", "object_parent_graph", "joint_hierarchy",
        "skin_joint_indices", "skin_joint_weights", "dominant_joint_id",
        "deformation_mode", "attachment_relation", "source_object_id",
        "owner_instance_id", "material_region_id", "canonical_owner_id",
        "canonical_material_id", "amodal_owner_mask", "amodal_material_mask",
        "ray_hit_stack", "pairwise_overlap_relation", "dynamic_pose_joint_matrices",
        "dynamic_owner_depth", "dynamic_overlap_relation", "dynamic_contact_relation",
    ),
    "F": (
        "authored_vertices", "authored_faces", "authored_vertex_normals",
        "surface_xyz", "surface_normal", "canonical_rgba", "canonical_depth",
        "canonical_normal",
    ),
    "T": (
        "authored_vertices", "authored_faces", "authored_vertex_normals",
        "face_object_id", "joint_hierarchy", "inverse_bind_matrices",
        "skin_joint_indices", "skin_joint_weights", "deformation_mode",
        "attachment_relation", "surface_xyz", "surface_normal", "source_face_id",
        "source_barycentric", "source_object_id", "owner_instance_id",
        "canonical_rgba", "canonical_depth", "canonical_normal", "canonical_face_id",
        "canonical_barycentric", "dynamic_pose_joint_matrices",
        "dynamic_pose_vertex_xyz", "dynamic_pose_vertex_normal",
        "dynamic_edge_stretch", "dynamic_face_area_ratio", "dynamic_face_condition",
        "dynamic_contact_relation",
    ),
    "CAO": (
        "authored_vertices", "authored_faces", "authored_vertex_normals",
        "authored_uv", "face_object_id", "face_material_id", "object_parent_graph",
        "material_manifest", "deformation_mode", "attachment_relation", "surface_xyz",
        "surface_normal", "surface_rgba", "source_face_id", "source_barycentric",
        "source_object_id", "source_material_id", "owner_instance_id",
        "material_region_id", "canonical_rgba", "canonical_depth", "canonical_normal",
        "canonical_owner_id", "canonical_material_id", "canonical_face_id",
        "canonical_barycentric", "amodal_owner_mask", "amodal_material_mask",
        "ray_hit_stack", "pairwise_overlap_relation", "dynamic_pose_vertex_xyz",
        "dynamic_pose_vertex_normal", "dynamic_owner_depth", "dynamic_overlap_relation",
        "dynamic_contact_relation",
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
    "FIT8FullTruthTeacherBundleManifestV1",
    "HEAD_TARGETS_V1",
    "OPTIONAL_DIAGNOSTIC",
    "REQUIRED",
    "REQUIRED_WITH_VALIDITY",
    "SCHEMA_VERSION",
    "TEACHER_CHANNELS_V1",
    "TeacherChannelSpecV1",
    "validate_schema_v1",
]
