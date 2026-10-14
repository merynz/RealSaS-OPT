from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "RealSaS.FIT8FullTruthTeacherBundle.v1"
CANONICAL_VIEW_COUNT = 8


@dataclass(frozen=True)
class TeacherChannelSpecV1:
    name: str
    domain: str
    dtype: str
    shape: str
    producer_truth: str
    consumed_by: tuple[str, ...]
    required: bool = True
    notes: str = ""

    def validate(self) -> None:
        if not self.name or not self.domain or not self.dtype or not self.shape:
            raise ValueError("FIT8_TEACHER_CHANNEL_IDENTITY_MISSING")
        if not self.producer_truth:
            raise ValueError("FIT8_TEACHER_CHANNEL_PRODUCER_MISSING")
        if not self.consumed_by:
            raise ValueError("FIT8_TEACHER_CHANNEL_CONSUMER_MISSING")


# The schema is deliberately factored by truth type rather than a single flat
# semantic class map. This lets Snake, Spider, Horse, Bat, Blob, Shark and Mech
# share one contract without forcing human anatomy or StdGEN's hair/cloth/body
# ontology onto every subject.
TEACHER_CHANNELS_V1 = (
    TeacherChannelSpecV1(
        "surface_xyz", "continuous_surface", "float32", "[N,3]",
        "AUTHORED_GEOMETRY", ("G", "A", "S", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "surface_normal", "continuous_surface", "float32", "[N,3]",
        "AUTHORED_GEOMETRY", ("G", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "surface_rgba", "continuous_surface", "float32", "[N,4]",
        "AUTHORED_MATERIAL_UV", ("A", "CAO"),
    ),
    TeacherChannelSpecV1(
        "source_face_id", "continuous_surface", "int64", "[N]",
        "AUTHORED_TOPOLOGY", ("T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "source_object_id", "continuous_surface", "int64", "[N]",
        "AUTHORED_OBJECT_HIERARCHY", ("S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "source_material_id", "continuous_surface", "int64", "[N]",
        "AUTHORED_MATERIAL_SLOTS", ("A", "S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "skin_joint_indices", "continuous_surface", "int32", "[N,K]",
        "AUTHORED_SKIN", ("S", "T"),
    ),
    TeacherChannelSpecV1(
        "skin_joint_weights", "continuous_surface", "float32", "[N,K]",
        "AUTHORED_SKIN", ("S", "T"),
    ),
    TeacherChannelSpecV1(
        "dominant_joint_id", "continuous_surface", "int32", "[N]",
        "DERIVED_FROM_AUTHORED_SKIN", ("S",),
    ),
    TeacherChannelSpecV1(
        "deformation_mode", "component", "enum", "[C]",
        "AUTHORED_SKIN_PARENT_CONSTRAINT", ("S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "attachment_relation", "component_graph", "json", "[E]",
        "AUTHORED_PARENT_CONSTRAINT_RIG", ("S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "owner_instance_id", "continuous_surface", "int32", "[N]",
        "AUTHORED_COMPONENT_IDENTITY", ("S", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "material_region_id", "continuous_surface", "int32", "[N]",
        "AUTHORED_MATERIAL_IDENTITY", ("A", "S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_rgb", "canonical_view", "uint8", "[8,H,W,4]",
        "DETERMINISTIC_RENDER", ("G", "A", "S", "F", "T"),
    ),
    TeacherChannelSpecV1(
        "canonical_depth", "canonical_view", "float32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("G", "F", "T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_normal", "canonical_view", "float32", "[8,H,W,3]",
        "DETERMINISTIC_RENDER", ("G", "F", "T"),
    ),
    TeacherChannelSpecV1(
        "canonical_owner_id", "canonical_view", "int32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_material_id", "canonical_view", "int32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("A", "S", "CAO"),
    ),
    TeacherChannelSpecV1(
        "canonical_face_id", "canonical_view", "int32", "[8,H,W]",
        "DETERMINISTIC_RENDER", ("T", "CAO"),
    ),
    TeacherChannelSpecV1(
        "amodal_owner_mask", "canonical_view_component", "bool", "[8,C,H,W]",
        "COMPONENT_ISOLATION_RENDER", ("S", "CAO"),
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
    TeacherChannelSpecV1(
        "dynamic_owner_depth", "dynamic_pose_view", "ragged", "[P,8,H,W]",
        "AUTHORED_ANIMATION_RENDER", ("S", "CAO"),
        notes="Representative motion poses; RGB is optional here, ownership/order truth is not.",
    ),
    TeacherChannelSpecV1(
        "dynamic_overlap_relation", "dynamic_pose_relation", "json", "[P,E_view]",
        "DERIVED_FROM_DYNAMIC_RAY_HITS", ("S", "CAO"),
    ),
)


HEAD_TARGETS_V1: Mapping[str, tuple[str, ...]] = {
    "G": (
        "surface_xyz", "surface_normal", "canonical_depth", "canonical_normal",
    ),
    "A": (
        "surface_rgba", "source_material_id", "material_region_id",
        "canonical_rgb", "canonical_material_id",
    ),
    "S": (
        "source_object_id", "skin_joint_indices", "skin_joint_weights",
        "dominant_joint_id", "deformation_mode", "attachment_relation",
        "owner_instance_id", "material_region_id", "canonical_owner_id",
        "canonical_material_id", "amodal_owner_mask", "ray_hit_stack",
        "pairwise_overlap_relation", "dynamic_owner_depth",
        "dynamic_overlap_relation",
    ),
    "F": (
        "surface_xyz", "surface_normal", "canonical_rgb", "canonical_depth",
        "canonical_normal",
    ),
    "T": (
        "surface_xyz", "surface_normal", "source_face_id", "source_object_id",
        "skin_joint_indices", "skin_joint_weights", "owner_instance_id",
        "canonical_rgb", "canonical_depth", "canonical_normal", "canonical_face_id",
    ),
    "CAO": (
        "surface_xyz", "surface_normal", "surface_rgba", "source_face_id",
        "source_object_id", "source_material_id", "deformation_mode",
        "attachment_relation", "owner_instance_id", "material_region_id",
        "canonical_depth", "canonical_owner_id", "canonical_material_id",
        "canonical_face_id", "amodal_owner_mask", "ray_hit_stack",
        "pairwise_overlap_relation", "dynamic_owner_depth",
        "dynamic_overlap_relation",
    ),
}


@dataclass(frozen=True)
class FIT8FullTruthTeacherBundleManifestV1:
    subject_key: str
    source_sha256: str
    canonical_camera_contract_sha256: str
    normalization_contract_sha256: str
    channel_artifacts: Mapping[str, str]
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
        missing = [
            spec.name for spec in TEACHER_CHANNELS_V1
            if spec.required and not self.channel_artifacts.get(spec.name)
        ]
        if missing:
            raise ValueError(f"FIT8_TEACHER_REQUIRED_CHANNEL_MISSING:{missing}")

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
    "CANONICAL_VIEW_COUNT",
    "FIT8FullTruthTeacherBundleManifestV1",
    "HEAD_TARGETS_V1",
    "SCHEMA_VERSION",
    "TEACHER_CHANNELS_V1",
    "TeacherChannelSpecV1",
    "validate_schema_v1",
]
