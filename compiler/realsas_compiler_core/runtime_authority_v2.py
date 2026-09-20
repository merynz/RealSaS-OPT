from __future__ import annotations

"""Typed deterministic Runtime V2 projection/package authority."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class RuntimeViewV2IR:
    view_index: int
    view_id: str
    camera: Json
    texture_path: str
    texture_sha256: str
    schema_version: str = "RealSaS.RuntimeViewIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RuntimeClipV2IR:
    clip_id: str
    duration_seconds: float
    loop: bool
    frame_count: int
    array_prefix: str
    schema_version: str = "RealSaS.RuntimeClipIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RuntimeProjectionV2IR:
    complete_puppet_binding_hash: str
    mechanical_state_binding_hash: str
    mesh_binding_hash: str
    dynamic_motion_binding_hash: str
    appearance_asset_binding_hash: str
    appearance_qualification_binding_hash: str
    camera_set_binding_hash: str
    visibility_contract_hash: str
    projection_npz_path: str
    projection_npz_sha256: str
    provenance_npz_path: str
    provenance_npz_sha256: str
    views: tuple[RuntimeViewV2IR, ...]
    clips: tuple[RuntimeClipV2IR, ...]
    projection_hash: str
    schema_version: str = "RealSaS.RuntimeProjectionIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def runtime_projection_hash(value: RuntimeProjectionV2IR) -> str:
    payload = value.to_dict()
    payload.pop("projection_hash", None)
    return content_sha256(payload)


def validate_runtime_projection(value: RuntimeProjectionV2IR) -> None:
    views = tuple(sorted(value.views, key=lambda row: row.view_index))
    if len(views) != 8 or tuple(row.view_index for row in views) != tuple(range(8)):
        raise QualificationError("RUNTIME_V2_REQUIRES_V0_V7")
    if tuple(row.view_id for row in views) != tuple(f"V{i}" for i in range(8)):
        raise QualificationError("RUNTIME_V2_VIEW_ID_DRIFT")
    if not value.clips:
        raise QualificationError("RUNTIME_V2_CLIP_SET_EMPTY")
    if len({row.clip_id for row in value.clips}) != len(value.clips):
        raise QualificationError("RUNTIME_V2_CLIP_ID_DUPLICATE")
    if any(row.frame_count <= 0 for row in value.clips):
        raise QualificationError("RUNTIME_V2_CLIP_FRAME_EMPTY")
    if value.projection_hash != runtime_projection_hash(value):
        raise QualificationError("RUNTIME_V2_PROJECTION_HASH_DRIFT")


@dataclass(frozen=True)
class RuntimePackageSealV2IR:
    projection_binding_hash: str
    archive_path: str
    archive_sha256: str
    archive_bytes: int
    package_format: str
    entry_names: tuple[str, ...]
    package_hash: str
    schema_version: str = "RealSaS.RuntimePackageSealIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def runtime_package_hash(value: RuntimePackageSealV2IR) -> str:
    payload = value.to_dict()
    payload.pop("package_hash", None)
    return content_sha256(payload)


@dataclass(frozen=True)
class NativePlaybackProbeV2IR:
    clip_id: str
    view_id: str
    frame_index: int
    rgba_raw_path: str
    rgba_raw_sha256: str
    provenance_raw_path: str
    provenance_raw_sha256: str
    owner_raw_path: str
    owner_raw_sha256: str
    stdout_sha256: str
    probe_hash: str
    schema_version: str = "RealSaS.NativePlaybackProbeIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def native_playback_probe_hash(value: NativePlaybackProbeV2IR) -> str:
    payload = value.to_dict()
    payload.pop("probe_hash", None)
    return content_sha256(payload)


@dataclass(frozen=True)
class NativePlaybackV2IR:
    package_binding_hash: str
    projection_binding_hash: str
    native_player_sha256: str
    probes: tuple[NativePlaybackProbeV2IR, ...]
    playback_hash: str
    schema_version: str = "RealSaS.NativePlaybackIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def native_playback_hash(value: NativePlaybackV2IR) -> str:
    payload = value.to_dict()
    payload.pop("playback_hash", None)
    return content_sha256(payload)


@dataclass(frozen=True)
class DynamicVisualIntegrityV2IR:
    package_binding_hash: str
    projection_binding_hash: str
    native_playback_binding_hash: str
    frame_count: int
    geometry_visible_pixel_count: int
    final_alpha_hole_pixel_count: int
    final_alpha_hole_fraction: float
    compiled_unobserved_visible_pixel_count: int
    compiled_unobserved_visible_fraction: float
    maximum_frame_compiled_unobserved_visible_fraction: float
    maximum_connected_compiled_unobserved_visible_fraction: float
    compiled_global_visible_pixel_count: int
    compiled_global_visible_fraction: float
    maximum_frame_micro_visible_pixel_fraction: float
    consequential_visible_face_count: int
    unmeasurable_consequential_visible_face_count: int
    exact_depth_ambiguous_pixel_count: int
    exact_depth_ambiguous_fraction: float
    maximum_frame_exact_depth_ambiguous_fraction: float
    native_reference_mismatch_pixel_count: int
    maximum_frame_native_reference_mismatch_fraction: float
    dynamic_conditioning_sample_count: int
    relative_conditioning_sample_count: int
    temporal_conditioning_sample_count: int
    maximum_uv_to_surface_condition_number: float
    maximum_relative_surface_condition_number: float
    maximum_relative_surface_principal_stretch: float
    maximum_adjacent_frame_surface_principal_stretch: float
    qualification_report: Json
    visual_integrity_hash: str
    schema_version: str = "RealSaS.DynamicVisualIntegrityIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def dynamic_visual_integrity_hash(value: DynamicVisualIntegrityV2IR) -> str:
    payload = value.to_dict()
    payload.pop("visual_integrity_hash", None)
    return content_sha256(payload)


@dataclass(frozen=True)
class ProductClosureV2IR:
    complete_puppet_binding_hash: str
    dynamic_motion_binding_hash: str
    runtime_projection_binding_hash: str
    runtime_package_binding_hash: str
    native_playback_binding_hash: str
    dynamic_visual_integrity_binding_hash: str
    editable_authoring_archive_path: str
    editable_authoring_archive_sha256: str
    qualification_report: Json
    product_closure_hash: str
    schema_version: str = "RealSaS.ProductClosureIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def product_closure_hash(value: ProductClosureV2IR) -> str:
    payload = value.to_dict()
    payload.pop("product_closure_hash", None)
    return content_sha256(payload)


def runtime_projection_from_dict(payload: Mapping[str, Any]) -> RuntimeProjectionV2IR:
    value = RuntimeProjectionV2IR(
        str(payload["complete_puppet_binding_hash"]),
        str(payload["mechanical_state_binding_hash"]),
        str(payload["mesh_binding_hash"]),
        str(payload["dynamic_motion_binding_hash"]),
        str(payload["appearance_asset_binding_hash"]),
        str(payload["appearance_qualification_binding_hash"]),
        str(payload["camera_set_binding_hash"]),
        str(payload["visibility_contract_hash"]),
        str(payload["projection_npz_path"]),
        str(payload["projection_npz_sha256"]),
        str(payload["provenance_npz_path"]),
        str(payload["provenance_npz_sha256"]),
        tuple(
            RuntimeViewV2IR(
                int(row["view_index"]),
                str(row["view_id"]),
                dict(row["camera"]),
                str(row["texture_path"]),
                str(row["texture_sha256"]),
                schema_version=str(row.get("schema_version") or "RealSaS.RuntimeViewIR.v2"),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in payload.get("views") or ()
        ),
        tuple(
            RuntimeClipV2IR(
                str(row["clip_id"]),
                float(row["duration_seconds"]),
                bool(row["loop"]),
                int(row["frame_count"]),
                str(row["array_prefix"]),
                schema_version=str(row.get("schema_version") or "RealSaS.RuntimeClipIR.v2"),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in payload.get("clips") or ()
        ),
        str(payload["projection_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.RuntimeProjectionIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_runtime_projection(value)
    return value


def runtime_package_seal_from_dict(payload: Mapping[str, Any]) -> RuntimePackageSealV2IR:
    value = RuntimePackageSealV2IR(
        projection_binding_hash=str(payload["projection_binding_hash"]),
        archive_path=str(payload["archive_path"]),
        archive_sha256=str(payload["archive_sha256"]),
        archive_bytes=int(payload["archive_bytes"]),
        package_format=str(payload["package_format"]),
        entry_names=tuple(map(str, payload.get("entry_names") or ())),
        package_hash=str(payload["package_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.RuntimePackageSealIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.package_hash != runtime_package_hash(value):
        raise QualificationError("RUNTIME_V2_PACKAGE_HASH_DRIFT")
    return value


def native_playback_from_dict(payload: Mapping[str, Any]) -> NativePlaybackV2IR:
    probes = tuple(
        NativePlaybackProbeV2IR(
            clip_id=str(row["clip_id"]),
            view_id=str(row["view_id"]),
            frame_index=int(row["frame_index"]),
            rgba_raw_path=str(row["rgba_raw_path"]),
            rgba_raw_sha256=str(row["rgba_raw_sha256"]),
            provenance_raw_path=str(row["provenance_raw_path"]),
            provenance_raw_sha256=str(row["provenance_raw_sha256"]),
            owner_raw_path=str(row["owner_raw_path"]),
            owner_raw_sha256=str(row["owner_raw_sha256"]),
            stdout_sha256=str(row["stdout_sha256"]),
            probe_hash=str(row["probe_hash"]),
            schema_version=str(row.get("schema_version") or "RealSaS.NativePlaybackProbeIR.v2"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("probes") or ()
    )
    for probe in probes:
        if probe.probe_hash != native_playback_probe_hash(probe):
            raise QualificationError("RUNTIME_V2_NATIVE_PROBE_HASH_DRIFT")
    value = NativePlaybackV2IR(
        package_binding_hash=str(payload["package_binding_hash"]),
        projection_binding_hash=str(payload["projection_binding_hash"]),
        native_player_sha256=str(payload["native_player_sha256"]),
        probes=probes,
        playback_hash=str(payload["playback_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.NativePlaybackIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.playback_hash != native_playback_hash(value):
        raise QualificationError("RUNTIME_V2_NATIVE_PLAYBACK_HASH_DRIFT")
    return value


def dynamic_visual_integrity_from_dict(payload: Mapping[str, Any]) -> DynamicVisualIntegrityV2IR:
    value = DynamicVisualIntegrityV2IR(
        package_binding_hash=str(payload["package_binding_hash"]),
        projection_binding_hash=str(payload["projection_binding_hash"]),
        native_playback_binding_hash=str(payload["native_playback_binding_hash"]),
        frame_count=int(payload["frame_count"]),
        geometry_visible_pixel_count=int(payload["geometry_visible_pixel_count"]),
        final_alpha_hole_pixel_count=int(payload["final_alpha_hole_pixel_count"]),
        final_alpha_hole_fraction=float(payload["final_alpha_hole_fraction"]),
        compiled_unobserved_visible_pixel_count=int(payload["compiled_unobserved_visible_pixel_count"]),
        compiled_unobserved_visible_fraction=float(payload["compiled_unobserved_visible_fraction"]),
        maximum_frame_compiled_unobserved_visible_fraction=float(payload["maximum_frame_compiled_unobserved_visible_fraction"]),
        maximum_connected_compiled_unobserved_visible_fraction=float(payload["maximum_connected_compiled_unobserved_visible_fraction"]),
        compiled_global_visible_pixel_count=int(payload["compiled_global_visible_pixel_count"]),
        compiled_global_visible_fraction=float(payload["compiled_global_visible_fraction"]),
        maximum_frame_micro_visible_pixel_fraction=float(payload["maximum_frame_micro_visible_pixel_fraction"]),
        consequential_visible_face_count=int(payload["consequential_visible_face_count"]),
        unmeasurable_consequential_visible_face_count=int(payload["unmeasurable_consequential_visible_face_count"]),
        exact_depth_ambiguous_pixel_count=int(payload["exact_depth_ambiguous_pixel_count"]),
        exact_depth_ambiguous_fraction=float(payload["exact_depth_ambiguous_fraction"]),
        maximum_frame_exact_depth_ambiguous_fraction=float(payload["maximum_frame_exact_depth_ambiguous_fraction"]),
        native_reference_mismatch_pixel_count=int(payload["native_reference_mismatch_pixel_count"]),
        maximum_frame_native_reference_mismatch_fraction=float(payload["maximum_frame_native_reference_mismatch_fraction"]),
        dynamic_conditioning_sample_count=int(payload["dynamic_conditioning_sample_count"]),
        relative_conditioning_sample_count=int(payload["relative_conditioning_sample_count"]),
        temporal_conditioning_sample_count=int(payload["temporal_conditioning_sample_count"]),
        maximum_uv_to_surface_condition_number=float(payload["maximum_uv_to_surface_condition_number"]),
        maximum_relative_surface_condition_number=float(payload["maximum_relative_surface_condition_number"]),
        maximum_relative_surface_principal_stretch=float(payload["maximum_relative_surface_principal_stretch"]),
        maximum_adjacent_frame_surface_principal_stretch=float(payload["maximum_adjacent_frame_surface_principal_stretch"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        visual_integrity_hash=str(payload["visual_integrity_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.DynamicVisualIntegrityIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.visual_integrity_hash != dynamic_visual_integrity_hash(value):
        raise QualificationError("RUNTIME_V2_VISUAL_INTEGRITY_HASH_DRIFT")
    return value
