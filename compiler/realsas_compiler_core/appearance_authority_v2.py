from __future__ import annotations

"""Typed Complete Appearance Authority artifacts for the RealSaS V2 product path."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]

CAA_BACKENDS = {
    "DETERMINISTIC_V1",
    "LEARNED_V2",
    "IM2SURFTEX_RESEARCH_ONLY",
}
CAA_PROVENANCE = {
    "DIRECT_SOURCE": 0,
    "OTHER_VIEW_SOURCE": 1,
    "COMPILED_NEAREST_SURFACE": 2,
    "COMPILED_GLOBAL_SURFACE": 3,
}
CAA_PROVENANCE_BY_CODE = {value: key for key, value in CAA_PROVENANCE.items()}


def _hash_without(value, field_name: str) -> str:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return content_sha256(payload)


@dataclass(frozen=True)
class CAACompilePreregistrationIR:
    backend_id: str
    shipping_eligible: bool
    contract_sha256: str
    observation_set_binding_hash: str
    camera_set_binding_hash: str
    output_direction_set_binding_hash: str
    candidate_mesh_binding_hash: str
    surface_addressing_binding_hash: str
    appearance_domain_binding_hash: str
    static_mesh_qualification_binding_hash: str
    compile_policy: Json
    source_lock_policy: Json
    completion_quality_policy: Json
    preregistration_hash: str
    schema_version: str = "RealSaS.CAACompilePreregistrationIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def caa_preregistration_hash(value: CAACompilePreregistrationIR) -> str:
    return _hash_without(value, "preregistration_hash")


def validate_caa_preregistration(value: CAACompilePreregistrationIR) -> None:
    if value.backend_id not in CAA_BACKENDS:
        raise QualificationError("CAA_BACKEND_UNSUPPORTED")
    if value.shipping_eligible != (
        value.backend_id != "IM2SURFTEX_RESEARCH_ONLY"
    ):
        raise QualificationError("CAA_BACKEND_SHIPPING_ELIGIBILITY_DRIFT")
    if len(value.contract_sha256) != 64:
        raise QualificationError("CAA_CONTRACT_SHA_INVALID")
    for name in (
        "observation_set_binding_hash",
        "camera_set_binding_hash",
        "output_direction_set_binding_hash",
        "candidate_mesh_binding_hash",
        "surface_addressing_binding_hash",
        "appearance_domain_binding_hash",
        "static_mesh_qualification_binding_hash",
    ):
        if not str(getattr(value, name)):
            raise QualificationError(f"CAA_PREREG_BINDING_MISSING:{name}")
    compile_policy = dict(value.compile_policy)
    tile_resolution = int(compile_policy.get("tile_resolution", 0))
    bleed_px = int(compile_policy.get("bleed_px", -1))
    if tile_resolution < 4 or bleed_px < 1:
        raise QualificationError("CAA_ATLAS_POLICY_INVALID")
    if str(compile_policy.get("atlas_layout")) != "UNIQUE_FACE_BARYCENTRIC_V1":
        raise QualificationError("CAA_ATLAS_LAYOUT_UNSUPPORTED")
    source_policy = dict(value.source_lock_policy)
    angle = float(source_policy.get("min_abs_normal_camera_cos", -1.0))
    erosion = int(source_policy.get("boundary_safe_erosion_px", -1))
    alpha = int(source_policy.get("min_source_alpha_u8", -1))
    if not (0.0 <= angle <= 1.0 and erosion >= 0 and 0 <= alpha <= 255):
        raise QualificationError("CAA_SOURCE_LOCK_POLICY_INVALID")
    if value.preregistration_hash != caa_preregistration_hash(value):
        raise QualificationError("CAA_PREREG_HASH_MISMATCH")


def build_caa_preregistration(
    *,
    backend_id: str,
    contract_sha256: str,
    observation_set_hash: str,
    camera_set_hash: str,
    output_direction_set_hash: str,
    candidate_mesh_hash: str,
    surface_addressing_hash: str,
    appearance_domain_hash: str,
    static_mesh_qualification_hash: str,
    compile_policy: Mapping[str, Any],
    source_lock_policy: Mapping[str, Any],
    completion_quality_policy: Mapping[str, Any],
) -> CAACompilePreregistrationIR:
    backend = str(backend_id)
    value = CAACompilePreregistrationIR(
        backend_id=backend,
        shipping_eligible=backend != "IM2SURFTEX_RESEARCH_ONLY",
        contract_sha256=str(contract_sha256),
        observation_set_binding_hash=str(observation_set_hash),
        camera_set_binding_hash=str(camera_set_hash),
        output_direction_set_binding_hash=str(output_direction_set_hash),
        candidate_mesh_binding_hash=str(candidate_mesh_hash),
        surface_addressing_binding_hash=str(surface_addressing_hash),
        appearance_domain_binding_hash=str(appearance_domain_hash),
        static_mesh_qualification_binding_hash=str(static_mesh_qualification_hash),
        compile_policy=dict(compile_policy),
        source_lock_policy=dict(source_lock_policy),
        completion_quality_policy=dict(completion_quality_policy),
        preregistration_hash="",
        metadata={
            "source_wins": True,
            "runtime_generation_forbidden": True,
            "presentation_warp_v1_forbidden": True,
            "internal_filtering": "PREMULTIPLIED_ALPHA",
            "transport_png_may_be_straight_alpha": True,
            "appearance_is_coequal_product_authority": True,
        },
    )
    value = replace(value, preregistration_hash=caa_preregistration_hash(value))
    validate_caa_preregistration(value)
    return value


@dataclass(frozen=True)
class CAACompileArtifactIR:
    backend_id: str
    preregistration_binding_hash: str
    candidate_mesh_binding_hash: str
    surface_addressing_binding_hash: str
    appearance_domain_binding_hash: str
    output_direction_set_binding_hash: str
    compile_npz_path: str
    compile_npz_sha256: str
    face_count: int
    direction_count: int
    tile_resolution: int
    sample_count_per_face: int
    total_sample_count: int
    direct_source_sample_count: int
    other_view_source_sample_count: int
    compiled_nearest_surface_sample_count: int
    compiled_global_surface_sample_count: int
    compile_hash: str
    schema_version: str = "RealSaS.CAACompileArtifactIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def caa_compile_hash(value: CAACompileArtifactIR) -> str:
    return _hash_without(value, "compile_hash")


def validate_caa_compile_artifact(value: CAACompileArtifactIR) -> None:
    if value.backend_id not in CAA_BACKENDS:
        raise QualificationError("CAA_COMPILE_BACKEND_UNSUPPORTED")
    if len(value.compile_npz_sha256) != 64:
        raise QualificationError("CAA_COMPILE_NPZ_SHA_INVALID")
    if (
        value.face_count <= 0
        or value.direction_count != 8
        or value.tile_resolution < 4
        or value.sample_count_per_face <= 0
    ):
        raise QualificationError("CAA_COMPILE_DIMENSION_INVALID")
    counted = (
        value.direct_source_sample_count
        + value.other_view_source_sample_count
        + value.compiled_nearest_surface_sample_count
        + value.compiled_global_surface_sample_count
    )
    if counted != value.total_sample_count:
        raise QualificationError("CAA_COMPILE_PROVENANCE_ACCOUNTING_DRIFT")
    if value.total_sample_count != (
        value.face_count * value.direction_count * value.sample_count_per_face
    ):
        raise QualificationError("CAA_COMPILE_TOTALITY_ACCOUNTING_DRIFT")
    if value.compile_hash != caa_compile_hash(value):
        raise QualificationError("CAA_COMPILE_HASH_MISMATCH")


@dataclass(frozen=True)
class CAACompileSealIR:
    compile_binding_hash: str
    preregistration_binding_hash: str
    compile_npz_sha256: str
    qualification_report: Json
    seal_hash: str
    schema_version: str = "RealSaS.CAACompileSealIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def caa_compile_seal_hash(value: CAACompileSealIR) -> str:
    return _hash_without(value, "seal_hash")


@dataclass(frozen=True)
class AppearanceTextureIR:
    direction_index: int
    direction_id: str
    transport_png_path: str
    transport_png_sha256: str
    width: int
    height: int
    schema_version: str = "RealSaS.AppearanceTextureIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CompleteAppearanceAssetIR:
    compile_seal_binding_hash: str
    candidate_mesh_binding_hash: str
    surface_addressing_binding_hash: str
    appearance_domain_binding_hash: str
    output_direction_set_binding_hash: str
    textures: tuple[AppearanceTextureIR, ...]
    uv_npz_path: str
    uv_npz_sha256: str
    provenance_npz_path: str
    provenance_npz_sha256: str
    atlas_layout: Json
    asset_hash: str
    schema_version: str = "RealSaS.CompleteAppearanceAssetIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def complete_appearance_asset_hash(value: CompleteAppearanceAssetIR) -> str:
    return _hash_without(value, "asset_hash")


def validate_complete_appearance_asset(value: CompleteAppearanceAssetIR) -> None:
    rows = tuple(sorted(value.textures, key=lambda row: row.direction_index))
    if len(rows) != 8 or tuple(row.direction_index for row in rows) != tuple(range(8)):
        raise QualificationError("CAA_ASSET_REQUIRES_V0_V7_TEXTURES")
    if tuple(row.direction_id for row in rows) != tuple(f"V{i}" for i in range(8)):
        raise QualificationError("CAA_ASSET_DIRECTION_ID_DRIFT")
    if any(len(row.transport_png_sha256) != 64 for row in rows):
        raise QualificationError("CAA_ASSET_TEXTURE_SHA_INVALID")
    if len(value.uv_npz_sha256) != 64 or len(value.provenance_npz_sha256) != 64:
        raise QualificationError("CAA_ASSET_AUXILIARY_SHA_INVALID")
    if value.asset_hash != complete_appearance_asset_hash(value):
        raise QualificationError("CAA_ASSET_HASH_MISMATCH")


@dataclass(frozen=True)
class CompleteAppearanceQualificationIR:
    asset_binding_hash: str
    preregistration_binding_hash: str
    source_lock_exact_fraction: float
    total_defined_fraction: float
    structured_holdout_sample_count: int
    structured_holdout_mean_rgba_l1: float
    structured_holdout_p95_rgba_l1: float
    provenance_boundary_pair_count: int
    provenance_boundary_mean_rgba_l1: float
    provenance_boundary_p95_rgba_l1: float
    provenance_boundary_gradient_pair_count: int
    provenance_boundary_mean_gradient_jump: float
    provenance_boundary_p95_gradient_jump: float
    qualification_report: Json
    qualification_hash: str
    schema_version: str = "RealSaS.CompleteAppearanceQualificationIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def complete_appearance_qualification_hash(
    value: CompleteAppearanceQualificationIR,
) -> str:
    return _hash_without(value, "qualification_hash")


@dataclass(frozen=True)
class CAARestViewProofIR:
    direction_index: int
    rendered_rgba_sha256: str
    rendered_alpha_pixel_count: int
    source_locked_pixel_count: int
    source_locked_exact_pixel_count: int
    source_locked_exact_fraction: float
    geometry_visible_pixel_count: int
    final_alpha_pixel_count: int
    geometry_visible_final_alpha_hole_count: int
    schema_version: str = "RealSaS.CAARestViewProofIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CAARestRenderProofIR:
    asset_binding_hash: str
    static_mesh_qualification_binding_hash: str
    camera_set_binding_hash: str
    views: tuple[CAARestViewProofIR, ...]
    qualification_report: Json
    proof_hash: str
    schema_version: str = "RealSaS.CAARestRenderProofIR.v2"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def caa_rest_render_proof_hash(value: CAARestRenderProofIR) -> str:
    return _hash_without(value, "proof_hash")


def caa_preregistration_from_dict(payload: Mapping[str, Any]) -> CAACompilePreregistrationIR:
    value = CAACompilePreregistrationIR(
        str(payload["backend_id"]),
        bool(payload["shipping_eligible"]),
        str(payload["contract_sha256"]),
        str(payload["observation_set_binding_hash"]),
        str(payload["camera_set_binding_hash"]),
        str(payload["output_direction_set_binding_hash"]),
        str(payload["candidate_mesh_binding_hash"]),
        str(payload["surface_addressing_binding_hash"]),
        str(payload["appearance_domain_binding_hash"]),
        str(payload["static_mesh_qualification_binding_hash"]),
        dict(payload.get("compile_policy") or {}),
        dict(payload.get("source_lock_policy") or {}),
        dict(payload.get("completion_quality_policy") or {}),
        str(payload["preregistration_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CAACompilePreregistrationIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_caa_preregistration(value)
    return value


def caa_compile_artifact_from_dict(payload: Mapping[str, Any]) -> CAACompileArtifactIR:
    value = CAACompileArtifactIR(
        str(payload["backend_id"]),
        str(payload["preregistration_binding_hash"]),
        str(payload["candidate_mesh_binding_hash"]),
        str(payload["surface_addressing_binding_hash"]),
        str(payload["appearance_domain_binding_hash"]),
        str(payload["output_direction_set_binding_hash"]),
        str(payload["compile_npz_path"]),
        str(payload["compile_npz_sha256"]),
        int(payload["face_count"]),
        int(payload["direction_count"]),
        int(payload["tile_resolution"]),
        int(payload["sample_count_per_face"]),
        int(payload["total_sample_count"]),
        int(payload["direct_source_sample_count"]),
        int(payload["other_view_source_sample_count"]),
        int(payload["compiled_nearest_surface_sample_count"]),
        int(payload["compiled_global_surface_sample_count"]),
        str(payload["compile_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CAACompileArtifactIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_caa_compile_artifact(value)
    return value


def caa_compile_seal_from_dict(payload: Mapping[str, Any]) -> CAACompileSealIR:
    value = CAACompileSealIR(
        str(payload["compile_binding_hash"]),
        str(payload["preregistration_binding_hash"]),
        str(payload["compile_npz_sha256"]),
        dict(payload.get("qualification_report") or {}),
        str(payload["seal_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CAACompileSealIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.seal_hash != caa_compile_seal_hash(value):
        raise QualificationError("CAA_COMPILE_SEAL_HASH_MISMATCH")
    return value


def complete_appearance_asset_from_dict(payload: Mapping[str, Any]) -> CompleteAppearanceAssetIR:
    textures = tuple(
        AppearanceTextureIR(
            int(row["direction_index"]),
            str(row["direction_id"]),
            str(row["transport_png_path"]),
            str(row["transport_png_sha256"]),
            int(row["width"]),
            int(row["height"]),
            schema_version=str(row.get("schema_version") or "RealSaS.AppearanceTextureIR.v2"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("textures") or ()
    )
    value = CompleteAppearanceAssetIR(
        str(payload["compile_seal_binding_hash"]),
        str(payload["candidate_mesh_binding_hash"]),
        str(payload["surface_addressing_binding_hash"]),
        str(payload["appearance_domain_binding_hash"]),
        str(payload["output_direction_set_binding_hash"]),
        textures,
        str(payload["uv_npz_path"]),
        str(payload["uv_npz_sha256"]),
        str(payload["provenance_npz_path"]),
        str(payload["provenance_npz_sha256"]),
        dict(payload.get("atlas_layout") or {}),
        str(payload["asset_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CompleteAppearanceAssetIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    validate_complete_appearance_asset(value)
    return value


def complete_appearance_qualification_from_dict(
    payload: Mapping[str, Any],
) -> CompleteAppearanceQualificationIR:
    value = CompleteAppearanceQualificationIR(
        str(payload["asset_binding_hash"]),
        str(payload["preregistration_binding_hash"]),
        float(payload["source_lock_exact_fraction"]),
        float(payload["total_defined_fraction"]),
        int(payload["structured_holdout_sample_count"]),
        float(payload["structured_holdout_mean_rgba_l1"]),
        float(payload["structured_holdout_p95_rgba_l1"]),
        int(payload["provenance_boundary_pair_count"]),
        float(payload["provenance_boundary_mean_rgba_l1"]),
        float(payload["provenance_boundary_p95_rgba_l1"]),
        int(payload["provenance_boundary_gradient_pair_count"]),
        float(payload["provenance_boundary_mean_gradient_jump"]),
        float(payload["provenance_boundary_p95_gradient_jump"]),
        dict(payload.get("qualification_report") or {}),
        str(payload["qualification_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.CompleteAppearanceQualificationIR.v2"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.qualification_hash != complete_appearance_qualification_hash(value):
        raise QualificationError("CAA_QUALIFICATION_HASH_MISMATCH")
    return value
