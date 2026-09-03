from __future__ import annotations

"""Exact current V4 proof/bake -> native runtime-v2 projection.

This service consumes only a PASS ProductProofBundleIR, the exact qualification-
owned motion bakes bound by its MOTION report, current appearance bindings, and
explicit texture payload identities. It never reruns a model, motion evaluator,
solver, proof operation, or repair.

Runtime rest XY comes from QualificationOwnedMotionBakeIR.rest_mesh_vertices_by_id,
never from mechanical mesh P.xy. Runtime UV is derived from the authoritative local
donor_raster_xy using the native C++ sampler's top-left endpoint-normalized contract.
Current material_uv is checked as a consistency witness, not reused as native UV.
"""

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.v4 import require_current_proof_bundle
from compiler.realsas_compiler_services.proof.motion_bake import (
    QualificationOwnedMotionBakeIR,
    assert_motion_bake_binding,
)
from .runtime_v2 import (
    RuntimeV2Clip,
    RuntimeV2Frame,
    RuntimeV2Mesh,
    RuntimeV2View,
    materialize_runtime_v2_archive,
)


PROJECTION_SCHEMA = "RealSaS.CurrentV4ProofBakeToNativeRuntimeV2.v1"
APPEARANCE_MATERIAL_UV_CONVENTION = "NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1"
NATIVE_RUNTIME_UV_CONVENTION = "PNG_TOP_LEFT_ENDPOINT_NORMALIZED_UV_V1"


@dataclass(frozen=True)
class RuntimeTexturePayloadV1:
    view_index: int
    image_relpath: str
    image_sha256: str
    image_crc32: int
    width: int
    height: int
    atlas_payload_hash: str
    schema_version: str = "RealSaS.RuntimeTexturePayload.v1"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RuntimeV2ProjectionResult:
    views: tuple[RuntimeV2View, ...]
    clips: tuple[RuntimeV2Clip, ...]
    projection_hash: str
    runtime_vertex_sources: Mapping[str, tuple[int, ...]]
    texture_bindings: tuple[RuntimeTexturePayloadV1, ...]
    source_motion_bake_hashes: Mapping[str, str]
    schema_version: str = PROJECTION_SCHEMA


def _motion_domain_report(proof_bundle):
    rows = [row for row in proof_bundle.domain_reports if row.proof_domain == "MOTION"]
    if len(rows) != 1:
        raise QualificationError("RUNTIME_V2_REQUIRES_EXACTLY_ONE_MOTION_PROOF")
    report = rows[0]
    if report.status != "PASS":
        raise QualificationError("RUNTIME_V2_REQUIRES_PASS_MOTION_PROOF")
    return report


def _texture_map(texture_bindings: Iterable[RuntimeTexturePayloadV1]):
    rows = tuple(sorted(texture_bindings, key=lambda row: int(row.view_index)))
    if tuple(int(row.view_index) for row in rows) != tuple(range(8)):
        raise QualificationError("RUNTIME_V2_REQUIRES_TEXTURE_BINDINGS_0_TO_7")
    out = {}
    for row in rows:
        if not row.image_relpath or len(str(row.image_sha256)) != 64:
            raise QualificationError("RUNTIME_V2_TEXTURE_IDENTITY_INVALID")
        if int(row.width) <= 0 or int(row.height) <= 0:
            raise QualificationError("RUNTIME_V2_TEXTURE_DIMENSIONS_INVALID")
        expected_atlas = content_sha256({
            "image_sha256": str(row.image_sha256),
            "image_relpath": str(row.image_relpath),
        })
        if expected_atlas != str(row.atlas_payload_hash):
            raise QualificationError("RUNTIME_V2_TEXTURE_ATLAS_LINEAGE_MISMATCH")
        out[int(row.view_index)] = row
    return rows, out


def _bake_map(product, proof_bundle, bakes: Iterable[QualificationOwnedMotionBakeIR]):
    motion_report = _motion_domain_report(proof_bundle)
    metadata = dict(motion_report.metadata or {})
    expected_hashes = dict(metadata.get("qualification_owned_bake_hashes") or {})
    expected_provider = str(metadata.get("qualified_motion_provider_hash") or "")
    expected_binding = str(metadata.get("directional_binding_set_hash") or "")
    expected_policy = str(metadata.get("directional_evaluator_policy_hash") or "")
    expected_evaluator = str(metadata.get("directional_evaluator_semantic_version") or "")
    if not expected_hashes or not all(len(x) == 64 for x in (expected_provider, expected_binding, expected_policy)) or not expected_evaluator:
        raise QualificationError("RUNTIME_V2_MOTION_PROOF_MISSING_EXACT_PROVIDER_BINDING")
    if metadata.get("export_solver_replay_forbidden") is not True:
        raise QualificationError("RUNTIME_V2_PROOF_MUST_FORBID_EXPORT_SOLVER_REPLAY")

    by_clip = {}
    for bake in bakes:
        if not isinstance(bake, QualificationOwnedMotionBakeIR):
            raise QualificationError("RUNTIME_V2_REQUIRES_TYPED_QUALIFICATION_BAKE")
        if bake.clip_id in by_clip:
            raise QualificationError("RUNTIME_V2_DUPLICATE_MOTION_BAKE_CLIP")
        assert_motion_bake_binding(
            bake,
            source_product_state_hash=product.product_state_hash,
            proof_plan_hash=motion_report.proof_plan_hash,
        )
        if expected_hashes.get(bake.clip_id) != bake.bake_hash:
            raise QualificationError("RUNTIME_V2_MOTION_BAKE_HASH_NOT_BOUND_BY_PROOF")
        bake_metadata = dict(bake.metadata or {})
        if bake_metadata.get("directional_binding_set_hash") != expected_binding:
            raise QualificationError("RUNTIME_V2_BAKE_DIRECTIONAL_BINDING_MISMATCH")
        if bake_metadata.get("evaluator_policy_hash") != expected_policy:
            raise QualificationError("RUNTIME_V2_BAKE_EVALUATOR_POLICY_MISMATCH")
        if bake.evaluator_semantic_version != expected_evaluator:
            raise QualificationError("RUNTIME_V2_BAKE_EVALUATOR_SEMANTIC_MISMATCH")
        if bake_metadata.get("export_solver_replay_forbidden") is not True:
            raise QualificationError("RUNTIME_V2_BAKE_MUST_FORBID_EXPORT_SOLVER_REPLAY")
        by_clip[bake.clip_id] = bake

    product_clip_ids = {str(clip.clip_id) for clip in product.motion_state.clips}
    if set(by_clip) != product_clip_ids or set(expected_hashes) != product_clip_ids:
        raise QualificationError("RUNTIME_V2_PROVEN_BAKE_CLIP_SET_MISMATCH")

    first = by_clip[sorted(by_clip)[0]]
    for clip_id, bake in sorted(by_clip.items()):
        if bake.rest_mesh_vertices_by_id != first.rest_mesh_vertices_by_id:
            raise QualificationError(f"RUNTIME_V2_REST_MESH_IDENTITY_DRIFT:{clip_id}")
        if bake.triangles_by_mesh_id != first.triangles_by_mesh_id:
            raise QualificationError(f"RUNTIME_V2_REST_TOPOLOGY_DRIFT:{clip_id}")
    return motion_report, metadata, by_clip, first


def _runtime_uv_from_local_corner(corner, texture: RuntimeTexturePayloadV1) -> tuple[float, float]:
    if corner.authority_class != "OBSERVED_LOCAL" or int(corner.donor_view_index) != int(texture.view_index):
        raise QualificationError("RUNTIME_V2_CROSS_VIEW_OR_COMPLETION_ATLAS_NOT_QUALIFIED")
    x, y = map(float, corner.donor_raster_xy)
    if not all(math.isfinite(v) for v in (x, y)):
        raise QualificationError("RUNTIME_V2_NONFINITE_DONOR_RASTER")
    width, height = int(texture.width), int(texture.height)
    if x < -1e-7 or x > float(width - 1) + 1e-7 or y < -1e-7 or y > float(height - 1) + 1e-7:
        raise QualificationError("RUNTIME_V2_DONOR_RASTER_OUTSIDE_NATIVE_SAMPLE_DOMAIN")

    expected_material = (
        (x + 0.5) / float(width),
        1.0 - (y + 0.5) / float(height),
    )
    if max(abs(float(a) - float(b)) for a, b in zip(corner.material_uv, expected_material)) > 1e-7:
        raise QualificationError("RUNTIME_V2_APPEARANCE_MATERIAL_UV_CONTRACT_MISMATCH")

    u = 0.0 if width == 1 else x / float(width - 1)
    v = 0.0 if height == 1 else y / float(height - 1)
    if not (-1e-7 <= u <= 1.0 + 1e-7 and -1e-7 <= v <= 1.0 + 1e-7):
        raise QualificationError("RUNTIME_V2_NATIVE_UV_OUTSIDE_UNIT_DOMAIN")
    return (float(min(1.0, max(0.0, u))), float(min(1.0, max(0.0, v))))


def _project_component(direction, component, texture: RuntimeTexturePayloadV1, reference_bake):
    appearance = component.appearance
    metadata = dict(getattr(appearance, "metadata", {}) or {})
    if metadata.get("material_uv_convention") != APPEARANCE_MATERIAL_UV_CONVENTION:
        raise QualificationError("RUNTIME_V2_APPEARANCE_UV_CONVENTION_MISMATCH")
    if int(appearance.target_view_index) != int(direction.view_index):
        raise QualificationError("RUNTIME_V2_APPEARANCE_TARGET_VIEW_MISMATCH")
    if appearance.atlas_payload_hash != texture.atlas_payload_hash:
        raise QualificationError("RUNTIME_V2_APPEARANCE_TEXTURE_LINEAGE_MISMATCH")
    if not bool(component.default_visible):
        raise QualificationError("RUNTIME_V2_DEFAULT_HIDDEN_COMPONENT_SEMANTICS_NOT_QUALIFIED")

    mesh_id = f"V{int(direction.view_index)}:{component.component_id}"
    rest_by_mesh = dict(reference_bake.rest_mesh_vertices_by_id)
    if mesh_id not in rest_by_mesh:
        raise QualificationError(f"RUNTIME_V2_BAKE_MISSING_COMPONENT_MESH:{mesh_id}")
    rest = tuple(rest_by_mesh[mesh_id])
    vertices = tuple(component.mesh.vertices)
    if len(rest) != len(vertices):
        raise QualificationError("RUNTIME_V2_BAKE_CANONICAL_VERTEX_COUNT_MISMATCH")
    canonical_index = {v.canonical_mesh_vertex_id: i for i, v in enumerate(vertices)}
    if len(canonical_index) != len(vertices):
        raise QualificationError("RUNTIME_V2_DUPLICATE_CANONICAL_MESH_VERTEX")

    corners = {(int(row.face_index), int(row.corner_index)): row for row in appearance.corner_bindings}
    expected_corner_count = sum(len(face) for face in component.mesh.faces)
    if len(corners) != expected_corner_count:
        raise QualificationError("RUNTIME_V2_APPEARANCE_CORNER_COVERAGE_MISMATCH")

    runtime_vertices = []
    runtime_sources = []
    runtime_index = {}
    triangles = []
    for face_index, face in enumerate(component.mesh.faces):
        if len(face) != 3:
            raise QualificationError("RUNTIME_V2_REQUIRES_TRIANGULATED_MESH")
        tri = []
        for corner_index, vertex_id in enumerate(face):
            if vertex_id not in canonical_index:
                raise QualificationError("RUNTIME_V2_FACE_REFERENCES_UNKNOWN_CANONICAL_VERTEX")
            corner = corners.get((face_index, corner_index))
            if corner is None:
                raise QualificationError("RUNTIME_V2_MISSING_APPEARANCE_CORNER")
            u, v = _runtime_uv_from_local_corner(corner, texture)
            source_index = canonical_index[vertex_id]
            key = (str(vertex_id), u, v)
            if key not in runtime_index:
                runtime_index[key] = len(runtime_vertices)
                x, y = rest[source_index]
                runtime_vertices.append((float(x), float(y), u, v))
                runtime_sources.append(source_index)
            tri.append(runtime_index[key])
        triangles.append(tuple(tri))
    return RuntimeV2Mesh(mesh_id, tuple(runtime_vertices), tuple(triangles)), tuple(runtime_sources)


def project_current_v4_proof_bakes_to_runtime_v2(
    *,
    product,
    proof_bundle,
    motion_bakes: Iterable[QualificationOwnedMotionBakeIR],
    texture_bindings: Iterable[RuntimeTexturePayloadV1],
) -> RuntimeV2ProjectionResult:
    require_current_proof_bundle(product, proof_bundle, require_pass=True)
    _, motion_metadata, bake_by_clip, reference_bake = _bake_map(product, proof_bundle, motion_bakes)
    texture_rows, texture_by_view = _texture_map(texture_bindings)

    directions = tuple(sorted(product.directional_renderables.directions, key=lambda row: int(row.view_index)))
    if tuple(int(row.view_index) for row in directions) != tuple(range(8)):
        raise QualificationError("RUNTIME_V2_REQUIRES_EXACT_DIRECTIONS_0_TO_7")

    views = []
    source_maps = {}
    expected_mesh_ids = set()
    for direction in directions:
        view_index = int(direction.view_index)
        texture = texture_by_view[view_index]
        meshes = []
        for component in sorted(direction.components, key=lambda row: (int(row.setup_order), str(row.component_id))):
            mesh, source_indices = _project_component(direction, component, texture, reference_bake)
            meshes.append(mesh)
            source_maps[mesh.mesh_id] = source_indices
            expected_mesh_ids.add(mesh.mesh_id)
        if not meshes:
            raise QualificationError("RUNTIME_V2_DIRECTION_HAS_NO_MESH")
        views.append(RuntimeV2View(
            f"V{view_index}",
            str(texture.image_relpath),
            str(texture.image_sha256),
            int(texture.image_crc32),
            int(texture.width),
            int(texture.height),
            tuple(meshes),
        ))

    reference_mesh_ids = {mid for mid, _ in reference_bake.rest_mesh_vertices_by_id}
    if reference_mesh_ids != expected_mesh_ids:
        raise QualificationError("RUNTIME_V2_BAKE_PRODUCT_MESH_SET_MISMATCH")

    clips = []
    clip_by_id = {str(clip.clip_id): clip for clip in product.motion_state.clips}
    for clip_id in sorted(clip_by_id):
        product_clip = clip_by_id[clip_id]
        bake = bake_by_clip[clip_id]
        frames = []
        for frame in bake.frames:
            canonical_meshes = dict(frame.mesh_vertices_by_id)
            if set(canonical_meshes) != expected_mesh_ids:
                raise QualificationError("RUNTIME_V2_FRAME_MESH_SET_MISMATCH")
            runtime_xy = {}
            for mesh_id in sorted(expected_mesh_ids):
                canonical_xy = tuple(canonical_meshes[mesh_id])
                source_indices = source_maps[mesh_id]
                if source_indices and max(source_indices) >= len(canonical_xy):
                    raise QualificationError("RUNTIME_V2_FRAME_CANONICAL_VERTEX_COUNT_MISMATCH")
                runtime_xy[mesh_id] = tuple(
                    (float(canonical_xy[i][0]), float(canonical_xy[i][1]))
                    for i in source_indices
                )
            draw = {str(view): tuple(map(str, order)) for view, order in frame.render_order_by_view}
            expected_views = {f"V{i}" for i in range(8)}
            if set(draw) != expected_views:
                raise QualificationError("RUNTIME_V2_FRAME_DRAW_ORDER_VIEW_SET_MISMATCH")
            for view in views:
                expected_order_set = {mesh.mesh_id for mesh in view.meshes}
                order = tuple(draw[view.view_id])
                if set(order) != expected_order_set or len(order) != len(expected_order_set):
                    raise QualificationError("RUNTIME_V2_FRAME_DRAW_ORDER_NOT_EXACT_PERMUTATION")
            frames.append(RuntimeV2Frame(float(frame.time_seconds), runtime_xy, draw))
        clip_metadata = dict(getattr(product_clip, "metadata", {}) or {})
        clips.append(RuntimeV2Clip(
            clip_id,
            str(clip_metadata.get("display_name") or clip_id),
            str(getattr(product_clip, "clip_kind", "MOTION")),
            float(bake.duration_seconds),
            float(bake.fps),
            bool(bake.loop),
            tuple(frames),
            True,
            1.0,
        ))

    projection_payload = {
        "schema": PROJECTION_SCHEMA,
        "source_product_state_hash": product.product_state_hash,
        "source_proof_bundle_hash": proof_bundle.proof_bundle_hash,
        "qualified_motion_provider_hash": motion_metadata["qualified_motion_provider_hash"],
        "directional_binding_set_hash": motion_metadata["directional_binding_set_hash"],
        "directional_evaluator_policy_hash": motion_metadata["directional_evaluator_policy_hash"],
        "directional_evaluator_semantic_version": motion_metadata["directional_evaluator_semantic_version"],
        "source_motion_bake_hashes": {clip_id: bake.bake_hash for clip_id, bake in sorted(bake_by_clip.items())},
        "runtime_uv_convention": NATIVE_RUNTIME_UV_CONVENTION,
        "runtime_uv_authority": "APPEARANCE_LOCAL_DONOR_RASTER_XY",
        "runtime_rest_xy_authority": "QUALIFICATION_OWNED_MOTION_BAKE_REST",
        "runtime_vertex_sources": {key: list(value) for key, value in sorted(source_maps.items())},
        "texture_bindings": [row.to_dict() for row in texture_rows],
        "solver_replay": False,
    }
    projection_hash = content_sha256(projection_payload)
    return RuntimeV2ProjectionResult(
        tuple(views),
        tuple(clips),
        projection_hash,
        source_maps,
        texture_rows,
        {clip_id: bake.bake_hash for clip_id, bake in sorted(bake_by_clip.items())},
    )


def materialize_current_v4_proof_bakes_runtime_v2(
    *,
    out_path,
    texture_root,
    product,
    proof_bundle,
    motion_bakes: Iterable[QualificationOwnedMotionBakeIR],
    texture_bindings: Iterable[RuntimeTexturePayloadV1],
):
    projection = project_current_v4_proof_bakes_to_runtime_v2(
        product=product,
        proof_bundle=proof_bundle,
        motion_bakes=motion_bakes,
        texture_bindings=texture_bindings,
    )
    result = materialize_runtime_v2_archive(
        out_path=out_path,
        texture_root=texture_root,
        views=projection.views,
        clips=projection.clips,
        source_product_state_hash=product.product_state_hash,
        source_proof_bundle_hash=proof_bundle.proof_bundle_hash,
        projection_hash=projection.projection_hash,
    )
    return projection, result
