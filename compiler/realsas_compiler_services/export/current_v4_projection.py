from __future__ import annotations

"""Proof-bound projection from current CanonicalPuppetGraph.v3 to native runtime v2.

This module does not rerun models, solvers, deformation, proof or repair. It
projects already-qualified topology/appearance plus the exact RuntimeDeployBake
sealed inside the MOTION MeasurementReport. Canonical editable meshes are never
mutated; face-corner UV seams become runtime-only duplicated vertices.
"""

from dataclasses import dataclass
from typing import Iterable

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.v4 import measurement_report_hash, require_current_proof_bundle
from compiler.realsas_compiler_services.export.runtime_deploy_bake import decode_runtime_deploy_bake
from compiler.realsas_compiler_services.export.runtime_v2 import (
    RuntimeV2Clip, RuntimeV2Frame, RuntimeV2Mesh, RuntimeV2View,
    materialize_runtime_v2_archive,
)

CURRENT_MATERIAL_UV_CONVENTION = "NATIVE_PIXEL_CENTER_TO_NORMALIZED_UV_V1"
RUNTIME_UV_CONVENTION = "PNG_TOP_LEFT_NORMALIZED_UV_V1"
PROJECTION_SCHEMA = "RealSaS.CurrentV4ToNativeRuntimeV2Projection.v1"


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
        return dict(self.__dict__)


@dataclass(frozen=True)
class RuntimeV2ProjectionResult:
    views: tuple[RuntimeV2View, ...]
    clips: tuple[RuntimeV2Clip, ...]
    projection_hash: str
    runtime_vertex_sources: dict[str, tuple[int, ...]]
    texture_bindings: tuple[RuntimeTexturePayloadV1, ...]
    schema_version: str = PROJECTION_SCHEMA


def _motion_domain_report(proof_bundle):
    matches = [r for r in proof_bundle.domain_reports if r.proof_domain == "MOTION"]
    if len(matches) != 1:
        raise ValueError("runtime projection requires exactly one MOTION domain proof")
    report = matches[0]
    if report.status != "PASS":
        raise ValueError("runtime projection requires PASS MOTION proof")
    return report


def _verify_motion_measurement(product, proof_bundle, motion_measurement_report):
    require_current_proof_bundle(product, proof_bundle, require_pass=True)
    domain = _motion_domain_report(proof_bundle)
    if motion_measurement_report.source_product_state_hash != product.product_state_hash:
        raise ValueError("runtime projection motion measurement is stale")
    if motion_measurement_report.proof_plan_hash != domain.proof_plan_hash:
        raise ValueError("runtime projection motion plan mismatch")
    if measurement_report_hash(motion_measurement_report) != motion_measurement_report.measurement_report_hash:
        raise ValueError("runtime projection motion measurement hash mismatch")
    if motion_measurement_report.measurement_report_hash != domain.measurement_report_hash:
        raise ValueError("runtime projection proof does not bind supplied motion measurement")
    measurements = dict(motion_measurement_report.measurements or {})
    if measurements.get("runtime_export_solver_replay") is not False:
        raise ValueError("runtime projection requires no-replay proof-owned bake")
    if not str(measurements.get("runtime_bake_status") or "").startswith("PASS_"):
        raise ValueError("runtime projection has no qualified runtime bake")
    bakes = list(measurements.get("runtime_deploy_bakes") or [])
    if not bakes:
        raise ValueError("runtime projection missing proof-owned runtime bake")
    return domain, measurements, bakes


def _texture_binding_map(texture_bindings: Iterable[RuntimeTexturePayloadV1]):
    rows = tuple(sorted(texture_bindings, key=lambda x: int(x.view_index)))
    by_view = {int(row.view_index): row for row in rows}
    if tuple(by_view) != tuple(range(8)) or len(rows) != 8:
        raise ValueError("runtime projection requires exact texture bindings for views 0..7")
    for row in rows:
        if len(str(row.image_sha256)) != 64:
            raise ValueError("runtime texture payload requires image sha256")
        expected_atlas = content_sha256({"image_sha256": str(row.image_sha256), "image_relpath": str(row.image_relpath)})
        if expected_atlas != str(row.atlas_payload_hash):
            raise ValueError("runtime texture payload atlas lineage mismatch")
        if int(row.width) <= 0 or int(row.height) <= 0:
            raise ValueError("runtime texture payload dimensions invalid")
    return rows, by_view


def _runtime_uv(material_uv):
    u, v = map(float, material_uv)
    ru, rv = u, 1.0 - v
    if not (0.0 <= ru <= 1.0 and 0.0 <= rv <= 1.0):
        raise ValueError("runtime projected UV outside [0,1]")
    return ru, rv


def _project_component(direction, component):
    appearance = component.appearance
    if dict(getattr(appearance, "metadata", {}) or {}).get("material_uv_convention") != CURRENT_MATERIAL_UV_CONVENTION:
        raise ValueError("runtime v2 projection requires explicit current material UV convention")
    vertices = tuple(component.mesh.vertices)
    canonical_index = {v.canonical_mesh_vertex_id: i for i, v in enumerate(vertices)}
    if len(canonical_index) != len(vertices):
        raise ValueError("runtime projection duplicate canonical mesh vertex")
    corners = {(int(c.face_index), int(c.corner_index)): c for c in appearance.corner_bindings}
    runtime_vertices = []
    runtime_sources = []
    runtime_index = {}
    triangles = []
    for face_index, face in enumerate(component.mesh.faces):
        if len(face) != 3:
            raise ValueError("runtime v2 projection requires triangulated mesh")
        tri = []
        for corner_index, vertex_id in enumerate(face):
            if vertex_id not in canonical_index:
                raise ValueError("runtime projection face references unknown canonical vertex")
            corner = corners.get((face_index, corner_index))
            if corner is None:
                raise ValueError("runtime projection missing face-corner appearance binding")
            u, v = _runtime_uv(corner.material_uv)
            key = (str(vertex_id), u, v)
            if key not in runtime_index:
                src = canonical_index[vertex_id]
                p = vertices[src].P
                runtime_index[key] = len(runtime_vertices)
                runtime_vertices.append((float(p[0]), float(p[1]), u, v))
                runtime_sources.append(src)
            tri.append(runtime_index[key])
        triangles.append(tuple(tri))
    runtime_id = f"V{int(direction.view_index)}:{component.component_id}"
    return RuntimeV2Mesh(runtime_id, tuple(runtime_vertices), tuple(triangles)), tuple(runtime_sources)


def project_current_v4_to_runtime_v2(*, product, proof_bundle, motion_measurement_report, texture_bindings: Iterable[RuntimeTexturePayloadV1]) -> RuntimeV2ProjectionResult:
    _, measurements, bake_rows = _verify_motion_measurement(product, proof_bundle, motion_measurement_report)
    texture_rows, texture_by_view = _texture_binding_map(texture_bindings)
    directions = tuple(sorted(product.directional_renderables.directions, key=lambda d: int(d.view_index)))
    if tuple(int(d.view_index) for d in directions) != tuple(range(8)):
        raise ValueError("runtime projection requires exact directions 0..7")

    views = []
    source_maps = {}
    for direction in directions:
        view_index = int(direction.view_index)
        texture = texture_by_view[view_index]
        components = tuple(sorted(direction.components, key=lambda c: (int(c.setup_order), str(c.component_id))))
        if not components:
            raise ValueError("runtime projection direction has no components")
        atlas_hashes = {str(c.appearance.atlas_payload_hash) for c in components}
        if atlas_hashes != {str(texture.atlas_payload_hash)}:
            raise ValueError("runtime v2 requires one exact source atlas per direction")
        meshes = []
        for component in components:
            mesh, source_indices = _project_component(direction, component)
            meshes.append(mesh)
            source_maps[mesh.mesh_id] = source_indices
        views.append(RuntimeV2View(
            f"V{view_index}", str(texture.image_relpath), str(texture.image_sha256), int(texture.image_crc32),
            int(texture.width), int(texture.height), tuple(meshes),
        ))

    decoded = [decode_runtime_deploy_bake(row) for row in bake_rows]
    by_clip = {str(row["clip_id"]): row for row in decoded}
    product_clips = tuple(sorted(product.motion_state.clips, key=lambda c: str(c.clip_id)))
    if set(by_clip) != {str(c.clip_id) for c in product_clips}:
        raise ValueError("runtime bake clip identity set differs from current motion state")

    runtime_clips = []
    for clip in product_clips:
        baked = by_clip[str(clip.clip_id)]
        frames = []
        for frame in baked["frames"]:
            mesh_xy = {}
            for view in views:
                for mesh in view.meshes:
                    canonical_xy = frame["mesh_vertices_by_id"].get(mesh.mesh_id)
                    if canonical_xy is None:
                        raise ValueError(f"runtime bake missing mesh:{mesh.mesh_id}")
                    source_indices = source_maps[mesh.mesh_id]
                    if source_indices and max(source_indices) >= len(canonical_xy):
                        raise ValueError("runtime bake canonical vertex count differs from projection")
                    mesh_xy[mesh.mesh_id] = tuple((float(canonical_xy[i][0]), float(canonical_xy[i][1])) for i in source_indices)
            draw = {str(view_id): tuple(map(str, ids)) for view_id, ids in dict(frame["render_order_by_view"]).items()}
            frames.append(RuntimeV2Frame(float(frame["time_seconds"]), mesh_xy, draw))
        metadata = dict(getattr(clip, "metadata", {}) or {})
        runtime_clips.append(RuntimeV2Clip(
            str(clip.clip_id), str(metadata.get("display_name") or clip.clip_id), str(getattr(clip, "clip_kind", "MOTION")),
            float(baked["duration_seconds"]), float(baked["fps"]), bool(baked["loop"]), tuple(frames), True, 1.0,
        ))

    projection_payload = {
        "schema": PROJECTION_SCHEMA,
        "source_product_state_hash": product.product_state_hash,
        "source_proof_bundle_hash": proof_bundle.proof_bundle_hash,
        "motion_measurement_report_hash": motion_measurement_report.measurement_report_hash,
        "runtime_uv_transform": "u=u_material;v=1-v_material",
        "runtime_uv_convention": RUNTIME_UV_CONVENTION,
        "texture_bindings": [row.to_dict() for row in texture_rows],
        "runtime_vertex_sources": {k:list(v) for k,v in sorted(source_maps.items())},
        "runtime_bake_payload_sha256": [row["payload_sha256"] for row in bake_rows],
        "solver_replay": False,
    }
    return RuntimeV2ProjectionResult(tuple(views), tuple(runtime_clips), content_sha256(projection_payload), source_maps, texture_rows)


def materialize_current_v4_runtime_v2(*, out_path, texture_root, product, proof_bundle, motion_measurement_report, texture_bindings: Iterable[RuntimeTexturePayloadV1]):
    projection = project_current_v4_to_runtime_v2(
        product=product, proof_bundle=proof_bundle, motion_measurement_report=motion_measurement_report,
        texture_bindings=texture_bindings,
    )
    result = materialize_runtime_v2_archive(
        out_path=out_path, texture_root=texture_root, views=projection.views, clips=projection.clips,
        source_product_state_hash=product.product_state_hash,
        source_proof_bundle_hash=proof_bundle.proof_bundle_hash,
        projection_hash=projection.projection_hash,
    )
    return projection, result


def materialize_current_v4_runtime_v2_from_evaluation(*, out_path, texture_root, product, proof_evaluation, texture_bindings: Iterable[RuntimeTexturePayloadV1]):
    """Convenience path from evidence-preserving current proof evaluation."""
    proof_bundle = proof_evaluation.proof_bundle
    motion_measurement_report = proof_evaluation.motion_measurement_report()
    return materialize_current_v4_runtime_v2(
        out_path=out_path, texture_root=texture_root, product=product,
        proof_bundle=proof_bundle, motion_measurement_report=motion_measurement_report,
        texture_bindings=texture_bindings,
    )
