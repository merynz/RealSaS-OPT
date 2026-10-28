from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.graphics_evidence_v2 import (
    AdmittedRayArtifactV2,
    CanonicalAddressV2,
    RayHitEvidenceV2,
    RayStackEvidenceV2,
)
from compiler.realsas_compiler_core.graphics_local_support_v2 import (
    bind_compiled_ray_support_v2,
    make_exact_raster_support_v2,
)
from compiler.realsas_compiler_core.graphics_v2 import compile_ray_graphics_v2
from compiler.realsas_compiler_core.hashing import content_sha256
from models.iris.truth.graphics_oracle_v1 import (
    OrthographicCameraV1,
    face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1,
)
from tools.demo.render_knight_m_caa_v1 import load_inputs


ROOT = Path(__file__).resolve().parents[2]
KNIGHT_BASELINE = ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json"


def _sha(value) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _camera(*, reverse=False):
    return OrthographicCameraV1(
        origin=(0.0, 0.0, 5.0 if reverse else -5.0),
        forward=(0.0, 0.0, -1.0 if reverse else 1.0),
        right=(-1.0 if reverse else 1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        half_extent=2.0,
        resolution=64,
        view_index=1 if reverse else 0,
    )


def _two_layers(*, tied=False):
    vertices = np.asarray(
        [
            [-1.5, -1.5, 0.0],
            [1.5, -1.5, 0.0],
            [0.0, 1.5, 0.0],
            [-1.5, -1.5, 0.0 if tied else 1.0],
            [1.5, -1.5, 0.0 if tied else 1.0],
            [0.0, 1.5, 0.0 if tied else 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    return vertices, faces


def test_ray_stack_preserves_local_depth_and_reverses_with_view():
    vertices, faces = _two_layers()
    front = measure_orthographic_ray_stack_v1(
        vertices, faces, _camera(), 31, 31, tie_epsilon=1e-8
    )
    back = measure_orthographic_ray_stack_v1(
        vertices, faces, _camera(reverse=True), 31, 31, tie_epsilon=1e-8
    )
    assert front.complete is True and back.complete is True
    assert [hit.face_index for hit in front.hits] == [0, 1]
    assert [hit.face_index for hit in back.hits] == [1, 0]
    assert front.has_depth_tie is False and back.has_depth_tie is False


def test_coplanar_hits_are_ties_not_face_id_tie_breaks():
    vertices, faces = _two_layers(tied=True)
    stack = measure_orthographic_ray_stack_v1(
        vertices, faces, _camera(), 31, 31, tie_epsilon=1e-6
    )
    assert len(stack.hits) == 2
    assert stack.has_depth_tie is True
    assert abs(stack.hits[1].depth - stack.hits[0].depth) <= stack.tie_epsilon


def test_camera_plane_inside_mesh_marks_stack_incomplete():
    vertices, faces = _two_layers()
    camera = OrthographicCameraV1(
        origin=(0.0, 0.0, 0.5),
        forward=(0.0, 0.0, 1.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        half_extent=2.0,
        resolution=64,
        view_index=0,
    )
    stack = measure_orthographic_ray_stack_v1(
        vertices, faces, camera, 31, 31, tie_epsilon=1e-8
    )
    assert stack.complete is False


def _knight_oracle_rows():
    baseline, arrays, _source, cameras, _caa = load_inputs(KNIGHT_BASELINE)
    assert baseline["canonical_geometry"] == "M"
    rest = np.asarray(arrays["rest"], dtype=np.float64)
    faces = np.asarray(arrays["faces"], dtype=np.int64)
    scale = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))
    tie_epsilon = max(scale * 1e-7, 1e-9)
    mesh_hash = _sha({
        "fixture": baseline["parent_witness_sha256"],
        "rest_sha": hashlib.sha256(rest.tobytes()).hexdigest(),
        "faces_sha": hashlib.sha256(faces.tobytes()).hexdigest(),
    })
    artifact_hash = _sha("KNIGHT_GRAPHICS_ORACLE_GEOMETRIC_RAYS_V1")
    qualification_hash = _sha("TEACHER_ORACLE_COURT_ONLY__NOT_REGISTRY_QUALIFICATION")
    skin_hash = _sha("KNIGHT_ORACLE_FIXED_SKIN_PARENT")
    source_graphics_hash = _sha("KNIGHT_SINGLE_M_CAA_SOURCE_GRAPHICS")
    camera_set_hash = _sha({
        "fixture": baseline["parent_witness_sha256"],
        "cameras": [dict(vars(camera)) for camera in cameras],
    })
    sampling_hash = _sha({
        "kind": "HALF_INTEGER_ORTHOGRAPHIC_PIXEL_RAY",
        "candidate_policy": "FACE_CENTROID_GEOMETRY_ONLY_MAX128",
        "tie_epsilon": tie_epsilon,
    })

    rows = []
    exact_support_coordinates = {}
    address_witnesses = set()
    per_view = {}
    for camera in cameras:
        candidates = face_centroid_candidate_pixels_v1(
            rest, faces, camera, maximum=128
        )
        kept = 0
        for pixel_x, pixel_y in candidates:
            stack = measure_orthographic_ray_stack_v1(
                rest,
                faces,
                camera,
                pixel_x,
                pixel_y,
                tie_epsilon=tie_epsilon,
            )
            if not stack.complete or len(stack.hits) < 2:
                continue
            hit_rows = []
            for hit in stack.hits:
                witness = _sha({
                    "mesh_hash": mesh_hash,
                    "face_index": hit.face_index,
                    "barycentric": [round(v, 12) for v in hit.barycentric],
                })
                address_witnesses.add(witness)
                hit_rows.append(
                    RayHitEvidenceV2(
                        CanonicalAddressV2(
                            mesh_hash=mesh_hash,
                            face_index=hit.face_index,
                            barycentric=hit.barycentric,
                            binding_witness_hash=witness,
                        ),
                        hit.depth,
                        None,
                        None,
                    )
                )
            row = RayStackEvidenceV2(
                scope_hash=_sha({"fixture": baseline["parent_witness_sha256"], "view": stack.view_index}),
                ray_key=f"diagnostic-only:V{stack.view_index}:{pixel_x}:{pixel_y}",
                view_index=stack.view_index,
                hits=tuple(hit_rows),
                complete=True,
                blend_law="SOURCE_OVER",
                tie_epsilon=tie_epsilon,
                evidence_artifact_hash=artifact_hash,
                qualification_hash=qualification_hash,
                measurement_valid=True,
                source_artistic_order_status="NO_AUTHORED_ORDER__DEPTH_QUALIFIED",
            )
            rows.append(row)
            exact_support_coordinates[row.record_hash()] = (
                int(stack.view_index), int(pixel_x), int(pixel_y), int(camera.resolution)
            )
            kept += 1
        per_view[int(camera.view_index)] = kept

    admitted = {
        artifact_hash: AdmittedRayArtifactV2(
            qualification_hash=qualification_hash,
            record_hashes=frozenset(row.record_hash() for row in rows),
        )
    }
    return (
        rest,
        faces,
        mesh_hash,
        skin_hash,
        source_graphics_hash,
        tuple(rows),
        frozenset(address_witnesses),
        admitted,
        per_view,
        camera_set_hash,
        sampling_hash,
        exact_support_coordinates,
    )


def _compiled_knight():
    (
        rest,
        faces,
        mesh_hash,
        skin_hash,
        source_graphics_hash,
        rows,
        address_witnesses,
        admitted,
        per_view,
        camera_set_hash,
        sampling_hash,
        exact_support_coordinates,
    ) = _knight_oracle_rows()
    compiled = compile_ray_graphics_v2(
        canonical_mesh_binding_hash=mesh_hash,
        canonical_mesh_skin_binding_hash=skin_hash,
        source_graphics_binding_hash=source_graphics_hash,
        face_count=len(faces),
        evidence=rows,
        address_witnesses=address_witnesses,
        admitted_artifacts=admitted,
    )
    return (
        rest, faces, compiled, per_view, camera_set_hash, sampling_hash,
        exact_support_coordinates,
    )


def test_knight_canonical_m_oracle_produces_consumable_local_order_without_fake_appearance_truth():
    _rest, _faces, compiled, per_view, _camera_set_hash, _sampling_hash, _coords = _compiled_knight()

    assert compiled.rays
    assert set(per_view) == set(range(8))
    assert all(count > 0 for count in per_view.values())
    constraint_count = sum(len(ray["order_constraints"]) for ray in compiled.rays)
    assert constraint_count > 0
    assert len(compiled.scopes) == 8
    assert all(ray["compositing"]["reason"] == "UNKNOWN_ALPHA" for ray in compiled.rays)
    assert all(ray["compositing"]["alpha_valid"] is False for ray in compiled.rays)
    assert all(ray["compositing"]["composite_valid"] is False for ray in compiled.rays)
    assert any(ray["order_constraints"] for ray in compiled.rays)


def test_knight_local_order_is_not_silently_promoted_to_one_global_layer_order():
    _rest, _faces, compiled, per_view, _camera_set_hash, _sampling_hash, _coords = _compiled_knight()
    assert set(per_view) == set(range(8))
    assert all(scope["edges_removed"] == 0 for scope in compiled.scopes)
    assert all(scope["unobserved_visibility"] == "UNKNOWN" for scope in compiled.scopes)
    assert all(
        scope["order_status"] in {"SCOPED_PARTIAL_ORDER", "LOCAL_INTERLEAVING_REQUIRED"}
        for scope in compiled.scopes
    )


def test_knight_oracle_rays_bind_to_explicit_exact_support_without_ray_key_authority():
    (
        _rest,
        _faces,
        compiled,
        per_view,
        camera_set_hash,
        sampling_hash,
        exact_support_coordinates,
    ) = _compiled_knight()
    supports = []
    for ray in compiled.rays:
        record_hash = ray["record_hash"]
        view, pixel_x, pixel_y, resolution = exact_support_coordinates[record_hash]
        supports.append(
            make_exact_raster_support_v2(
                record_hash=record_hash,
                camera_set_binding_hash=camera_set_hash,
                sampling_contract_hash=sampling_hash,
                view_index=view,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                resolution=resolution,
                metadata={"fixture": "single_m_caa_knight_v1"},
            )
        )

    bound = bind_compiled_ray_support_v2(
        compiled=compiled,
        supports=supports,
        camera_set_binding_hash=camera_set_hash,
        sampling_contract_hash=sampling_hash,
    )
    assert bound.observed_sample_count == len(compiled.rays)
    assert bound.consequence_sample_count > 0
    assert bound.rules
    assert {rule.view_index for rule in bound.rules} == {
        ray["view_index"] for ray in compiled.rays if ray["order_constraints"]
    }
    assert set(per_view) == set(range(8))
    assert bound.coverage_policy == "EXACT_SAMPLES_ONLY__NO_EXTRAPOLATION"
    assert bound.interpolation_authorized is False
    assert bound.unobserved_support_status == "UNKNOWN"
    assert bound.product_activation_allowed is False
    assert bound.metadata["ray_key_parsing_as_authority_forbidden"] is True
    # The typed consumer result and portable JSON must retain the same address
    # and content identity; repairing the Python boundary must not re-key truth.
    wire = json.loads(json.dumps(bound.to_dict()))
    binding_hash = wire.pop("support_binding_hash")
    assert content_sha256(wire) == binding_hash
    assert {row["record_hash"] for row in wire["rules"]} == {
        rule.record_hash for rule in bound.rules
    }


def test_missing_exact_support_fails_closed_instead_of_extrapolating_neighbor_rule():
    (
        _rest,
        _faces,
        compiled,
        _per_view,
        camera_set_hash,
        sampling_hash,
        exact_support_coordinates,
    ) = _compiled_knight()
    supports = []
    for ray in compiled.rays[:-1]:
        record_hash = ray["record_hash"]
        view, pixel_x, pixel_y, resolution = exact_support_coordinates[record_hash]
        supports.append(
            make_exact_raster_support_v2(
                record_hash=record_hash,
                camera_set_binding_hash=camera_set_hash,
                sampling_contract_hash=sampling_hash,
                view_index=view,
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                resolution=resolution,
            )
        )
    try:
        bind_compiled_ray_support_v2(
            compiled=compiled,
            supports=supports,
            camera_set_binding_hash=camera_set_hash,
            sampling_contract_hash=sampling_hash,
        )
    except Exception as exc:
        assert "GRAPHICS_V2_EXECUTABLE_SUPPORT_REQUIRED" in str(exc)
    else:
        raise AssertionError("missing exact support must fail closed")
