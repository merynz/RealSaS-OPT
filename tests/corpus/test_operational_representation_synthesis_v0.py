"""Research-only executable synthesis and CAA puppet regression tests."""
from dataclasses import dataclass
import numpy as np
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.operational_representation_synthesis_v0 import (
    SourceGraphicsEvidenceV0, PuppetCapabilityContractV0,
    RepresentationPolicyV0, synthesize_operational_representations_v0,
    validate_candidate_v0,
)
from compiler.realsas_compiler_core.operational_puppet_runtime_v0 import OperationalPuppetRuntimeV0
from compiler.realsas_compiler_core.types import QualificationError


def e(face_count=6, groups=(0, 0, 0, 1, 1, 1)):
    return SourceGraphicsEvidenceV0(
        mesh_binding_hash="a" * 64,
        skin_binding_hash="b" * 64,
        source_graphics_binding_hash="c" * 64,
        face_count=face_count,
        source_face_group_ids=tuple(groups),
        provenance_refs=("synthetic:source-not-authority",),
        input_mode="SYNTHETIC_FIXTURE",
    )


def c(*required):
    return PuppetCapabilityContractV0(
        independent_edit_address_count=2,
        required_operations=required or (
            "INDEPENDENT_ATLAS_EDITS",
            "POSE_AWARE_SOURCE_UV",
            "LOCAL_DEPTH_SOURCE_OVER",
            "HIDDEN_EDIT_STATE_PERSISTENCE",
        ),
    )


def test_candidate_generation_is_subject_free_and_retains_strong_alternatives():
    out = synthesize_operational_representations_v0(
        e(), c(), RepresentationPolicyV0(max_faces_per_packet=2))
    assert len(out.candidates) == 4
    assert {v.candidate_id: len(v.packets) for v in out.candidates} == {
        "addressed_aggregate": 1,
        "source_group_shared_carrier": 2,
        "bounded_hybrid": 3,
        "per_face_shared_carrier": 6,
    }
    assert len(out.executable_candidate_ids) == 4
    assert all(v == "SATISFIED" for v in out.capability_states.values())
    assert out.source_artist_ownership == "UNKNOWN"
    assert out.physical_source_qualification == "UNKNOWN"
    assert out.measured_runtime_cost == "UNMEASURED_COST"
    assert not any(out.authority.values())
    for v in out.candidates:
        validate_candidate_v0(v, face_count=6, address_count=2)
        assert v.edit_control_address_indices == (0, 1)


def test_no_false_qualification_when_contract_exceeds_runtime():
    out = synthesize_operational_representations_v0(
        e(), c("INDEPENDENT_ATLAS_EDITS", "INDEPENDENT_ATTACHMENT_RELEASE",
               "SECONDARY_MOTION", "NONBINARY_ALPHA_SOURCE_PARITY"))
    assert out.capability_states["INDEPENDENT_ATTACHMENT_RELEASE"] == "UNKNOWN"
    assert out.capability_states["SECONDARY_MOTION"] == "UNKNOWN"
    assert out.capability_states["NONBINARY_ALPHA_SOURCE_PARITY"] == "UNKNOWN"
    assert not out.executable_candidate_ids
    assert out.candidates  # alternatives remain available for research


def test_policy_changes_packing_not_source_evidence_or_control_count():
    a = synthesize_operational_representations_v0(
        e(), c(), RepresentationPolicyV0(max_faces_per_packet=2))
    b = synthesize_operational_representations_v0(
        e(), c(), RepresentationPolicyV0(max_faces_per_packet=4))
    assert a.evidence == b.evidence and a.contract == b.contract
    assert a.source_artist_ownership == b.source_artist_ownership == "UNKNOWN"
    assert a.candidates[0].candidate_hash == b.candidates[0].candidate_hash
    assert a.candidates[0].edit_control_address_indices == (0, 1)
    assert a.measured_runtime_cost == b.measured_runtime_cost == "UNMEASURED_COST"


def test_reject_corrupt_source_and_missing_control_authority():
    with pytest.raises(QualificationError, match="SHA"):
        synthesize_operational_representations_v0(
            SourceGraphicsEvidenceV0("bad", "b" * 64, "c" * 64, 2), c())
    with pytest.raises(QualificationError, match="SOURCE_GROUPS"):
        synthesize_operational_representations_v0(
            e(face_count=2, groups=(0,)), c())
    with pytest.raises(QualificationError, match="ATLAS_CONVENTION"):
        synthesize_operational_representations_v0(
            e(), PuppetCapabilityContractV0(2, ("INDEPENDENT_ATLAS_EDITS",),
                                            atlas_coordinate_convention="BOTTOM_LEFT"))


@dataclass(frozen=True)
class Vertex:
    canonical_mesh_vertex_id: str
    P: tuple[float, float, float]


@dataclass(frozen=True)
class Mesh:
    vertices: tuple[Vertex, ...]
    faces: tuple[tuple[str, str, str], ...]


def fixture():
    # Two independently UV-addressed overlapping triangles on the same
    # canonical carrier; closer triangle changes with authored posed XYZ.
    xyz = [
        (-0.7, -0.7, 0.0), (0.7, -0.7, 0.0), (0.0, 0.7, 0.0),
        (-0.7, -0.7, 0.5), (0.7, -0.7, 0.5), (0.0, 0.7, 0.5),
    ]
    verts = tuple(Vertex(f"v{i}", p) for i, p in enumerate(xyz))
    mesh = Mesh(verts, (("v0", "v1", "v2"), ("v3", "v4", "v5")))
    uv = np.array([
        [(0.10, 0.1), (0.38, 0.1), (0.23, 0.9)],
        [(0.62, 0.1), (0.90, 0.1), (0.77, 0.9)],
    ], dtype=np.float64)
    tex = np.full((16, 16, 4), (120, 40, 90, 255), dtype=np.uint8)
    camera = qualify_camera_v3({
        "view_index": 0, "origin": (0, 0, -4),
        "right": (1, 0, 0), "screen_up": (0, 1, 0),
        "forward": (0, 0, 1), "half_extent": 1.0,
        "resolution": 24,
    }, view_id="v0", view_index=0)
    return mesh, uv, tex, camera


def runtimes():
    evidence = e(face_count=2, groups=(0, 1))
    synthesis = synthesize_operational_representations_v0(
        evidence, c(), RepresentationPolicyV0(max_faces_per_packet=1))
    mesh, uv, tex, camera = fixture()
    result = [
        OperationalPuppetRuntimeV0(
            evidence=evidence, candidate=candidate,
            mesh=mesh, face_uv=uv, texture_rgba_u8=tex,
            edit_rectangles=((0, 0, 8, 16), (8, 0, 16, 16)),
        )
        for candidate in synthesis.candidates
    ]
    return result, camera, mesh


def test_real_executable_reference_pose_transport_and_hidden_edit_persistence():
    rs, camera, mesh = runtimes()
    runtime = rs[0]
    rest = runtime.render(camera=camera, control_bits=(0, 0))
    visible_edit = runtime.render(camera=camera, control_bits=(1, 0))
    hidden_edit = runtime.render(camera=camera, control_bits=(0, 1))
    # The second triangle is initially occluded, but its edit state is kept.
    assert np.any(rest.owner_face_index == 0)
    assert np.array_equal(rest.premultiplied_rgba, hidden_edit.premultiplied_rgba)
    assert not np.array_equal(rest.premultiplied_rgba, visible_edit.premultiplied_rgba)
    posed = np.array([v.P for v in mesh.vertices], dtype=np.float64)
    posed[3:, 2] = -0.5  # depth-swap: previously hidden triangle becomes front
    newly_visible = runtime.render(
        camera=camera, posed_positions=posed, control_bits=(0, 1))
    no_edit = runtime.render(
        camera=camera, posed_positions=posed, control_bits=(0, 0))
    assert np.any(newly_visible.owner_face_index == 1)
    assert not np.array_equal(newly_visible.premultiplied_rgba, no_edit.premultiplied_rgba)
    assert rest.edit_state == (0, 0) and newly_visible.edit_state == (0, 1)
    assert newly_visible.qualification == "REFERENCE_EXECUTED__SOURCE_PARITY_UNKNOWN"
    # Source texture is never mutated by controls.
    assert np.all(runtime.base_texture[..., 3] == 255)


def test_competing_strong_packings_preserve_executable_reference_pixels():
    rs, camera, mesh = runtimes()
    posed = np.array([v.P for v in mesh.vertices], dtype=np.float64)
    posed[3:, 2] = -0.5
    for bits in ((0, 0), (1, 0), (0, 1), (1, 1)):
        images = [r.render(camera=camera, posed_positions=posed,
                           control_bits=bits).premultiplied_rgba for r in rs]
        assert all(np.array_equal(images[0], other) for other in images[1:])
    assert len({len(x.candidate.packets) for x in rs}) == 2


def test_edit_region_overlap_fails_closed_and_alpha_is_not_inverted():
    rs, _camera, _mesh = runtimes()
    base = rs[0].base_texture.copy()
    edited = rs[0].edited_texture((1, 1))
    assert np.array_equal(base[..., 3], edited[..., 3])
    assert np.array_equal(edited[..., :3], 255 - base[..., :3])
    with pytest.raises(QualificationError, match="OVERLAP"):
        OperationalPuppetRuntimeV0(
            evidence=e(face_count=2, groups=(0, 1)),
            candidate=rs[0].candidate, mesh=rs[0].mesh,
            face_uv=rs[0].face_uv, texture_rgba_u8=base,
            edit_rectangles=((0, 0, 9, 16), (8, 0, 16, 16)))
