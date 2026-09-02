from __future__ import annotations

from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate
from compiler.realsas_compiler_core.types import ObservationEvidenceIR, RiggingSurfaceIR, SurfaceNode
from experiments.iris_reprojection_v2_20260831.persistence_adapter_v2 import attach_observed_local_relations_v2


def _node(i: int, x: float, y: float, *, ambiguous: bool = False) -> SurfaceNode:
    gid = f"Q{i:06d}:M00"
    flags = ("AMBIGUOUS_RAY",) if ambiguous else ()
    return SurfaceNode(
        surface_id=f"S:{i:04d}",
        P=(x * 0.001, y * 0.001, 0.0),
        support_views=(0, 1, 2, 3),
        provenance_refs=(f"P:{i}",),
        source_observation_ids=(f"O:{i}:V0",),
        raster_bindings=((0, (100.0 + x, 200.0 + y)),),
        persistence_group_id=f"QPV2:{gid}",
        validity_flags=flags,
    )


def _evidence() -> ObservationEvidenceIR:
    anchors = {
        "Q000000:M00": {"view_index": 0, "raster_xy": (100.0, 200.0)},
        "Q000001:M00": {"view_index": 0, "raster_xy": (104.0, 200.0)},
        "Q000002:M00": {"view_index": 0, "raster_xy": (100.0, 204.0)},
        "Q000003:M00": {"view_index": 0, "raster_xy": (104.0, 204.0)},
    }
    return ObservationEvidenceIR((), metadata={
        "hypothesis_anchor_raster": anchors,
        "raster_coordinate_system": "PIXEL_CENTER_XY",
        "resolution": 1024,
    })


def _surface(nodes) -> RiggingSurfaceIR:
    return RiggingSurfaceIR(
        tuple(nodes),
        geometry_lineage_hash="BASE_SURFACE_HASH",
        metadata={"raster_coordinate_system": "PIXEL_CENTER_XY", "resolution": 1024},
    )


def test_observed_anchor_locality_forms_mwb2_safe_face_complex():
    nodes = (_node(0, 0, 0), _node(1, 4, 0), _node(2, 0, 4), _node(3, 4, 4))
    bound = attach_observed_local_relations_v2(_evidence(), _surface(nodes))
    assert len(bound.local_relations) == 6
    assert all(r.relation_kind == "OBSERVED_LOCAL_RASTER_NEIGHBOR" for r in bound.local_relations)
    assert all(r.metadata["source_mesh_used"] is False for r in bound.local_relations)
    assert all(r.metadata["teacher_truth_used"] is False for r in bound.local_relations)
    candidate = build_mwb2_candidate(bound, view_index=0, camera_binding_hash="CAM:0")
    assert len(candidate.faces) >= 2
    assert candidate.metadata["source_mesh_used"] is False


def test_relation_operator_is_surface_enumeration_invariant():
    nodes = (_node(0, 0, 0), _node(1, 4, 0), _node(2, 0, 4), _node(3, 4, 4))
    a = attach_observed_local_relations_v2(_evidence(), _surface(nodes))
    b = attach_observed_local_relations_v2(_evidence(), _surface(tuple(reversed(nodes))))
    assert a.geometry_lineage_hash == b.geometry_lineage_hash
    assert [r.to_dict() for r in a.local_relations] == [r.to_dict() for r in b.local_relations]


def test_ambiguous_ray_cannot_become_safe_local_bridge():
    nodes = (_node(0, 0, 0), _node(1, 4, 0), _node(2, 0, 4), _node(3, 4, 4, ambiguous=True))
    bound = attach_observed_local_relations_v2(_evidence(), _surface(nodes))
    assert len(bound.local_relations) == 3
    assert all("S:0003" not in (r.a_surface_id, r.b_surface_id) for r in bound.local_relations)
    candidate = build_mwb2_candidate(bound, view_index=0, camera_binding_hash="CAM:0")
    assert len(candidate.faces) == 1
