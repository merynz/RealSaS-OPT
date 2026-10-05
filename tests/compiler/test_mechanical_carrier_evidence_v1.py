from types import SimpleNamespace

import pytest

from compiler.realsas_compiler_core.mechanical_carrier_evidence_v1 import (
    build_mechanical_carrier_evidence_v1,
    mechanical_carrier_evidence_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError


def _candidate(*, diagonal: str):
    vertices = (
        SimpleNamespace(candidate_vertex_id="a", P=(0.0, 0.0, 0.0)),
        SimpleNamespace(candidate_vertex_id="b", P=(1.0, 0.0, 0.0)),
        SimpleNamespace(candidate_vertex_id="c", P=(1.0, 1.0, 0.0)),
        SimpleNamespace(candidate_vertex_id="d", P=(0.0, 1.0, 0.0)),
    )
    if diagonal == "ac":
        faces = (("a", "b", "c"), ("a", "c", "d"))
    elif diagonal == "bd":
        faces = (("a", "b", "d"), ("b", "c", "d"))
    else:
        raise ValueError(diagonal)
    return SimpleNamespace(
        vertices=vertices,
        faces=faces,
        candidate_lineage_hash=("a" if diagonal == "ac" else "b") * 64,
        surface_binding_hash="s" * 64,
    )


def _qualification(candidate):
    return SimpleNamespace(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash="x" * 64,
        qualification_hash="q" * 64,
    )


def _addressing(candidate):
    return SimpleNamespace(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        addressing_hash="x" * 64,
    )


def _build(diagonal: str):
    candidate = _candidate(diagonal=diagonal)
    return build_mechanical_carrier_evidence_v1(
        candidate,
        static_qualification=_qualification(candidate),
        surface_addressing=_addressing(candidate),
    )


def test_carrier_evidence_is_deterministic_and_roundtrips():
    a = _build("ac")
    b = _build("ac")
    assert a == b
    assert a.carrier_evidence_hash == b.carrier_evidence_hash
    assert a.topology_hash == b.topology_hash
    assert a.geometry_hash == b.geometry_hash
    assert all(a.normal_valid)
    assert set(a.normals) == {(0.0, 0.0, 1.0)}
    assert mechanical_carrier_evidence_from_dict(a.to_dict()) == a


def test_connectivity_change_changes_carrier_even_when_geometry_is_identical():
    ac = _build("ac")
    bd = _build("bd")

    assert ac.positions == bd.positions
    assert ac.normals == bd.normals
    assert ac.geometry_hash == bd.geometry_hash

    assert ac.face_vertex_indices != bd.face_vertex_indices
    assert ac.topology_hash != bd.topology_hash
    assert ac.carrier_evidence_hash != bd.carrier_evidence_hash


def test_static_qualification_must_bind_exact_candidate():
    candidate = _candidate(diagonal="ac")
    bad_qualification = SimpleNamespace(
        candidate_mesh_binding_hash="z" * 64,
        surface_addressing_binding_hash="x" * 64,
        qualification_hash="q" * 64,
    )
    with pytest.raises(
        QualificationError, match="MECHANICAL_CARRIER_STATIC_BINDING_DRIFT"
    ):
        build_mechanical_carrier_evidence_v1(
            candidate,
            static_qualification=bad_qualification,
            surface_addressing=_addressing(candidate),
        )


def test_surface_addressing_must_bind_exact_candidate():
    candidate = _candidate(diagonal="ac")
    bad_addressing = SimpleNamespace(
        candidate_mesh_binding_hash="z" * 64,
        addressing_hash="x" * 64,
    )
    with pytest.raises(
        QualificationError, match="MECHANICAL_CARRIER_ADDRESSING_BINDING_DRIFT"
    ):
        build_mechanical_carrier_evidence_v1(
            candidate,
            static_qualification=_qualification(candidate),
            surface_addressing=bad_addressing,
        )
