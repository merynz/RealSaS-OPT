from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.mesh.direct_model_skin import (
    DIRECT_MODEL_TRANSFER_METHOD,
    qualify_direct_model_mesh_skin,
)
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    qualify_supported_mesh,
)
from compiler.realsas_compiler_core.types import (
    MeshDiscretizationCandidateIR,
    MeshVertexCandidate,
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)


def _fixture():
    surface = RiggingSurfaceIR(
        surface_nodes=(
            SurfaceNode("S0", (0.0, 0.0, 0.0), (0,), (), ()),
            SurfaceNode("S1", (1.0, 0.0, 0.0), (0,), (), ()),
            SurfaceNode("S2", (0.0, 1.0, 0.0), (0,), (), ()),
        ),
        geometry_lineage_hash="SURFACE_NEW",
    )
    skeleton = QualifiedSkeletonIR(
        joints=(
            QualifiedJoint("J0", (0.0, 0.0, 0.0), None),
            QualifiedJoint("J1", (1.0, 0.0, 0.0), "J0"),
        ),
        root_id="J0",
        qualification_report={},
        skeleton_lineage_hash="SKELETON",
    )
    # This authority intentionally belongs to an older surface. Direct-model
    # qualification may use its learned-skin lineage, but must never consume rows.
    old_skin_authority = QualifiedSkinIR(
        rows=(),
        surface_binding_hash="SURFACE_OLD",
        skeleton_binding_hash="SKELETON",
        qualification_report={"status": "SEALED_MODEL_SKIN_AUTHORITY"},
        skin_lineage_hash="SKIN_V5",
    )
    verts = tuple(
        MeshVertexCandidate(
            f"C{i}",
            surface.surface_nodes[i].P,
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((f"S{i}", 1.0),)),
            metadata={"raster_xy": (float(i), 0.0)},
        )
        for i in range(3)
    )
    candidate = MeshDiscretizationCandidateIR(
        vertices=verts,
        faces=(("C0", "C1", "C2"),),
        edges=(("C0", "C1"), ("C1", "C2"), ("C0", "C2")),
        surface_binding_hash=surface.geometry_lineage_hash,
        view_index=0,
        camera_binding_hash="CAM0",
        candidate_lineage_hash="",
    )
    candidate = replace(candidate, candidate_lineage_hash=mesh_candidate_lineage_hash(candidate))
    mesh = qualify_supported_mesh(surface, candidate)
    return surface, skeleton, old_skin_authority, mesh


def test_direct_model_exact_mesh_weights_pass_without_source_row_transfer():
    surface, skeleton, old_skin, mesh = _fixture()
    weights = {
        "MV:00000": (0.75, 0.25),
        "MV:00001": (0.20, 0.80),
        "MV:00002": (0.50, 0.50),
    }
    bound = qualify_direct_model_mesh_skin(
        surface,
        skeleton,
        old_skin,
        mesh,
        joint_ids=("J0", "J1"),
        predicted_weights_by_vertex=weights,
        model_provenance={"architecture": "ArachneV5", "checkpoint": "SEALED"},
    )
    assert bound.transfer_method == DIRECT_MODEL_TRANSFER_METHOD
    assert bound.mesh_binding_hash == mesh.mesh_lineage_hash
    assert bound.skin_binding_hash == "SKIN_V5"
    assert bound.qualification_report["source_skin_rows_consumed"] is False
    assert bound.qualification_report["historical_barycentric_weight_transfer_used"] is False
    assert bound.qualification_report["direct_model_query"] is True
    assert len(bound.rows) == len(mesh.vertices) == 3


def test_direct_model_rejects_missing_or_extra_mesh_vertex_rows():
    surface, skeleton, old_skin, mesh = _fixture()
    with pytest.raises(QualificationError, match="DIRECT_MODEL_WEIGHT_VERTEX_SET_MISMATCH"):
        qualify_direct_model_mesh_skin(
            surface,
            skeleton,
            old_skin,
            mesh,
            joint_ids=("J0", "J1"),
            predicted_weights_by_vertex={"MV:00000": (0.5, 0.5)},
            model_provenance={},
        )


def test_direct_model_rejects_wrong_joint_set_and_material_negative_weights():
    surface, skeleton, old_skin, mesh = _fixture()
    exact = {vid: (0.5, 0.5) for vid in ("MV:00000", "MV:00001", "MV:00002")}
    with pytest.raises(QualificationError, match="DIRECT_MODEL_JOINT_SET_MISMATCH"):
        qualify_direct_model_mesh_skin(
            surface,
            skeleton,
            old_skin,
            mesh,
            joint_ids=("J0", "JX"),
            predicted_weights_by_vertex=exact,
            model_provenance={},
        )
    bad = dict(exact)
    bad["MV:00001"] = (-0.1, 1.1)
    with pytest.raises(QualificationError, match="DIRECT_MODEL_WEIGHT_MATERIAL_NEGATIVE"):
        qualify_direct_model_mesh_skin(
            surface,
            skeleton,
            old_skin,
            mesh,
            joint_ids=("J0", "J1"),
            predicted_weights_by_vertex=bad,
            model_provenance={},
        )


def test_direct_model_rejects_skin_authority_from_different_skeleton():
    surface, skeleton, old_skin, mesh = _fixture()
    drift = replace(old_skin, skeleton_binding_hash="OTHER_G")
    exact = {vid: (0.5, 0.5) for vid in ("MV:00000", "MV:00001", "MV:00002")}
    with pytest.raises(QualificationError, match="DIRECT_MODEL_SKIN_AUTHORITY_SKELETON_MISMATCH"):
        qualify_direct_model_mesh_skin(
            surface,
            skeleton,
            drift,
            mesh,
            joint_ids=("J0", "J1"),
            predicted_weights_by_vertex=exact,
            model_provenance={},
        )
