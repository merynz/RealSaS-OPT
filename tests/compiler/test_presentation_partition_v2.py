from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.appearance_bake_v2 import bake_direction_atlas
from compiler.realsas_compiler_core.presentation_partition_v2 import (
    build_presentation_partition_evidence,
    presentation_partition_evidence_from_dict,
)
from compiler.realsas_compiler_core.product_state_v2 import (
    build_presentation_structure_v2,
)


POLICY = {
    "mechanical_binding_policy": {
        "min_rigid_owner_weight": 0.999,
        "max_rigid_other_mass": 0.001,
    },
    "appearance_boundary_policy": {
        "edge_samples_per_direction": 5,
        "edge_inset_fraction": 0.15,
        "min_source_backed_sample_pairs": 8,
        "mean_premultiplied_rgba_l1_cut": 0.18,
        "p95_premultiplied_rgba_l1_cut": 0.30,
        "source_backed_provenance": ["DIRECT_SOURCE", "OTHER_VIEW_SOURCE"],
    },
}


def _mesh():
    return SimpleNamespace(
        vertices=(
            SimpleNamespace(
                canonical_mesh_vertex_id="v0", component_id="c0", P=(0.0, 0.0, 0.0)
            ),
            SimpleNamespace(
                canonical_mesh_vertex_id="v1", component_id="c0", P=(1.0, 0.0, 0.0)
            ),
            SimpleNamespace(
                canonical_mesh_vertex_id="v2", component_id="c0", P=(0.0, 1.0, 0.0)
            ),
            SimpleNamespace(
                canonical_mesh_vertex_id="v3", component_id="c0", P=(1.0, 1.0, 0.0)
            ),
        ),
        faces=(("v0", "v1", "v2"), ("v1", "v3", "v2")),
        mesh_lineage_hash="m" * 64,
    )


def _baked(face1_rgba):
    sample_count = 8 * 9 // 2
    rgba = np.zeros((2 * sample_count, 4), dtype=np.uint8)
    rgba[:sample_count] = np.asarray([220, 40, 30, 255], dtype=np.uint8)
    rgba[sample_count:] = np.asarray(face1_rgba, dtype=np.uint8)
    provenance = np.zeros((2 * sample_count,), dtype=np.uint8)
    atlas, prov, uv, _layout = bake_direction_atlas(
        face_sample_rgba=rgba,
        face_sample_provenance=provenance,
        face_count=2,
        tile_resolution=8,
        bleed_px=2,
    )
    return atlas, prov, uv


def _structure(mesh, evidence):
    skeleton = SimpleNamespace(
        joints=(SimpleNamespace(canonical_joint_id="j0"),),
        root_id="j0",
        skeleton_lineage_hash="s" * 64,
    )
    mesh_skin = SimpleNamespace(
        rows=tuple(
            SimpleNamespace(
                canonical_mesh_vertex_id=f"v{i}",
                influences=(("j0", 1.0),),
            )
            for i in range(4)
        ),
        mesh_skin_lineage_hash="w" * 64,
    )
    partition = SimpleNamespace(
        components=(SimpleNamespace(component_id="c0"),),
        partition_lineage_hash="p" * 64,
    )
    carrier = SimpleNamespace(
        decisions=(SimpleNamespace(component_id="c0", carrier_class="MESH"),),
        carrier_policy_lineage_hash="c" * 64,
    )
    return build_presentation_structure_v2(
        skeleton=skeleton,
        mesh=mesh,
        mesh_skin=mesh_skin,
        partition=partition,
        carrier_policy=carrier,
        min_rigid_owner_weight=0.999,
        max_rigid_other_mass=0.001,
        presentation_cut_face_pairs=evidence.cut_face_pairs,
        presentation_partition_evidence_hash=evidence.evidence_hash,
    )


def test_role_free_partition_splits_connected_faces_on_source_backed_visual_boundary():
    mesh = _mesh()
    atlas, prov, uv = _baked([20, 80, 230, 255])
    evidence = build_presentation_partition_evidence(
        mesh=mesh,
        appearance_asset_hash="a" * 64,
        appearance_qualification_hash="q" * 64,
        face_uv=uv,
        textures_by_direction={i: atlas for i in range(8)},
        provenance_by_direction=np.stack([prov for _ in range(8)], axis=0),
        policy=POLICY,
    )
    assert evidence.cut_face_pairs == ((0, 1),)
    assert evidence.source_supported_edge_count == 1
    assert evidence.boundary_measurements[0]["decision"] == (
        "CUT_SOURCE_BACKED_APPEARANCE_BOUNDARY"
    )
    decoded = presentation_partition_evidence_from_dict(evidence.to_dict())
    assert decoded.evidence_hash == evidence.evidence_hash

    structure = _structure(mesh, evidence)
    assert len(structure.slots) == 2
    assert len(structure.attachments) == 2
    assert structure.metadata["appearance_boundary_evidence_consumed"] is True
    assert structure.metadata["categorical_recognition_used"] is False
    assert structure.metadata["conceptual_object_identity_claimed"] is False


def test_role_free_partition_keeps_connected_faces_when_source_backed_art_is_continuous():
    mesh = _mesh()
    atlas, prov, uv = _baked([220, 40, 30, 255])
    evidence = build_presentation_partition_evidence(
        mesh=mesh,
        appearance_asset_hash="a" * 64,
        appearance_qualification_hash="q" * 64,
        face_uv=uv,
        textures_by_direction={i: atlas for i in range(8)},
        provenance_by_direction=np.stack([prov for _ in range(8)], axis=0),
        policy=POLICY,
    )
    assert evidence.cut_face_pairs == ()
    assert evidence.source_supported_edge_count == 1

    structure = _structure(mesh, evidence)
    assert len(structure.slots) == 1
    assert len(structure.attachments) == 1
