from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    build_mechanical_core_target_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)


def _teacher_arrays():
    parents = np.asarray([-1, 0, 1, 2, 1], dtype=np.int64)
    deform = np.ones(5, dtype=bool)
    skin = np.zeros((4, 5), dtype=np.float32)
    skin[0, 1] = 1.0
    skin[1, 3] = 1.0
    heads_world = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    return parents, deform, skin, heads_world


def test_mechanical_core_uses_support_and_required_bridge_only():
    parents, deform, skin, heads_world = _teacher_arrays()
    out = build_mechanical_core_target_v1(
        parents=parents,
        deform_mask=deform,
        skin=skin,
        bone_heads_world=heads_world,
    )
    assert out.count == 3
    assert set(out.source_indices_provenance_only.tolist()) == {1, 2, 3}
    assert 0 not in out.source_indices_provenance_only
    assert 4 not in out.source_indices_provenance_only
    assert int(out.root_mask.sum()) == 1
    assert np.array_equal(out.parent_indices, np.asarray([-1, 0, 1], dtype=np.int64))


def test_mechanical_core_is_source_row_permutation_invariant():
    parents, deform, skin, heads_world = _teacher_arrays()
    ref = build_mechanical_core_target_v1(
        parents=parents,
        deform_mask=deform,
        skin=skin,
        bone_heads_world=heads_world,
    )
    perm = np.asarray([4, 2, 0, 3, 1], dtype=np.int64)
    inv = np.empty(len(perm), dtype=np.int64)
    inv[perm] = np.arange(len(perm), dtype=np.int64)
    p2 = np.asarray(
        [-1 if parents[old] < 0 else inv[int(parents[old])] for old in perm],
        dtype=np.int64,
    )
    got = build_mechanical_core_target_v1(
        parents=p2,
        deform_mask=deform[perm],
        skin=skin[:, perm],
        bone_heads_world=heads_world[perm],
    )
    assert np.array_equal(ref.positions_world, got.positions_world)
    assert np.array_equal(ref.parent_indices, got.parent_indices)
    assert np.array_equal(ref.root_mask, got.root_mask)


def _surface(*, teacher_truth_used: bool = False) -> RiggingSurfaceIR:
    nodes = (
        SurfaceNode(
            surface_id="s0",
            P=(0.0, 0.0, 0.0),
            support_views=(0,),
            provenance_refs=("IRIS_PROMOTED",),
            source_observation_ids=(),
            raster_bindings=((0, (511.5, 511.5)),),
            persistence_group_id="pg0",
            derived_normal=(1.0, 0.0, 0.0),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": teacher_truth_used},
        ),
        SurfaceNode(
            surface_id="s1",
            P=(1.0, 0.0, 0.0),
            support_views=(),
            provenance_refs=("IRIS_PROMOTED",),
            source_observation_ids=(),
            raster_bindings=(),
            persistence_group_id="pg1",
            derived_normal=(1.0, 0.0, 0.0),
            validity_flags=("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
    )
    relations = (
        SurfaceRelation(
            relation_id="r0",
            a_surface_id="s0",
            b_surface_id="s1",
            relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
            score=1.0,
            metadata={
                "world_distance": 1.0,
                "crosses_unknown": False,
                "unknown_bridge": False,
                "teacher_truth_used": False,
            },
        ),
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash="surface_hash",
        builder_id="RealSaS.GeometricSubstrateAssembler.SceneFirstSigned.v1",
        metadata={
            "scene_first_signed_geometry": True,
            "raster_coordinate_system": "PIXEL_CENTER_XY",
            "resolution": 1024,
            "Nd_operator_sha256": "operator_hash",
            "teacher_truth_used": False,
        },
    )


def test_rigging_surface_tensorization_preserves_direct_evidence_and_topology():
    out = tensorize_rigging_surface_v1(_surface())
    assert out.node_count == 2
    assert out.edge_count == 1
    assert out.surface_ids == ("s0", "s1")
    assert out.positions_world.shape == (2, 3)
    assert out.normals.shape == (2, 3)
    assert out.support.shape == (2, 8)
    assert out.raster_xy_normalized.shape == (2, 8, 2)
    assert bool(out.support[0, 0])
    assert bool(out.raster_valid[0, 0])
    assert np.allclose(out.raster_xy_normalized[0, 0], np.zeros(2), atol=0, rtol=0)
    assert bool(out.observed[0])
    assert bool(out.completed[1])
    assert np.array_equal(out.edge_index, np.asarray([[0, 1]], dtype=np.int64))
    assert np.array_equal(out.degree, np.asarray([1, 1], dtype=np.int64))
    assert out.relation_kind_vocab == ("SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",)


def test_rigging_surface_tensorization_fails_closed_on_teacher_contamination():
    with pytest.raises(ValueError, match="teacher truth contamination"):
        tensorize_rigging_surface_v1(_surface(teacher_truth_used=True))
