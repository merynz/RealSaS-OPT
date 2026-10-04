from types import SimpleNamespace

import numpy as np
import torch

from compiler.realsas_compiler_core.mechanical_carrier_skin_v1 import (
    qualify_mechanical_carrier_skin_v1,
    validate_mechanical_carrier_skin_v1,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from models.mira.carrier_query_v1 import (
    build_mira_mechanical_carrier_query_v1,
    transport_surface_memory_to_carrier_v1,
)


def _binding(mode, coeffs, metadata=None):
    return SimpleNamespace(mode=mode, coefficients=tuple(coeffs), metadata=metadata or {})


def _fixture():
    candidate = SimpleNamespace(
        candidate_lineage_hash="c" * 64,
        vertices=(
            SimpleNamespace(candidate_vertex_id="a", P=(0.0, 0.0, 0.0),
                            support_binding=_binding("IDENTITY_SURFACE_NODE", (("s0", 1.0),))),
            SimpleNamespace(candidate_vertex_id="b", P=(1.0, 0.0, 0.0),
                            support_binding=_binding("LOCAL_CONVEX_INTERPOLATION", (("s0", 0.25), ("s1", 0.75)))),
            SimpleNamespace(candidate_vertex_id="c", P=(0.0, 1.0, 0.0),
                            support_binding=_binding("IDENTITY_SURFACE_NODE", (("s1", 1.0),))),
        ),
        faces=(("a", "b", "c"),),
    )
    carrier = SimpleNamespace(
        candidate_mesh_binding_hash="c" * 64,
        ordered_vertex_ids=("a", "b", "c"),
        positions=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        normals=((0.0, 0.0, 1.0),) * 3,
        normal_valid=(True, True, True),
        carrier_evidence_hash="e" * 64,
        topology_hash="t" * 64,
    )
    tensor = SimpleNamespace(
        surface_ids=("s0", "s1"),
        normalization_center=np.asarray([0.0, 0.0, 0.0], np.float64),
        normalization_scale=1.0,
        tensorization_hash="g" * 64,
    )
    conditioning = SimpleNamespace(
        surface_ids=(("s0", "s1"),),
        surface_tensorization_hashes=("g" * 64,),
        joint_ids=(("j0", "j1"),),
        joint_positions_normalized=np.asarray([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], np.float32),
        parent_indices=np.asarray([[ -1, 0]], np.int64),
        source_skeleton_hashes=("k" * 64,),
    )
    skeleton = SimpleNamespace(
        skeleton_lineage_hash="k" * 64,
        joints=(
            SimpleNamespace(canonical_joint_id="j0"),
            SimpleNamespace(canonical_joint_id="j1"),
        ),
    )
    surface = SimpleNamespace(
        geometry_lineage_hash="s" * 64,
        surface_nodes=(SimpleNamespace(surface_id="s0"), SimpleNamespace(surface_id="s1")),
    )
    return candidate, carrier, tensor, conditioning, skeleton, surface


def test_carrier_query_uses_exact_geometry_and_support_bound_memory_transport():
    candidate, carrier, tensor, conditioning, _, _ = _fixture()
    query = build_mira_mechanical_carrier_query_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        surface_tensor=tensor,
        conditioning=conditioning,
    )

    np.testing.assert_allclose(query.positions_normalized, np.asarray(carrier.positions))
    np.testing.assert_allclose(query.normals, np.asarray(carrier.normals))
    assert query.pair_geometry.shape == (3, 2, 10)
    assert query.legal_pair.all()

    memory = torch.tensor([[1.0, 10.0], [5.0, 30.0]], dtype=torch.float32)
    transported = transport_surface_memory_to_carrier_v1(memory, query)
    expected = torch.tensor([
        [1.0, 10.0],
        [4.0, 25.0],
        [5.0, 30.0],
    ])
    torch.testing.assert_close(transported, expected)


def test_direct_carrier_skin_bypasses_surface_weight_transfer():
    candidate, carrier, _, _, skeleton, surface = _fixture()
    weights = np.asarray([
        [1.0, 0.0],
        [0.2, 0.8],
        [0.0, 1.0],
    ], np.float64)
    skin = qualify_mechanical_carrier_skin_v1(
        candidate=candidate,
        carrier_evidence=carrier,
        skeleton=skeleton,
        vertex_ids=carrier.ordered_vertex_ids,
        joint_ids=("j0", "j1"),
        weights=weights,
    )
    validate_mechanical_carrier_skin_v1(
        skin,
        candidate=candidate,
        carrier_evidence=carrier,
        skeleton=skeleton,
    )
    rest, actual, faces = _candidate_skin_matrix(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    np.testing.assert_allclose(rest, np.asarray(carrier.positions))
    np.testing.assert_allclose(actual, weights)
    assert faces == ((0, 1, 2),)
    assert skin.qualification_report["surface_skin_transfer_used"] is False
