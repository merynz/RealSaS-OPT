from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from compiler.realsas_compiler_services.platform_worker import shape_capability_v2 as shape_v2


class _Court:
    def to_dict(self):
        return {
            "schema": "RealSaS.TESSADecodedTopologyCourt.v2",
            "grammar_valid": True,
            "decode_failure": None,
            "vertex_count": 3,
            "face_count": 1,
            "declared_component_count": 1,
            "geometric_component_count": 1,
            "restart_count": 1,
            "unique_edge_count": 3,
            "boundary_edge_count": 3,
            "boundary_loop_count": 1,
            "open_boundary_chain_count": 0,
            "nonmanifold_edge_count": 0,
            "zero_area_face_count": 0,
            "euler_characteristic": 1,
            "finite_geometry": True,
            "within_scale_policy": True,
            "scientific_pass_claimed": False,
            "training_authorized": False,
        }


def _binding(family: str):
    return SimpleNamespace(
        model_ref=SimpleNamespace(
            model_release_id=family.lower() + "-release",
            checkpoint_content_sha256=("a" if family == "IRIS" else "b") * 64,
            architecture_id=family + ".vX",
        )
    )


def _surface():
    count = 3
    return SimpleNamespace(
        surface_xyz_tessa=torch.tensor(
            [[[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.0, 0.01, 0.0]]],
            dtype=torch.float32,
        ),
        surface_normal=torch.tensor(
            [[[0.0, 0.0, 1.0]] * count], dtype=torch.float32
        ),
        surface_h=torch.zeros(1, count, 8),
        mechanical={
            "stretch_major": torch.ones(1, count),
            "stretch_minor": torch.zeros(1, count),
            "tangent_direction_2theta": torch.zeros(1, count, 2),
            "bend_demand": torch.zeros(1, count),
            "rigidity_probability": torch.zeros(1, count),
            "contact_persistence_probability": torch.zeros(1, count),
        },
        contact_persistence_valid=torch.zeros(1, count, dtype=torch.bool),
    )


def test_shape_v2_roundtrip_serializes_topology_without_mesh_authority():
    decoded = SimpleNamespace(
        vertices_normalized=np.asarray(
            [[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        vertex_component_indices=(0, 0, 0),
        face_component_indices=(0,),
        component_count=1,
        restart_count=1,
        orientation_qualified=False,
    )
    proposal = SimpleNamespace(
        token_ids=(1, 2, 3, 4),
        topology_court=_Court(),
        completed=True,
        stop_reason="EOS",
    )
    payload = shape_v2.encode_shape_proposal_v2(
        iris_binding=_binding("IRIS"),
        tessa_binding=_binding("TESSA"),
        surface=_surface(),
        proposal=proposal,
        decoded=decoded,
    )
    value = shape_v2.decode_shape_proposal_v2(payload)
    assert value.metadata["schema"] == "RealSaS.ShapeProposal.v2"
    assert value.metadata["authority"] == "LEARNED_SHAPE_PROPOSAL_ONLY"
    assert value.metadata["canonical_mesh_authority"] is False
    assert value.metadata["mesh_authority"] is False
    assert value.metadata["decoded_topology_is_proposal_evidence"] is True
    assert value.metadata["mesh_object_frame_transform"] == (
        "REALSAS_OBJECT_FRAME_XYZ_EQUALS_TESSA_XYZ_TIMES_2"
    )
    assert value.decoded_faces.tolist() == [[0, 1, 2]]
    assert value.decoded_vertex_component_indices.tolist() == [0, 0, 0]
    assert value.metadata["decoded_geometry_sha256"]


def test_shape_v2_decode_rejects_geometry_hash_drift():
    decoded = SimpleNamespace(
        vertices_normalized=np.asarray(
            [[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        vertex_component_indices=(0, 0, 0),
        face_component_indices=(0,),
        component_count=1,
        restart_count=1,
        orientation_qualified=False,
    )
    proposal = SimpleNamespace(
        token_ids=(1, 2, 3, 4), topology_court=_Court(), completed=True, stop_reason="EOS"
    )
    payload = shape_v2.encode_shape_proposal_v2(
        iris_binding=_binding("IRIS"),
        tessa_binding=_binding("TESSA"),
        surface=_surface(),
        proposal=proposal,
        decoded=decoded,
    )
    # The codec itself is the authority boundary; normal roundtrip must remain exact.
    value = shape_v2.decode_shape_proposal_v2(payload)
    mutated = value.decoded_vertices_tessa.copy()
    mutated[0, 0] += 0.001
    assert shape_v2._decoded_geometry_sha256(
        mutated,
        value.decoded_faces,
        value.decoded_vertex_component_indices,
        value.decoded_face_component_indices,
    ) != value.metadata["decoded_geometry_sha256"]
