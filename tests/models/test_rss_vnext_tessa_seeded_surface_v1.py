from __future__ import annotations

"""Real tensor/gradient proof of one-surface TESSA topology identity."""
from dataclasses import replace
import pytest
import torch

from models.rss_vnext.tessa_seeded_surface_v1 import (
    build_tessa_seeded_surface_v1, tessa_adaptive_split_edges_v1,
)


def _surface():
    x = torch.tensor(
        [[[0., 1., 0.], [1., 1., 0.], [0., 0., 0.], [1., 0., 0.]]],
        requires_grad=True,
    )
    faces = torch.tensor([[0, 2, 1], [1, 2, 3]], dtype=torch.long)
    components = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    return x, faces, components


def _prepare(*, refine=False, xyz=None, faces=None, components=None):
    a, b, c = _surface()
    return build_tessa_seeded_surface_v1(
        iris_surface_xyz=a if xyz is None else xyz,
        iris_surface_faces=b if faces is None else faces,
        component_ids_by_vertex=c if components is None else components,
        scene_sha256="a" * 64,
        original_surface_sha256="b" * 64,
        one_to_four_refine=refine,
    )


def test_identity_keeps_exact_iris_vertices_and_faces_for_axis_mira():
    x, f, c = _surface()
    r = build_tessa_seeded_surface_v1(
        iris_surface_xyz=x, iris_surface_faces=f,
        component_ids_by_vertex=c,
        scene_sha256="a" * 64, original_surface_sha256="b" * 64,
    )
    assert r.xyz is x
    assert r.faces is f
    assert r.original_vertex_count == 4
    assert r.parent_seed_edges == tuple((i, i) for i in range(4))
    assert not r.canonical_mesh_authority
    assert r.source_authority == "ONE_IRIS_SURFACE_TESSA_REFINEMENT_CANDIDATE"


def test_refinement_preserves_shared_edge_and_support_and_gradients_to_same_H():
    x, f, c = _surface()
    r = build_tessa_seeded_surface_v1(
        iris_surface_xyz=x, iris_surface_faces=f,
        component_ids_by_vertex=c,
        scene_sha256="a" * 64, original_surface_sha256="b" * 64,
        one_to_four_refine=True,
    )
    # Five unique source edges, no independent copied visual geometry.
    assert r.xyz.shape == (1, 9, 3)
    assert r.faces.shape == (8, 3)
    assert r.parent_seed_edges.count((1, 2)) == 1
    assert r.parent_seed_weights[-1] == (0.5, 0.5)
    for i, pair in enumerate(r.parent_seed_edges):
        a, b = pair
        expected = x[0, a] if a == b else (x[0, a] + x[0, b]) * 0.5
        assert torch.allclose(r.xyz[0, i], expected)
    # Geometry is *not* detached: a posed/final-mesh loss can reach IRIS.
    loss = (r.xyz.square().sum() +
            torch.linalg.vector_norm(
                torch.cross(
                    r.xyz[:, r.faces][:, :, 1] - r.xyz[:, r.faces][:, :, 0],
                    r.xyz[:, r.faces][:, :, 2] - r.xyz[:, r.faces][:, :, 0],
                    dim=-1,
                ), dim=-1,
            ).sum())
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    # This is the final topology to pass to AXIS/MIRA; no rebinding.
    assert torch.equal(
        r.vertex_component_ids[r.faces],
        r.face_component_ids[:, None].expand(-1, 3),
    )


def test_cannot_weld_separate_semantic_components():
    x, f, c = _surface()
    with pytest.raises(ValueError, match="CROSS_COMPONENT_TRIANGLE"):
        build_tessa_seeded_surface_v1(
            iris_surface_xyz=x, iris_surface_faces=f,
            component_ids_by_vertex=torch.tensor([0, 0, 1, 0]),
            scene_sha256="a" * 64, original_surface_sha256="b" * 64,
        )


def test_cannot_invent_canonical_authority_or_silent_mesh_rebind():
    r = _prepare()
    with pytest.raises(ValueError, match="TESSA_SEEDED_SURFACE_DOMAIN_INVALID"):
        replace(r, canonical_mesh_authority=True).validate()
    with pytest.raises(ValueError, match="PARENT_VERTEX_SUPPORT_DRIFT"):
        replace(
            r, parent_seed_edges=((0, 0), (1, 1), (2, 2), (3, 0)),
        ).validate()
    x, f, c = _surface()
    with pytest.raises(ValueError, match="DUPLICATE_TRIANGLE_CORNER"):
        build_tessa_seeded_surface_v1(
            iris_surface_xyz=x,
            iris_surface_faces=torch.tensor([[0, 0, 1]], dtype=torch.long),
            component_ids_by_vertex=c,
            scene_sha256="a" * 64, original_surface_sha256="b" * 64,
        )


def test_adaptive_refinement_keeps_parent_support_areas_gradients_and_no_t_junction():
    x, f, c = _surface()
    seed = build_tessa_seeded_surface_v1(
        iris_surface_xyz=x, iris_surface_faces=f,
        component_ids_by_vertex=c,
        scene_sha256="a"*64, original_surface_sha256="b"*64,
    )
    metric = torch.tensor([[0., 2., 2., 0.]],requires_grad=True)
    m = tessa_adaptive_split_edges_v1(
        seed,importance_logits=metric,max_added_vertices=2,
        length_gate_ratio=0.7,
    )
    assert m.original_vertex_count == 4
    assert m.original_face_count == 2
    assert m.xyz.shape[1] in (5,6)
    assert len(m.parent_seed_edges) == m.xyz.shape[1]
    faces = m.faces.detach().cpu().numpy()
    pairs = m.parent_seed_edges[4:]
    for i, (a,b) in enumerate(pairs,start=4):
        torch.testing.assert_close(
            m.xyz[0,i],0.5*(x[0,a]+x[0,b]),
        )
    orig_tri=x[0,f]
    actual_tri=m.xyz[0,m.faces]
    area=lambda t:0.5*torch.linalg.vector_norm(
        torch.cross(t[:,1]-t[:,0],t[:,2]-t[:,0],dim=-1),
        dim=-1,
    ).sum()
    torch.testing.assert_close(
        area(orig_tri),area(actual_tri),atol=1e-6,rtol=1e-6,
    )
    # No original edge is split on one incident triangle but not another.
    from collections import Counter
    sides=Counter()
    for a,b,c0 in m.faces.detach().cpu().tolist():
        for pair in ((a,b),(b,c0),(c0,a)):
            sides[tuple(sorted(pair))]+=1
    assert all(count <= 2 for count in sides.values())
    assert not any(count == 0 for count in sides.values())
    area(actual_tri).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_adaptive_refinement_rejects_repeated_parent_reconstruction():
    x,f,c=_surface()
    seed=build_tessa_seeded_surface_v1(
        iris_surface_xyz=x,iris_surface_faces=f,
        component_ids_by_vertex=c,
        scene_sha256="a"*64,original_surface_sha256="b"*64,
        one_to_four_refine=True,
    )
    with pytest.raises(ValueError,match="ADAPTIVE_SOURCE_OR_POLICY"):
        tessa_adaptive_split_edges_v1(
            seed,importance_logits=torch.zeros(1,seed.xyz.shape[1]),
            max_added_vertices=1,
        )
