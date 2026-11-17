from __future__ import annotations

"""Real pinned NVIDIA FlexiCubes dense ↔ sparse parity and fail-closed proof.

Small analytical signed sphere is only a regression: it does not qualify
a production high-resolution student field, source closedness, or dataset.
"""
import os

import pytest
import torch

from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from models.rss_vnext.experts.iris.vnext.flexicubes_sparse_v1 import (
    exhaustive_surface_cube_ids_v1,compact_active_grid_v1,
    globally_addressed_case_ids_v1,PinnedSparseFlexiCubesV1,
)


@pytest.fixture(autouse=True)
def cpu_seed():
    torch.manual_seed(101)
    torch.set_num_threads(1)


def _mesh(res=(7,8,9)):
    adapter=PinnedFlexiCubesV1.__new__(PinnedFlexiCubesV1)
    adapter.device=torch.device("cpu")
    xyz,cubes=adapter.construct_iris_grid(res)
    # Slightly off-center to exercise anisotropic boundary cases.
    field=torch.linalg.vector_norm(
        xyz-torch.tensor([0.03,-0.07,0.09]),dim=-1,
    )-0.63
    negatives=(field<0).reshape(
        res[0]+1,res[1]+1,res[2]+1,
    )
    ids=exhaustive_surface_cube_ids_v1(
        negatives,resolution=res,x_slab=2,
    )
    ids1=exhaustive_surface_cube_ids_v1(
        negatives,resolution=res,x_slab=5,
    )
    torch.testing.assert_close(ids,ids1)
    unique,compact,coords=compact_active_grid_v1(ids,res)
    return res,xyz,cubes,field,negatives,ids,unique,compact,coords


def test_global_sparse_grid_exact_corner_ids_no_seams_or_coarse_miss():
    res,xyz,cubes,field,neg,ids,unique,compact,coords=_mesh()
    active=(field[cubes]<0)
    direct=active.any(-1)&~active.all(-1)
    torch.testing.assert_close(ids,torch.nonzero(direct).flatten())
    torch.testing.assert_close(cubes[ids],unique[compact])
    torch.testing.assert_close(coords,xyz[unique],atol=1e-6,rtol=0)
    assert len(unique)<len(xyz)
    assert int(compact.min())==0
    assert int(compact.max())==len(unique)-1
    with pytest.raises(ValueError,match="COMPACT_CUBE_DOMAIN"):
        compact_active_grid_v1(
            torch.flip(ids,dims=(0,)),res,
        )


def _pinned():
    root=os.environ.get("REALSAS_FLEXICUBES_ROOT")
    if not root:
        pytest.skip("Exact pinned NVIDIA checkout not present")
    return PinnedFlexiCubesV1(root,device="cpu")


def test_real_pinned_ambiguous_case_ids_dense_equals_sparse_neighbor_index():
    pinned=_pinned()
    res,xyz,cubes,field,neg,ids,unique,compact,coords=_mesh()
    surf,occupancy=pinned.operator._identify_surf_cubes(field,cubes)
    dense=pinned.operator._get_case_id(occupancy,surf,res)
    assert bool(surf.any())
    sparse=globally_addressed_case_ids_v1(
        pinned.operator,occupancy[ids],
        torch.ones(len(ids),dtype=torch.bool),
        res,ids,
    )
    assert int(surf.sum())==len(ids)
    torch.testing.assert_close(dense,sparse)
    # The full set of cubes (including inactive ones) also reproduces pinned
    # case decisions with a different internal search domain.
    glob=globally_addressed_case_ids_v1(
        pinned.operator,occupancy,surf,
        res,torch.arange(len(cubes),dtype=torch.long),
    )
    torch.testing.assert_close(dense,glob)


def test_actual_pinned_mesh_and_gradients_match_sparse_without_welding():
    pinned=_pinned()
    sparse=PinnedSparseFlexiCubesV1(pinned)
    res,xyz,cubes,field,neg,ids,unique,compact,coords=_mesh()
    dense_raw=torch.randn(len(cubes),21)*.05
    full=pinned.extract(
        vertices_iris=xyz,field=field,cubes=cubes,
        raw_parameters=dense_raw,resolution=res,training=False,
    )
    little=sparse.extract(
        vertices_iris=coords,field=field[unique],cubes=compact,
        raw_parameters=dense_raw[ids],
        global_cube_ids=ids,global_vertex_ids=unique,
        exhaustive_negative_grid=neg,resolution=res,training=False,
    )
    assert len(full[0])==len(little[0])
    assert len(full[1])==len(little[1])
    # cdist's default float32 MM shortcut yields a spurious ~3e-4
    # nonzero diagonal even for equal coordinates. Use exact difference
    # distances, not cancellation of dot products.
    dist=torch.cdist(
        full[0],little[0],
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    assert float(dist.min(dim=-1).values.max())<1.e-5
    assert float(dist.min(dim=0).values.max())<1.e-5

    dense_field=field.detach().clone().requires_grad_(True)
    dense_params=dense_raw.detach().clone().requires_grad_(True)
    dense_out=pinned.extract(
        vertices_iris=xyz,field=dense_field,cubes=cubes,
        raw_parameters=dense_params,resolution=res,training=True,
    )
    dense_loss=dense_out[0].square().sum()+.01*dense_out[2].sum()
    dense_loss.backward()
    active_field=field[unique].detach().clone().requires_grad_(True)
    active_params=dense_raw[ids].detach().clone().requires_grad_(True)
    sparse_out=sparse.extract(
        vertices_iris=coords,field=active_field,cubes=compact,
        raw_parameters=active_params,
        global_cube_ids=ids,global_vertex_ids=unique,
        exhaustive_negative_grid=neg,resolution=res,training=True,
    )
    sparse_loss=sparse_out[0].square().sum()+.01*sparse_out[2].sum()
    sparse_loss.backward()
    torch.testing.assert_close(
        dense_loss,sparse_loss,atol=1e-3,rtol=1e-5,
    )
    torch.testing.assert_close(
        dense_field.grad[unique],active_field.grad,
        atol=1e-3,rtol=2e-4,
    )
    torch.testing.assert_close(
        dense_params.grad[ids],active_params.grad,
        atol=1e-3,rtol=2e-4,
    )


def test_sparse_extraction_rejects_missing_cube_coverage():
    pinned=_pinned()
    sparse=PinnedSparseFlexiCubesV1(pinned)
    res,xyz,cubes,field,neg,ids,unique,compact,coords=_mesh()
    with pytest.raises(ValueError,match="SIGN_COVERAGE_OR_INDEX_DRIFT"):
        sparse.extract(
            vertices_iris=coords,field=field[unique],cubes=compact,
            raw_parameters=torch.zeros(len(ids),21),
            global_cube_ids=ids,
            global_vertex_ids=unique,
            exhaustive_negative_grid=(
                neg.clone().index_put_(
                    (torch.tensor([0]),torch.tensor([0]),torch.tensor([0])),
                    ~neg[0,0,0].reshape(1),
                )
            ),
            resolution=res,training=False,
        )
