from __future__ import annotations

"""Fine-grid sign coverage and opt-in IRIS product sparse integration."""

import os
import pytest
import torch
from torch import nn

from models.iris.vnext.flexicubes_v1 import PinnedFlexiCubesV1
from models.rss_vnext.sparse_layer_sign_scan_v1 import (
    global_grid_xyz_v1,scan_exhaustive_signed_grid_v1,
    qualify_sparse_resolution_v1,
)
from models.rss_vnext.experts.iris.vnext.flexicubes_sparse_v1 import (
    exhaustive_surface_cube_ids_v1,
)
from models.rss_vnext.flexicubes_surface_evidence import (
    RSSvNextFlexiCubesEvidenceBridge,
)
from models.rss_vnext.semantic_geometry_prior import (
    RSSvNextSemanticLayerSpec,realsas_style_generic_schema,
)


@pytest.fixture(autouse=True)
def set_cpu():
    torch.manual_seed(127)
    torch.set_num_threads(1)


def test_exhaustive_sign_scan_equals_all_dense_cube_crossings_and_global_xyz():
    resolution=(7,8,9)
    stub=PinnedFlexiCubesV1.__new__(PinnedFlexiCubesV1)
    stub.device=torch.device("cpu")
    dense_xyz,dense_cubes=stub.construct_iris_grid(resolution)
    all_ids=torch.arange(len(dense_xyz))
    torch.testing.assert_close(
        global_grid_xyz_v1(all_ids,resolution=resolution),
        dense_xyz,atol=1.e-7,rtol=0,
    )
    max_chunk=[0]
    def sphere(points):
        max_chunk[0]=max(max_chunk[0],points.shape[1])
        return points.norm(dim=-1)-0.64
    neg=scan_exhaustive_signed_grid_v1(
        sphere,resolution=resolution,device=torch.device("cpu"),
        scene_dtype=torch.float32,query_chunk=17,
    )
    dense_inside=sphere(dense_xyz[None])[0]<0
    cube_signs=dense_inside[dense_cubes]
    expected=torch.nonzero(
        cube_signs.any(-1)&~cube_signs.all(-1),
    ).flatten()
    actual=exhaustive_surface_cube_ids_v1(
        neg,resolution=resolution,x_slab=3,
    )
    torch.testing.assert_close(actual,expected)
    assert max_chunk[0]<=17


def test_sign_scanner_rejects_forged_incomplete_or_nonfinite_output():
    kw=dict(resolution=(3,4,5),device=torch.device("cpu"),
            scene_dtype=torch.float32,query_chunk=9)
    with pytest.raises(ValueError,match="FIELD_DOMAIN"):
        scan_exhaustive_signed_grid_v1(
            lambda x: torch.full(x.shape[:2],float("nan")),**kw,
        )
    with pytest.raises(ValueError,match="FIELD_DOMAIN"):
        scan_exhaustive_signed_grid_v1(
            lambda x: torch.zeros(x.shape[1]),**kw,
        )
    with pytest.raises(ValueError,match="NO_SIGN_CHANGE_ABSTAIN"):
        scan_exhaustive_signed_grid_v1(
            lambda x: x.new_ones(x.shape[:2]),**kw,
        )
    with pytest.raises(ValueError,match="GRID_BUDGET"):
        qualify_sparse_resolution_v1((512,512,512))


class _IRIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.cube_extraction=nn.Linear(64,21)
        self.max_query=0

    def query(self,planes,points):
        self.max_query=max(self.max_query,points.shape[1])
        xyz=points.float()
        h=torch.cat((xyz,xyz.square(),xyz[...,:2]),dim=-1)
        return dict(
            H=h,field=xyz.norm(dim=-1)-0.62,
            extraction_delta=torch.zeros_like(xyz),
        )


def _args():
    roles=realsas_style_generic_schema()
    return dict(
        scene_triplanes=torch.ones((1,3,2,2,2)),
        role_head=nn.Linear(8,roles.count).eval(),
        schema=roles,layer=RSSvNextSemanticLayerSpec("all",roles.roles),
        field_policy="SIGNED_HYPOTHESIS_ONLY",
    )


def test_sparse_product_is_explicitly_opt_in_and_never_training_fallback():
    iris=_IRIS().eval()
    bridge=RSSvNextFlexiCubesEvidenceBridge(
        iris,resolution=3,inference_resolution=(7,8,9),
        field_query_chunk=20,cube_query_chunk=5,
        surface_query_chunk=8,max_cubes=512,
    ).eval()
    args=_args()
    with torch.no_grad(),pytest.raises(
        ValueError,match="SPARSE_INFERENCE_ONLY_PINNED",
    ):
        bridge(extractor=object(),sparse_product_resolution=(7,8,9),**args)
    bridge.train()
    with torch.no_grad(),pytest.raises(
        ValueError,match="SPARSE_INFERENCE_ONLY_PINNED",
    ):
        bridge(extractor=object(),sparse_product_resolution=(7,8,9),**args)


def test_pinned_sparse_product_matches_dense_live_iris_field_and_surface():
    root=os.environ.get("REALSAS_FLEXICUBES_ROOT")
    if not root:
        pytest.skip("Requires independently pinned NVIDIA checkout")
    pinned=PinnedFlexiCubesV1(root,device="cpu")
    args=_args()
    iris=_IRIS().eval()
    bridge=RSSvNextFlexiCubesEvidenceBridge(
        iris,resolution=3,inference_resolution=(7,8,9),
        field_query_chunk=20,cube_query_chunk=5,
        surface_query_chunk=8,max_cubes=512,
    ).eval()
    with torch.no_grad():
        dense=bridge(extractor=pinned,**args)
        sparse=bridge(
            extractor=pinned,sparse_product_resolution=(7,8,9),**args,
        )
    assert sparse.sparse_product_resolution==(7,8,9)
    assert sparse.sparse_sign_sha256 is not None
    assert sparse.sparse_global_cube_ids is not None
    assert sparse.sparse_global_vertex_ids is not None
    assert sparse.sparse_global_cube_ids.numel()<7*8*9
    assert sparse.sample_grid_xyz.shape[1]<dense.sample_grid_xyz.shape[1]
    assert sparse.cube_parameters.shape[0]<dense.cube_parameters.shape[0]
    torch.testing.assert_close(dense.faces,sparse.faces)
    torch.testing.assert_close(
        dense.xyz,sparse.xyz,atol=2.e-5,rtol=1.e-5,
    )
    assert iris.max_query<=max(20,5*8,8)
