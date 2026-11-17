from __future__ import annotations

"""Synthetic source-native open cape regression, not real corpus qualification."""
from dataclasses import replace
import pytest
import torch

from models.rss_vnext.native_face_layer_truth_v1 import NativeFaceLayerSamplesV1
from models.rss_vnext.native_layer_open_boundary_v1 import (
    SCHEMA,DERIVATION,NativeLayerOpenBoundaryV1,
    exact_open_boundary_edges_v1,validate_native_open_boundary_v1,
)


def _witness():
    base=torch.tensor([
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
    ])
    vertices=torch.cat((base,base),dim=0)
    faces=torch.tensor([
        [0,1,2],[1,3,2], [4,6,5],[5,6,7],
    ],dtype=torch.long)
    native=NativeFaceLayerSamplesV1(
        vertices=vertices,faces=faces,
        face_layer_ids=torch.tensor([0,0,1,1],dtype=torch.long),
        face_component_ids=torch.tensor([5,5,9,9],dtype=torch.long),
        sample_face_ids=torch.tensor([0,0,1,1,2,2,3,3],dtype=torch.long),
        sample_barycentric=torch.tensor([
            [1.,0.,0.],[0.,1.,0.],[0.,1.,0.],[0.,1.,0.],
            [1.,0.,0.],[0.,1.,0.],[0.,1.,0.],[0.,1.,0.],
        ]),
        source_asset_sha256="a"*64,
        layer_namespace_sha256="b"*64,
    )
    # Deliberately two independent superposed open sheets.
    e=exact_open_boundary_edges_v1(faces)
    ids=torch.unique(e.reshape(-1),sorted=True)
    witness=NativeLayerOpenBoundaryV1(
        boundary_edges=e,boundary_vertex_ids=ids,
        open_layer_ids=torch.tensor([0,1],dtype=torch.long),
        source_asset_sha256=native.source_asset_sha256,
        layer_namespace_sha256=native.layer_namespace_sha256,
    )
    xyz,role=witness.replay(native)
    receipt={
        "source_native_open_boundary_schema":SCHEMA,
        "source_native_open_boundary_sha256":witness.receipt_sha256(),
        "source_native_open_boundary_derivation":DERIVATION,
    }
    return native,witness,xyz,role,receipt


def test_open_body_cape_overlapping_xyz_remain_distinct_native_edges():
    native,w,xyz,roles,receipt=_witness()
    assert w.boundary_edges.shape==(8,2)
    assert roles.tolist()==[0]*4+[1]*4
    torch.testing.assert_close(xyz[:4],xyz[4:])
    out=validate_native_open_boundary_v1(
        w,native,source_boundary_xyz=xyz,
        source_boundary_mask=torch.ones(len(xyz),dtype=torch.bool),
        source_asset_sha256="a"*64,
        layer_namespace_sha256="b"*64,receipt=receipt,
    )
    torch.testing.assert_close(out[0],xyz)
    assert out[2].tolist()==[0,1]


def test_missing_or_extra_open_edge_and_layer_forgery_fail_closed():
    native,w,xyz,roles,receipt=_witness()
    with pytest.raises(ValueError,match="INCOMPLETE_EDGE_COURT"):
        replace(w,boundary_edges=w.boundary_edges[:-1]).replay(native)
    with pytest.raises(ValueError,match="VERTEX_ADDRESS_DRIFT"):
        replace(w,boundary_vertex_ids=w.boundary_vertex_ids[:-1]).replay(native)
    with pytest.raises(ValueError,match="LAYER_COVERAGE_DRIFT"):
        replace(w,open_layer_ids=torch.tensor([0])).replay(native)
    with pytest.raises(ValueError,match="BOUNDARY_RECEIPT_DRIFT"):
        validate_native_open_boundary_v1(
            replace(w,source_asset_sha256="f"*64),native,
            source_boundary_xyz=xyz,
            source_boundary_mask=torch.ones(len(xyz),dtype=torch.bool),
            source_asset_sha256="a"*64,
            layer_namespace_sha256="b"*64,receipt=receipt,
        )


def test_partial_or_spoofed_hem_not_accepted_as_complete_source_truth():
    native,w,xyz,roles,receipt=_witness()
    kw=dict(
        witness=w,native_faces=native,
        source_asset_sha256="a"*64,layer_namespace_sha256="b"*64,
        receipt=receipt,
    )
    with pytest.raises(ValueError,match="BOUNDARY_REPLAY_DRIFT"):
        validate_native_open_boundary_v1(
            source_boundary_xyz=xyz,
            source_boundary_mask=torch.tensor(
                [1,1,1,0,1,1,1,1],dtype=torch.bool,
            ),**kw,
        )
    with pytest.raises(ValueError,match="BOUNDARY_REPLAY_DRIFT"):
        validate_native_open_boundary_v1(
            source_boundary_xyz=xyz.roll(1,0),
            source_boundary_mask=torch.ones(len(xyz),dtype=torch.bool),
            **kw,
        )


def test_nonmanifold_or_duplicate_native_triangles_cannot_prove_open_hem():
    repeated=torch.tensor([
        [0,1,2],[1,0,3],[0,1,4],
    ],dtype=torch.long)
    with pytest.raises(ValueError,match="NONMANIFOLD_EDGE"):
        exact_open_boundary_edges_v1(repeated)
    duplicate=torch.tensor([[0,1,2],[2,1,0]],dtype=torch.long)
    with pytest.raises(ValueError,match="DUPLICATE_TRIANGLE"):
        exact_open_boundary_edges_v1(duplicate)
