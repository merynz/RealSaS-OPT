from types import SimpleNamespace

import numpy as np
import pytest

from models.shared.surface_evidence_input_v1 import (
    assert_no_downstream_authority_fields_v1,
    build_surface_evidence_encoder_input_v1,
)


def _surface():
    edge_index=np.asarray([[0,1],[1,2]],dtype=np.int64)
    degree=np.asarray([1,2,1],dtype=np.int64)
    return SimpleNamespace(
        positions_normalized=np.asarray([[0,0,0],[1,0,0],[1,1,0]],dtype=np.float32),
        normals=np.asarray([[0,0,1],[0,0,1],[0,0,1]],dtype=np.float32),
        normal_valid=np.asarray([1,1,1],dtype=bool),
        support=np.asarray([
            [1,1,0,0,0,0,0,0],
            [1,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
        ],dtype=bool),
        raster_xy_normalized=np.zeros((3,8,2),dtype=np.float32),
        raster_valid=np.asarray([
            [1,1,0,0,0,0,0,0],
            [1,1,1,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
        ],dtype=bool),
        observed=np.asarray([1,1,1],dtype=bool),
        completed=np.asarray([0,0,0],dtype=bool),
        edge_index=edge_index,
        edge_score=np.asarray([0.9,0.8],dtype=np.float32),
        edge_distance_normalized=np.asarray([0.2,0.3],dtype=np.float32),
        edge_crosses_unknown=np.asarray([0,1],dtype=bool),
        edge_unknown_bridge=np.asarray([0,0],dtype=bool),
        degree=degree,
        source_surface_hash="surface-hash",
        tensorization_hash="tensor-hash",
    )


def test_shared_contract_derives_task_neutral_evidence_deterministically():
    yaw=np.asarray([
        [np.sin(i),np.cos(i),np.sin(2*i),np.cos(2*i)] for i in range(8)
    ],dtype=np.float32)
    a=build_surface_evidence_encoder_input_v1(_surface(),view_yaw_fourier=yaw)
    b=build_surface_evidence_encoder_input_v1(_surface(),view_yaw_fourier=yaw)
    assert a.contract_hash==b.contract_hash
    assert a.node_count==3 and a.edge_count==2
    np.testing.assert_array_equal(a.degree,[1,2,1])
    np.testing.assert_allclose(a.support_fraction[:,0],[0.25,0.375,0.25])
    np.testing.assert_allclose(a.raster_fraction,a.support_fraction)
    assert a.edge_features.shape==(2,4)


def test_shared_contract_rejects_degree_drift():
    s=_surface()
    s.degree=np.asarray([1,1,1],dtype=np.int64)
    with pytest.raises(ValueError,match="DERIVED_DEGREE_DRIFT"):
        build_surface_evidence_encoder_input_v1(
            s,view_yaw_fourier=np.zeros((8,4),dtype=np.float32)
        )


@pytest.mark.parametrize("key",[
    "teacher_skin","skin_weights","bone_id","parent_index","qualified_skeleton",
])
def test_shared_contract_forbids_downstream_authority(key):
    with pytest.raises(ValueError,match="DOWNSTREAM_AUTHORITY_FORBIDDEN"):
        assert_no_downstream_authority_fields_v1({key:object()})


def test_shared_contract_allows_surface_only_payload():
    assert_no_downstream_authority_fields_v1({
        "positions_normalized":None,
        "normals":None,
        "support_views":None,
        "raster_xy":None,
        "edge_features":None,
        "view_yaw_fourier":None,
    })
