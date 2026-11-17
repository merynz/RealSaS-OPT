from __future__ import annotations

import numpy as np
import pytest
import torch

from models.rss_vnext.dcudf2_open_backend_v1 import (
    BACKEND_ID,DCUDF2OpenPolicyV1,
    materialize_dcudf2_open_mesh_v1,
    validate_dcudf2_checkout_v1,
)


def _case():
    xyz=np.array([
        [-.2,-.2,-.5],[.2,-.2,-.5],
        [-.2,.2,-.5],[.2,.2,-.5],
    ],dtype=np.float32)
    faces=np.array([[0,1,2],[1,3,2]],dtype=np.int64)
    volume=np.full((64,64,64),.5,dtype=np.float32)
    scene=torch.ones((1,2,4,4,4))
    udf=lambda x:(x[:,2]+.5).abs()
    policy=DCUDF2OpenPolicyV1(resolution=64)
    return xyz,faces,volume,scene,udf,policy


def _materialize(xyz,faces,volume,scene,udf,policy,role=True):
    return materialize_dcudf2_open_mesh_v1(
        optimized_vertices=xyz,optimized_faces=faces,
        sampled_field=volume,scene_triplanes=scene,
        query_open_component_udf=udf,
        policy=policy,source_role_conditioned=role,
    )


def test_paper_backend_output_is_one_real_open_sheet():
    x,f,volume,scene,udf,policy=_case()
    witness=_materialize(x,f,volume,scene,udf,policy)
    witness.validate()
    assert witness.backend_id==BACKEND_ID
    assert witness.surface.xyz.shape==(1,4,3)
    assert witness.surface.faces.shape==(2,3)
    assert len(witness.surface.boundary_edges)==4
    assert witness.maximum_residual==0.
    assert not witness.open_surface_qualified
    assert not witness.graphics_or_mechanics_qualified


def test_no_silent_closed_double_cover_success_even_when_optimizer_does_not_raise():
    x,f,volume,scene,udf,policy=_case()
    tetra_x=np.array([
        [0.,0.,-.5],[.1,0.,-.5],[0.,.1,-.5],[0.,0.,-.4],
    ],dtype=np.float32)
    tetra_f=np.array([
        [0,2,1],[0,1,3],[1,2,3],[2,0,3],
    ],dtype=np.int64)
    with pytest.raises(ValueError,match="NOT_AN_OPEN_SHEET"):
        _materialize(tetra_x,tetra_f,volume,scene,udf,policy)


def test_global_unconditioned_udf_is_not_a_specific_open_component():
    x,f,volume,scene,udf,policy=_case()
    with pytest.raises(ValueError,match="ROLE_CONDITIONED_H_REQUIRED"):
        _materialize(x,f,volume,scene,udf,policy,role=False)


def test_model_field_residual_stops_false_sheet_even_with_valid_faces():
    x,f,volume,scene,udf,policy=_case()
    x[:,2]=-.3
    with pytest.raises(ValueError,match="FIELD_RESIDUAL_TOO_HIGH"):
        _materialize(x,f,volume,scene,udf,policy)


def test_graph_domain_and_no_orphan_verts_are_enforced():
    x,f,volume,scene,udf,policy=_case()
    with pytest.raises(ValueError,match="ORPHAN_VERTEX"):
        _materialize(
            np.concatenate((x,x[[0]]),axis=0),
            f,volume,scene,udf,policy,
        )
    with pytest.raises(ValueError,match="BACKEND_RESULT_DOMAIN"):
        _materialize(x,f,volume[:32],scene,udf,policy)


def test_licensed_checkout_requires_exact_upstream_git_blobs(tmp_path):
    (tmp_path/"dcudf2").mkdir()
    (tmp_path/"dcudf2"/"mesh_extraction.py").write_text("stub")
    with pytest.raises(ValueError,match="PINNED_SOURCE_DRIFT"):
        validate_dcudf2_checkout_v1(tmp_path)


def test_no_unsigned_open_cut_bypass_via_policy():
    with pytest.raises(ValueError,match="OPEN_QUALITY_POLICY_INVALID"):
        DCUDF2OpenPolicyV1(is_cut=False).validate()
    with pytest.raises(ValueError,match="OPEN_QUALITY_POLICY_INVALID"):
        DCUDF2OpenPolicyV1(watertight_separate=True).validate()


def test_every_output_component_requires_its_own_true_open_boundary():
    x,f,volume,scene,udf,policy=_case()
    # Attach a completely closed double-covered remnant next to a valid open
    # sheet; a global "there exists a boundary" test would falsely PASS.
    tetra_x=np.array([
        [.4,.4,-.5],[.6,.4,-.5],[.4,.6,-.5],[.4,.4,-.3],
    ],dtype=np.float32)
    tetra_f=np.array([
        [0,2,1],[0,1,3],[1,2,3],[2,0,3],
    ],dtype=np.int64)+len(x)
    combined=np.concatenate((x,tetra_x),axis=0)
    combined_f=np.concatenate((f,tetra_f),axis=0)
    with pytest.raises(ValueError,match="CLOSED_COMPONENT_LEAK"):
        _materialize(combined,combined_f,volume,scene,udf,policy)


def test_two_independent_open_sheets_can_coexist_without_cross_welding():
    x,f,volume,scene,udf,policy=_case()
    # The two independent sheets may be exactly coincident in 3D; their
    # indexed component identity must remain distinct, never welded by XYZ.
    together_x=np.concatenate((x,x),axis=0)
    together_f=np.concatenate((f,f+len(x)),axis=0)
    candidate=_materialize(
        together_x,together_f,volume,scene,udf,policy,
    ).surface
    assert candidate.component_ids[:4].unique().numel()==1
    assert candidate.component_ids[4:].unique().numel()==1
    assert candidate.component_ids[0]!=candidate.component_ids[4]
    assert candidate.boundary_edges.shape==(8,2)
