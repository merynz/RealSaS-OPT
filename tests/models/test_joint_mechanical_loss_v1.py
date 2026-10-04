import math

import torch

from models.shared.joint_mechanical_loss_v1 import (
    JointMechanicalLossConfigV1,
    joint_mechanical_loss_v1,
    lbs_points_v1,
)


DTYPE=torch.float64


def _fixture():
    rest=torch.tensor([
        [0.0,0.0,0.0],  # a
        [1.0,0.0,0.0],  # b
        [1.0,1.0,0.0],  # c
        [0.0,1.0,0.0],  # d
    ],dtype=DTYPE)
    # Same quad geometry; only the diagonal differs.
    ac=torch.tensor([[0,1,2],[0,2,3]],dtype=torch.long)
    bd=torch.tensor([[0,1,3],[1,2,3]],dtype=torch.long)

    theta=1.4
    c=math.cos(theta); s=math.sin(theta)
    ident=torch.eye(4,dtype=DTYPE)
    moved=torch.tensor([
        [c,-s,0.0,0.3],
        [s, c,0.0,-0.1],
        [0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,1.0],
    ],dtype=DTYPE)
    mats=torch.stack([ident,moved],dim=0).unsqueeze(0)

    # Exact same per-vertex skin field in both topology arms.
    bone1=torch.tensor([0.77,0.74,0.14,0.18],dtype=DTYPE)
    weights=torch.stack([1.0-bone1,bone1],dim=1)
    return rest,ac,bd,weights,mats


def test_identity_is_zero_inside_mechanical_limits():
    rest,ac,_,weights,_=_fixture()
    ident=torch.eye(4,dtype=DTYPE).repeat(1,2,1,1)
    out=joint_mechanical_loss_v1(rest,ac,weights,ident)
    assert float(out["loss"]) == 0.0
    assert abs(float(out["max_condition"])-1.0) < 1e-10
    assert abs(float(out["min_area_ratio"])-1.0) < 1e-10
    assert abs(float(out["max_edge_ratio"])-1.0) < 1e-10


def test_diagonal_flip_changes_mechanics_with_same_geometry_rig_weights_and_pose():
    rest,ac,bd,weights,mats=_fixture()

    # Posed vertices are topology-independent and therefore exactly identical.
    posed=lbs_points_v1(rest,weights,mats)
    assert posed.shape==(1,4,3)

    out_ac=joint_mechanical_loss_v1(rest,ac,weights,mats)
    out_bd=joint_mechanical_loss_v1(rest,bd,weights,mats)

    # Only face connectivity changed. AC nearly collapses one triangle while BD
    # remains within the frozen V1 mechanical limits.
    assert float(out_ac["max_condition"]) > 100.0
    assert float(out_ac["min_area_ratio"]) < 0.01
    assert float(out_ac["loss"]) > 1.0

    assert float(out_bd["max_condition"]) < 4.0
    assert float(out_bd["min_area_ratio"]) > 0.20
    assert float(out_bd["max_edge_ratio"]) < 2.0
    assert float(out_bd["loss"]) == 0.0


def test_joint_mechanical_loss_backpropagates_to_skin_logits():
    rest,ac,_,weights,mats=_fixture()
    logits=torch.log(weights).detach().clone().requires_grad_(True)
    live=torch.softmax(logits,dim=1)
    out=joint_mechanical_loss_v1(rest,ac,live,mats)
    out["loss"].backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(torch.linalg.vector_norm(logits.grad)) > 1e-6


def test_threshold_configuration_is_explicit():
    cfg=JointMechanicalLossConfigV1(
        max_condition=16.0,
        min_area_ratio=0.05,
        max_area_ratio=20.0,
        max_edge_ratio=4.0,
    )
    assert cfg.max_condition==16.0
    assert cfg.min_area_ratio==0.05
    assert cfg.max_area_ratio==20.0
    assert cfg.max_edge_ratio==4.0
