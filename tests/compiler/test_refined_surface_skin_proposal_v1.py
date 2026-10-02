from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.refined_surface_skin_proposal_v1 import propose_refined_surface_skin_v1
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.types import (
    QualificationError, QualifiedJoint, QualifiedSkeletonIR, QualifiedSkinIR,
    QualifiedSkinRow, RiggingSurfaceIR, SurfaceNode,
)


def fixture():
    xyz = np.array([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.]])
    def surface(ids, positions, lineage):
        return RiggingSurfaceIR(tuple(SurfaceNode(s, tuple(p), (), (), ())
                                      for s, p in zip(ids, positions)), geometry_lineage_hash=lineage)
    source = surface(("a", "b"), ((1., 0., 0.), (5., 0., 0.)), "source")
    target = surface(("u", "v", "w"), ((0., 0., 0.), (2., 0., 0.), (5., 0., 0.)), "target")
    skeleton = QualifiedSkeletonIR((QualifiedJoint("j0", (0., 0., 0.), None),
                                    QualifiedJoint("j1", (1., 0., 0.), "j0")), "j0", {}, "rig")
    skin = QualifiedSkinIR((QualifiedSkinRow("a", (("j0", .25), ("j1", .75)), 0., 0.),
                            QualifiedSkinRow("b", (("j0", 1.),), 0., 0.)), "source", "rig", {}, "skin")
    return dict(source_surface=source, target_surface=target, skeleton=skeleton, source_skin=skin,
                dense_positions=xyz, source_inverse=np.array([0, 0, 1, 1]),
                target_inverse=np.array([0, 1, 2, 2]), dense_source_sha256="a" * 64)


def test_preserves_parent_field_and_requires_new_qualification():
    args = fixture()
    proposal, receipt = propose_refined_surface_skin_v1(**args)
    skin = qualify_skin(args["target_surface"], args["skeleton"], proposal)
    assert dict(skin.rows[0].influences) == {"j0": .25, "j1": .75}
    assert dict(skin.rows[1].influences) == {"j0": .25, "j1": .75}
    assert dict(skin.rows[2].influences) == {"j0": 1.}
    assert receipt["product_authority_minted"] is False
    assert receipt["new_position_prediction_claimed"] is False
    assert receipt["requires_dynamic_requalification"] is True
    assert propose_refined_surface_skin_v1(**args)[1] == receipt


@pytest.mark.parametrize("fault,reason", [
    ("stale_surface", "SOURCE_SURFACE_DRIFT"),
    ("stale_rig", "SKELETON_DRIFT"),
    ("missing_row", "ROW_ACCOUNTING"),
    ("negative_weight", "SIMPLEX"),
    ("reordered_nodes", "NODE_ORDER_OR_POSITION"),
    ("float_inverse", "INVERSE_INVALID"),
    ("missing_cluster", "CLUSTER_ACCOUNTING"),
])
def test_rejects_invalid_transport(fault, reason):
    args = fixture()
    skin = args["source_skin"]
    if fault == "stale_surface": args["source_skin"] = replace(skin, surface_binding_hash="stale")
    if fault == "stale_rig": args["source_skin"] = replace(skin, skeleton_binding_hash="stale")
    if fault == "missing_row": args["source_skin"] = replace(skin, rows=skin.rows[:1])
    if fault == "negative_weight":
        args["source_skin"] = replace(skin, rows=(replace(skin.rows[0], influences=(("j0", -1.), ("j1", 2.))), skin.rows[1]))
    if fault == "reordered_nodes":
        args["target_surface"] = replace(args["target_surface"], surface_nodes=tuple(reversed(args["target_surface"].surface_nodes)))
    if fault == "float_inverse": args["target_inverse"] = args["target_inverse"].astype(float)
    if fault == "missing_cluster": args["target_inverse"] = np.array([0, 0, 2, 2])
    with pytest.raises(QualificationError, match=reason): propose_refined_surface_skin_v1(**args)


def test_rejects_cross_parent_merge_even_when_centroids_match():
    args = fixture()
    args["target_inverse"] = np.array([0, 1, 0, 2])
    nodes = args["target_surface"].surface_nodes
    args["target_surface"] = replace(args["target_surface"], surface_nodes=(
        replace(nodes[0], P=(2., 0., 0.)), replace(nodes[1], P=(2., 0., 0.)), replace(nodes[2], P=(6., 0., 0.))))
    with pytest.raises(QualificationError, match="CROSS_SOURCE_CLUSTER_MERGE"):
        propose_refined_surface_skin_v1(**args)
