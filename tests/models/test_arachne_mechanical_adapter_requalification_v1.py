from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    RiggingSurfaceIR,
    SurfaceNode,
)
from tools.requalify_arachne_mechanical_adapter_v1 import adapted_skin_proposal


def _fixture():
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("s0",(0.0,0.0,0.0),(0,),("src",),("o0",)),
            SurfaceNode("s1",(1.0,0.0,0.0),(0,),("src",),("o1",)),
        ),
        (),
        "surface-hash",
    )
    skeleton=QualifiedSkeletonIR(
        (
            QualifiedJoint("j0",(0.0,0.0,0.0),None,("s0",),"p0"),
            QualifiedJoint("j1",(1.0,0.0,0.0),"j0",("s1",),"p1"),
        ),
        "j0",
        {"status":"PASS"},
        "skeleton-hash",
    )
    return surface,skeleton


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adapted_skin_proposal_binds_receipt_and_preserves_support(tmp_path):
    surface,skeleton=_fixture()
    p=tmp_path/"adapted.npz"
    np.savez_compressed(
        p,
        weights=np.asarray([[1.0,0.0],[0.25,0.75]],np.float32),
        row_joint_mask=np.asarray([[1,0],[1,1]],np.uint8),
        surface_ids=np.asarray(["s0","s1"],dtype="U8"),
        joint_ids=np.asarray(["j0","j1"],dtype="U8"),
    )
    receipt={
        "status":"PASS_FIT_COMPLETED__AWAIT_COMPILER_REQUALIFICATION",
        "adapted_weights_sha256":_sha(p),
        "bundle_sha256":"b"*64,
        "best_step":12,
        "best_semantic_budgets_passed":True,
    }
    proposal,meta=adapted_skin_proposal(
        surface=surface,skeleton=skeleton,adapted_npz=p,fit_receipt=receipt
    )
    assert proposal.surface_binding_hash==surface.geometry_lineage_hash
    assert proposal.skeleton_binding_hash==skeleton.skeleton_lineage_hash
    assert meta["adapter_artifact_sha256"]==_sha(p)
    rows={(x.surface_id,x.canonical_joint_id):x.weight for x in proposal.influences}
    assert ("s0","j1") not in rows
    assert rows[("s0","j0")]==1.0
    assert abs(rows[("s1","j0")]-0.25)<1e-12
    assert abs(rows[("s1","j1")]-0.75)<1e-12


def test_adapted_skin_proposal_rejects_forbidden_support_mint(tmp_path):
    surface,skeleton=_fixture()
    p=tmp_path/"bad.npz"
    np.savez_compressed(
        p,
        weights=np.asarray([[0.9,0.1],[0.25,0.75]],np.float32),
        row_joint_mask=np.asarray([[1,0],[1,1]],np.uint8),
        surface_ids=np.asarray(["s0","s1"],dtype="U8"),
        joint_ids=np.asarray(["j0","j1"],dtype="U8"),
    )
    with pytest.raises(RuntimeError,match="MINTED_FORBIDDEN_SUPPORT"):
        adapted_skin_proposal(
            surface=surface,skeleton=skeleton,adapted_npz=p,fit_receipt=None
        )


def test_adapted_skin_proposal_rejects_receipt_hash_drift(tmp_path):
    surface,skeleton=_fixture()
    p=tmp_path/"adapted.npz"
    np.savez_compressed(
        p,
        weights=np.asarray([[1.0,0.0],[0.25,0.75]],np.float32),
        row_joint_mask=np.asarray([[1,0],[1,1]],np.uint8),
        surface_ids=np.asarray(["s0","s1"],dtype="U8"),
        joint_ids=np.asarray(["j0","j1"],dtype="U8"),
    )
    receipt={
        "status":"PASS_FIT_COMPLETED__AWAIT_COMPILER_REQUALIFICATION",
        "adapted_weights_sha256":"0"*64,
        "bundle_sha256":"b"*64,
        "best_step":1,
        "best_semantic_budgets_passed":True,
    }
    with pytest.raises(RuntimeError,match="RECEIPT_HASH_DRIFT"):
        adapted_skin_proposal(
            surface=surface,skeleton=skeleton,adapted_npz=p,fit_receipt=receipt
        )
