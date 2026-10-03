from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from models.arachne.v4.train_mechanical_residual_adapter_v1 import (
    _aggregate_carrier_mechanical,
    _load_bundle,
)


def _write_common(path, *, multi: bool):
    N,J,P=3,2,2
    base=np.asarray([[1.,0.],[0.5,0.5],[0.,1.]],np.float32)
    teacher=base.copy()
    payload={
        "base_weights":base,
        "teacher_weights":teacher,
        "teacher_valid_mask":np.asarray([1,1,1],np.uint8),
        "geometry7":np.zeros((N,7),np.float32),
        "pair_geometry":np.zeros((N,J,10),np.float32),
        "surface_mask":np.ones((N,),np.uint8),
        "joint_mask":np.ones((J,),np.uint8),
        "probe_transforms":np.tile(np.eye(4,dtype=np.float32),(P,J,1,1)),
        "probe_ids":np.asarray(["REST","P1"],dtype="U16"),
    }
    if multi:
        payload.update({
            "carrier_ids":np.asarray(["base","child"],dtype="U16"),
            "carrier_00_support_indices":np.asarray([[0],[1],[2]],np.int64),
            "carrier_00_support_coefficients":np.ones((3,1),np.float32),
            "carrier_00_rest_vertices":np.asarray(
                [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32
            ),
            "carrier_00_faces":np.asarray([[0,1,2]],np.int64),
            "carrier_01_support_indices":np.asarray(
                [[0,1],[1,2],[0,2],[2,-1]],np.int64
            ),
            "carrier_01_support_coefficients":np.asarray(
                [[0.5,0.5],[0.25,0.75],[0.75,0.25],[1.0,0.0]],np.float32
            ),
            "carrier_01_rest_vertices":np.asarray(
                [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]],np.float32
            ),
            "carrier_01_faces":np.asarray([[0,1,2],[1,3,2]],np.int64),
        })
    else:
        payload.update({
            "candidate_support_indices":np.asarray([[0],[1],[2]],np.int64),
            "candidate_support_coefficients":np.ones((3,1),np.float32),
            "candidate_rest_vertices":np.asarray(
                [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],np.float32
            ),
            "candidate_faces":np.asarray([[0,1,2]],np.int64),
        })
    np.savez_compressed(path,**payload)


def test_multi_carrier_bundle_loader_preserves_independent_shapes(tmp_path):
    path=tmp_path/"multi.npz"
    _write_common(path,multi=True)
    b=_load_bundle(path,torch.device("cpu"))
    assert b["carrier_ids"]==("base","child")
    assert len(b["carriers"])==2
    assert tuple(b["carriers"][0]["rest"].shape)==(1,3,3)
    assert tuple(b["carriers"][0]["faces"].shape)==(1,3)
    assert tuple(b["carriers"][1]["rest"].shape)==(1,4,3)
    assert tuple(b["carriers"][1]["faces"].shape)==(2,3)
    assert tuple(b["carriers"][1]["support_idx"].shape)==(1,4,2)


def test_single_carrier_bundle_remains_backward_compatible(tmp_path):
    path=tmp_path/"single.npz"
    _write_common(path,multi=False)
    b=_load_bundle(path,torch.device("cpu"))
    assert b["carrier_ids"]==("single",)
    assert len(b["carriers"])==1
    assert tuple(b["carriers"][0]["support_idx"].shape)==(1,3,1)


def test_carrier_mechanical_aggregation_emphasizes_worst_without_hiding_mean():
    rows=[
        {"mechanical":torch.tensor(1.0)},
        {"mechanical":torch.tensor(3.0)},
        {"mechanical":torch.tensor(2.0)},
    ]
    aggregate,mean,worst=_aggregate_carrier_mechanical(rows)
    assert float(mean)==2.0
    assert float(worst)==3.0
    assert float(aggregate)==2.5


def test_multi_carrier_loader_filters_nonadmitted_training_carriers(tmp_path):
    path=tmp_path/"multi_admission.npz"
    _write_common(path,multi=True)
    with np.load(path,allow_pickle=False) as z:
        payload={k:np.asarray(z[k]) for k in z.files}
    payload["carrier_training_admitted"]=np.asarray([1,0],np.uint8)
    np.savez_compressed(path,**payload)
    b=_load_bundle(path,torch.device("cpu"))
    assert b["diagnostic_carrier_ids"]==("base","child")
    assert b["carrier_ids"]==("base",)
    assert len(b["carriers"])==1
    assert b["carriers"][0]["training_admitted"] is True


def test_trainer_entrypoint_completes_one_cpu_step(tmp_path):
    bundle=tmp_path/"fit.npz"
    _write_common(bundle,multi=True)
    with np.load(bundle,allow_pickle=False) as z:
        payload={k:np.asarray(z[k]) for k in z.files}
    payload["carrier_training_admitted"]=np.asarray([1,1],np.uint8)
    payload["carrier_lineage_hashes"]=np.asarray(
        ["lineage-a","lineage-b"],dtype="U32"
    )
    payload["row_joint_mask"]=(payload["base_weights"]>1e-8).astype(np.uint8)
    np.savez_compressed(bundle,**payload)
    out=tmp_path/"out"
    subprocess.run(
        [
            sys.executable,
            "-m","models.arachne.v4.train_mechanical_residual_adapter_v1",
            "--bundle",str(bundle),
            "--out-dir",str(out),
            "--device","cpu",
            "--steps","1",
            "--lr","0.0001",
            "--probe-batch","1",
            "--surface-chunk","2",
            "--mechanical-weight","1.0",
            "--teacher-weight","1.0",
            "--trust-weight","0.25",
            "--max-base-p95-l1","1.0",
            "--max-teacher-valid-p95-l1","1.0",
            "--min-training-carriers","2",
            "--eval-probe-count","2",
            "--checkpoint-every","1",
            "--log-every","1",
        ],
        check=True,
    )
    receipt=json.loads((out/"FIT_RECEIPT.json").read_text())
    assert receipt["status"]=="PASS_FIT_COMPLETED__AWAIT_COMPILER_REQUALIFICATION"
    assert receipt["carrier_count"]==2
    assert receipt["unique_training_carrier_lineages"]==2
    assert receipt["semantic_support_hard_mask"] is True
    assert (out/"ARACHNE_MECHANICAL_RESIDUAL_ADAPTER_V1.pt").is_file()
    assert (out/"ARACHNE_MECHANICAL_RESIDUAL_ADAPTER_RESUME.pt").is_file()
    assert (out/"ADAPTED_SURFACE_WEIGHTS_V1.npz").is_file()
    assert receipt["adapted_weights_sha256"]
