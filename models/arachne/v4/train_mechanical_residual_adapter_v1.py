"""Train only the small Arachne mechanical residual adapter on a sealed V9 bundle.

The base Arachne field is frozen data.  This script never updates the historical
Arachne checkpoint.  It optimizes the zero-init residual conditioner against a
product-space mechanical consequence objective with explicit semantic/trust
budgets.  It is intended for A100/Colab after CPU preflight has sealed the bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from models.arachne.v4.mechanical_consequence_loss_v1 import mechanical_consequence_loss_v1
from models.arachne.v4.mechanical_residual_adapter_v1 import MechanicalResidualAdapterV1


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def q(x: torch.Tensor,p:float)->float:
    if x.numel()==0:return 0.0
    return float(torch.quantile(x.float(),torch.tensor(p,device=x.device)).detach().cpu())


def correction_metrics(pred,base,teacher,valid):
    base_l1=(pred-base).abs().sum(-1)[0]
    teach_l1=(pred-teacher).abs().sum(-1)[0]
    vm=valid[0].bool()
    return {
        "base_mean_l1":float(base_l1.mean().detach().cpu()),
        "base_p95_l1":q(base_l1,0.95),
        "base_max_l1":float(base_l1.max().detach().cpu()),
        "teacher_valid_mean_l1":float(teach_l1[vm].mean().detach().cpu()),
        "teacher_valid_p95_l1":q(teach_l1[vm],0.95),
        "teacher_valid_max_l1":float(teach_l1[vm].max().detach().cpu()),
    }


def _load_bundle(path: Path, dev: torch.device):
    with np.load(path,allow_pickle=False) as z:
        def T(name,dtype=None):
            arr=np.asarray(z[name])
            t=torch.from_numpy(arr.copy())
            if dtype is not None:t=t.to(dtype)
            return t.to(dev)
        common={
            "base":T("base_weights",torch.float32)[None],
            "teacher":T("teacher_weights",torch.float32)[None],
            "valid":T("teacher_valid_mask",torch.bool)[None],
            "geometry7":T("geometry7",torch.float32)[None],
            "pair":T("pair_geometry",torch.float32)[None],
            "surface_mask":T("surface_mask",torch.bool)[None],
            "joint_mask":T("joint_mask",torch.bool)[None],
            "probes":T("probe_transforms",torch.float32)[None],
            "probe_ids":tuple(map(str,z["probe_ids"].tolist())),
        }
        if "carrier_ids" in z.files:
            all_carrier_ids=tuple(map(str,z["carrier_ids"].tolist()))
            if not all_carrier_ids:
                raise ValueError("multi-carrier bundle has empty carrier_ids")
            admitted=(
                np.asarray(z["carrier_training_admitted"]).astype(bool).tolist()
                if "carrier_training_admitted" in z.files
                else [True]*len(all_carrier_ids)
            )
            if len(admitted)!=len(all_carrier_ids):
                raise ValueError("carrier admission axis drift")
            carriers=[]
            diagnostic_carriers=[]
            for i,(cid,is_admitted) in enumerate(zip(all_carrier_ids,admitted)):
                p=f"carrier_{i:02d}_"
                row={
                    "id":cid,
                    "support_idx":T(p+"support_indices",torch.long)[None],
                    "support_coeff":T(p+"support_coefficients",torch.float32)[None],
                    "rest":T(p+"rest_vertices",torch.float32)[None],
                    "faces":T(p+"faces",torch.long),
                    "training_admitted":bool(is_admitted),
                }
                diagnostic_carriers.append(row)
                if is_admitted:
                    carriers.append(row)
            carrier_ids=tuple(row["id"] for row in carriers)
            diagnostic_carrier_ids=tuple(row["id"] for row in diagnostic_carriers)
        else:
            carrier_ids=("single",)
            diagnostic_carrier_ids=carrier_ids
            carriers=[{
                "id":"single",
                "support_idx":T("candidate_support_indices",torch.long)[None],
                "support_coeff":T("candidate_support_coefficients",torch.float32)[None],
                "rest":T("candidate_rest_vertices",torch.float32)[None],
                "faces":T("candidate_faces",torch.long),
                "training_admitted":True,
            }]
    common["carrier_ids"]=carrier_ids
    common["diagnostic_carrier_ids"]=diagnostic_carrier_ids
    common["carriers"]=tuple(carriers)
    return common


def _aggregate_carrier_mechanical(rows):
    if not rows:
        raise ValueError("carrier mechanical rows empty")
    values=torch.stack([row["mechanical"] for row in rows])
    mean=values.mean()
    worst=values.max()
    aggregate=0.5*mean+0.5*worst
    return aggregate,mean,worst


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--steps",type=int,required=True)
    ap.add_argument("--lr",type=float,required=True)
    ap.add_argument("--weight-decay",type=float,default=0.0)
    ap.add_argument("--probe-batch",type=int,required=True)
    ap.add_argument("--surface-chunk",type=int,default=512)
    ap.add_argument("--mechanical-weight",type=float,required=True)
    ap.add_argument("--teacher-weight",type=float,required=True)
    ap.add_argument("--trust-weight",type=float,required=True)
    ap.add_argument("--max-base-p95-l1",type=float,required=True)
    ap.add_argument("--max-teacher-valid-p95-l1",type=float,required=True)
    ap.add_argument("--seed",type=int,default=20261003)
    ap.add_argument("--log-every",type=int,default=25)
    ap.add_argument("--min-training-carriers",type=int,default=2)
    a=ap.parse_args()

    if a.steps<1 or a.lr<=0 or a.probe_batch<1 or a.surface_chunk<1 or a.min_training_carriers<1:
        raise ValueError("invalid optimization hyperparameters")
    if min(a.mechanical_weight,a.teacher_weight,a.trust_weight)<0:
        raise ValueError("negative objective weight forbidden")
    if a.mechanical_weight<=0:
        raise ValueError("mechanical objective must be active")
    if min(a.max_base_p95_l1,a.max_teacher_valid_p95_l1)<0:
        raise ValueError("negative correction budget forbidden")

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    dev=torch.device(a.device)
    if dev.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    bundle=_load_bundle(a.bundle,dev)
    base=bundle["base"]; teacher=bundle["teacher"]; valid=bundle["valid"]
    geometry7=bundle["geometry7"]; pair=bundle["pair"]
    sm=bundle["surface_mask"]; jm=bundle["joint_mask"]
    probes=bundle["probes"]; probe_ids=bundle["probe_ids"]
    carriers=bundle["carriers"]; carrier_ids=bundle["carrier_ids"]
    diagnostic_carrier_ids=bundle["diagnostic_carrier_ids"]
    if len(carriers)<int(a.min_training_carriers):
        raise RuntimeError(
            f"insufficient training-admitted carriers: {len(carriers)} < {a.min_training_carriers}"
        )

    P=probes.shape[1]
    if a.probe_batch>P:
        raise ValueError(f"probe batch {a.probe_batch} exceeds frozen bank {P}")

    model=MechanicalResidualAdapterV1().to(dev)
    if model.parameter_count!=1_582_849:
        raise RuntimeError("adapter parameter contract drift")
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)

    a.out_dir.mkdir(parents=True,exist_ok=True)
    receipt={
        "schema":"RealSaS.ArachneMechanicalResidualAdapterFitReceipt.v1",
        "status":"RUNNING",
        "bundle_sha256":sha256(a.bundle),
        "seed":a.seed,
        "trainable_parameters":model.parameter_count,
        "frozen_base_field":True,
        "optimizer":"AdamW",
        "steps":a.steps,
        "lr":a.lr,
        "weight_decay":a.weight_decay,
        "probe_batch":a.probe_batch,
        "surface_chunk":a.surface_chunk,
        "carrier_ids":list(carrier_ids),
        "carrier_count":len(carrier_ids),
        "diagnostic_carrier_ids":list(diagnostic_carrier_ids),
        "diagnostic_carrier_count":len(diagnostic_carrier_ids),
        "min_training_carriers":int(a.min_training_carriers),
        "carrier_mechanical_aggregation":"0.5_MEAN_PLUS_0.5_WORST",
        "objective_weights":{
            "mechanical":a.mechanical_weight,
            "teacher_valid":a.teacher_weight,
            "base_trust":a.trust_weight,
        },
        "budgets":{
            "base_p95_l1":a.max_base_p95_l1,
            "teacher_valid_p95_l1":a.max_teacher_valid_p95_l1,
        },
        "history":[],
        "claim_boundary":[
            "Knight integration fit only; no genericity claim.",
            "Historical Arachne/base field is frozen.",
            "Only MechanicalResidualAdapterV1 parameters are trainable.",
            "Mechanical fit is evaluated across a frozen carrier ensemble when present.",
            "Carrier objective is 0.5 mean + 0.5 worst; one topology cannot hide another.",
            "Compiler exact G3B/G3/motion courts remain final authority.",
        ],
    }

    best=None
    full_order=np.arange(P,dtype=np.int64)
    rng=np.random.default_rng(a.seed)
    # Pre-register one deterministic permutation stream; no metric-conditioned resampling.
    schedules=[]
    while sum(len(x) for x in schedules)<a.steps*a.probe_batch:
        schedules.append(rng.permutation(full_order))
    stream=np.concatenate(schedules)
    stream=stream[:a.steps*a.probe_batch].reshape(a.steps,a.probe_batch)

    for step in range(a.steps):
        idx=torch.as_tensor(stream[step],device=dev,dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=(dev.type=="cuda")):
            pred,delta=model(
                base_weights=base,
                surface_geometry7=geometry7,
                pair_geometry=pair,
                surface_mask=sm,
                joint_mask=jm,
                surface_chunk_size=a.surface_chunk,
            )
        carrier_loss_rows=[]
        for carrier in carriers:
            carrier_loss_rows.append(mechanical_consequence_loss_v1(
                pred.float(),
                candidate_support_indices=carrier["support_idx"],
                candidate_support_coefficients=carrier["support_coeff"],
                candidate_rest_vertices=carrier["rest"],
                candidate_faces=carrier["faces"],
                probe_transforms=probes[:,idx],
                base_surface_weights=None,
                teacher_surface_weights=None,
                teacher_valid_mask=None,
                max_edge_ratio=4.0,
                min_area_ratio=0.05,
                max_area_ratio=20.0,
                max_condition_number=16.0,
                mechanical_weight=1.0,
                teacher_weight=0.0,
                trust_weight=0.0,
            ))
        mechanical,mechanical_mean,mechanical_worst=_aggregate_carrier_mechanical(
            carrier_loss_rows
        )
        trust=(pred.float()-base).abs().sum(-1).mean()
        teacher_row=(pred.float()-teacher).abs().sum(-1)
        vm=valid.bool()
        supervised=teacher_row[vm].mean() if bool(vm.any()) else teacher_row.new_zeros(())
        total=(
            a.mechanical_weight*mechanical
            + a.teacher_weight*supervised
            + a.trust_weight*trust
        )
        if not torch.isfinite(total):
            raise RuntimeError(f"nonfinite loss at step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if step==0 or (step+1)%a.log_every==0 or step+1==a.steps:
            with torch.no_grad():
                metrics=correction_metrics(pred.float(),base,teacher,valid)
                per_carrier={
                    carrier["id"]:{
                        "mechanical":float(row_loss["mechanical"].detach().cpu()),
                        "edge":float(row_loss["edge"].detach().cpu()),
                        "area_hi":float(row_loss["area_hi"].detach().cpu()),
                        "area_lo":float(row_loss["area_lo"].detach().cpu()),
                        "condition":float(row_loss["condition"].detach().cpu()),
                        "max_edge_ratio":float(row_loss["maximum_edge_ratio"].detach().cpu()),
                        "min_area_ratio":float(row_loss["minimum_area_ratio"].detach().cpu()),
                        "max_area_ratio":float(row_loss["maximum_area_ratio"].detach().cpu()),
                        "max_condition_number":float(row_loss["maximum_condition_number"].detach().cpu()),
                    }
                    for carrier,row_loss in zip(carriers,carrier_loss_rows)
                }
                worst_carrier=max(
                    per_carrier,
                    key=lambda cid:(per_carrier[cid]["mechanical"],cid),
                )
                row={
                    "step":step+1,
                    "probe_indices":stream[step].tolist(),
                    "probe_ids":[probe_ids[i] for i in stream[step]],
                    "total":float(total.detach().cpu()),
                    "mechanical":float(mechanical.detach().cpu()),
                    "mechanical_mean":float(mechanical_mean.detach().cpu()),
                    "mechanical_worst":float(mechanical_worst.detach().cpu()),
                    "worst_carrier_id":worst_carrier,
                    "per_carrier":per_carrier,
                    **metrics,
                }
                receipt["history"].append(row)
                print("ARACHNE_MECH_FIT_STEP="+json.dumps(row,sort_keys=True),flush=True)
                admissible=(
                    metrics["base_p95_l1"]<=a.max_base_p95_l1
                    and metrics["teacher_valid_p95_l1"]<=a.max_teacher_valid_p95_l1
                )
                key=(0 if admissible else 1,float(row["mechanical"]),float(row["total"]),step)
                if best is None or key<best[0]:
                    best=(key,step+1,{k:v.detach().cpu() for k,v in model.state_dict().items()},row)

    if best is None:
        raise RuntimeError("no adapter checkpoint candidate")
    _,best_step,best_state,best_row=best
    torch.save({
        "schema":"RealSaS.ArachneMechanicalResidualAdapterCheckpoint.v1",
        "state_dict":best_state,
        "parameter_count":model.parameter_count,
        "bundle_sha256":receipt["bundle_sha256"],
        "best_step":best_step,
        "best_metrics":best_row,
        "config":model.config.__dict__,
    },a.out_dir/"ARACHNE_MECHANICAL_RESIDUAL_ADAPTER_V1.pt")
    receipt["status"]="PASS_FIT_COMPLETED__AWAIT_COMPILER_REQUALIFICATION"
    receipt["best_step"]=best_step
    receipt["best_metrics"]=best_row
    receipt["best_semantic_budgets_passed"]=bool(
        best_row["base_p95_l1"]<=a.max_base_p95_l1
        and best_row["teacher_valid_p95_l1"]<=a.max_teacher_valid_p95_l1
    )
    (a.out_dir/"FIT_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    print("ARACHNE_MECH_FIT_RESULT="+json.dumps({
        "status":receipt["status"],
        "best_step":best_step,
        "best_metrics":best_row,
        "semantic_budgets_passed":receipt["best_semantic_budgets_passed"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
