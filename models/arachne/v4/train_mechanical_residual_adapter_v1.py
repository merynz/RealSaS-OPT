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
            "row_joint_mask":(
                T("row_joint_mask",torch.bool)[None]
                if "row_joint_mask" in z.files
                else None
            ),
            "probes":T("probe_transforms",torch.float32)[None],
            "probe_ids":tuple(map(str,z["probe_ids"].tolist())),
            "surface_ids":(
                tuple(map(str,z["surface_ids"].tolist()))
                if "surface_ids" in z.files else None
            ),
            "joint_ids":(
                tuple(map(str,z["joint_ids"].tolist()))
                if "joint_ids" in z.files else None
            ),
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
            lineage_hashes=(
                tuple(map(str,z["carrier_lineage_hashes"].tolist()))
                if "carrier_lineage_hashes" in z.files
                else all_carrier_ids
            )
            if len(lineage_hashes)!=len(all_carrier_ids):
                raise ValueError("carrier lineage axis drift")
            if len(admitted)!=len(all_carrier_ids):
                raise ValueError("carrier admission axis drift")
            carriers=[]
            diagnostic_carriers=[]
            for i,(cid,is_admitted,lineage_hash) in enumerate(
                zip(all_carrier_ids,admitted,lineage_hashes)
            ):
                p=f"carrier_{i:02d}_"
                row={
                    "id":cid,
                    "support_idx":T(p+"support_indices",torch.long)[None],
                    "support_coeff":T(p+"support_coefficients",torch.float32)[None],
                    "rest":T(p+"rest_vertices",torch.float32)[None],
                    "faces":T(p+"faces",torch.long),
                    "training_admitted":bool(is_admitted),
                    "lineage_hash":str(lineage_hash),
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
                "lineage_hash":"single",
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
    ap.add_argument("--checkpoint-every",type=int,default=100)
    ap.add_argument("--resume-checkpoint",type=Path)
    ap.add_argument("--eval-probe-count",type=int,default=32)
    a=ap.parse_args()

    if (
        a.steps<1 or a.lr<=0 or a.probe_batch<1 or a.surface_chunk<1
        or a.min_training_carriers<1 or a.checkpoint_every<1
        or a.eval_probe_count<1
    ):
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
    row_jm=bundle["row_joint_mask"]
    if row_jm is None:
        # Backward-compatible local/unit bundle path only. Sealed A100 bundles
        # must carry an explicit row_joint_mask from CPU preflight.
        row_jm=(base>1e-8)
    probes=bundle["probes"]; probe_ids=bundle["probe_ids"]
    carriers=bundle["carriers"]; carrier_ids=bundle["carrier_ids"]
    diagnostic_carrier_ids=bundle["diagnostic_carrier_ids"]
    if len(carriers)<int(a.min_training_carriers):
        raise RuntimeError(
            f"insufficient training-admitted carriers: {len(carriers)} < {a.min_training_carriers}"
        )
    unique_training_lineages={str(x["lineage_hash"]) for x in carriers}
    if len(unique_training_lineages)<int(a.min_training_carriers):
        raise RuntimeError(
            "insufficient distinct training carrier lineages: "
            f"{len(unique_training_lineages)} < {a.min_training_carriers}"
        )

    P=probes.shape[1]
    if a.probe_batch>P:
        raise ValueError(f"probe batch {a.probe_batch} exceeds frozen bank {P}")
    if a.eval_probe_count>P:
        raise ValueError(
            f"eval probe count {a.eval_probe_count} exceeds frozen bank {P}"
        )

    model=MechanicalResidualAdapterV1().to(dev)
    if model.parameter_count!=1_582_849:
        raise RuntimeError("adapter parameter contract drift")
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)

    a.out_dir.mkdir(parents=True,exist_ok=True)
    start_step=0
    resumed_from=None
    resume_best=None
    resume_history=[]
    if a.resume_checkpoint is not None:
        ck=torch.load(a.resume_checkpoint,map_location="cpu",weights_only=False)
        if str(ck.get("bundle_sha256") or "")!=sha256(a.bundle):
            raise RuntimeError("resume checkpoint bundle hash mismatch")
        if int(ck.get("parameter_count") or -1)!=model.parameter_count:
            raise RuntimeError("resume checkpoint parameter contract drift")
        model.load_state_dict(ck["state_dict"])
        if "optimizer_state_dict" in ck:
            opt.load_state_dict(ck["optimizer_state_dict"])
        start_step=int(ck.get("next_step") or 0)
        if start_step<0 or start_step>a.steps:
            raise RuntimeError("resume checkpoint step outside requested schedule")
        resumed_from=str(a.resume_checkpoint)
        resume_best=ck.get("best")
        resume_history=list(ck.get("history") or ())
    receipt={
        "schema":"RealSaS.ArachneMechanicalResidualAdapterFitReceipt.v1",
        "status":"RUNNING",
        "bundle_sha256":sha256(a.bundle),
        "seed":a.seed,
        "trainable_parameters":model.parameter_count,
        "frozen_base_field":True,
        "optimizer":"AdamW",
        "steps":a.steps,
        "start_step":start_step,
        "resumed_from":resumed_from,
        "checkpoint_every":a.checkpoint_every,
        "lr":a.lr,
        "weight_decay":a.weight_decay,
        "probe_batch":a.probe_batch,
        "eval_probe_count":a.eval_probe_count,
        "surface_chunk":a.surface_chunk,
        "carrier_ids":list(carrier_ids),
        "carrier_count":len(carrier_ids),
        "diagnostic_carrier_ids":list(diagnostic_carrier_ids),
        "diagnostic_carrier_count":len(diagnostic_carrier_ids),
        "min_training_carriers":int(a.min_training_carriers),
        "unique_training_carrier_lineages":len(unique_training_lineages),
        "carrier_mechanical_aggregation":"0.5_MEAN_PLUS_0.5_WORST",
        "semantic_support_hard_mask":True,
        "objective_weights":{
            "mechanical":a.mechanical_weight,
            "teacher_valid":a.teacher_weight,
            "base_trust":a.trust_weight,
        },
        "budgets":{
            "base_p95_l1":a.max_base_p95_l1,
            "teacher_valid_p95_l1":a.max_teacher_valid_p95_l1,
        },
        "history":resume_history,
        "claim_boundary":[
            "Knight integration fit only; no genericity claim.",
            "Historical Arachne/base field is frozen.",
            "Only MechanicalResidualAdapterV1 parameters are trainable.",
            "Residual influence redistribution is hard-limited by the sealed per-row semantic support mask.",
            "Mechanical fit is evaluated across a frozen carrier ensemble when present.",
            "Carrier objective is 0.5 mean + 0.5 worst; one topology cannot hide another.",
            "Compiler exact G3B/G3/motion courts remain final authority.",
        ],
    }

    best=None
    if resume_best is not None:
        rb=resume_best
        best=(
            tuple(rb["key"]),
            int(rb["step"]),
            {k:v.detach().cpu() for k,v in rb["state_dict"].items()},
            dict(rb["row"]),
        )
    full_order=np.arange(P,dtype=np.int64)
    rng=np.random.default_rng(a.seed)
    # Pre-register one deterministic permutation stream; no metric-conditioned resampling.
    schedules=[]
    while sum(len(x) for x in schedules)<a.steps*a.probe_batch:
        schedules.append(rng.permutation(full_order))
    stream=np.concatenate(schedules)
    stream=stream[:a.steps*a.probe_batch].reshape(a.steps,a.probe_batch)
    eval_rng=np.random.default_rng(a.seed+1)
    eval_stream=eval_rng.permutation(full_order)[:a.eval_probe_count]
    receipt["eval_probe_indices"]=eval_stream.tolist()
    receipt["eval_probe_ids"]=[probe_ids[i] for i in eval_stream]

    for step in range(start_step,a.steps):
        idx=torch.as_tensor(stream[step],device=dev,dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=(dev.type=="cuda")):
            pred,delta=model(
                base_weights=base,
                surface_geometry7=geometry7,
                pair_geometry=pair,
                surface_mask=sm,
                joint_mask=jm,
                row_joint_mask=row_jm,
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

        if (
            step==start_step
            or (step+1)%a.log_every==0
            or (step+1)%a.checkpoint_every==0
            or step+1==a.steps
        ):
            # Metrics and checkpoint must describe the exact same post-update
            # model state. Re-evaluate after opt.step(); do not reuse the
            # pre-update training prediction.
            with torch.no_grad():
                eval_idx=torch.as_tensor(
                    eval_stream,device=dev,dtype=torch.long
                )
                with torch.autocast(
                    device_type=dev.type,dtype=torch.bfloat16,
                    enabled=(dev.type=="cuda")
                ):
                    pred_eval,_=model(
                        base_weights=base,
                        surface_geometry7=geometry7,
                        pair_geometry=pair,
                        surface_mask=sm,
                        joint_mask=jm,
                        row_joint_mask=row_jm,
                        surface_chunk_size=a.surface_chunk,
                    )
                eval_rows=[]
                for carrier in carriers:
                    eval_rows.append(mechanical_consequence_loss_v1(
                        pred_eval.float(),
                        candidate_support_indices=carrier["support_idx"],
                        candidate_support_coefficients=carrier["support_coeff"],
                        candidate_rest_vertices=carrier["rest"],
                        candidate_faces=carrier["faces"],
                        probe_transforms=probes[:,eval_idx],
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
                eval_mech,eval_mean,eval_worst=_aggregate_carrier_mechanical(eval_rows)
                eval_trust=(pred_eval.float()-base).abs().sum(-1).mean()
                eval_teacher_row=(pred_eval.float()-teacher).abs().sum(-1)
                eval_supervised=(
                    eval_teacher_row[vm].mean()
                    if bool(vm.any()) else eval_teacher_row.new_zeros(())
                )
                eval_total=(
                    a.mechanical_weight*eval_mech
                    + a.teacher_weight*eval_supervised
                    + a.trust_weight*eval_trust
                )
                metrics=correction_metrics(pred_eval.float(),base,teacher,valid)
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
                    for carrier,row_loss in zip(carriers,eval_rows)
                }
                worst_carrier=max(
                    per_carrier,
                    key=lambda cid:(per_carrier[cid]["mechanical"],cid),
                )
                row={
                    "step":step+1,
                    "train_probe_indices":stream[step].tolist(),
                    "train_probe_ids":[probe_ids[i] for i in stream[step]],
                    "eval_probe_indices":eval_stream.tolist(),
                    "eval_probe_ids":[probe_ids[i] for i in eval_stream],
                    "total":float(eval_total.detach().cpu()),
                    "mechanical":float(eval_mech.detach().cpu()),
                    "mechanical_mean":float(eval_mean.detach().cpu()),
                    "mechanical_worst":float(eval_worst.detach().cpu()),
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
                    best=(
                        key,step+1,
                        {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        row,
                    )

                receipt["last_completed_step"]=step+1
                (a.out_dir/"FIT_RECEIPT_PARTIAL.json").write_text(
                    json.dumps(receipt,indent=2,sort_keys=True)+"\n"
                )
                if (step+1)%a.checkpoint_every==0 or step+1==a.steps:
                    best_payload=None
                    if best is not None:
                        best_payload={
                            "key":list(best[0]),
                            "step":int(best[1]),
                            "state_dict":best[2],
                            "row":best[3],
                        }
                    torch.save({
                        "schema":"RealSaS.ArachneMechanicalResidualAdapterResumeCheckpoint.v1",
                        "state_dict":{
                            k:v.detach().cpu() for k,v in model.state_dict().items()
                        },
                        "optimizer_state_dict":opt.state_dict(),
                        "parameter_count":model.parameter_count,
                        "bundle_sha256":receipt["bundle_sha256"],
                        "next_step":step+1,
                        "best":best_payload,
                        "history":receipt["history"],
                        "seed":a.seed,
                    },a.out_dir/"ARACHNE_MECHANICAL_RESIDUAL_ADAPTER_RESUME.pt")

    if best is None:
        raise RuntimeError("no adapter checkpoint candidate")
    _,best_step,best_state,best_row=best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_pred,_=model(
            base_weights=base,
            surface_geometry7=geometry7,
            pair_geometry=pair,
            surface_mask=sm,
            joint_mask=jm,
            row_joint_mask=row_jm,
            surface_chunk_size=a.surface_chunk,
        )
    best_weights=best_pred[0].float().cpu().numpy().astype(np.float32)
    if not np.allclose(best_weights.sum(1),1.0,atol=1e-6,rtol=0.0):
        raise RuntimeError("best adapted skin simplex drift")
    allowed=row_jm[0].detach().cpu().numpy().astype(bool)
    if np.any(best_weights[~allowed]!=0.0):
        raise RuntimeError("best adapted skin minted forbidden support")
    np.savez_compressed(
        a.out_dir/"ADAPTED_SURFACE_WEIGHTS_V1.npz",
        weights=best_weights,
        base_weights=base[0].detach().cpu().numpy().astype(np.float32),
        row_joint_mask=allowed.astype(np.uint8),
        surface_ids=np.asarray(
            bundle["surface_ids"] if bundle["surface_ids"] is not None
            else tuple(str(i) for i in range(best_weights.shape[0])),
            dtype="U128",
        ),
        joint_ids=np.asarray(
            bundle["joint_ids"] if bundle["joint_ids"] is not None
            else tuple(str(i) for i in range(best_weights.shape[1])),
            dtype="U128",
        ),
        bundle_sha256=np.asarray([receipt["bundle_sha256"]],dtype="U64"),
        best_step=np.asarray([best_step],dtype=np.int64),
    )
    torch.save({
        "schema":"RealSaS.ArachneMechanicalResidualAdapterCheckpoint.v1",
        "state_dict":best_state,
        "parameter_count":model.parameter_count,
        "bundle_sha256":receipt["bundle_sha256"],
        "best_step":best_step,
        "best_metrics":best_row,
        "config":model.config.__dict__,
    },a.out_dir/"ARACHNE_MECHANICAL_RESIDUAL_ADAPTER_V1.pt")
    if not (
        best_row["base_p95_l1"]<=a.max_base_p95_l1
        and best_row["teacher_valid_p95_l1"]<=a.max_teacher_valid_p95_l1
    ):
        receipt["status"]="FAIL_FIT_NO_SEMANTICALLY_ADMISSIBLE_CHECKPOINT"
    else:
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
    if receipt["status"]!="PASS_FIT_COMPLETED__AWAIT_COMPILER_REQUALIFICATION":
        raise RuntimeError(receipt["status"])


if __name__=="__main__":
    main()
