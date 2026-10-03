from __future__ import annotations

"""Audit Geppetto articulation capacity implied by target-construction policy.

Training/evaluation only. Builds K0/K1/K3 anonymous target policies from the
exact frozen Knight teacher source. K2 is deliberately not synthesized here:
it requires an independent marginal-deformation necessity court.

No model is trained and no product authority is minted.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.geppetto_reference_strength_fullstack_v1.mechanical_capacity_target_v2 import (
    POLICY_K0,
    POLICY_K1,
    POLICY_K3,
    build_mechanical_capacity_target_v2,
)
from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    world_heads_from_rest_world_source_v1,
)


def _load_source(path: Path):
    with np.load(path,allow_pickle=False) as z:
        required={"parents","deform_mask","skin","rest_world_source"}
        missing=required-set(z.files)
        if missing:
            raise RuntimeError(
                "GEPPETTO_CAPACITY_SOURCE_ARRAYS_MISSING:"
                + ",".join(sorted(missing))
            )
        return (
            np.asarray(z["parents"],np.int64),
            np.asarray(z["deform_mask"],np.uint8).astype(bool),
            np.asarray(z["skin"],np.float64),
            np.asarray(z["rest_world_source"],np.float64),
        )


def _depths(parents):
    p=np.asarray(parents,np.int64)
    out=np.zeros(len(p),np.int64)
    for i in range(len(p)):
        seen=set()
        n=i
        d=0
        while int(p[n])>=0:
            if n in seen:
                raise RuntimeError("GEPPETTO_CAPACITY_TARGET_CYCLE")
            seen.add(n)
            n=int(p[n]); d+=1
        out[i]=d
    return out


def _target_report(target):
    depth=_depths(target.parent_indices)
    mass=np.asarray(target.skin_mass,np.float64)
    deform=np.asarray(target.source_deform_mask,bool)
    pos=np.asarray(target.positions_world,np.float64)
    parent=np.asarray(target.parent_indices,np.int64)
    seg=[]
    for i,p in enumerate(parent.tolist()):
        if p>=0:
            seg.append(float(np.linalg.norm(pos[i]-pos[p])))
    return {
        "policy":target.policy,
        "count":int(target.count),
        "root_count":int(np.count_nonzero(target.root_mask)),
        "source_deform_count":int(np.count_nonzero(deform)),
        "zero_direct_skin_mass_count":int(np.count_nonzero(mass<=1e-8)),
        "positive_direct_skin_mass_count":int(np.count_nonzero(mass>1e-8)),
        "skin_mass_sum":float(mass.sum()),
        "max_depth":int(depth.max(initial=0)),
        "mean_depth":float(depth.mean()) if len(depth) else 0.0,
        "segment_length_mean":float(np.mean(seg)) if seg else 0.0,
        "segment_length_p95":float(np.quantile(seg,0.95)) if seg else 0.0,
        "policy_receipt_hash":target.policy_receipt_hash,
        "source_indices_provenance_only":
            [int(x) for x in target.source_indices_provenance_only.tolist()],
    }


def main(args):
    args.out_dir.mkdir(parents=True,exist_ok=True)
    parents,deform,skin,rest=_load_source(args.teacher_source_npz)
    heads=world_heads_from_rest_world_source_v1(rest)
    source_mass=np.asarray(skin,np.float64).sum(axis=0)
    policies=(POLICY_K0,POLICY_K1,POLICY_K3)
    targets={}
    reports={}
    for policy in policies:
        t=build_mechanical_capacity_target_v2(
            parents=parents,
            deform_mask=deform,
            skin=skin,
            bone_heads_world=heads,
            policy=policy,
        )
        targets[policy]=t
        reports[policy]=_target_report(t)

    sets={
        p:set(map(int,targets[p].source_indices_provenance_only.tolist()))
        for p in policies
    }
    k0,k1,k3=POLICY_K0,POLICY_K1,POLICY_K3
    added_k1=sorted(sets[k1]-sets[k0])
    added_k3=sorted(sets[k3]-sets[k1])

    def classify(indices):
        return {
            "count":len(indices),
            "deform_count":int(sum(bool(deform[i]) for i in indices)),
            "nondeform_count":int(sum(not bool(deform[i]) for i in indices)),
            "positive_skin_mass_count":int(
                sum(float(source_mass[i])>1e-8 for i in indices)
            ),
            "zero_skin_mass_count":int(
                sum(float(source_mass[i])<=1e-8 for i in indices)
            ),
            "source_indices_provenance_only":[int(i) for i in indices],
        }

    report={
        "schema":"RealSaS.GeppettoArticulationTargetCapacityCourt.v1",
        "status":"MEASURED__K2_PENDING_MARGINAL_DEFORMATION_COURT",
        "source_joint_count":int(len(parents)),
        "source_deform_count":int(np.count_nonzero(deform)),
        "source_positive_skin_mass_count":int(
            np.count_nonzero(source_mass>1e-8)
        ),
        "policies":reports,
        "k0_to_k1_added":classify(added_k1),
        "k1_to_k3_added":classify(added_k3),
        "count_deltas":{
            "K1_minus_K0":int(len(sets[k1])-len(sets[k0])),
            "K3_minus_K1":int(len(sets[k3])-len(sets[k1])),
            "K3_minus_K0":int(len(sets[k3])-len(sets[k0])),
        },
        "k2_state":{
            "status":"NOT_CONSTRUCTED",
            "reason":"REQUIRES_INDEPENDENT_MARGINAL_ARTICULATION_GAIN_EVIDENCE",
        },
        "training_used":False,
        "product_authority_minted":False,
        "claim_boundary":[
            "Counts measure target-policy capacity only, not final rig quality.",
            "K1 adds all source deform controls plus required bridges without targeting a requested count.",
            "K3 is a diagnostic ceiling and is not a shipping recommendation.",
            "K2 remains blocked until control-removal deformation consequence is measured.",
        ],
    }
    (args.out_dir/"REPORT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n"
    )
    print(
        "GEPPETTO_ARTICULATION_TARGET_CAPACITY_RESULT="
        +json.dumps(report,sort_keys=True),
        flush=True,
    )


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--teacher-source-npz",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args()
    try:
        main(a)
    except Exception as exc:
        a.out_dir.mkdir(parents=True,exist_ok=True)
        (a.out_dir/"ERROR.json").write_text(
            json.dumps(
                {"type":type(exc).__name__,"message":str(exc)},
                indent=2,sort_keys=True,
            )+"\n"
        )
        raise
