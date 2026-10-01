from __future__ import annotations

import json, math, subprocess
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    robust_zero_surface_normals_v1,
)

ROOT=Path(__file__).resolve().parents[1]


def angle_deg(a,b):
    aa=np.asarray(a,dtype=np.float64)
    bb=np.asarray(b,dtype=np.float64)
    aa=aa/np.maximum(np.linalg.norm(aa,axis=1,keepdims=True),1e-12)
    bb=bb/np.maximum(np.linalg.norm(bb,axis=1,keepdims=True),1e-12)
    dot=np.clip(np.sum(aa*bb,axis=1),-1.0,1.0)
    return np.degrees(np.arccos(dot))


def sheet_grid(n:int=33, spacing:float=0.04):
    c=(np.arange(n,dtype=np.float64)-(n-1)/2.0)*spacing
    x,y=np.meshgrid(c,c,indexing="xy")
    return x.reshape(-1),y.reshape(-1)


def main():
    x,y=sheet_grid()
    # Sheet A: horizontal.
    a=np.column_stack((x,y,np.zeros_like(x)))
    na=np.tile(np.asarray((0.0,0.0,1.0),dtype=np.float64),(len(a),1))

    # Sheet B: distinct connected sheet passing very near A, tilted 35 degrees.
    theta=math.radians(35.0)
    slope=math.tan(theta)
    b=np.column_stack((x,y,0.012+slope*x))
    nb0=np.asarray((-slope,0.0,1.0),dtype=np.float64)
    nb0/=np.linalg.norm(nb0)
    nb=np.tile(nb0,(len(b),1))

    points=np.concatenate((a,b),axis=0)
    truth=np.concatenate((na,nb),axis=0)

    current=robust_zero_surface_normals_v1(points,truth,k=64)
    # Oracle uses the exact same operator but forbids cross-component neighbours.
    oracle=np.concatenate((
        robust_zero_surface_normals_v1(a,na,k=64),
        robust_zero_surface_normals_v1(b,nb,k=64),
    ),axis=0)

    current_err=angle_deg(current,truth)
    oracle_err=angle_deg(oracle,truth)
    n=len(a)
    # Near-contact band: where sheet separation is < one in-sheet spacing.
    separation=np.abs(0.012+slope*x)
    band=np.concatenate((separation<0.04,separation<0.04))

    payload={
      "schema":"RealSaS.GSANormalTopologyAdversary.v1",
      "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
      "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
      "construction":{
        "sheet_count":2,
        "points_per_sheet":int(n),
        "spacing":0.04,
        "minimum_offset":0.012,
        "relative_tilt_deg":35.0,
        "k":64,
        "components_are_distinct":True,
      },
      "current_global_knn":{
        "mean_angle_error_deg":float(np.mean(current_err)),
        "p95_angle_error_deg":float(np.percentile(current_err,95)),
        "max_angle_error_deg":float(np.max(current_err)),
        "near_contact_mean_angle_error_deg":float(np.mean(current_err[band])),
        "near_contact_p95_angle_error_deg":float(np.percentile(current_err[band],95)),
      },
      "component_aware_same_operator_oracle":{
        "mean_angle_error_deg":float(np.mean(oracle_err)),
        "p95_angle_error_deg":float(np.percentile(oracle_err,95)),
        "max_angle_error_deg":float(np.max(oracle_err)),
        "near_contact_mean_angle_error_deg":float(np.mean(oracle_err[band])),
        "near_contact_p95_angle_error_deg":float(np.percentile(oracle_err[band],95)),
      },
      "finding":{
        "id":"GSA_NORMAL_GLOBAL_KNN_CROSS_SHEET_CONTAMINATION",
        "confirmed":bool(
          np.percentile(current_err[band],95) >
          np.percentile(oracle_err[band],95) + 5.0
        ),
        "claim_boundary":"Synthetic falsification of topology-blind Euclidean normal neighbourhoods; does not choose the production repair operator.",
      },
    }
    out=ROOT/"canonical"/"GSA_NORMAL_TOPOLOGY_ADVERSARY_V1_20260928.json"
    out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
