from __future__ import annotations
import json, subprocess
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _triangle_metrics

ROOT=Path(__file__).resolve().parents[1]
rest=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=np.float64)
posed=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,-1.,0.]],dtype=np.float64)
metric=_triangle_metrics(rest,posed)
rest_n=np.cross(rest[1]-rest[0],rest[2]-rest[0])
posed_n=np.cross(posed[1]-posed[0],posed[2]-posed[0])
orientation_dot=float(np.dot(rest_n,posed_n))
payload={
  "schema":"RealSaS.G3OrientationInversionAdversary.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "construction":"unit right triangle reflected across X axis in its rest tangent plane",
  "current_g3_triangle_metrics":{
    "area_ratio":float(metric[0]),
    "condition_number":float(metric[1]),
    "min_edge_ratio":float(metric[2]),
    "max_edge_ratio":float(metric[3]),
    "sigma_min":float(metric[4])
  },
  "orientation":{
    "rest_normal":rest_n.tolist(),
    "posed_normal":posed_n.tolist(),
    "normal_dot":orientation_dot,
    "inverted":orientation_dot<0.0
  },
  "finding":{
    "id":"G3_SVD_METRICS_DO_NOT_DETECT_TRIANGLE_INVERSION",
    "confirmed":bool(orientation_dot<0.0 and abs(metric[0]-1.0)<1e-12 and abs(metric[1]-1.0)<1e-12),
    "class":"IMPLEMENTATION_DEVIATION_FROM_FROZEN_G3_NO_INVERSION_CONTRACT",
    "claim_boundary":"Tiny deterministic reflection adversary. Stage45 may separately catch projected flips, but G3 itself does not implement its frozen no-inversion/foldover invariant."
  }
}
out=ROOT/"canonical"/"G3_ORIENTATION_INVERSION_ADVERSARY_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
if not payload["finding"]["confirmed"]:
    raise SystemExit("G3_INVERSION_ADVERSARY_NOT_CONFIRMED")
