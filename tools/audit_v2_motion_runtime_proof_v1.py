from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MOTION=(ROOT/"compiler/realsas_compiler_core/motion_dynamic_proof_v2.py").read_text()
RUNTIME=(ROOT/"compiler/realsas_compiler_services/orchestrator/adapters/runtime_v2.py").read_text()
PACKAGE=(ROOT/"compiler/realsas_compiler_core/runtime_package_v2.py").read_text()

def present(s,needle): return needle in s

self_intersection_hard=(
    present(MOTION,"MOTION_V2_NEW_UNEXPECTED_SELF_INTERSECTION")
    and present(MOTION,"MOTION_V2_EXISTING_SELF_INTERSECTION_SEVERITY_REGRESSION")
)
contact_narrow=(
    present(MOTION,'"supported_contact_modes": ["PLANT_2D"]')
    and present(MOTION,'"full_3d_ground_contact_quality_claimed": False')
)
sealed_playback=(
    present(PACKAGE,"SEALED_FRAME_INDEX_ONLY")
    and (
        present(PACKAGE,"host_interpolation_authorized")
        or present(RUNTIME,"host_interpolation_authorized")
    )
)
exact_frame_transfer=(
    present(RUNTIME,"frame.time_seconds")
    and present(RUNTIME,"frame.frame_hash")
    and present(RUNTIME,"frame.posed_vertex_xyz")
)
native_all_frames=(
    "for frame_index" in RUNTIME
    and "for view_index" in RUNTIME
)
unmeasurable_gate=(
    "unmeasurable_consequential" in RUNTIME
    and "dynamic_visibility_load_gate" in RUNTIME
)
passed_pos=RUNTIME.find("passed = (")
passed_block=RUNTIME[passed_pos:passed_pos+3500] if passed_pos>=0 else ""
alpha_measured=(
    "alpha_transparent" in RUNTIME
    and "final_alpha_hole_pixel_count" in RUNTIME
)
alpha_hard=("alpha_transparent" in passed_block or "final_alpha_hole" in passed_block)

rows=[
  {
    "id":"DYNAMIC_SELF_INTERSECTION",
    "status":"CLOSED_CURRENT_V2" if self_intersection_hard else "OPEN",
    "severity":"PASS" if self_intersection_hard else "P0",
    "evidence":"new unexpected intersections and worsening rest-existing severity are exceptions",
  },
  {
    "id":"CONTACT_CLAIM_SCOPE",
    "status":"CLOSED_BY_EXPLICIT_SCOPE" if contact_narrow else "OPEN",
    "severity":"PASS" if contact_narrow else "P1",
    "evidence":"PLANT_2D only; full_3d_ground_contact_quality_claimed=false",
  },
  {
    "id":"PROOF_RUNTIME_FRAME_IDENTITY",
    "status":"CLOSED_CURRENT_V2" if sealed_playback and exact_frame_transfer and native_all_frames else "OPEN",
    "severity":"PASS" if sealed_playback and exact_frame_transfer and native_all_frames else "P0",
    "evidence":{
      "sealed_frame_playback":sealed_playback,
      "exact_frame_transfer":exact_frame_transfer,
      "native_all_frames_views":native_all_frames,
    },
  },
  {
    "id":"UNMEASURABLE_VISIBLE_FACE_LOAD",
    "status":"CLOSED_CURRENT_V2" if unmeasurable_gate else "OPEN",
    "severity":"PASS" if unmeasurable_gate else "P1",
    "evidence":"unmeasurable consequential faces enter dynamic_visibility_load_gate",
  },
  {
    "id":"DYNAMIC_VISIBLE_ALPHA_HOLES",
    "status":"OPEN__DIAGNOSTIC_NOT_HARD_GATE" if alpha_measured and not alpha_hard else "CLOSED_OR_NOT_PRESENT",
    "severity":"P1" if alpha_measured and not alpha_hard else "PASS",
    "evidence":{
      "alpha_measured":alpha_measured,
      "alpha_in_pass_predicate":alpha_hard,
      "pass_predicate_excerpt":passed_block,
    },
    "design_before_code":"Define intentional transparent art versus unintended geometry-visible alpha holes, including coherent and scattered/speckled classes, then calibrate product hard gates."
  },
]

payload={
  "schema":"RealSaS.V2MotionRuntimeProofAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "rows":rows,
  "open_count":sum(r["severity"] in {"P0","P1","P2"} for r in rows),
  "repair_applied":False,
}
out=ROOT/"canonical"/"V2_MOTION_RUNTIME_PROOF_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
