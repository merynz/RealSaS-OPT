from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MOTION_PATH=ROOT/"compiler/realsas_compiler_core/motion_dynamic_proof_v2.py"
RUNTIME_PATH=ROOT/"compiler/realsas_compiler_services/orchestrator/adapters/runtime_v2.py"
PACKAGE_PATH=ROOT/"compiler/realsas_compiler_core/runtime_package_v2.py"
MOTION=MOTION_PATH.read_text()
RUNTIME=RUNTIME_PATH.read_text()
PACKAGE=PACKAGE_PATH.read_text()


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def function_source(source: str, name: str) -> str:
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name:
            return ast.get_source_segment(source,node) or ""
    return ""


def assignment_source(function_text: str, name: str, *, last: bool=False) -> str:
    tree=ast.parse(function_text)
    rows=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign):
            if any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
                rows.append(ast.get_source_segment(function_text,node) or "")
    if not rows:
        return ""
    return rows[-1] if last else rows[0]


motion_compact=compact(MOTION)
runtime_build=function_source(RUNTIME,"build_runtime_projection_stage")
runtime_dvi=function_source(RUNTIME,"prove_dynamic_visual_integrity_stage")
package_compact=compact(PACKAGE)
dvi_pass_assignment=assignment_source(runtime_dvi,"passed",last=True)

self_intersection_hard=(
    "MOTION_V2_DYNAMIC_NEW_SELF_INTERSECTION" in MOTION
    and "MOTION_V2_DYNAMIC_REST_INTERSECTION_WORSENED" in MOTION
    and "unexpected_intersection_pairs(" in MOTION
)
contact_narrow=(
    '"supported_contact_modes":["PLANT_2D"]' in motion_compact
    and '"full_3d_ground_contact_quality_claimed":False' in motion_compact
)
sealed_playback=(
    "playback_sampling_contract=SEALED_FRAME_INDEX_ONLY" in PACKAGE
    and "host_interpolation_authorized=0" in PACKAGE
)
exact_frame_transfer=(
    "times = np.asarray([frame.time_seconds for frame in clip.frames]" in runtime_build
    and "for vertex_id, xyz in frame.posed_vertex_xyz" in runtime_build
    and '"posed_xyz_from_stage41_exact": True' in runtime_build
    and "dynamic_motion_binding_hash=dynamic.dynamic_motion_hash" in runtime_build
)
native_all_frames=(
    "for clip in projection.clips:" in runtime_dvi
    and "for frame_index in range(clip.frame_count):" in runtime_dvi
    and "views = tuple(projection.views)" in runtime_dvi
    and "_run_native_many(" in runtime_dvi
    and "parity_mismatch" in runtime_dvi
)
unmeasurable_gate=(
    "unmeasurable_visible_face_count" in runtime_dvi
    and "dynamic_visibility_load_gate" in runtime_dvi
    and "conditioning_passed" in runtime_dvi
)
alpha_measured=(
    "alpha_transparent" in runtime_dvi
    and "final_alpha_hole_pixel_count=alpha_transparent" in compact(runtime_dvi)
)
alpha_hard=(
    "alpha_transparent" in dvi_pass_assignment
    or "transparent_fraction" in dvi_pass_assignment
    or "final_alpha_hole" in dvi_pass_assignment
)

rows=[
  {
    "id":"DYNAMIC_SELF_INTERSECTION",
    "status":"CLOSED_CURRENT_V2" if self_intersection_hard else "OPEN",
    "severity":"PASS" if self_intersection_hard else "P0",
    "evidence":"new unexpected intersections and worsening rest-existing intersection persistence are hard QualificationErrors",
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
      "exact_stage41_times_and_posed_xyz_transfer":exact_frame_transfer,
      "native_every_sealed_frame_every_view_parity":native_all_frames,
    },
  },
  {
    "id":"UNMEASURABLE_VISIBLE_FACE_LOAD",
    "status":"CLOSED_CURRENT_V2" if unmeasurable_gate else "OPEN",
    "severity":"PASS" if unmeasurable_gate else "P1",
    "evidence":"unmeasurable consequential faces enter the dynamic visibility/conditioning gate",
  },
  {
    "id":"DYNAMIC_VISIBLE_ALPHA_HOLES",
    "status":"OPEN__DIAGNOSTIC_NOT_HARD_GATE" if alpha_measured and not alpha_hard else "CLOSED_OR_NOT_PRESENT",
    "severity":"P1" if alpha_measured and not alpha_hard else "PASS",
    "evidence":{
      "alpha_measured":alpha_measured,
      "alpha_in_final_pass_predicate":alpha_hard,
      "final_pass_assignment":dvi_pass_assignment,
    },
    "design_before_code":"Define intentional transparent art versus unintended geometry-visible alpha holes, including coherent and scattered/speckled classes, then calibrate product hard gates."
  },
]

payload={
  "schema":"RealSaS.V2MotionRuntimeProofAudit.v2",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "rows":rows,
  "open_count":sum(r["severity"] in {"P0","P1","P2"} for r in rows),
  "repair_applied":False,
}
out=ROOT/"canonical"/"V2_MOTION_RUNTIME_PROOF_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
