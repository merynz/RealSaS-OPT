from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load_json(rel):
    return json.loads((ROOT/rel).read_text(encoding="utf-8"))

def text(rel):
    return (ROOT/rel).read_text(encoding="utf-8")

rows=[]

def add(*,id,classification,status,owner,knobs,evidence,quality_role,performance_role,decision):
    rows.append({
        "id":id,
        "classification":classification,
        "status":status,
        "owner":owner,
        "knobs":knobs,
        "evidence":evidence,
        "quality_role":quality_role,
        "performance_role":performance_role,
        "design_before_code":decision,
    })

perf=load_json("canonical/PRODUCT_COMPILE_PERFORMANCE_CONTRACT_V1_20260927.json")
gsa=load_json("canonical/STAGE14_SUBSTRATE_ADEQUACY_POLICY_V2_20260920.json")
mesh=load_json("canonical/QUALIFIED_MESH_PRODUCT_POLICY_V2_20260919.json")
dvi=load_json("canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json")
run_contract=load_json("canonical/RUN_MANIFEST_PRODUCT_AUTHORITY_CONTRACT_V1.json")
visual=text("compiler/realsas_compiler_core/visual_mesh_arap_v1.py")
v2arch=text("compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture.py")
skincompat=text("compiler/realsas_compiler_core/mesh/skin_topology_compatibility_v1.py")
cdt=text("compiler/realsas_compiler_core/canonical_cdt_adapter_v1.py")
motion=text("compiler/realsas_compiler_core/motion_compile_v2.py")
motionproof=text("compiler/realsas_compiler_core/motion_dynamic_proof_v2.py")

add(
 id="CAL_GSA_SUBSTRATE_POLICY",
 classification="CALIBRATED_SUBJECT_FREE",
 status="KEEP",
 owner="STAGE14_GSA",
 knobs={
   "normal_k":gsa["gsa"]["normal_k"],
   "visibility_depth_tolerance_norm":gsa["gsa"]["visibility_depth_tolerance_norm"],
   "max_candidate_nodes":gsa["gsa"]["adequacy_policy"]["max_candidate_nodes"],
   "refinement_rounds":gsa["gsa"]["adequacy_policy"]["refinement_rounds"],
 },
 evidence=[gsa["calibration_authority"],"canonical/STAGE14_SUBSTRATE_ADEQUACY_POLICY_V2_20260920.json"],
 quality_role="Mechanical-carrier adequacy and component preservation.",
 performance_role="Bounded search budget; selection is minimum passing node count.",
 decision="Preserve unless a new subject-free calibration supersedes it. Do not change merely for speed."
)
add(
 id="CAL_MESH_G3_G5_POLICY",
 classification="CALIBRATED_SUBJECT_FREE",
 status="KEEP",
 owner="MECHANICAL_MESH_QUALIFICATION",
 knobs={
   "minimum_rest_triangle_angle_deg":mesh["g3"]["minimum_rest_triangle_angle_deg"],
   "maximum_rest_aspect":mesh["g3"]["maximum_rest_aspect_longest_edge_over_min_altitude"],
   "minimum_dynamic_area_ratio":mesh["g3"]["minimum_dynamic_area_ratio"],
   "maximum_dynamic_area_ratio":mesh["g3"]["maximum_dynamic_area_ratio"],
   "maximum_dynamic_condition_number":mesh["g3"]["maximum_dynamic_condition_number"],
   "g5_mesh":mesh["g5"]["MESH"],
 },
 evidence=[mesh["calibration_authority"],mesh["g3"]["calibration_result"]],
 quality_role="Frozen mechanical geometry admission.",
 performance_role="Not a performance knob; quality authority.",
 decision="Never relax for compile time. Optimize algorithms around the thresholds."
)
add(
 id="CAL_DYNAMIC_APPEARANCE_CONDITIONING",
 classification="CALIBRATED_SUBJECT_FREE",
 status="KEEP",
 owner="STAGE45_DVI",
 knobs=dvi["selected_policy"],
 evidence=["canonical/DYNAMIC_APPEARANCE_CONDITIONING_CALIBRATION_V1_20260921.json"],
 quality_role="Intrinsic dynamic visual conditioning and measurable visibility.",
 performance_role="Proof cost may be optimized, thresholds may not be weakened.",
 decision="Preserve numerical limits; fuse/vectorize/accelerate evaluation instead."
)

cdt_budget=run_contract["sections"]["mesh"]
add(
 id="REVIEW_CANONICAL_CDT_ITERATION_BUDGET",
 classification="CONTRACT_BOUND__PARETO_CALIBRATION_NOT_FOUND",
 status="DESIGN_REQUIRED",
 owner="CANONICAL_CDT",
 knobs={
   "max_constraint_recovery_iterations":cdt_budget["max_constraint_recovery_iterations"],
   "max_quality_iterations":cdt_budget["max_quality_iterations"],
 },
 evidence=[
   "canonical/RUN_MANIFEST_PRODUCT_AUTHORITY_CONTRACT_V1.json",
   "compiler/realsas_compiler_core/canonical_cdt_adapter_v1.py",
 ],
 quality_role="Iteration ceilings determine whether the constrained refiner reaches the frozen mesh-quality target before fail-close.",
 performance_role="Potentially large compile-time multiplier per parent patch.",
 decision="Benchmark convergence/latency distribution on subject-free geometry controls. Select the minimum ceilings that preserve the frozen G3 outcome distribution; do not infer optimum from 96."
)

if "target_min_angle_deg=0.0" not in visual or "max_quality_iterations=0" not in visual:
    raise RuntimeError("visual CDT shortcut signature drift")
if "target_edge_px=16" not in v2arch:
    raise RuntimeError("Stage18 visual target-edge signature drift")
add(
 id="OPEN_VISUAL_MESH_SPEED_SHORTCUTS",
 classification="UNCALIBRATED_SPEED_SHORTCUT_AFFECTING_QUALITY",
 status="OPEN",
 owner="SOURCE_OWNED_VISUAL_MESH",
 knobs={
   "stage18_target_edge_px":16,
   "target_min_angle_deg":0.0,
   "max_quality_iterations":0,
   "max_constraint_recovery_iterations":192,
 },
 evidence=[
   "compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture.py",
   "compiler/realsas_compiler_core/visual_mesh_arap_v1.py",
 ],
 quality_role="Controls source-owned visual primitive density/shape before deformation; exact silhouette alone does not bound deformation conditioning.",
 performance_role="Code comment explicitly disables quality insertion to keep a fast iteration loop.",
 decision="Calibrate an adaptive tessellation/refinement criterion against source curvature, deformation-field error, dynamic flip/stretch rate and compile latency. Do not choose a replacement count yet."
)

for token in (
 "DEFAULT_RISK_L1_MIN = 0.5",
 "DEFAULT_MAX_EDGE_RATIO = 4.0",
 "DEFAULT_STRESS_ANGLE_DEG = 120.0",
 "DEFAULT_MAX_REPAIR_ITERATIONS = 4",
):
    if token not in skincompat:
        raise RuntimeError("skin topology knob signature drift:"+token)
add(
 id="OPEN_SKIN_TOPOLOGY_COMPATIBILITY_THRESHOLDS",
 classification="UNCALIBRATED_HARD_GATE_AND_REPAIR_BUDGET",
 status="OPEN",
 owner="STAGE35_G3B",
 knobs={
   "risk_l1_min":0.5,
   "max_edge_ratio":4.0,
   "stress_angle_deg":120.0,
   "max_repair_iterations":4,
 },
 evidence=["compiler/realsas_compiler_core/mesh/skin_topology_compatibility_v1.py"],
 quality_role="Determines which topology×skin discontinuities are attacked and whether a candidate passes.",
 performance_role="Probe angle/repair budget directly affect stress-test and recompile cost.",
 decision="Build subject-free seam/non-seam controls with known artist topology and skin discontinuities. Select thresholds by separation and repair efficacy; repair budget should follow convergence evidence, not an arbitrary count."
)

if "alternative_cost - primary_cost < 0.01" not in motion:
    raise RuntimeError("retarget ambiguity margin signature drift")
add(
 id="REVIEW_MOTION_RETARGET_AMBIGUITY_MARGIN",
 classification="UNEXPLAINED_FAIL_CLOSED_MARGIN",
 status="DESIGN_REVIEW",
 owner="STAGE40_RETARGET",
 knobs={"global_second_best_cost_margin":0.01},
 evidence=["compiler/realsas_compiler_core/motion_compile_v2.py"],
 quality_role="Fails ambiguous topology-constrained source→target skeleton mappings.",
 performance_role="Requires a second exact MILP solve; still bounded for preset-sized skeletons.",
 decision="Calibrate cost-margin semantics on subject-free skeleton perturbations / known ambiguous controls. Preserve fail-closed behavior while replacing unexplained scale-sensitive constants if needed."
)

if "_UNIFORM_SAMPLE_COUNT=17" not in motionproof:
    raise RuntimeError("motion proof uniform count signature drift")
add(
 id="REVIEW_MOTION_EXTRA_UNIFORM_PROOF_DENSITY",
 classification="PROOF_DENSITY_BUDGET__NOT_OUTPUT_FPS",
 status="DESIGN_REVIEW",
 owner="STAGE41_DYNAMIC_MOTION_PROOF",
 knobs={"extra_uniform_sample_count":17},
 evidence=[
   "compiler/realsas_compiler_core/motion_dynamic_proof_v2.py",
   "compiler/realsas_compiler_core/runtime_package_v2.py",
 ],
 quality_role="Adds samples beyond all source keyframe and contact-boundary times. Runtime plays sealed frame indices only.",
 performance_role="Linear multiplier on FK/LBS/intersection/visibility proof work for extra samples.",
 decision="Measure marginal defect detection versus wall time on subject-free high-curvature motion controls. This is proof density, not output frame-rate; optimize only after measurement."
)

summary={
 "schema":"RealSaS.PerformanceQualityKnobAudit.v1",
 "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
 "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "performance_contract":{
   "hard_total_compile_max_seconds":perf["latency_budget_seconds"]["hard_total_compile_max"],
   "quality_threshold_relaxation_for_speed_forbidden":perf["principles"]["quality_threshold_relaxation_for_speed_forbidden"],
   "logical_stages_may_be_fused":True,
   "note":perf["implementation_note"],
 },
 "rows":rows,
 "counts":{
   "calibrated_keep":sum(r["classification"]=="CALIBRATED_SUBJECT_FREE" for r in rows),
   "open":sum(r["status"]=="OPEN" for r in rows),
   "design_required":sum(r["status"] in {"DESIGN_REQUIRED","DESIGN_REVIEW"} for r in rows),
 },
 "repair_applied":False,
 "new_numeric_value_selected":False,
}
out=ROOT/"canonical"/"V2_PERFORMANCE_QUALITY_KNOB_AUDIT_V1_20260928.json"
out.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("KNOB_AUDIT",json.dumps(summary["counts"],sort_keys=True))
for r in rows:
    print("KNOB_ROW",json.dumps({k:r[k] for k in ("id","classification","status")},sort_keys=True))
