from __future__ import annotations
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def has(path,needle):
    return needle in (ROOT/path).read_text(encoding="utf-8")

def snippet(path,needle,span=700):
    s=(ROOT/path).read_text(encoding="utf-8");i=s.find(needle)
    return "" if i<0 else s[max(0,i-span):i+span]

findings=[]

# P0 visual authority wiring
ap="compiler/realsas_compiler_services/orchestrator/adapters/appearance_v2.py"
rt="compiler/realsas_compiler_services/orchestrator/adapters/runtime_v2.py"
ps="compiler/realsas_compiler_core/product_state_v2.py"
vm="compiler/realsas_compiler_core/visual_mesh_arap_v1.py"
gsa="compiler/realsas_compiler_core/substrate/scene_first_signed.py"
cm="compiler/realsas_compiler_core/canonical_mesh_candidate_v1.py"
st="compiler/realsas_compiler_core/mesh/skin_topology_compatibility_v1.py"

if has(ap,'source_owned_visual_mesh_mode": True') and has(ps,'"mechanical_mesh_render_authority": (\n                False if visual_mode else True') and not has(rt,"visual_mesh_set"):
    findings.append({
      "id":"P0_SOURCE_OWNED_VISUAL_AUTHORITY_NOT_EXECUTED_BY_STAGE42_45",
      "severity":"P0","status":"CONFIRMED_OPEN","class":"AUTHORITY_WIRING_GAP",
      "evidence":{
        "stage23_source_owned":snippet(ap,'source_owned_visual_mesh_mode": True'),
        "stage38_declares_mechanical_not_render":snippet(ps,'"mechanical_mesh_render_authority": ('),
        "runtime_visual_mesh_token_present":False,
      },
      "consequence":"Stage38 seals visual mesh as render authority but Stage42-45 runtime projects/renders the mechanical mesh.",
      "design_before_code":"Define exact visual-deformation/visibility/runtime authority and bind Stage42 to the sealed VisualMeshSet plus its qualified mechanical correspondence."
    })

if has(ap,'uv_path = root / "visual_uv_binding.npz"') and has(rt,'if "face_uv" not in data.files:'):
    findings.append({
      "id":"P0_SOURCE_OWNED_STAGE23_STAGE42_UV_SCHEMA_MISMATCH",
      "severity":"P0","status":"CONFIRMED_OPEN","class":"IR_SCHEMA_COMPOSITION_GAP",
      "evidence":{
        "source_visual_uv":snippet(ap,'uv_path = root / "visual_uv_binding.npz"'),
        "runtime_requirement":snippet(rt,'if "face_uv" not in data.files:')
      },
      "consequence":"Source-owned Stage23 emits view/count/hash visual UV binding, while Stage42 unconditionally requires mechanical face_uv.",
      "design_before_code":"Create one explicit runtime visual projection IR; do not overload legacy mechanical CAA UV schema."
    })

# GSA information loss/invention
if has(gsa,"return cp,cn,edges") and has(cm,"combinations(neighbors[a], 2)"):
    findings.append({
      "id":"P0_GSA_FACE_INCIDENCE_LOST_THEN_REINVENTED_AS_CLIQUE",
      "severity":"P0","status":"CONFIRMED_OPEN","class":"INFORMATION_LOSS_AND_INVENTION",
      "evidence":{
        "gsa_compaction":snippet(gsa,"return cp,cn,edges"),
        "stage18_face_reconstruction":snippet(cm,"combinations(neighbors[a], 2)")
      },
      "consequence":"Dense triangle/cell incidence collapses to pairwise edges; Stage18 can mint a face from a 3-clique that was never one admitted surface cell.",
      "design_before_code":"Determine minimum higher-order GSA incidence needed by mechanics without turning GSA into product-mesh authority."
    })

# GSA relation epistemic flags
if has(gsa,"crosses_unknown=False") and has(gsa,"unknown_bridge=False") and has(gsa,"score=1.0"):
    findings.append({
      "id":"P1_GSA_RELATION_EPISTEMIC_STATE_FORCED_SAFE",
      "severity":"P1","status":"CONFIRMED_OPEN","class":"EPISTEMIC_INFORMATION_LOSS",
      "evidence":snippet(gsa,"crosses_unknown=False"),
      "consequence":"Consumers have UNKNOWN/score channels, but scene-first relation producer can mark every derived edge certain even when endpoints/support are model-completed.",
      "design_before_code":"Specify relation confidence/UNKNOWN derivation from observed/completed support and topology provenance, then recalibrate Stage15/17 consumers."
    })

# topology-unaware normals
if has(gsa,"cKDTree") and has(gsa,"robust_zero_surface_normals_v1"):
    findings.append({
      "id":"P1_GSA_NORMAL_NEIGHBORHOOD_EUCLIDEAN_NOT_SURFACE_AWARE",
      "severity":"P1","status":"DESIGN_VALIDATION_REQUIRED","class":"GEOMETRIC_OPERATOR_ASSUMPTION",
      "evidence":snippet(gsa,"def robust_zero_surface_normals_v1"),
      "consequence":"Nearby distinct sheets/components can enter one normal PCA neighborhood unless separately excluded.",
      "design_before_code":"Benchmark topology/component-aware geodesic/local-cell neighborhoods against current k=64 Euclidean operator before changing it."
    })

# visual mesh perf shortcut
if has(vm,"target_min_angle_deg=0.0") and has(vm,"max_quality_iterations=0"):
    findings.append({
      "id":"P1_VISUAL_CDT_SHAPE_QUALITY_DISABLED_FOR_SPEED",
      "severity":"P1","status":"CONFIRMED_OPEN","class":"PERFORMANCE_SHORTCUT_QUALITY_GAP",
      "evidence":snippet(vm,"target_min_angle_deg=0.0"),
      "consequence":"Exact silhouette constraints are preserved, but no visual triangle shape refinement is attempted before deformation.",
      "design_before_code":"Calibrate adaptive visual triangulation quality vs source curvature/deformation error/compile latency; do not guess an iteration count."
    })
if has(vm,"target_edge_px: int = 20") and has(vm,"step = max(4, int(target_edge_px))"):
    findings.append({
      "id":"P2_VISUAL_MESH_RESOLUTION_FIXED_HEURISTIC",
      "severity":"P2","status":"DESIGN_VALIDATION_REQUIRED","class":"QUALITY_PERFORMANCE_KNOB",
      "evidence":snippet(vm,"step = max(4, int(target_edge_px))"),
      "consequence":"Uniform source-space support spacing is not tied to local silhouette curvature, texture frequency, or deformation field.",
      "design_before_code":"Derive adaptive criterion and Pareto-calibrate under <=60s total compile contract."
    })

# visual affine coherence
if has(vm,"bound_faces[int(vertex_index)]") and has(vm,"VISUAL_REGION_CROSS_REGION_FACE_FORBIDDEN"):
    findings.append({
      "id":"P1_VISUAL_TRIANGLE_SINGLE_DEFORMATION_DOMAIN_NOT_GUARANTEED",
      "severity":"P1","status":"CONFIRMED_CONTRACT_GAP","class":"COMPOSITION_INVARIANT_GAP",
      "evidence":{
        "per_vertex_binding":snippet(vm,"bound_faces[int(vertex_index)]"),
        "only_region_face_gate":snippet(vm,"VISUAL_REGION_CROSS_REGION_FACE_FORBIDDEN"),
      },
      "consequence":"A visual triangle can stay inside one chart yet have corners driven by different mechanical affine faces.",
      "design_before_code":"Choose one-face containment, seam-aware subdivision, or a continuous qualified transfer field based on deformation quality and cost."
    })

# repair quality
if has(st,"SEAM_CUT_BY_UNSAFE_FACE_REMOVAL") and has(st,'"local_cdt_required":False'):
    findings.append({
      "id":"P1_STAGE35_REPAIR_DELETES_FACES_INSTEAD_OF_RETOPOLOGIZING",
      "severity":"P1","status":"CONFIRMED_OPEN","class":"REPAIR_OPERATOR_GAP",
      "evidence":snippet(st,"SEAM_CUT_BY_UNSAFE_FACE_REMOVAL"),
      "consequence":"Diagnostic smear removal creates holes/fragmentation rather than a product-quality seam.",
      "design_before_code":"Select optimal seam-preserving topology repair after topology/cell-incidence design is frozen."
    })

# alpha holes
if has(rt,"final_alpha_hole_pixel_count=alpha_transparent"):
    pass_pos=(ROOT/rt).read_text().find("passed = (")
    pass_block=(ROOT/rt).read_text()[pass_pos:pass_pos+1800]
    if "alpha_transparent" not in pass_block and "transparent_fraction" not in pass_block:
        findings.append({
          "id":"P1_STAGE45_DYNAMIC_ALPHA_HOLES_NOT_HARD_GATE",
          "severity":"P1","status":"CONFIRMED_OPEN","class":"PROOF_COVERAGE_GAP",
          "evidence":pass_block,
          "consequence":"Visible geometry can have final-alpha holes while DVI passes if other visibility/provenance gates pass.",
          "design_before_code":"Define topology-aware alpha-hole classes (coherent, scattered/speckled, intentional transparent art) and calibrate separate hard budgets."
        })

# source-owned visual quality validation only at rest
if has(ap,"CAA_VISUAL_REST") and not has(rt,"load_visual_mesh_view"):
    findings.append({
      "id":"P0_SOURCE_OWNED_VISUAL_DYNAMIC_PROOF_MISSING_FROM_CANONICAL_RUNTIME",
      "severity":"P0","status":"CONFIRMED_OPEN","class":"PROOF_EXECUTION_GAP",
      "evidence":"Stage25 loads VisualMeshSet and proves exact source-owned rest; Stage42-45 never load the VisualMeshSet.",
      "consequence":"Rest visual authority is proven, then discarded before dynamic/runtime proof.",
      "design_before_code":"Dynamic proof must exercise exactly the same visual mesh/deformation state exported to runtime."
    })

out={
 "schema":"RealSaS.V2SemanticCompositionAudit.v1",
 "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
 "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "findings":findings,
 "counts":{k:sum(x["severity"]==k for x in findings) for k in ("P0","P1","P2")},
 "repair_applied":False,
 "optimization_selected":False,
}
path=ROOT/"canonical"/"V2_SEMANTIC_COMPOSITION_AUDIT_V1_20260928.json"
path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("SEMANTIC_AUDIT",json.dumps(out["counts"],sort_keys=True))
for f in findings: print("SEMANTIC_FINDING",json.dumps({k:f[k] for k in ("id","severity","status","class","consequence")},sort_keys=True))
