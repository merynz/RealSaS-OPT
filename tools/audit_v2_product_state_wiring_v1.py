from __future__ import annotations
import json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def T(rel): return (ROOT/rel).read_text(encoding="utf-8")
def has(rel,s): return s in T(rel)

v2a="compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture.py"
ps="compiler/realsas_compiler_core/product_state_v2.py"
psta="compiler/realsas_compiler_services/orchestrator/adapters/product_state_v2.py"
rt="compiler/realsas_compiler_services/orchestrator/adapters/runtime_v2.py"
cl="compiler/realsas_compiler_services/orchestrator/adapters/closure_v2.py"
vm="compiler/realsas_compiler_core/visual_mesh_arap_v1.py"
ap="compiler/realsas_compiler_services/orchestrator/adapters/appearance_v2.py"

findings=[]

def add(fid,severity,cls,evidence,consequence,design):
    findings.append({
      "id":fid,"severity":severity,"class":cls,
      "evidence":evidence,"consequence":consequence,
      "design_before_code":design,
    })

stage18_visual = has(v2a,'"mechanical_mesh_render_authority":False') or has(v2a,'"mechanical_mesh_render_authority": False')
stage38_visual_hash = has(ps,"visual_mesh_set_binding_hash")
stage42_consumes_visual = (
    has(rt,"visual_mesh_set_binding_hash")
    or has(rt,"VisualMeshSet")
    or has(rt,"load_visual_mesh_view")
)
stage46_consumes_visual = (
    has(cl,"visual_mesh_set_binding_hash")
    or has(cl,"VisualMeshSet")
    or has(cl,"visual_mesh/")
)

if stage18_visual and stage38_visual_hash and not stage42_consumes_visual:
    add(
      "P0_VISUAL_AUTHORITY_HASHED_BUT_NOT_EXECUTED_AT_RUNTIME",
      "P0","AUTHORITY_CARRIER_GAP",
      {
        "stage18_declares_mechanical_not_render_authority":stage18_visual,
        "stage38_carries_visual_hash":stage38_visual_hash,
        "stage42_visual_payload_consumer":stage42_consumes_visual,
      },
      "Product state knows which VisualMeshSet is authoritative, but runtime projection has no typed visual-geometry input and falls back to the mechanical mesh path.",
      "Define a typed runtime visual state carrying exact VisualMeshSet geometry plus a qualified deformation/visibility binding; Stage42 must consume that exact state."
    )

if stage18_visual and not stage46_consumes_visual:
    add(
      "P0_VISUAL_AUTHORITY_NOT_PRESENT_IN_EDITABLE_EXPORT",
      "P0","AUTHORING_STATE_GAP",
      {
        "visual_authority_declared":stage18_visual,
        "stage46_visual_payload_consumer":stage46_consumes_visual,
      },
      "The editable authoring/export state cannot contain or edit the actual declared visual topology.",
      "Make render-authoritative visual topology a first-class editable artifact or explicitly reject the dual-topology product architecture."
    )

if has(ap,'uv_path = root / "visual_uv_binding.npz"') and has(rt,'if "face_uv" not in data.files:'):
    add(
      "P0_SOURCE_OWNED_VISUAL_UV_RUNTIME_SCHEMA_DISCONNECT",
      "P0","SCHEMA_COMPOSITION_GAP",
      {
        "stage23_source_uv_binding":"visual_uv_binding.npz",
        "stage42_legacy_requirement":"face_uv required",
      },
      "Source-owned appearance emits a visual binding contract while runtime still requires per-mechanical-face UVs.",
      "Freeze one runtime visual projection IR; do not overload or silently coerce the legacy mechanical face_uv contract."
    )

declared = has(v2a,"BARYCENTRIC_BINDING_PLUS_ARAP_2D")
canonical_calls=[]
for rel in (v2a,psta,rt):
    src=T(rel)
    if "bind_region_visual_vertices_to_mechanical_affine_v1(" in src:
        canonical_calls.append(rel)
if declared and not canonical_calls:
    add(
      "P0_DECLARED_VISUAL_DEFORMATION_CONTRACT_HAS_NO_CANONICAL_EXECUTOR",
      "P0","DECLARED_VS_EXECUTABLE_CONTRACT_GAP",
      {
        "declared_contract":"BARYCENTRIC_BINDING_PLUS_ARAP_2D",
        "canonical_invocations":canonical_calls,
      },
      "VisualMeshSet metadata declares a deformation contract, but the canonical 46-stage path does not materialize or qualify that binding.",
      "Introduce an explicit typed visual-to-mechanical deformation binding stage/artifact and prove the exact binding before runtime/export."
    )

if has(vm,"bound_faces[int(vertex_index)]") and has(vm,"VISUAL_REGION_CROSS_REGION_FACE_FORBIDDEN"):
    add(
      "P1_VISUAL_TRIANGLE_DEFORMATION_DOMAIN_COHERENCE_NOT_PROVEN",
      "P1","COMPOSITION_INVARIANT_GAP",
      {
        "binding_granularity":"per visual vertex -> mechanical face",
        "existing_face_gate":"cross-region face forbidden",
      },
      "A visual triangle may remain inside one region while its corners are driven by different mechanical affine faces.",
      "Choose and qualify either single-affine-domain visual triangles, seam-aware subdivision, or a continuous transfer field; do not select by intuition."
    )

slot_visual=False
for rel in (psta,ps,cl,rt,v2a):
    src=T(rel).lower()
    if "visual_mesh" in src and ("slot" in src or "attachment" in src):
        slot_visual=True
if stage18_visual and not slot_visual:
    add(
      "P1_VISUAL_TO_PRESENTATION_SLOT_BINDING_ABSENT",
      "P1","EDITABILITY_SEMANTIC_BINDING_GAP",
      {"visual_to_slot_binding_detected":False},
      "Separate source-owned visual topology is not explicitly assigned to editable presentation slots or attachments.",
      "Define whether editability operates on visual primitives, mechanical primitives, or a stable cross-domain attachment map and prove round-trip edits."
    )

out={
  "schema":"RealSaS.V2ProductStateWiringAudit.v1",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "findings":findings,
  "counts":{k:sum(x["severity"]==k for x in findings) for k in ("P0","P1","P2")},
  "repair_applied":False,
  "architecture_choice_frozen":False,
}
p=ROOT/"canonical"/"V2_PRODUCT_STATE_WIRING_AUDIT_V1_20260928.json"
p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(out,indent=2,sort_keys=True))
