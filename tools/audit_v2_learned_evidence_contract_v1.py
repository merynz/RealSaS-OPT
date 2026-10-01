from __future__ import annotations
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def txt(rel): return (ROOT/rel).read_text(encoding="utf-8")
def contains(rel,s): return s in txt(rel)
def snip(rel,s,span=900):
    t=txt(rel);i=t.find(s)
    return "" if i<0 else t[max(0,i-span):i+span]

findings=[]

# IRIS V5 vs current ownership/Stage12
v5_files=list((ROOT/"models/iris/v5").glob("*.py"))
v5_all="\n".join(p.read_text(encoding="utf-8") for p in v5_files)
owner=txt("canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md")
codec=txt("compiler/realsas_compiler_core/geometry_artifact_codec_v2.py")
if "uncertainty" in owner.lower() and "uncertainty" not in v5_all.lower():
    findings.append({
      "id":"P1_IRIS_V5_EPISTEMIC_CONTRACT_DRIFT",
      "severity":"P1","status":"DESIGN_RECONCILIATION_REQUIRED",
      "class":"LEARNED_OUTPUT_CONTRACT_DRIFT",
      "evidence":{
        "ownership_claim":"IRIS learned envelope includes support/confidence/uncertainty evidence",
        "v5_uncertainty_token_present":False,
        "stage12_zero_surface_uncertainty_present":"uncertainty" in codec.lower(),
      },
      "consequence":"Current V5 signed-field route and canonical IRIS ownership text disagree on whether uncertainty/support is an authoritative learned output.",
      "design_before_code":"Freeze the actual V5 product evidence ontology. If uncertainty is still required, define its typed path through Stage12→GSA; if retired, update ownership/contracts and downstream expectations."
    })

# Geppetto uncertainty decision-effectiveness
rig="compiler/realsas_compiler_core/rig.py"
opt="compiler/vendor/realsas_synthesis/canonical_graph_optimizer.py"
if contains(rig,"confidence01=float(j.confidence)") and contains(opt,"selected_nodes=nodes") and contains(opt,"sum(float(edge.weight) for edge in selected_real_edges)"):
    findings.append({
      "id":"P1_GEPPETTO_NODE_UNCERTAINTY_INERT_IN_NORMAL_QUALIFICATION",
      "severity":"P1","status":"CONFIRMED_OPEN",
      "class":"EVIDENCE_DECISION_EFFECTIVENESS_GAP",
      "evidence":{
        "proposal_to_candidate":snip(rig,"confidence01=float(j.confidence)"),
        "normal_solver_fixed_nodes":snip(opt,"selected_nodes=nodes"),
        "normal_objective":snip(opt,"sum(float(edge.weight) for edge in selected_real_edges)")
      },
      "consequence":"Geppetto position/existence-derived confidence reaches graph candidates but normal arborescence selects the full proposed node set and does not use node confidence in its objective.",
      "design_before_code":"Decide explicit confidence/abstention semantics for control admission. Do not silently make Compiler a second cardinality predictor."
    })

# Arachne confidence/coverage typed seam
types="compiler/realsas_compiler_core/types.py"
skin="compiler/realsas_compiler_core/skin.py"
prop="models/arachne/v4/proposal_v4.py"
if contains(types,"class SkinInfluenceProposal") and not contains(types,"row_confidence") and not contains(prop,"row_confidence"):
    findings.append({
      "id":"P1_ARACHNE_PER_ROW_UNCERTAINTY_NOT_TYPED",
      "severity":"P1","status":"CONFIRMED_OPEN",
      "class":"LEARNED_EVIDENCE_TYPE_GAP",
      "evidence":{
        "typed_influence":snip(types,"class SkinInfluenceProposal"),
        "proposal_builder":snip(prop,"return SkinProposalIR"),
        "qualifier_optional_confidence":snip(skin,"row_confidence_available")
      },
      "consequence":"Dense weights are typed, but per-surface-row confidence/uncertainty/abstention is not. Stage32 can PASS legal simplex rows while product_skin_evidence_complete=False.",
      "design_before_code":"Define whether row uncertainty is required product evidence; if yes, type it and define fail/abstain semantics, especially for weak/uncovered rows."
    })

if contains(skin,"product_skin_evidence_complete") and contains(skin,"return QualifiedSkinIR") and "product_skin_evidence_complete" not in txt("compiler/realsas_compiler_services/orchestrator/adapters/learned_mechanics_v2.py").split('return {"status":"PASS"',1)[0]:
    pass
# Explicit check: Stage32 always returns PASS regardless report evidence completeness.
lm=txt("compiler/realsas_compiler_services/orchestrator/adapters/learned_mechanics_v2.py")
qpos=lm.find("def qualify_skin_stage")
qend=lm.find("def seal_arachne_checkpoint_stage",qpos)
qbody=lm[qpos:qend]
if '"status":"PASS"' in qbody and "product_skin_evidence_complete" in qbody and "if" not in qbody[qbody.find("product_skin_evidence_complete")-120:qbody.find("product_skin_evidence_complete")+120]:
    findings.append({
      "id":"P1_STAGE32_EVIDENCE_INCOMPLETE_CAN_STILL_PASS",
      "severity":"P1","status":"CONFIRMED_OPEN",
      "class":"QUALIFICATION_SCOPE_GAP",
      "evidence":"Stage32 reports product_skin_evidence_complete diagnostically but returns PASS based on legality/correction only.",
      "consequence":"Legally normalized skin is conflated with evidentially complete skin; later mechanical proof becomes the only guard for unsupported semantic rows.",
      "design_before_code":"Split LEGAL_SKIN qualification from PRODUCT_EVIDENCE_COMPLETENESS, or make downstream dependency explicit and fail-close at the first product authority point."
    })

# GSA relation epistemic fields
gsa="compiler/realsas_compiler_core/substrate/scene_first_signed.py"
if contains(gsa,"crosses_unknown=False") and contains(gsa,"unknown_bridge=False") and contains(gsa,"score=1.0"):
    findings.append({
      "id":"P1_GSA_RELATION_EPISTEMIC_FIELDS_FORCED_CERTAIN",
      "severity":"P1","status":"CONFIRMED_OPEN",
      "class":"EPISTEMIC_INFORMATION_LOSS",
      "evidence":snip(gsa,"crosses_unknown=False"),
      "consequence":"Geppetto/Arachne/Stage17 have relation uncertainty channels, but current GSA producer can force all mapped topology relations to certain/non-UNKNOWN.",
      "design_before_code":"Derive relation confidence/UNKNOWN from node support, completion state and preserved surface provenance before calibrating consumers."
    })

out={
 "schema":"RealSaS.V2LearnedEvidenceContractAudit.v1",
 "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
 "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "findings":findings,
 "counts":{k:sum(f["severity"]==k for f in findings) for k in ("P0","P1","P2")},
 "repair_applied":False,
 "model_retrain_authorized":False,
}
p=ROOT/"canonical"/"V2_LEARNED_EVIDENCE_CONTRACT_AUDIT_V1_20260928.json"
p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print("LEARNED_EVIDENCE_AUDIT",json.dumps(out["counts"],sort_keys=True))
for f in findings: print("LEARNED_EVIDENCE_FINDING",json.dumps({k:f[k] for k in ("id","severity","status","class","consequence")},sort_keys=True))
