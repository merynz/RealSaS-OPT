from __future__ import annotations

import json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def loadj(p): return json.loads((ROOT/p).read_text(encoding="utf-8"))
def txt(p): return (ROOT/p).read_text(encoding="utf-8")

arch=loadj("canonical/REALSAS_CANONICAL_ARCHITECTURE_V2_20260920.json")
dag=loadj("canonical/V2_DAG_TRUTHFULNESS_AUDIT_V1_20260928.json")
prod=loadj("canonical/V2_PRODUCT_STATE_WIRING_AUDIT_V1_20260928.json")
learn=loadj("canonical/V2_LEARNED_EVIDENCE_CONTRACT_AUDIT_V1_20260928.json")
knob=loadj("canonical/V2_PERFORMANCE_QUALITY_KNOB_AUDIT_V1_20260928.json")
motion=loadj("canonical/V2_MOTION_RUNTIME_PROOF_AUDIT_V1_20260928.json")
gsa_obs=loadj("canonical/GSA_OBSERVATION_SEMANTICS_AUDIT_V1_20260928.json")
gsa_coh=loadj("canonical/GSA_SURFACE_COHERENCE_ADVERSARIES_V1_20260928.json")
gsa_norm=loadj("canonical/GSA_NORMAL_TOPOLOGY_ADVERSARY_V1_20260928.json")
gsa_face=loadj("canonical/GSA_FACE_INCIDENCE_CLIQUE_ADVERSARY_V1_20260928.json")
subj=loadj("canonical/V2_SUBJECT_GENERIC_EXECUTABLE_CLOSURE_AUDIT_V1_20260928.json")
fp=loadj("canonical/V2_FINGERPRINT_COMPLETENESS_AUDIT_V1_20260928.json")

A=[]  # intended correct, implementation deviates
B=[]  # intended direction retained, contract/invariant/calibration incomplete
C=[]  # intended decision itself falsified
G=[]  # validated guardrails

def add(bucket, **kw): bucket.append(kw)

# A — direct implementation deviations from frozen intended architecture.
add(A,
 id="A01_STAGE18_CLIQUE_FACE_INVENTION_VIOLATES_MWB",
 severity="P0",
 owner="GSA_TO_STAGE18",
 evidence=[gsa_face["finding"], "canonical/MESH_WEIGHT_BINDING_CONTRACT_V1.md §4.2"],
 intended="A triangle must not become surface evidence merely because a triangulator can connect its vertices.",
 actual="Dense face incidence collapses to pairwise edges; Stage18 relation baseline can mint any admissible 3-clique as a face.",
 action_class="RESTORE_IMPLEMENTATION_TO_FROZEN_CONTRACT"
)

add(A,
 id="A02_DUAL_VISUAL_TOPOLOGY_DEVIATES_FROM_SINGLE_QUALIFIED_MESH_AUTHORITY",
 severity="P0",
 owner="STAGE18_STAGE23_STAGE38_RUNTIME_EXPORT",
 evidence=prod["findings"][:4],
 intended=arch["authorities"]["mesh"],
 actual="Current branch introduced a second VisualMeshSet topology and declares mechanical_mesh_render_authority=False while the frozen architecture says QualifiedMeshIR is the single product geometry authority.",
 action_class="QUARANTINE_EXPERIMENTAL_DEVIATION_BEFORE_MAIN"
)

add(A,
 id="A03_STAGE35_TO_STAGE18_HIDDEN_FILESYSTEM_FEEDBACK",
 severity="P0",
 owner="ORCHESTRATOR",
 evidence=dag["stage18_stage35_cycle_witness"],
 intended=arch["repair_policy"],
 actual="Stage18 reads Stage35 repair artifacts through a same-run filesystem side-channel not represented by the declared dataflow/fingerprint.",
 action_class="RESTORE_EXPLICIT_LINEAGE_AND_INVALIDATION"
)

add(A,
 id="A04_FINGERPRINT_READSET_INCOMPLETE",
 severity="P0",
 owner="ORCHESTRATOR",
 evidence=[x for x in fp["findings"] if x.get("class") in {"MANIFEST_FINGERPRINT_GAP","FINGERPRINT_DEPENDENCY_GAP","FILESYSTEM_SIDE_CHANNEL"}],
 intended="Every consumed authority edge is exact and invalidation follows the true dependency subgraph.",
 actual="Several adapters consume undeclared stage outputs/manifest sections; some read truth is absent from fingerprints.",
 action_class="RESTORE_EXACT_CONSUMED_AUTHORITY_GRAPH"
)

alpha_rows=[r for r in motion["rows"] if r["id"]=="DYNAMIC_VISIBLE_ALPHA_HOLES"]
if alpha_rows:
    add(A,
     id="A05_DYNAMIC_ALPHA_HOLE_METRIC_NOT_HARD_GATE",
     severity="P1",
     owner="STAGE45",
     evidence=alpha_rows[0],
     intended="Reclosed visual-fidelity policy requires consequential visible alpha holes to be shipping-fail evidence.",
     actual="Alpha-hole metric is computed diagnostically but is not part of the final pass predicate.",
     action_class="RESTORE_GATE_TO_RECLOSED_POLICY"
    )

add(A,
 id="A06_DEMO_FALLBACK_SCHEMA_INSIDE_CANONICAL_ADAPTERS",
 severity="P1",
 owner="FRONT_HALF_ADAPTER_HYGIENE",
 evidence=subj["violations"],
 intended="Subject-agnostic product implementation; witness/demo apparatus cannot own canonical truth.",
 actual="No Knight-specific algorithmic branch was found, but canonical adapters retain two named demo fallback schema paths.",
 action_class="SEPARATE_DEMO_APPARATUS_FROM_CANONICAL_IMPLEMENTATION"
)

# B — intended direction retained, but the contract/operator semantics are insufficiently frozen.
add(B,
 id="B01_GSA_NORMAL_NEIGHBORHOOD_SEMANTICS_UNDERSPECIFIED",
 severity="P0",
 owner="GSA",
 evidence=gsa_norm,
 intended_gap="Stage14 freezes k=64 and error thresholds but does not require topology/component/geodesic-safe neighborhoods for normal estimation.",
 demonstrated_failure="Synthetic close-sheet adversary: current global Euclidean kNN near-contact p95 normal error 20.94 degrees versus 0 degrees for a component-aware same-operator oracle.",
 action_class="FREEZE_OPERATOR_SEMANTICS_WITHIN_EXISTING_GSA_ROLE"
)

add(B,
 id="B02_GSA_COMPACTION_SAME_COMPONENT_SHEET_COHERENCE_UNDERSPECIFIED",
 severity="P0",
 owner="GSA",
 evidence=gsa_coh["findings"][0],
 intended_gap="Component-aware voxel compaction forbids cross-component aliasing but does not state a surface/geodesic coherence invariant for folds/sheets within one connected component.",
 action_class="FREEZE_COMPACTION_COHERENCE_INVARIANT"
)

add(B,
 id="B03_GSA_VISIBILITY_AND_OBSERVED_SEMANTICS_UNDERSPECIFIED",
 severity="P1",
 owner="GSA",
 evidence=[gsa_coh["findings"][1],gsa_obs["finding"]],
 intended_gap="Contract does not cleanly distinguish source-observed support from self-visible predicted geometry nor fully specify continuous-surface visibility.",
 action_class="FREEZE_VISIBILITY_AND_OBSERVATION_SEMANTICS"
)

gsa_epi=[x for x in learn["findings"] if x["id"]=="P1_GSA_RELATION_EPISTEMIC_FIELDS_FORCED_CERTAIN"]
if gsa_epi:
    add(B,
     id="B04_GSA_RELATION_EPISTEMIC_SEMANTICS_MISSING",
     severity="P1",
     owner="GSA_TO_GEPPETTO_ARACHNE",
     evidence=gsa_epi[0],
     intended_gap="Typed relation uncertainty/UNKNOWN channels exist but Stage14 contract does not define how relation epistemic state is derived from support/completion/provenance.",
     action_class="FREEZE_RELATION_EPISTEMIC_CONTRACT"
    )

add(B,
 id="B05_ARACHNE_ROW_UNCERTAINTY_PRODUCT_SEMANTICS_MISSING",
 severity="P1",
 owner="ARACHNE_STAGE32",
 evidence=[x for x in learn["findings"] if "ARACHNE" in x["id"] or "STAGE32" in x["id"]],
 intended_gap="Qualified skin legality and mesh-weight transfer are frozen, but per-surface-row learned confidence/abstention/product-evidence completeness are not frozen as typed product semantics.",
 action_class="FREEZE_EVIDENCE_COMPLETENESS_AND_ABSTENTION"
)

add(B,
 id="B06_GEPPETTO_UNCERTAINTY_ADMISSION_ROLE_MISSING",
 severity="P1",
 owner="GEPPETTO_STAGE28",
 evidence=[x for x in learn["findings"] if "GEPPETTO" in x["id"]],
 intended_gap="Geppetto uncertainty reaches graph candidates but frozen product contract does not define how it affects node admission/abstention in the normal qualification route.",
 action_class="FREEZE_UNCERTAINTY_DECISION_SEMANTICS"
)

for row in knob["rows"]:
    if row["status"] in {"OPEN","DESIGN_REQUIRED","DESIGN_REVIEW"}:
        add(B,
            id="B_KNOB_"+row["id"],
            severity="P1" if row["status"]=="OPEN" else "P2",
            owner=row["owner"],
            evidence=row,
            intended_gap="Shipping-relevant numeric value lacks enough calibration/provenance to claim an optimum.",
            action_class="CALIBRATE_BEFORE_IMPLEMENTATION_CHANGE"
        )

# C — intended decision itself is wrong. Deliberately empty unless falsified.
# Promotion rule is strict: implementation must be shown conformant to the intended
# contract, and the intended decision must still produce a reproducible failure that
# disappears under a competing design while downstream authority is held fixed.
C_promotion_rule={
 "required":True,
 "conditions":[
   "IMPLEMENTATION_CONFORMS_TO_THE_INTENDED_CONTRACT",
   "FAILURE_REPRODUCES_ON_THE_CONFORMANT_IMPLEMENTATION",
   "FAILURE_IS_CAUSALLY_ATTRIBUTED_TO_THE_INTENDED_DECISION_ITSELF",
   "COMPETING_DESIGN_REMOVES_THE_FAILURE_WITH_OTHER_AUTHORITIES_HELD_FIXED",
   "NO_WEAKER_A_OR_B_REPAIR_EXPLAINS_THE_RESULT",
 ],
 "current_promoted_count":0,
 "note":"No current finding is promoted to C. In particular, the frozen single QualifiedMeshIR mechanics+appearance decision is not falsified merely because an experimental dual-visual-topology branch was introduced."
}

add(G,
 id="G01_SUBJECT_ALGORITHMIC_GENERICITY",
 status="PASS_WITH_HYGIENE_EXCEPTIONS",
 evidence={"violation_count":subj["violation_count"],"finding":subj["finding"]},
 note="No Knight-specific algorithmic branch was found in current 46-stage adapters/core/promoted models."
)
add(G,
 id="G02_DECLARED_GRAPH_INCREMENTAL_MECHANISM",
 status="STRUCTURALLY_USEFUL_BUT_NOT_YET_SEMANTICALLY_TRUSTWORTHY",
 evidence=dag["declared_graph"],
 note="Declared dependency graph is acyclic and supports targeted invalidation, but incremental resume must remain untrusted until A03/A04 close."
)

payload={
 "schema":"RealSaS.IntendedArchitectureConformanceAudit.v2",
 "status":"AUDIT_ONLY__NO_REPAIR_APPLIED__NO_ARCHITECTURE_REPLACEMENT_AUTHORIZED",
 "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "policy":{
   "architecture_replacement_authorized":False,
   "repair_authorized":False,
   "main_merge_authorized":False,
   "priority":"FIRST_RESTORE_AND_PROVE_THE_EXISTING_FROZEN_INTENDED_ARCHITECTURE",
   "post_repair_rule":"Only after the implementation is conformant to intended contracts and the exact witness still fails may category C be populated or alternative architecture be considered."
 },
 "categories":{
   "A_IMPLEMENTATION_DEVIATES_FROM_INTENDED":A,
   "B_INTENDED_DIRECTION_RETAINED_BUT_CONTRACT_OR_CALIBRATION_INCOMPLETE":B,
   "C_INTENDED_DECISION_ITSELF_FALSIFIED":C,
 },
 "C_promotion_rule":C_promotion_rule,
 "validated_guardrails":G,
 "counts":{
   "A_P0":sum(x["severity"]=="P0" for x in A),
   "A_P1":sum(x["severity"]=="P1" for x in A),
   "B_P0":sum(x["severity"]=="P0" for x in B),
   "B_P1":sum(x["severity"]=="P1" for x in B),
   "B_P2":sum(x["severity"]=="P2" for x in B),
   "C_total":len(C),
 },
}
out=ROOT/"canonical"/"V2_INTENDED_ARCHITECTURE_CONFORMANCE_AUDIT_V2_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("INTENDED_CONFORMANCE_V2",json.dumps(payload["counts"],sort_keys=True))
for x in A: print("A",x["severity"],x["id"])
for x in B: print("B",x["severity"],x["id"])
print("C",len(C),"PROMOTED")
