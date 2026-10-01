from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/"canonical/MAINLINE_EXECUTION_PLAN_V2.json").read_text())

TOKENS=("KNIGHT","SUBJECT2","QUATERNIUS_KNIGHT","MAGE_FIT","PROMOTED_MAGE")
DEMO_SCHEMA_MARKERS=("KnightDemoStage14FallbackPreregistration","KnightDemoStage18MeshFallbackPreregistration")
GUARD_MARKERS=(
    "NO_KNIGHT","NOT_EMPIRICAL_KNIGHT","BEFORE_ANY_NEW_KNIGHT","BEFORE_KNIGHT","PRE_KNIGHT",
    "thresholds_from_knight","without Knight-dependent tuning","No Knight-tuned threshold","Knight result",
    "KNIGHT_RESULT","Knight may not tune","forbidden",
)
METADATA_MARKERS=(
    "first_knight_v2_reuses_exact_output_camera_geometry",
    "canonical Knight policy",
)

def module_path(module:str):
    p=ROOT/Path(*module.split(".")).with_suffix(".py")
    return p if p.is_file() else None

files=set()
for stage in PLAN["stages"]:
    module,_,_=str(stage["adapter"]).partition(":")
    p=module_path(module)
    if p: files.add(p)
for base in (
    ROOT/"compiler/realsas_compiler_core",
    ROOT/"models/iris",
    ROOT/"models/geppetto",
    ROOT/"models/arachne",
):
    if base.exists(): files.update(base.rglob("*.py"))

rows=[]
algorithmic=[]
demo_embedded=[]
guardrails=[]
metadata=[]
for path in sorted(files):
    rel=str(path.relative_to(ROOT))
    src=path.read_text(encoding="utf-8",errors="replace")
    lines=src.splitlines()
    for lineno,line in enumerate(lines,1):
        upper=line.upper()
        if not any(t in upper for t in TOKENS):
            continue
        row={"path":rel,"line":lineno,"text":line.strip()[:600]}
        if any(m.lower() in line.lower() for m in DEMO_SCHEMA_MARKERS):
            row["classification"]="DEMO_ONLY_SUBJECT_NAMED_APPARATUS_EMBEDDED_IN_CANONICAL_ADAPTER"
            demo_embedded.append(row)
        elif any(m.lower() in line.lower() for m in GUARD_MARKERS):
            row["classification"]="SUBJECT_FREE_GUARDRAIL_OR_PROVENANCE"
            guardrails.append(row)
        elif any(m.lower() in line.lower() for m in METADATA_MARKERS):
            row["classification"]="NON_BEHAVIORAL_METADATA_OR_DOCSTRING"
            metadata.append(row)
        else:
            # A genuine product contamination requires a subject token in executable
            # behavior, not a comment/provenance sentence. Treat comments/docstrings
            # as review-only and code expressions as algorithmic candidates.
            stripped=line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                row["classification"]="COMMENT_OR_DOCSTRING_REVIEW"
                metadata.append(row)
            else:
                row["classification"]="ALGORITHMIC_SUBJECT_CONTAMINATION_CANDIDATE"
                algorithmic.append(row)
        rows.append(row)

# Semantic post-classification for known generic anti-contamination constructs.
semantic_algorithmic=[]
reclassified=[]
for row in algorithmic:
    text=row["text"]
    if (
        "issubset(forbidden)" in text
        or "knight_result_used_for_threshold_selection" in text.lower()
        or "thresholds_from_knight_results_forbidden" in text.lower()
        or "authority" in text.lower() and ("KNIGHT" in text.upper() or "MAGE" in text.upper())
        or "status" in text.lower() and ("KNIGHT" in text.upper() or "MAGE" in text.upper())
    ):
        rr=dict(row)
        rr["classification"]="SUBJECT_FREE_GUARDRAIL_OR_PROVENANCE"
        guardrails.append(rr);reclassified.append(row)
    else:
        semantic_algorithmic.append(row)
algorithmic=semantic_algorithmic

payload={
  "schema":"RealSaS.SubjectGenericExecutableClosureAudit.v2",
  "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
  "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  "scanned_file_count":len(files),
  "algorithmic_subject_contamination":algorithmic,
  "algorithmic_subject_contamination_count":len(algorithmic),
  "demo_only_subject_named_apparatus_embedded_in_canonical_adapters":demo_embedded,
  "demo_only_embedded_count":len(demo_embedded),
  "subject_free_guardrail_or_provenance_hits":guardrails,
  "non_behavioral_metadata_or_docstring_hits":metadata,
  "findings":[
    {
      "id":"PRODUCT_EXECUTABLE_ALGORITHMIC_SUBJECT_GENERICITY",
      "severity":"P0" if algorithmic else "PASS",
      "passed":not algorithmic,
      "claim":"No current 46-stage/core/promoted-model product behavior may branch on Knight/Mage identity.",
    },
    {
      "id":"DEMO_APPARATUS_SEPARATION_FROM_CANONICAL_ADAPTERS",
      "severity":"P1" if demo_embedded else "PASS",
      "passed":not demo_embedded,
      "claim":"Subject-named demo fallback schemas should not live inside canonical generic adapters even when execution_class=DEMO_WITNESS and product authority is forbidden.",
    },
  ],
  "claim_boundary":"Text+semantic source audit over current 46-stage adapters, compiler core, promoted IRIS/Geppetto/Arachne. Subject-free guard strings and historical provenance labels are not algorithmic contamination.",
}
out=ROOT/"canonical"/"V2_SUBJECT_GENERIC_EXECUTABLE_CLOSURE_AUDIT_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2,sort_keys=True))
if algorithmic:
    raise SystemExit(2)
