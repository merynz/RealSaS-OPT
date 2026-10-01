from __future__ import annotations

import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GSA=ROOT/"compiler/realsas_compiler_core/substrate/scene_first_signed.py"
STAGE18=ROOT/"compiler/realsas_compiler_core/canonical_mesh_candidate_v1.py"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pair(a,b):
    return (a,b) if a<b else (b,a)

# A valid simplicial surface fragment with a triangular 1-skeleton cycle (A,B,C)
# that is NOT itself a 2-simplex. GSA edge projection cannot distinguish this
# from a filled ABC triangle once face incidence is discarded.
vertices=("A","B","C","D","E","F")
dense_faces=(
    ("A","B","D"),
    ("B","C","E"),
    ("A","C","F"),
)
dense_face_set={tuple(sorted(f)) for f in dense_faces}

edges={
    pair(f[i],f[j])
    for f in dense_faces
    for i,j in ((0,1),(1,2),(2,0))
}
neighbors={v:set() for v in vertices}
for a,b in edges:
    neighbors[a].add(b);neighbors[b].add(a)

# Exact semantic rule used by Stage18 relation baseline: every 3-clique is a face
# if its geometric area passes the floor. Give all nodes generic non-collinear
# positions so ABC passes.
pos={
    "A":np.asarray((0.0,0.0,0.0)),
    "B":np.asarray((1.0,0.0,0.0)),
    "C":np.asarray((0.5,0.8,0.0)),
    "D":np.asarray((0.5,-0.7,0.0)),
    "E":np.asarray((1.3,0.8,0.0)),
    "F":np.asarray((-0.3,0.8,0.0)),
}

def double_area(a,b,c):
    return float(np.linalg.norm(np.cross(b-a,c-a)))

minted=[]
for a in sorted(vertices):
    for b,c in combinations(sorted(neighbors[a]),2):
        if pair(b,c) not in edges:
            continue
        tri=tuple(sorted((a,b,c)))
        if tri[0]!=a:
            continue
        pts=[pos[x] for x in tri]
        lengths=[float(np.linalg.norm(pts[i]-pts[j])) for i,j in ((0,1),(1,2),(2,0))]
        scale=max(lengths)
        rel=double_area(*pts)/(scale*scale)
        if rel>=1e-8:
            minted.append(tri)

minted_set=set(minted)
invented=sorted(minted_set-dense_face_set)

gsa_src=GSA.read_text(encoding="utf-8")
stage18_src=STAGE18.read_text(encoding="utf-8")
signatures={
    "gsa_collapses_faces_to_edges":(
        "mapped = inverse[f]" in gsa_src
        and "edges = np.concatenate" in gsa_src
        and "return cp, cn, edges.astype" in gsa_src
    ),
    "stage18_mints_clique_faces":(
        "for b, c in combinations(sorted(neighbors[a]), 2):" in stage18_src
        and "if _pair(b, c) not in safe_edges:" in stage18_src
        and "faces_surface.append(tri)" in stage18_src
    ),
}

payload={
 "schema":"RealSaS.GSAFaceIncidenceCliqueAdversary.v1",
 "status":"AUDIT_ONLY__NO_REPAIR_APPLIED",
 "repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
 "source_hashes":{
   str(GSA.relative_to(ROOT)):sha(GSA),
   str(STAGE18.relative_to(ROOT)):sha(STAGE18),
 },
 "code_signatures":signatures,
 "construction":{
   "dense_faces":dense_faces,
   "pairwise_edges":sorted(edges),
   "missing_dense_face":["A","B","C"],
 },
 "stage18_rule_result":{
   "minted_faces":sorted(minted_set),
   "invented_faces":invented,
   "invented_face_count":len(invented),
 },
 "finding":{
   "id":"GSA_FACE_INCIDENCE_LOST_THEN_REINVENTED_AS_CLIQUE",
   "confirmed":bool(all(signatures.values()) and ("A","B","C") in invented),
   "class":"INFORMATION_LOSS_AND_HIGHER_ORDER_TOPOLOGY_INVENTION",
   "claim_boundary":"This proves pairwise adjacency is insufficient to preserve admitted 2-simplex incidence under the current Stage18 clique rule. It does not prescribe the replacement surface complex or final mesh algorithm.",
 },
}
out=ROOT/"canonical"/"GSA_FACE_INCIDENCE_CLIQUE_ADVERSARY_V1_20260928.json"
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2,sort_keys=True))
if not payload["finding"]["confirmed"]:
    raise SystemExit("ADVERSARY_DID_NOT_CONFIRM")
