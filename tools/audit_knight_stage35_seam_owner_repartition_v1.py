from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,deformation_envelope_from_dict,mechanical_partition_from_dict,
    mesh_policy_from_dict,qualified_camera_set_from_dict,qualified_skeleton_from_dict,
    qualified_skin_from_dict,rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    run_skin_topology_compatibility_v1,propose_mechanical_repartition_directive_v2,
)


def load(p,codec): return codec(json.loads(Path(p).read_text()))

def owner_sid(v):
    mode=str(v.support_binding.mode)
    coeff=tuple(v.support_binding.coefficients)
    if mode=="IDENTITY_SURFACE_NODE" and len(coeff)==1 and abs(float(coeff[0][1])-1.0)<=1e-12:
        return str(coeff[0][0]),"IDENTITY"
    if mode=="SEAM_GEOMETRY_INTERPOLATION":
        md=dict(v.support_binding.metadata or {})
        s=tuple(md.get("skin_support_coefficients") or ())
        if len(s)==1 and abs(float(s[0][1])-1.0)<=1e-12:
            return str(s[0][0]),"SEAM_OWNER"
    return None,mode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--surface",type=Path,required=True)
    ap.add_argument("--partition",type=Path,required=True)
    ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--skin",type=Path,required=True)
    ap.add_argument("--skeleton",type=Path,required=True)
    ap.add_argument("--envelope",type=Path,required=True)
    ap.add_argument("--cameras",type=Path,required=True)
    ap.add_argument("--policy",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    surface=load(a.surface,rigging_surface_from_dict)
    partition=load(a.partition,mechanical_partition_from_dict)
    candidate=load(a.candidate,canonical_mesh_candidate_from_dict)
    skin=load(a.skin,qualified_skin_from_dict)
    skeleton=load(a.skeleton,qualified_skeleton_from_dict)
    envelope=load(a.envelope,deformation_envelope_from_dict)
    cameras=tuple(sorted(load(a.cameras,qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    policy=load(a.policy,mesh_policy_from_dict)
    comp=run_skin_topology_compatibility_v1(
        candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy
    )
    directive=propose_mechanical_repartition_directive_v2(
        candidate,surface=surface,skeleton=skeleton,skin=skin,partition=partition,compatibility_report=comp
    )
    _,weights,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    faces=np.asarray(faces,dtype=np.int64)
    modes=Counter()
    identity_resolved=0
    owner_resolved=0
    owner_pair_set=set()
    unresolved_owner=0
    verts=tuple(candidate.vertices)
    for fi in comp["unsafe_face_indices"]:
        idx=tuple(map(int,faces[int(fi)].tolist()))
        modes[tuple(sorted(str(verts[i].support_binding.mode) for i in idx))]+=1
        edge_rows=[]
        for ia,ib in ((0,1),(1,2),(2,0)):
            u,v=idx[ia],idx[ib]
            l1=float(np.abs(weights[u]-weights[v]).sum())
            edge_rows.append((l1,u,v))
        edge_rows.sort(reverse=True)
        _,u,v=edge_rows[0]
        def ident(i):
            vv=verts[i]; coeff=tuple(vv.support_binding.coefficients)
            if str(vv.support_binding.mode)=="IDENTITY_SURFACE_NODE" and len(coeff)==1 and abs(float(coeff[0][1])-1.0)<=1e-12:
                return str(coeff[0][0])
            return None
        iu,iv=ident(u),ident(v)
        if iu and iv and iu!=iv: identity_resolved+=1
        ou,ku=owner_sid(verts[u]); ov,kv=owner_sid(verts[v])
        if ou and ov and ou!=ov:
            owner_resolved+=1
            owner_pair_set.add(tuple(sorted((ou,ov))))
        else:
            unresolved_owner+=1
    report={
      "schema":"RealSaS.KnightStage35SeamOwnerRepartitionAudit.v1",
      "status":"MEASURED",
      "g3b_unsafe_face_count":int(comp["unsafe_face_count"]),
      "current_directive_candidate_pair_count":int(directive["candidate_separate_pair_count"]),
      "current_directive_unresolved_unsafe_face_count":int(directive["unresolved_unsafe_face_count"]),
      "unsafe_face_support_mode_histogram":{"|".join(k):v for k,v in sorted(modes.items())},
      "strongest_edge_identity_resolved_count":identity_resolved,
      "strongest_edge_owner_resolved_count":owner_resolved,
      "strongest_edge_owner_unresolved_count":unresolved_owner,
      "owner_resolved_unique_pair_count":len(owner_pair_set),
      "finding":{
        "seam_owner_support_recovers_additional_unsafe_faces":owner_resolved>identity_resolved,
        "current_identity_only_directive_leaves_repairable_seam_owner_evidence":int(directive["unresolved_unsafe_face_count"])>unresolved_owner,
      },
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("SEAM_OWNER_AUDIT="+json.dumps(report,sort_keys=True))

if __name__=="__main__":main()
