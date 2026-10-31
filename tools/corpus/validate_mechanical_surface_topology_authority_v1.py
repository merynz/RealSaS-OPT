#!/usr/bin/env python3
from __future__ import annotations

"""Independent court for modular mechanical surface topology authority.

This module intentionally does not import the producer. It reconstructs the raw
owner-preserving quotient, boundary-edge evidence, trajectory-qualified seam unions,
face validity and edge-connected face partition from source tensors.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

PARENT_SCHEMA = "RealSaS.MechanicalSurfaceTopologyAuthority.v1"
PARENT_STATUS = "PASS_MECHANICAL_SURFACE_TOPOLOGY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.MechanicalSurfaceTopologyAuthorityCourt.v1"
STATUS = "PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(p):
    p = np.asarray(p, dtype=np.float64).copy(); p[p == 0.0] = 0.0
    return p.tobytes(order="C")


def _partition(labels, members):
    groups = {}
    for member, label in zip(members, labels):
        groups.setdefault(int(label), []).append(int(member))
    return {frozenset(v) for v in groups.values()}


def _face_partition(faces):
    edge_to_faces = {}
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b)
            edge_to_faces.setdefault((min(a, b), max(a, b)), []).append(fi)
    graph = [set() for _ in range(len(faces))]
    for ids in edge_to_faces.values():
        ids = sorted(set(map(int, ids)))
        for i, a in enumerate(ids):
            for b in ids[i + 1:]: graph[a].add(b); graph[b].add(a)
    labels = np.full(len(faces), -1, np.int32); cid = 0
    for start in range(len(faces)):
        if labels[start] >= 0: continue
        labels[start] = cid; stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in graph[cur]:
                if labels[nxt] < 0: labels[nxt] = cid; stack.append(nxt)
        cid += 1
    return labels


def _double_area(v, f):
    tri = v[f]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _expected(rest, faces, owner, posed):
    raw_owner = np.full(len(rest), -1, np.int32)
    for fi, tri in enumerate(faces):
        for raw in tri:
            raw = int(raw); oid = int(owner[fi])
            if raw_owner[raw] == -1: raw_owner[raw] = oid
            elif int(raw_owner[raw]) != oid: raise ValueError("MECHANICAL_COURT_RAW_OWNER_CROSS")

    referenced = np.unique(faces)
    token_to_q = {}; q_members = []; q_pos = []; q_owner = []; raw_to_q = np.full(len(rest), -1, np.int32)
    for raw in referenced:
        raw = int(raw); token = (int(raw_owner[raw]), _key(rest[raw]))
        if token not in token_to_q:
            token_to_q[token] = len(q_pos); q_pos.append(rest[raw].copy()); q_owner.append(int(raw_owner[raw])); q_members.append([])
        qid = token_to_q[token]; raw_to_q[raw] = qid; q_members[qid].append(raw)
    q_pos = np.asarray(q_pos, np.float64); qf = raw_to_q[faces]

    edge_rows = {}
    for fi, tri in enumerate(qf):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a), int(b); edge_rows.setdefault((min(a,b),max(a,b)), []).append((fi,(a,b)))
    boundary = []
    for rows in edge_rows.values():
        if len(rows) == 1:
            fi, (a,b) = rows[0]
            boundary.append({"fi":int(fi),"owner":int(owner[fi]),"a":a,"b":b,"ka":_key(q_pos[a]),"kb":_key(q_pos[b])})
    groups = {}
    for row in boundary: groups.setdefault(tuple(sorted((row["ka"],row["kb"]))), []).append(row)

    parent = np.arange(len(q_pos), dtype=np.int32)
    def find(x):
        x=int(x)
        while int(parent[x]) != x: parent[x]=parent[int(parent[x])]; x=int(parent[x])
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)] = min(a,b)
    scale=max(float(np.linalg.norm(rest.max(0)-rest.min(0))),1e-12)
    seam_count=0
    for rows in groups.values():
        if len(rows)!=2: continue
        l,r=rows
        if l["owner"]==r["owner"]: continue
        if not (l["ka"]==r["kb"] and l["kb"]==r["ka"]): continue
        pairs=((l["a"],r["b"]),(l["b"],r["a"]))
        ok=True
        for qa,qb in pairs:
            raw=sorted(set(q_members[qa])|set(q_members[qb])); p=posed[:,raw,:]
            drift=float(np.max(np.linalg.norm(p-p[:,:1,:],axis=-1),initial=0.0))/scale
            if drift>1e-9: ok=False; break
        if not ok: continue
        for qa,qb in pairs: union(qa,qb)
        seam_count += 1

    roots=[find(i) for i in range(len(q_pos))]; lut={r:i for i,r in enumerate(sorted(set(roots)))}
    q_to_m=np.asarray([lut[r] for r in roots],np.int32); raw_to_m=np.full(len(rest),-1,np.int32)
    raw_to_m[referenced]=q_to_m[raw_to_q[referenced]]
    mv=np.asarray([q_pos[r] for r in sorted(set(roots))],np.float64); mf=raw_to_m[faces]
    raw_area=_double_area(rest,faces); mech_area=_double_area(mv,mf)
    collapsed=(mf[:,0]==mf[:,1])|(mf[:,1]==mf[:,2])|(mf[:,2]==mf[:,0])
    valid=np.isfinite(raw_area)&(raw_area>1e-12)&np.isfinite(mech_area)&(mech_area>1e-12)&~collapsed
    components=_face_partition(mf)
    return referenced, raw_to_m, mv, mf.astype(np.int32), valid.astype(np.bool_), components, seam_count, raw_owner


def validate(manifest_path: Path) -> dict:
    manifest_path=manifest_path.resolve(); manifest=json.loads(manifest_path.read_text())
    if manifest.get("schema")!=PARENT_SCHEMA or manifest.get("status")!=PARENT_STATUS: raise ValueError("MECHANICAL_COURT_PARENT")
    source=Path(manifest["source"]["geometry_and_response"]["path"]).resolve()
    if not source.is_file() or sha256(source)!=manifest["source"]["geometry_and_response"]["sha256"]: raise ValueError("MECHANICAL_COURT_SOURCE_DRIFT")
    output=Path(manifest["output"]["path"]).resolve()
    if not output.is_file() or sha256(output)!=manifest["output"]["sha256"]: raise ValueError("MECHANICAL_COURT_OUTPUT_DRIFT")
    with np.load(source,allow_pickle=False) as z:
        rest=np.asarray(z["rest_vertices_source_world"],np.float64); faces=np.asarray(z["faces"],np.int64); owner=np.asarray(z["face_owner_privileged"],np.int32); posed=np.asarray(z["isolated_position"],np.float64)
    referenced, expected_map, expected_v, expected_f, expected_valid, expected_component, seam_count, raw_owner = _expected(rest,faces,owner,posed)
    with np.load(output,allow_pickle=False) as z:
        required={"raw_to_mechanical_vertex","mechanical_vertices_source_world","mechanical_faces","face_owner","face_valid","surface_component_id","mechanical_vertex_owner_offsets","mechanical_vertex_owner_values"}
        if not required.issubset(z.files): raise ValueError("MECHANICAL_COURT_OUTPUT_ARRAYS")
        mapping=np.asarray(z["raw_to_mechanical_vertex"],np.int32); mv=np.asarray(z["mechanical_vertices_source_world"],np.float64); mf=np.asarray(z["mechanical_faces"],np.int32); out_owner=np.asarray(z["face_owner"],np.int32); valid=np.asarray(z["face_valid"],bool); component=np.asarray(z["surface_component_id"],np.int32); offsets=np.asarray(z["mechanical_vertex_owner_offsets"],np.int32); values=np.asarray(z["mechanical_vertex_owner_values"],np.int32)
    if mapping.shape!=(len(rest),) or not np.array_equal(mapping<0, expected_map<0): raise ValueError("MECHANICAL_COURT_RAW_MAP_LAYOUT")
    if _partition(mapping[referenced],referenced)!=_partition(expected_map[referenced],referenced): raise ValueError("MECHANICAL_COURT_VERTEX_PARTITION")
    if not np.array_equal(mf,mapping[faces]): raise ValueError("MECHANICAL_COURT_FACE_REMAP")
    if not np.array_equal(out_owner,owner): raise ValueError("MECHANICAL_COURT_FACE_OWNER")
    if not np.array_equal(valid,expected_valid): raise ValueError("MECHANICAL_COURT_FACE_VALIDITY")
    if _partition(component,np.arange(len(faces)))!=_partition(expected_component,np.arange(len(faces))): raise ValueError("MECHANICAL_COURT_SURFACE_COMPONENT_PARTITION")
    if int(manifest["counts"]["qualified_cross_owner_seam_edge_count"])!=seam_count: raise ValueError("MECHANICAL_COURT_SEAM_COUNT")
    if offsets.shape!=(len(mv)+1,) or offsets[0]!=0 or offsets[-1]!=len(values) or np.any(np.diff(offsets)<0): raise ValueError("MECHANICAL_COURT_OWNER_CSR")
    for mid in range(len(mv)):
        raws=referenced[mapping[referenced]==mid]
        expected_owners=sorted(set(map(int,raw_owner[raws].tolist())))
        actual=values[offsets[mid]:offsets[mid+1]].tolist()
        if actual!=expected_owners: raise ValueError("MECHANICAL_COURT_OWNER_MEMBERSHIP")
    return {
        "schema":SCHEMA,"status":STATUS,"subject_id":manifest["subject_id"],
        "counts":{"mechanical_vertex_count":int(len(mv)),"surface_component_count":int(len(np.unique(component))),"qualified_cross_owner_seam_edge_count":int(seam_count)},
        "family_candidate_outcomes":{"SURFACE_TOPOLOGY":"QUALIFIED_CORE_CANDIDATE"},
        "authority":{"independent_court_passed":True,"final_family_authority_minted":False,"training_authorized":False},
        "parent_manifest":{"path":str(manifest_path),"sha256":sha256(manifest_path)},
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    result=validate(a.manifest); a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(result,sort_keys=True,indent=2)+"\n"); print(json.dumps({"status":result["status"],"subject_id":result["subject_id"],"counts":result["counts"]},sort_keys=True))


if __name__=="__main__": main()
