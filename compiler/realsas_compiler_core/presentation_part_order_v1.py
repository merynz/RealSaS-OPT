"""Explicit whole-part overlap ownership on the existing canonical M.

Connected M components supply bounded part evidence; one attachment remains one
part even when it contains several disconnected material pieces. Per-view depth
orders parts, then canonical geometric depth orders triangles inside a part.
This is an editable presentation policy, not a semantic segmentation proof.
"""
import numpy as np


def compile_part_ids(vertex_count, faces, attachment_owners, weights=None):
    faces=np.asarray(faces,dtype=np.int64)
    owners=np.asarray(attachment_owners,dtype=np.int32)
    parent=np.arange(vertex_count,dtype=np.int64)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for face in faces:
        roots=[find(int(i)) for i in face]
        for root in roots[1:]:parent[root]=roots[0]
    roots=np.array([find(i) for i in range(vertex_count)])
    component_owner={}
    if weights is not None:
        W=np.asarray(weights,dtype=float)
        if W.shape[0]!=vertex_count or not np.isfinite(W).all():
            raise ValueError("PART_SKIN_DOMAIN_INVALID")
        for component in np.unique(roots):
            ids=(roots==component)&(owners==0)
            if np.any(ids):component_owner[int(component)]=int(W[ids].sum(axis=0).argmax())
    keys=[("attachment",int(owners[i])) if owners[i]>0 else
          (("skin_region",component_owner[int(roots[i])]) if weights is not None else ("component",int(roots[i])))
          for i in range(vertex_count)]
    ordered=sorted(set(keys))
    lookup={key:i for i,key in enumerate(ordered)}
    vertex_part=np.array([lookup[k] for k in keys],dtype=np.int32)
    if not np.all(vertex_part[faces]==vertex_part[faces][:,:1]):
        raise ValueError("PART_FACE_CROSSES_OWNERSHIP_BOUNDARY")
    return vertex_part,vertex_part[faces[:,0]],ordered


def part_depth_order(poses, vertex_part, camera):
    parts=np.unique(vertex_part)
    output=[]
    for pose in poses:
        centers=np.array([np.mean(pose[vertex_part==p],axis=0) for p in parts])
        depth=(centers-np.asarray(camera.origin))@np.asarray(camera.forward)
        # Stable tie priority is explicit in the ordered part IDs. It is never
        # advertised as artist-qualified semantics or independent depth proof.
        ordered=np.lexsort((parts,depth))
        rank=np.empty(len(parts),dtype=np.int32);rank[ordered]=np.arange(len(parts))
        output.append(rank)
    return np.array(output)
