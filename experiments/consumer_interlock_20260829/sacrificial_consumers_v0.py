from __future__ import annotations
import numpy as np


def _grid_positions(vol):
    R=vol.resolution; b0=np.asarray(vol.bounds_min,np.float32); b1=np.asarray(vol.bounds_max,np.float32)
    ax=[np.linspace(b0[i],b1[i],R,dtype=np.float32) for i in range(3)]
    X,Y,Z=np.meshgrid(*ax,indexing='ij'); return np.stack([X,Y,Z],-1)

def build_g0_proposal(surface, vol, interior, types_module, max_leaf_axes:int=3):
    T=types_module
    posgrid=_grid_positions(vol)
    valid=vol.high_confidence_interior
    if valid.sum()<8: valid=vol.strict_hull
    pts=posgrid[valid]; w=interior.distance_to_strict_boundary[valid]
    if len(pts)<8: raise RuntimeError('insufficient interior for G0')
    # Root = thickest strict-interior voxel, tie broken lexicographically by flat index.
    flat=np.flatnonzero(valid); vals=interior.distance_to_strict_boundary.reshape(-1)[flat]; root_flat=int(flat[int(np.argmax(vals))]); root=np.asarray(posgrid.reshape(-1,3)[root_flat],np.float32)
    # PCA over strict interior, weighted by local radius; axis sign deterministic.
    ww=np.maximum(w,1e-6); ctr=np.average(pts,axis=0,weights=ww); X=pts-ctr; cov=(X*ww[:,None]).T@X/ww.sum(); evals,evecs=np.linalg.eigh(cov); order=np.argsort(evals)[::-1]; axes=evecs[:,order]
    for k in range(axes.shape[1]):
        j=int(np.argmax(np.abs(axes[:,k])))
        if axes[j,k]<0: axes[:,k]*=-1
    # Surface support lookup.
    SP=np.asarray([n.P for n in surface.surface_nodes],np.float32); SIDS=[n.surface_id for n in surface.surface_nodes]
    def supports(p,k=6):
        d=np.linalg.norm(SP-p[None],axis=1); ids=np.argsort(d)[:min(k,len(d))]; return tuple(SIDS[int(i)] for i in ids)
    joints=[]; edges=[]
    joints.append(T.SkeletonProposalJoint('G0_ROOT',tuple(map(float,root)),root_score=1.0,confidence=1.0,support_surface_ids=supports(root),metadata={'generator':'INTERIOR_PCA_V0','role':'ROOT'}))
    # For each PCA axis choose positive/negative high-medial endpoint; add mid joint when arm is long.
    jid=1
    for aidx in range(min(max_leaf_axes,3)):
        axis=axes[:,aidx]; proj=(pts-root)@axis
        for sign in (-1,1):
            score=sign*proj + 0.35*(w/max(float(w.max()),1e-8))
            q=int(np.argmax(score)); end=np.asarray(pts[q],np.float32); length=float(np.linalg.norm(end-root))
            if length < 2.5*vol.voxel_size: continue
            mid=root+0.5*(end-root)
            mid_id=f'G0_J{jid:02d}'; leaf_id=f'G0_J{jid+1:02d}'; jid+=2
            conf=float(min(0.95,0.55+0.4*w[q]/max(float(w.max()),1e-8)))
            joints.append(T.SkeletonProposalJoint(mid_id,tuple(map(float,mid)),root_score=0.05,confidence=conf,support_surface_ids=supports(mid),metadata={'generator':'INTERIOR_PCA_V0','axis':aidx,'sign':sign,'role':'MID'}))
            joints.append(T.SkeletonProposalJoint(leaf_id,tuple(map(float,end)),root_score=0.01,confidence=conf,support_surface_ids=supports(end),metadata={'generator':'INTERIOR_PCA_V0','axis':aidx,'sign':sign,'role':'LEAF'}))
            edges.append(T.SkeletonProposalEdge(f'G0_E_ROOT_{mid_id}','G0_ROOT',mid_id,score=0.95,confidence=0.95,hard_required=True,reason='sacrificial_interior_axis'))
            edges.append(T.SkeletonProposalEdge(f'G0_E_{mid_id}_{leaf_id}',mid_id,leaf_id,score=0.9,confidence=0.9,hard_required=True,reason='sacrificial_interior_axis'))
    if len(joints)<3: raise RuntimeError('G0 produced too few joints')
    return T.SkeletonProposalIR(tuple(joints),tuple(edges),surface.geometry_lineage_hash,model_provenance='SACRIFICIAL_G0_INTERIOR_PCA_V0',metadata={'volume_lineage':vol.lineage_sha256,'interior_lineage':interior.lineage_sha256,'canonical_authority':False})

def build_a0_proposal(surface, skeleton, types_module, sigma_scale:float=0.35, topk:int=4):
    T=types_module; JP=np.asarray([j.position for j in skeleton.joints],np.float32); JID=[j.canonical_joint_id for j in skeleton.joints]
    SP=np.asarray([n.P for n in surface.surface_nodes],np.float32); diag=np.linalg.norm(SP.max(0)-SP.min(0)); sigma=max(diag*sigma_scale,1e-5)
    inf=[]
    for n,p in zip(surface.surface_nodes,SP):
        d=np.linalg.norm(JP-p[None],axis=1); idx=np.argsort(d)[:min(topk,len(d))]; logits=np.exp(-0.5*(d[idx]/sigma)**2); logits=np.maximum(logits,1e-12); ww=logits/logits.sum()
        for i,w in zip(idx,ww): inf.append(T.SkinInfluenceProposal(n.surface_id,JID[int(i)],float(w)))
    return T.SkinProposalIR(tuple(inf),surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,model_provenance='SACRIFICIAL_A0_DISTANCE_SOFTMAX_V0',metadata={'sigma_scale':sigma_scale,'topk':topk,'canonical_authority':False})
