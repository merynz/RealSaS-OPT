from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from compiler.realsas_compiler_core.substrate.scene_first_signed import _adaptive_voxel_compact

OUT=Path(os.environ.get("REALSAS_STAGE14_CA_OUT","stage14_ca_out")); OUT.mkdir(parents=True,exist_ok=True)

def ell(a,b,c,nlat,nlon,center):
    cx,cy,cz=center; v=[(cx,cy,cz+c)]; n=[(0,0,1)]
    for i in range(1,nlat):
        t=math.pi*i/nlat; st,ct=math.sin(t),math.cos(t)
        for j in range(nlon):
            p=2*math.pi*j/nlon; cp,sp=math.cos(p),math.sin(p)
            x,y,z=a*st*cp,b*st*sp,c*ct
            v.append((cx+x,cy+y,cz+z))
            g=np.asarray([x/(a*a),y/(b*b),z/(c*c)],float);g/=np.linalg.norm(g);n.append(tuple(g))
    bot=len(v);v.append((cx,cy,cz-c));n.append((0,0,-1));f=[]
    for j in range(nlon):f.append((0,1+j,1+(j+1)%nlon))
    for i in range(nlat-2):
        r0=1+i*nlon;r1=r0+nlon
        for j in range(nlon):
            a0=r0+j;a1=r0+(j+1)%nlon;b0=r1+j;b1=r1+(j+1)%nlon
            f.extend(((a0,b0,b1),(a0,b1,a1)))
    last=1+(nlat-2)*nlon
    for j in range(nlon):f.append((bot,last+(j+1)%nlon,last+j))
    return np.asarray(v,np.float64),np.asarray(f,np.int64),np.asarray(n,np.float64)

def dense():
    a=ell(.31,.38,.52,64,96,(-.17,0,0));b=ell(.20,.24,.34,64,96,(.19,0,.03))
    v=np.concatenate((a[0],b[0]));f=np.concatenate((a[1],b[1]+len(a[0])));n=np.concatenate((a[2],b[2]))
    labels=np.concatenate((np.zeros(len(a[0]),np.int64),np.ones(len(b[0]),np.int64)))
    return v,f,n,labels

def component_compact(p,f,n,labels,target):
    lo=p.min(0);span=np.maximum(p.max(0)-lo,1e-12)
    def inv_for(div):
        xyz=np.floor((p-lo)/span*div).astype(np.int64);xyz=np.clip(xyz,0,div-1)
        keys=np.column_stack((labels,xyz))
        _,inv=np.unique(keys,axis=0,return_inverse=True)
        return inv
    low,high=1,512;best=None
    while low<=high:
        mid=(low+high)//2;inv=inv_for(mid);count=int(inv.max())+1
        if count<=target:best=(mid,inv);low=mid+1
        else:high=mid-1
    if best is None:raise RuntimeError("no component-aware compaction")
    div,inv=best;count=int(inv.max())+1;cnt=np.bincount(inv,minlength=count).astype(float)
    cp=np.zeros((count,3));cn=np.zeros((count,3));np.add.at(cp,inv,p);np.add.at(cn,inv,n);cp/=cnt[:,None]
    cn/=np.linalg.norm(cn,axis=1,keepdims=True).clip(min=1e-12)
    mapped=inv[f];edges=np.concatenate((mapped[:,[0,1]],mapped[:,[1,2]],mapped[:,[2,0]]));edges=np.sort(edges,axis=1);edges=np.unique(edges[edges[:,0]!=edges[:,1]],axis=0)
    return cp,cn,edges,div,inv

def mixed(inv,labels):
    owners={}
    for i,c in enumerate(inv):owners.setdefault(int(c),set()).add(int(labels[i]))
    return sum(len(x)>1 for x in owners.values())

def spatial(p,cp):
    d,_=cKDTree(cp).query(p,k=1,workers=-1)
    return {"p95":float(np.quantile(d,.95)),"max":float(np.max(d))}

def projected(p,cp):
    vals=[]
    for vi in range(8):
        yaw=math.radians(45*vi);r=np.array([-math.cos(yaw),math.sin(yaw),0.]);u=np.array([0,0,1.])
        a=np.column_stack((((p@r)+1)*512-.5, (-(p@u)+1)*512-.5))
        b=np.column_stack((((cp@r)+1)*512-.5, (-(cp@u)+1)*512-.5))
        d,_=cKDTree(b).query(a,k=1,workers=-1);vals.extend(d.tolist())
    vals=np.asarray(vals)
    return {"p95_px":float(np.quantile(vals,.95)),"max_px":float(np.max(vals))}

def main():
    p,f,n,labels=dense();target=8192
    bp,bn,be,bd,bi=_adaptive_voxel_compact(p,f,n,target_nodes=target)
    cp,cn,ce,cd,ci=component_compact(p,f,n,labels,target)
    out={
      "schema":"RealSaS.Stage14ComponentAwareCompactionCausalProbe.v1",
      "status":"PASS" if mixed(ci,labels)==0 else "FAIL",
      "subject_inputs_used":False,"knight_result_used":False,"mage_result_used":False,
      "single_changed_variable":"VOXEL_KEY_INCLUDES_DENSE_CONNECTED_COMPONENT_LABEL",
      "target_nodes":target,"dense_vertex_count":len(p),
      "baseline":{"actual_nodes":len(bp),"divisions":bd,"mixed_component_voxels":mixed(bi,labels),"spatial":spatial(p,bp),"projected":projected(p,bp)},
      "component_aware":{"actual_nodes":len(cp),"divisions":cd,"mixed_component_voxels":mixed(ci,labels),"spatial":spatial(p,cp),"projected":projected(p,cp)},
      "acceptance":{"mixed_component_voxels_must_be_zero":True,"node_count_increase_fraction_max":0.05}
    }
    out["node_count_increase_fraction"]=(len(cp)-len(bp))/max(len(bp),1)
    if out["node_count_increase_fraction"]>0.05:out["status"]="FAIL_NODE_COST"
    (OUT/"STAGE14_COMPONENT_AWARE_COMPACTION_CAUSAL_PROBE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return 0 if out["status"]=="PASS" else 2
if __name__=="__main__":raise SystemExit(main())
