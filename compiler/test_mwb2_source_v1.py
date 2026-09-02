from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation, QualifiedJoint, QualifiedSkeletonIR, QualifiedSkinRow, QualifiedSkinIR, QualificationError
from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh
from compiler.realsas_compiler_core.mwb2_skin import bind_mwb2_mesh_skin
import pytest

def surface():
    pts={'a':(-.4,-.4,0.),'b':(.4,-.4,0.),'c':(.4,.4,0.),'d':(-.4,.4,0.)}
    nodes=tuple(SurfaceNode(s,p,(0,),('p',),('o',),((0,(p[0],p[1])),),f'pg-{s}',derived_normal=(0,0,1)) for s,p in pts.items())
    edges=[('a','b'),('b','c'),('a','c'),('a','d'),('c','d')]
    rel=tuple(SurfaceRelation(f'r{i}',a,b,'LOCAL_NEIGHBOR',1.,{}) for i,(a,b) in enumerate(edges))
    return RiggingSurfaceIR(nodes,rel,'S-HASH')

def skeleton_skin(s):
    sk=QualifiedSkeletonIR((QualifiedJoint('J:0',(0,0,0),None),QualifiedJoint('J:1',(0.5,0,0),'J:0')), 'J:0', {'status':'PASS'}, 'G-HASH')
    rows=tuple(QualifiedSkinRow(n.surface_id,(('J:0',0.75),('J:1',0.25)),0.,0.) for n in s.surface_nodes)
    return sk,QualifiedSkinIR(rows,s.geometry_lineage_hash,sk.skeleton_lineage_hash,{'status':'PASS'},'W-HASH')

def test_mwb2_full_qualification_and_skin_lineage():
    s=surface(); cand=build_mwb2_candidate(s,view_index=0,camera_binding_hash='CAM'); mesh=qualify_mwb2_mesh(s,cand); sk,w=skeleton_skin(s); b=bind_mwb2_mesh_skin(s,sk,w,mesh)
    assert len(mesh.faces)==2 and b.mesh_binding_hash==mesh.mesh_lineage_hash and b.skin_binding_hash==w.skin_lineage_hash
    assert all(abs(sum(x[1] for x in r.influences)-1.)<1e-9 for r in b.rows)

def test_unknown_bridge_is_rejected():
    s=surface(); rel=list(s.local_relations); rel[2]=SurfaceRelation('bad','a','c','UNKNOWN_GAP',1.,{'crosses_unknown':True}); bad=RiggingSurfaceIR(s.surface_nodes,tuple(rel),s.geometry_lineage_hash)
    with pytest.raises(QualificationError): build_mwb2_candidate(bad,view_index=0,camera_binding_hash='CAM')
