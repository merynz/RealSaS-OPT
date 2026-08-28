import bpy, sys, json, math, os
from pathlib import Path
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv=sys.argv[sys.argv.index('--')+1:]
src=Path(argv[0]); out=Path(argv[1])

MAX_TRI_SAMPLES=1024
SAMPLE_BARY=np.asarray([
    [1/3,1/3,1/3],
    [0.5,0.5,0.0],
    [0.5,0.0,0.5],
    [0.0,0.5,0.5],
],dtype=np.float64)

def import_source(src):
    ext=src.suffix.lower()
    if ext=='.blend':
        bpy.ops.wm.open_mainfile(filepath=str(src))
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if ext in {'.glb','.gltf'}:
            bpy.ops.import_scene.gltf(filepath=str(src))
        elif ext=='.fbx':
            try: bpy.ops.wm.fbx_import(filepath=str(src))
            except Exception: bpy.ops.import_scene.fbx(filepath=str(src))
        elif ext in {'.usd','.usda','.usdc','.usdz'}:
            bpy.ops.wm.usd_import(filepath=str(src))
        elif ext=='.dae':
            bpy.ops.wm.collada_import(filepath=str(src))
        elif ext=='.obj':
            try: bpy.ops.wm.obj_import(filepath=str(src))
            except Exception: bpy.ops.import_scene.obj(filepath=str(src))
        else:
            raise RuntimeError('unsupported extension '+ext)

def world_vertices(obj, me):
    M=np.asarray(obj.matrix_world,dtype=np.float64)
    if len(me.vertices)==0: return np.zeros((0,3),np.float64)
    V=np.array([list(v.co)+[1.0] for v in me.vertices],dtype=np.float64)
    return (V@M.T)[:,:3]

def fan_faces(me):
    fs=[]
    for p in me.polygons:
        vs=list(p.vertices)
        for k in range(1,len(vs)-1): fs.append((vs[0],vs[k],vs[k+1]))
    return np.asarray(fs,dtype=np.int64).reshape((-1,3)) if fs else np.zeros((0,3),np.int64)

def looptri_faces(me):
    try: me.calc_loop_triangles()
    except Exception: pass
    return np.asarray([list(t.vertices) for t in me.loop_triangles],dtype=np.int64).reshape((-1,3)) if len(me.loop_triangles) else np.zeros((0,3),np.int64)

def bbox_diag(V):
    if len(V)==0: return 0.0
    return float(np.linalg.norm(np.max(V,axis=0)-np.min(V,axis=0)))

def sample_surface_points(V,F,max_tri=MAX_TRI_SAMPLES):
    if len(F)==0 or len(V)==0: return np.zeros((0,3),np.float64)
    if len(F)>max_tri:
        idx=np.linspace(0,len(F)-1,max_tri,dtype=np.int64)
        F=F[idx]
    tri=V[F]
    pts=np.einsum('bc,tcd->tbd',SAMPLE_BARY,tri).reshape((-1,3))
    return pts

def bvh_from(V,F):
    if len(V)==0 or len(F)==0: return None
    verts=[Vector((float(x),float(y),float(z))) for x,y,z in V]
    polys=[tuple(map(int,t)) for t in F]
    try: return BVHTree.FromPolygons(verts,polys,all_triangles=True)
    except Exception: return None

def nearest_distances(points,bvh):
    if bvh is None or len(points)==0: return np.zeros((0,),np.float64)
    ds=[]
    for p in points:
        hit=bvh.find_nearest(Vector((float(p[0]),float(p[1]),float(p[2]))))
        if hit is not None and hit[0] is not None and hit[3] is not None:
            ds.append(float(hit[3]))
    return np.asarray(ds,dtype=np.float64)

def bidirectional_surface(Va,Fa,Vb,Fb,scale):
    if len(Fa)==0 or len(Fb)==0 or scale<=1e-12: return None
    ba=bvh_from(Va,Fa); bb=bvh_from(Vb,Fb)
    if ba is None or bb is None: return None
    pa=sample_surface_points(Va,Fa); pb=sample_surface_points(Vb,Fb)
    da=nearest_distances(pa,bb); db=nearest_distances(pb,ba)
    if len(da)==0 and len(db)==0: return None
    ds=np.concatenate([x for x in [da,db] if len(x)])
    return {
        'count':int(len(ds)),
        'p50':float(np.quantile(ds,.50)),
        'p95':float(np.quantile(ds,.95)),
        'p99':float(np.quantile(ds,.99)),
        'max':float(np.max(ds)),
        'p50_rel':float(np.quantile(ds,.50)/scale),
        'p95_rel':float(np.quantile(ds,.95)/scale),
        'p99_rel':float(np.quantile(ds,.99)/scale),
        'max_rel':float(np.max(ds)/scale),
    }

def active_shape_keys(me):
    sk=getattr(me,'shape_keys',None)
    if sk is None: return []
    rows=[]
    for i,kb in enumerate(sk.key_blocks):
        if i==0: continue
        val=float(kb.value)
        if abs(val)>1e-8: rows.append({'name':kb.name,'value':val})
    return rows

def object_probe(obj,deps):
    me=obj.data
    V=world_vertices(obj,me); Ffan=fan_faces(me); Floop=looptri_faces(me)
    scale=max(bbox_diag(V),1e-12)
    tri_surface=bidirectional_surface(V,Ffan,V,Floop,scale)
    ev=obj.evaluated_get(deps); eme=ev.to_mesh()
    try:
        EV=world_vertices(ev,eme); EF=looptri_faces(eme)
        escale=max(scale,bbox_diag(EV),1e-12)
        same_counts=(len(V)==len(EV) and len(Floop)==len(EF))
        delta=None
        if len(V)==len(EV):
            delta=float(np.max(np.linalg.norm(V-EV,axis=1))) if len(V) else 0.0
        eval_surface=bidirectional_surface(V,Floop,EV,EF,escale)
    finally:
        ev.to_mesh_clear()
    mods=[]
    for m in obj.modifiers:
        mods.append({'type':m.type,'name':m.name,'show_viewport':bool(getattr(m,'show_viewport',True)),'show_render':bool(getattr(m,'show_render',True))})
    return {
        'name':obj.name,
        'vertices':int(len(V)),
        'fan_triangles':int(len(Ffan)),
        'blender_loop_triangles':int(len(Floop)),
        'bbox_diag':float(scale),
        'triangulation_surface':tri_surface,
        'modifiers':mods,
        'non_armature_modifier_types':sorted({m['type'] for m in mods if m['type']!='ARMATURE' and m['show_viewport']}),
        'active_shape_keys':active_shape_keys(me),
        'evaluated':{
            'vertices':int(len(EV)),
            'triangles':int(len(EF)),
            'same_counts':bool(same_counts),
            'indexwise_world_vertex_max_delta':delta,
            'indexwise_world_vertex_max_delta_rel':None if delta is None else float(delta/escale),
            'surface':eval_surface,
        }
    }

res={'source':str(src),'extension':src.suffix.lower(),'errors':[]}
try:
    import_source(src)
    for o in bpy.data.objects:
        if o.type=='ARMATURE':
            try: o.data.pose_position='REST'
            except Exception: pass
    bpy.context.view_layer.update(); deps=bpy.context.evaluated_depsgraph_get()
    objs=[]
    for o in bpy.context.scene.objects:
        if o.type=='MESH' and len(o.data.vertices)>0:
            try: objs.append(object_probe(o,deps))
            except Exception as e: res['errors'].append(f'object {o.name}: {type(e).__name__}: {e}')
    res['mesh_objects']=objs; res['mesh_object_count']=len(objs)
    res['asset_max_eval_surface_p95_rel']=max([((x.get('evaluated') or {}).get('surface') or {}).get('p95_rel',0.0) for x in objs] or [0.0])
    res['asset_max_eval_surface_max_rel']=max([((x.get('evaluated') or {}).get('surface') or {}).get('max_rel',0.0) for x in objs] or [0.0])
    res['asset_eval_topology_change_object_count']=sum(not bool((x.get('evaluated') or {}).get('same_counts',True)) for x in objs)
    res['asset_max_triang_surface_p95_rel']=max([((x.get('triangulation_surface') or {}).get('p95_rel',0.0)) for x in objs] or [0.0])
    res['asset_max_triang_surface_max_rel']=max([((x.get('triangulation_surface') or {}).get('max_rel',0.0)) for x in objs] or [0.0])
    res['asset_non_armature_modifier_object_count']=sum(bool(x.get('non_armature_modifier_types')) for x in objs)
    res['asset_active_shape_key_object_count']=sum(bool(x.get('active_shape_keys')) for x in objs)
except Exception as e:
    res['errors'].append(f'{type(e).__name__}: {e}')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(res,indent=2),encoding='utf-8')
