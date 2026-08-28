import bpy, sys, json
from pathlib import Path
import numpy as np

argv=sys.argv[sys.argv.index('--')+1:]
src=Path(argv[0]); out_npz=Path(argv[1]); out_json=Path(argv[2])

bpy.ops.wm.open_mainfile(filepath=str(src))

# Canonical repair policy: armatures in REST; active non-Basis shape-key assets are
# excluded by the driver and must never reach this extractor.
for a in bpy.context.scene.objects:
    if a.type=='ARMATURE':
        try: a.data.pose_position='REST'
        except Exception: pass
bpy.context.view_layer.update()

meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and len(o.data.vertices)>0 and not getattr(o,'hide_render',False)]
if not meshes: raise RuntimeError('No mesh objects found')

# Fail closed if any visible mesh has an active non-Basis key; Stage-B5 quarantines these.
active=[]
for o in meshes:
    sk=getattr(o.data,'shape_keys',None)
    if sk:
        for i,k in enumerate(sk.key_blocks):
            if i>0 and abs(float(k.value))>1e-8:
                active.append({'object':o.name,'key':k.name,'value':float(k.value)})
if active:
    raise RuntimeError('active_nonbasis_shape_keys_forbidden_in_auto_repair: '+json.dumps(active[:20]))

# Keep the same primary-armature authority as V4.3: choose the armature influencing
# the most BASE mesh vertices. This avoids silently changing rig identity during repair.
arm_scores={}
for o in meshes:
    arms=[]
    for m in o.modifiers:
        if m.type=='ARMATURE' and getattr(m,'object',None): arms.append(m.object)
    if o.parent and o.parent.type=='ARMATURE': arms.append(o.parent)
    for a in set(arms): arm_scores[a]=arm_scores.get(a,0)+len(o.data.vertices)
arm=max(arm_scores,key=arm_scores.get) if arm_scores else None
bones=list(arm.data.bones) if arm else []
bone_index={b.name:i for i,b in enumerate(bones)}

deps=bpy.context.evaluated_depsgraph_get()
V=[]; F=[]; skin_rows=[]; object_ranges=[]; offset=0
for o in meshes:
    ev=o.evaluated_get(deps)
    eme=ev.to_mesh(preserve_all_data_layers=True,depsgraph=deps)
    try:
        if len(eme.vertices)==0: continue
        M=ev.matrix_world
        verts=np.array([tuple(M @ v.co) for v in eme.vertices],dtype=np.float32)
        eme.calc_loop_triangles()
        faces=np.asarray([[int(i)+offset for i in t.vertices] for t in eme.loop_triangles],dtype=np.int32)
        if faces.size==0: continue

        W=np.zeros((len(verts),len(bones)),dtype=np.float32)
        group_elements=0
        if arm and bones:
            vg_to_bone={g.index:bone_index[g.name] for g in ev.vertex_groups if g.name in bone_index}
            for vi,v in enumerate(eme.vertices):
                for g in v.groups:
                    group_elements += 1
                    bi=vg_to_bone.get(int(g.group))
                    if bi is not None:
                        w=float(g.weight)
                        if np.isfinite(w) and w>=0: W[vi,bi]+=w
            # Same V4.3 rigid bone-parented fallback.
            rigid=(o.parent_bone if o.parent==arm and getattr(o,'parent_type','')=='BONE' and getattr(o,'parent_bone','') in bone_index else None)
            if rigid is not None:
                zero=W.sum(1)<=1e-8
                W[zero,bone_index[rigid]]=1.0
            rs=W.sum(1,keepdims=True)
            nz=rs[:,0]>1e-8
            W[nz]/=rs[nz]

        vstart=offset
        V.append(verts); F.append(faces); skin_rows.append(W)
        object_ranges.append({
            'name':o.name,
            'vertex_start':int(vstart),'vertex_count':int(len(verts)),
            'face_count':int(len(faces)),
            'base_vertices':int(len(o.data.vertices)),
            'evaluated_vertices':int(len(eme.vertices)),
            'topology_changed':bool(len(o.data.vertices)!=len(eme.vertices)),
            'non_armature_modifiers':sorted({m.type for m in o.modifiers if m.type!='ARMATURE' and bool(getattr(m,'show_viewport',True))}),
            'evaluated_vertex_group_element_count':int(group_elements),
            'zero_weight_fraction':None if not bones else float(np.mean(W.sum(1)<=1e-8)),
        })
        offset += len(verts)
    finally:
        ev.to_mesh_clear()

if not V: raise RuntimeError('No evaluated geometry')
V=np.concatenate(V,axis=0).astype(np.float32)
F=np.concatenate(F,axis=0).astype(np.int32)
W=np.concatenate(skin_rows,axis=0).astype(np.float32) if bones else None

if arm and bones:
    A=arm.matrix_world
    heads=np.array([tuple(A @ b.head_local) for b in bones],dtype=np.float32)
    tails=np.array([tuple(A @ b.tail_local) for b in bones],dtype=np.float32)
    parents=np.array([bone_index[b.parent.name] if b.parent else -1 for b in bones],dtype=np.int32)
    deform=np.array([1 if b.use_deform else 0 for b in bones],dtype=np.uint8)
    rest_local=np.stack([np.asarray(b.matrix_local,dtype=np.float32) for b in bones])
    rest_world=np.stack([np.asarray(A @ b.matrix_local,dtype=np.float32) for b in bones])
    names=[b.name for b in bones]
else:
    heads=tails=parents=deform=rest_local=rest_world=None; names=[]

out_npz.parent.mkdir(parents=True,exist_ok=True)
arr={'vertices_source':V,'faces':F}
if heads is not None:
    arr.update(bone_heads_source=heads,bone_tails_source=tails,parents=parents,deform_mask=deform,
               rest_local_source=rest_local,rest_world_source=rest_world,skin=W)
np.savez_compressed(out_npz,**arr)
out_json.write_text(json.dumps({
    'schema':'RealSaS.BlenderEvaluatedExtract.v1','source':str(src),
    'primary_armature':arm.name if arm else None,'armature_scores':{a.name:int(v) for a,v in arm_scores.items()},
    'bone_names':names,'bone_count':len(names),'mesh_objects':object_ranges,
    'active_nonbasis_shape_keys':active,
},indent=2),encoding='utf-8')
