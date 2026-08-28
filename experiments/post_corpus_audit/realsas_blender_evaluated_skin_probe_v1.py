import bpy, sys, json, math
from pathlib import Path
import numpy as np

argv=sys.argv[sys.argv.index('--')+1:]
src=Path(argv[0]); out=Path(argv[1])


def open_source(p):
    bpy.ops.wm.open_mainfile(filepath=str(p))


def active_shape_keys(me):
    sk=getattr(me,'shape_keys',None)
    if sk is None: return []
    rows=[]
    for i,k in enumerate(sk.key_blocks):
        if i==0: continue
        v=float(k.value)
        if abs(v)>1e-8: rows.append({'name':k.name,'value':v})
    return rows


def select_primary_armature(meshes):
    scores={}
    for o in meshes:
        arms=[]
        for m in o.modifiers:
            if m.type=='ARMATURE' and getattr(m,'object',None): arms.append(m.object)
        if o.parent and o.parent.type=='ARMATURE': arms.append(o.parent)
        for a in set(arms): scores[a]=scores.get(a,0)+len(o.data.vertices)
    return max(scores,key=scores.get) if scores else None, {a.name:int(v) for a,v in scores.items()}


def weight_matrix(vertices, groups, bone_index, rigid_bone=None):
    W=np.zeros((len(vertices),len(bone_index)),np.float64)
    gi_to_bone={g.index:bone_index[g.name] for g in groups if g.name in bone_index}
    group_elem_count=0
    bad_weight_count=0
    for vi,v in enumerate(vertices):
        for ge in v.groups:
            group_elem_count += 1
            bi=gi_to_bone.get(int(ge.group))
            if bi is None: continue
            w=float(ge.weight)
            if not np.isfinite(w) or w<0: bad_weight_count+=1; continue
            W[vi,bi]+=w
    if rigid_bone is not None and rigid_bone in bone_index:
        z=W.sum(1)<=1e-8
        W[z,bone_index[rigid_bone]]=1.0
    rs=W.sum(1)
    nz=rs>1e-8
    if np.any(nz): W[nz] /= rs[nz,None]
    return W, {'vertex_group_element_count':int(group_elem_count),'bad_weight_count':int(bad_weight_count)}


def stats(W):
    if W is None or W.shape[1]==0:
        return {'vertices':0 if W is None else int(W.shape[0]),'bones':0 if W is None else int(W.shape[1]),'zero_row_fraction':None,'nonzero_rows':0}
    rs=W.sum(1)
    return {
        'vertices':int(W.shape[0]),'bones':int(W.shape[1]),
        'zero_row_fraction':float(np.mean(rs<=1e-8)) if len(rs) else 0.0,
        'nonzero_rows':int(np.sum(rs>1e-8)),
        'row_sum_min_nonzero':float(np.min(rs[rs>1e-8])) if np.any(rs>1e-8) else None,
        'row_sum_max':float(np.max(rs)) if len(rs) else 0.0,
        'finite':bool(np.isfinite(W).all()),
        'nonnegative':bool((W>=-1e-12).all()),
    }

res={'source':str(src),'errors':[]}
try:
    open_source(src)
    for a in bpy.context.scene.objects:
        if a.type=='ARMATURE':
            try: a.data.pose_position='REST'
            except Exception: pass
    bpy.context.view_layer.update()
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and len(o.data.vertices)>0 and not getattr(o,'hide_render',False)]
    arm,arm_scores=select_primary_armature(meshes)
    bones=list(arm.data.bones) if arm else []
    bone_index={b.name:i for i,b in enumerate(bones)}
    deps=bpy.context.evaluated_depsgraph_get()
    rows=[]; baseWs=[]; evalWs=[]
    for o in meshes:
        rigid=(o.parent_bone if arm and o.parent==arm and getattr(o,'parent_type','')=='BONE' else None)
        BW,bmeta=weight_matrix(o.data.vertices,o.vertex_groups,bone_index,rigid)
        ev=o.evaluated_get(deps)
        eme=ev.to_mesh(preserve_all_data_layers=True,depsgraph=deps)
        try:
            EW,emeta=weight_matrix(eme.vertices,ev.vertex_groups,bone_index,rigid)
            row={
                'name':o.name,
                'base_vertices':int(len(o.data.vertices)),
                'evaluated_vertices':int(len(eme.vertices)),
                'base_faces':int(len(o.data.polygons)),
                'evaluated_polygons':int(len(eme.polygons)),
                'base_skin':stats(BW),
                'evaluated_skin':stats(EW),
                'base_group_meta':bmeta,
                'evaluated_group_meta':emeta,
                'non_armature_modifiers':sorted({m.type for m in o.modifiers if m.type!='ARMATURE' and bool(getattr(m,'show_viewport',True))}),
                'active_shape_keys':active_shape_keys(o.data),
                'topology_changed':bool(len(o.data.vertices)!=len(eme.vertices)),
                'rigid_parent_bone':rigid,
            }
            rows.append(row); baseWs.append(BW); evalWs.append(EW)
        finally:
            ev.to_mesh_clear()
    baseW=np.concatenate(baseWs,axis=0) if baseWs and len(bones) else np.zeros((sum(len(o.data.vertices) for o in meshes),0))
    evalW=np.concatenate(evalWs,axis=0) if evalWs and len(bones) else np.zeros((sum(r['evaluated_vertices'] for r in rows),0))
    res.update({
        'armature':arm.name if arm else None,
        'armature_scores':arm_scores,
        'bone_count':len(bones),
        'objects':rows,
        'base_global_skin':stats(baseW),
        'evaluated_global_skin':stats(evalW),
        'topology_changed_object_count':sum(r['topology_changed'] for r in rows),
        'objects_with_eval_group_elements':sum(r['evaluated_group_meta']['vertex_group_element_count']>0 for r in rows),
        'objects_with_nonarm_modifier':sum(bool(r['non_armature_modifiers']) for r in rows),
        'objects_with_active_shape_keys':sum(bool(r['active_shape_keys']) for r in rows),
    })
except Exception as e:
    res['errors'].append(f'{type(e).__name__}: {e}')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(res,indent=2),encoding='utf-8')
