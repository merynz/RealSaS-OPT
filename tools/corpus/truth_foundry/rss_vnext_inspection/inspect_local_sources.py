#!/usr/bin/env python3
"""Read-only, source-hashed, conservative audit of locally mounted verified derivative pairs.

DO NOT interpret normalized derivatives as creator-original assets, do not infer
character identity, anatomical front, scene appearance, rights or training PASS.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,pathlib,collections,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

VERSION = 'RealSaS.DeepDerivativeStructuralCourt.v1'


def sha256(path: pathlib.Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as inp:
        for data in iter(lambda:inp.read(1<<20),b''):digest.update(data)
    return digest.hexdigest()


def judge(source: pathlib.Path, structure: pathlib.Path, version: str) -> dict:
    k=source.stem
    receipt={'schema':VERSION,'source_key_fragment':k,'court_version':version,
             'derivative_npz_sha256':sha256(source),'derivative_structure_sha256':sha256(structure),
             'derivative_npz_bytes':source.stat().st_size,'derivative_structure_bytes':structure.stat().st_size,
             'original_asset_byte_parity':'NOT_TESTED', 'rights_training_eligibility':'UNKNOWN',
             'character_classification':'UNKNOWN', 'morphology':'UNKNOWN',
             'anatomical_front_axis':'UNKNOWN','native_pose_semantics':'UNKNOWN',
             'native_scene_materials_textures':'NOT_COURTED_FROM_ORIGINAL',
             'two_view_render_suitability':'NOT_COURTED','supervision_admission':'NOT_COURTED'}
    errors=[];flags=[]
    try:
        s=json.loads(structure.read_text())
        with np.load(source,allow_pickle=False) as npz:
            arrays={name:npz[name] for name in npz.files}
        required={'vertices','vertices_source','faces','vertex_normals','canonical_transform',
                  'inverse_canonical_transform','parents','deform_mask','skin',
                  'rest_local_source','rest_world_source','bone_heads','bone_tails'}
        missing=required-set(arrays)
        if missing: raise ValueError('MISSING_ARRAYS:'+','.join(sorted(missing)))
        v=np.asarray(arrays['vertices']);vs=np.asarray(arrays['vertices_source']);f=np.asarray(arrays['faces']);
        n=np.asarray(arrays['vertex_normals']);p=np.asarray(arrays['parents']);w=np.asarray(arrays['skin']);
        d=np.asarray(arrays['deform_mask']);M=np.asarray(arrays['canonical_transform']);Minv=np.asarray(arrays['inverse_canonical_transform'])
        V=len(v);F=len(f);J=len(p)
        if v.shape!=(V,3) or vs.shape!=(V,3) or f.shape!=(F,3) or w.shape!=(V,J):errors.append('GEOMETRY_SKIN_SHAPE_MISMATCH')
        if n.shape!=(V,3):errors.append('NORMAL_SHAPE_MISMATCH')
        if not np.isfinite(v).all() or not np.isfinite(vs).all() or not np.isfinite(n).all() or not np.isfinite(w).all():errors.append('NONFINITE_ARRAY_VALUE')
        if (f<0).any() or (f>=V).any():errors.append('FACE_VERTEX_INDEX_OOB')
        if not np.allclose(M@Minv,np.eye(4),atol=1e-3):errors.append('TRANSFORM_INVERSE_MISMATCH')
        obj=s.get('mesh_objects',[])
        if not isinstance(obj,list):errors.append('MESH_OBJECTS_NOT_ARRAY');obj=[]
        declared_v=sum(int(x.get('vertex_count',0)) for x in obj)
        declared_f=sum(int(x.get('face_count',0)) for x in obj)
        if declared_v!=V:errors.append('MESH_OBJECT_VERTEX_TOTAL_MISMATCH')
        if declared_f!=F:errors.append('MESH_OBJECT_FACE_TOTAL_MISMATCH')
        if int(s.get('bone_count',-1))!=J:errors.append('STRUCTURE_BONE_COUNT_MISMATCH')
        if len(s.get('bone_names',[]))!=J:errors.append('STRUCTURE_BONE_NAMES_COUNT_MISMATCH')
        if not len(obj):flags.append('NO_MESH_OBJECTS')
        if not J:flags.append('NO_RIG_JOINTS')
        if not F:flags.append('NO_FACES')
        # object partition: counts alone are not a proof that each face is within
        # an object. Check source index ranges on each face and count violations.
        starts=np.array([int(x.get('vertex_start',0)) for x in obj],dtype=np.int64)
        ends=starts+np.array([int(x.get('vertex_count',0)) for x in obj],dtype=np.int64)
        if len(starts):
            order=np.argsort(starts,kind='stable')
            if not np.array_equal(order,np.arange(len(order))):errors.append('OBJECT_RANGES_NOT_SORTED')
            if starts[0]!=0 or ends[-1]!=V or (starts[1:]!=ends[:-1]).any():errors.append('OBJECT_RANGES_NOT_CONTIGUOUS')
        cross=0;actual_face_counts=[]
        if len(starts) and not (f<0).any() and not (f>=V).any():
            owner=np.searchsorted(ends,f,side='right')
            cross=int(np.count_nonzero((owner[:,0]!=owner[:,1])|(owner[:,1]!=owner[:,2])))
            if cross:flags.append('FACE_CROSSES_SOURCE_MESH_OBJECT_RANGE')
            actual_face_counts=np.bincount(np.minimum(owner[:,0],len(obj)-1),minlength=len(obj)).tolist()
            if actual_face_counts!=[int(x.get('face_count',0)) for x in obj]:
                flags.append('OBJECT_FACE_COUNT_ORDER_OR_PARTITION_CONFLICT')
        # Strict adjacency evidence is not equivalent to a rig, anatomical part or
        # drawable owner. We only count topological islands and edge incidence.
        islands=None;isolated=None;boundary=None;nonmanifold=None
        if V and F and V<=300000 and F<=600000 and not errors:
            edges=np.concatenate([f[:,(0,1)],f[:,(1,2)],f[:,(2,0)]],axis=0).astype(np.int64,copy=False)
            sorted_edges=np.sort(edges,axis=1)
            nonloop=sorted_edges[:,0]!=sorted_edges[:,1]
            edge_uni,counts=np.unique(sorted_edges[nonloop],axis=0,return_counts=True)
            boundary=int(np.count_nonzero(counts==1));nonmanifold=int(np.count_nonzero(counts>2))
            e=edge_uni
            graph=coo_matrix((np.ones(2*len(e),dtype=np.uint8),
                            (np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(V,V)).tocsr()
            _,labels=connected_components(graph,directed=False,return_labels=True)
            used=np.zeros(V,dtype=bool);used[f.reshape(-1)]=True
            islands=int(len(np.unique(labels[used])))
            isolated=int(np.count_nonzero(~used))
        else:
            flags.append('ISLAND_ADJACENCY_NOT_COURTED_DUE_TO_SIZE_OR_INTEGRITY')
        skin_row_sum=w.sum(axis=1) if J else np.zeros(V)
        positive_mask=w>1e-6
        nonzero=skin_row_sum>1e-8
        nonzero_bad=int(np.count_nonzero(nonzero&(~np.isclose(skin_row_sum,1.0,atol=1e-3))))
        zero=int(np.count_nonzero(~nonzero))
        if zero:flags.append('ZERO_SKIN_ROW_EXISTS')
        if nonzero_bad:flags.append('NONNORMALIZED_NONZERO_SKIN_ROW_EXISTS')
        if nonmanifold:flags.append('NONMANIFOLD_EDGES_EXIST')
        if boundary:flags.append('OPEN_BOUNDARY_EDGES_EXIST')
        if cross:flags.append('SOURCE_OBJECT_TOPOLOGY_CROSSING')
        native_extension=pathlib.Path(str(s.get('source',''))).suffix.lower()
        re=np.asarray(arrays['rest_world_source'])
        rn=np.asarray(arrays['rest_local_source'])
        rest_hierarchy_error=int(np.count_nonzero(~np.isfinite(rn)))+int(np.count_nonzero(~np.isfinite(re)))
        if rest_hierarchy_error:errors.append('NONFINITE_REST_FRAME')
        rc={'native_format_claim_from_structure':native_extension or 'UNKNOWN',
            'mesh_object_count':len(obj),'mesh_object_names':[str(x.get('name','')) for x in obj],
            'declared_object_vertex_counts':[int(x.get('vertex_count',0)) for x in obj],
            'declared_object_triangle_counts':[int(x.get('face_count',0)) for x in obj],
            'vertices':V,'triangles':F,'joints':J,'deform_joint_count':int(np.count_nonzero(d)),
            'control_or_nondeform_joint_count':int(J-np.count_nonzero(d)),
            'root_joints':int(np.count_nonzero(p<0)),
            'source_aabb_min':vs.min(axis=0).round(6).tolist() if V else [],
            'source_aabb_max':vs.max(axis=0).round(6).tolist() if V else [],
            'normalized_aabb_min':v.min(axis=0).round(6).tolist() if V else [],
            'normalized_aabb_max':v.max(axis=0).round(6).tolist() if V else [],
            'skin_zero_sum_vertices':zero,'skin_unnormalized_nonzero_rows':nonzero_bad,
            'skin_positive_entries':int(np.count_nonzero(positive_mask)),
            'skin_max_positive_influences_per_vertex':int(positive_mask.sum(axis=1).max()) if V and J else 0,
            'source_object_crossing_faces':cross,
            'face_count_by_vertex_range':actual_face_counts,
            'connected_triangle_surface_islands':islands,'unused_or_isolated_vertices':isolated,
            'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,
            'normalized_payload_has_uv':bool(any('uv' in name.lower() for name in arrays)),
            'normalized_payload_has_texture_pixels':bool(any('texture' in name.lower() or 'image' in name.lower() for name in arrays)),
            'original_render_materials_verified':False,
            'joint_name_inventory_count':len(s.get('bone_names',[]))}
        receipt.update(rc)
    except Exception as e:
        errors.append('INSPECT_EXCEPTION:'+str(e)[:400])
    receipt['fail_closed_errors']=sorted(set(errors))
    receipt['quality_flags']=sorted(set(flags))
    receipt['status']='FAIL_DERIVATIVE_STRUCTURE' if errors else 'OBSERVED_DERIVATIVE_ONLY_NOT_TRAINING_AUTHORIZED'
    return receipt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--versions',nargs='+',type=int,default=[24,25,26])
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/mnt/data'))
    p.add_argument('--out',type=pathlib.Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for version in a.versions:
        folder=a.root/f'realsas_v{version}'
        sources=sorted((folder/'sources').glob('*.npz'))
        for source in sources:
            structure=folder/'structures'/(source.stem+'.json')
            if not structure.exists():
                rows.append({'source_key_fragment':source.stem,'court_version':f'V{version}',
                             'status':'FAIL_DERIVATIVE_STRUCTURE','fail_closed_errors':['MISSING_STRUCTURE_JSON']})
            else:
                rows.append(judge(source,structure,f'V{version}'))
    if len(rows)!=len({r['source_key_fragment'] for r in rows}):raise ValueError('DUPLICATE_SOURCE_KEYS')
    target=a.out/'DEEP_DERIVATIVE_SUBJECT_AUDIT_V1.jsonl'
    with target.open('w') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n')
    cols=['source_key_fragment','court_version','status','native_format_claim_from_structure','mesh_object_count','vertices','triangles','joints','deform_joint_count','control_or_nondeform_joint_count','root_joints','connected_triangle_surface_islands','unused_or_isolated_vertices','source_object_crossing_faces','skin_zero_sum_vertices','skin_unnormalized_nonzero_rows','skin_positive_entries','skin_max_positive_influences_per_vertex','boundary_edges','nonmanifold_edges','normalized_payload_has_uv','normalized_payload_has_texture_pixels','fail_closed_errors','quality_flags','derivative_npz_sha256','derivative_structure_sha256','original_asset_byte_parity','character_classification','morphology','anatomical_front_axis','native_pose_semantics','native_scene_materials_textures','rights_training_eligibility','two_view_render_suitability','supervision_admission']
    with (a.out/'DEEP_DERIVATIVE_SUBJECT_AUDIT_V1.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');wr.writeheader()
        for r in rows:
            wr.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k,'') for k in cols})
    stat=collections.Counter(r['status'] for r in rows)
    numeric=lambda key:sum(int(r.get(key,0) or 0) for r in rows)
    summary={'schema':VERSION,'local_source_count':len(rows),'per_version':dict(collections.Counter(r['court_version'] for r in rows)),
             'status':dict(stat),'source_numeric_counts':{k:numeric(k) for k in ['mesh_object_count','vertices','triangles','joints','deform_joint_count','control_or_nondeform_joint_count','connected_triangle_surface_islands','unused_or_isolated_vertices','source_object_crossing_faces','skin_zero_sum_vertices','skin_unnormalized_nonzero_rows','boundary_edges','nonmanifold_edges']},
             'sources_with_multiple_mesh_objects':sum((r.get('mesh_object_count') or 0)>1 for r in rows),
             'sources_with_multi_island_topology':sum((r.get('connected_triangle_surface_islands') or 0)>1 for r in rows),
             'source_originals_independently_verified_here':0,'native_pose_morphology_graphics_rights_admitted_here':0,
             'notes':'Only V24-V26 verified normalized derivative source pairs currently available locally. Strong numeric court unchanged; this pass adds structural audit, not creator-original or training authority.'}
    (a.out/'DEEP_DERIVATIVE_SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()