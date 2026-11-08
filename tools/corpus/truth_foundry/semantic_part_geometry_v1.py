#!/usr/bin/env python3
"""Source-addressed processed-GLB PartVerse semantic-part *evidence*, not labels.

Consumes frozen bundles already stored in private Drive, validates bundle SHA,
per-face cardinality and contiguous IDs, and writes per-asset/per-part/per-adjacency
observations. Part ID is provider segmentation only; ORIGINAL SOURCE, anatomy,
2D art, drawable, edit consequence, license and training eligibility remain UNKNOWN.
No source geometry bytes are duplicated in output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import trimesh

SCHEMA = 'RealSaS.PartVerseProcessedPartGeometryCourt.v1'
ASSET_FIELDS = ('source_uid','provider_split','claimed_original_source_sha256',
                'archive_sha256','processed_glb_sha256','face_map_sha256','info_sha256',
                'processed_glb_primitive_count','processed_glb_triangle_count','processed_glb_vertex_count',
                'processed_glb_part_count','processed_part_id0_face_count','processed_glb_face_coverage_ratio',
                'processed_part_adjacent_different_label_face_pairs','processed_part_boundary_pair_count',
                'processed_part_label_connected_piece_count','processed_part_label_spanning_primitive_count',
                'processed_info_listed_part_count','processed_info_unlisted_face_count',
                'processed_info_listed_parts_exact_face_sets_match','processed_vertex_local_transform_authority',
                'processed_glb_uv_primitive_count','processed_glb_texture_visual_primitive_count',
                'geometry_observation_court','original_source_geometry_parity_court',
                'source_face_order_authority','anatomical_semantic_court','2d_game_art_court',
                'drawable_edit_court','source_rights_court','training_authorized','error')
PART_FIELDS = ('source_uid','part_id','is_zero_label','processed_glb_primitive_count',
               'part_triangle_count','part_local_surface_area','part_area_fraction',
               'part_face_id_min','part_face_id_max','part_face_ids_sha256',
               'part_mesh_primitive_count','part_intra_label_connected_pieces',
               'part_neighbor_distinct_ids','part_external_adjacent_face_pairs',
               'part_local_area_centroid_xyz','part_original_source_name',
               'part_anatomical_or_functional_concept','part_semantic_truth_status',
               'source_face_parity_court','training_mask_valid')
EDGE_FIELDS = ('source_uid','part_a','part_b','processed_adjacent_face_pairs','source_surface_causality','training_mask_valid')
MAX_SOURCE_BYTES = 80 * 1024 * 1024
MAX_SOURCE_FACES = 1_000_000


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def _uf_counts(num_faces, face_adj, labels):
    parents=np.arange(num_faces,dtype=np.int32)
    rank=np.zeros(num_faces,dtype=np.uint8)
    def root(v):
        v=int(v)
        while parents[v]!=v:
            parents[v]=parents[parents[v]]
            v=int(parents[v])
        return v
    for a,b in face_adj:
        a=int(a);b=int(b)
        if labels[a]!=labels[b]:continue
        a=root(a);b=root(b)
        if a==b:continue
        if rank[a]<rank[b]:a,b=b,a
        parents[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    sets=defaultdict(set)
    for i,lab in enumerate(labels): sets[int(lab)].add(root(i))
    return {lab:len(s) for lab,s in sets.items()}


def inspect(glb_bytes, face_bytes, info_bytes, uid, provider, archive_sha):
    data=json.loads(face_bytes)
    if not isinstance(data,dict) or not data:
        raise ValueError('EMPTY_OR_INVALID_FACE_MAP')
    scene=trimesh.load(io.BytesIO(glb_bytes),file_type='glb',force='scene',process=False)
    meshes=list(scene.geometry.values())
    if not meshes or any(not hasattr(m,'faces') for m in meshes):
        raise ValueError('EMPTY_MESH_GEOMETRY')
    total_faces=sum(len(m.faces) for m in meshes)
    if total_faces<=0 or total_faces>MAX_SOURCE_FACES:
        raise ValueError('FACE_COUNT_LIMIT')
    try:
        faces={int(k):int(v) for k,v in data.items()}
    except (ValueError,TypeError) as exc:
        raise ValueError('NON_INTEGER_FACE_OR_PART_ID') from exc
    if any(k<0 or v<0 for k,v in faces.items()):
        raise ValueError('NEGATIVE_FACE_OR_LABEL')
    if len(faces)!=total_faces or set(faces)!=set(range(total_faces)):
        raise ValueError('FACE_COUNT_AND_CONTIGUITY_MISMATCH')
    labels=np.array([faces[i] for i in range(total_faces)],dtype=np.int32)
    unique=np.unique(labels)
    if np.any(labels<0):raise ValueError('NEGATIVE_LABEL')
    tri_area=np.empty(total_faces,dtype=np.float64)
    tri_center=np.empty((total_faces,3),dtype=np.float64)
    face_adj=[]
    mesh_index=np.empty(total_faces,dtype=np.int32)
    total_vertices=0
    material_count=0
    uv_count=0
    offset=0
    for mi,mesh in enumerate(meshes):
        count=len(mesh.faces)
        tri_area[offset:offset+count]=mesh.area_faces
        tri_center[offset:offset+count]=mesh.triangles_center
        mesh_index[offset:offset+count]=mi
        if len(mesh.face_adjacency):
            face_adj.append(np.asarray(mesh.face_adjacency,dtype=np.int32)+offset)
        total_vertices+=len(mesh.vertices)
        material_count+=int(mesh.visual.kind=='texture')
        uv_count+=int(getattr(mesh.visual,'uv',None) is not None)
        offset+=count
    if not np.all(np.isfinite(tri_area)) or not np.all(np.isfinite(tri_center)):
        raise ValueError('NON_FINITE_MESH')
    face_adj=np.concatenate(face_adj,axis=0) if face_adj else np.zeros((0,2),dtype=np.int32)
    edge_pairs=Counter()
    ext_face_pairs=Counter()
    for a,b in face_adj:
        va,vb=int(labels[a]),int(labels[b])
        if va==vb:continue
        edge_pairs[(min(va,vb),max(va,vb))]+=1
        ext_face_pairs[va]+=1
        ext_face_pairs[vb]+=1
    components=_uf_counts(total_faces,face_adj,labels)
    info_match=None
    info_labels=set()
    unlisted_faces=None
    if info_bytes is not None:
        info=json.loads(info_bytes)
        names=[int(i) for i in info['ordered_face_label']]
        source_sets=[{int(i) for i in part} for part in info['ordered_faceid']]
        info_labels=set(names)
        info_match=(len(names)==len(source_sets)==len(info_labels))
        if info_match:
            for part_id, face_ids in zip(names,source_sets):
                observed=set(np.flatnonzero(labels==part_id).tolist())
                if face_ids!=observed:
                    info_match=False
                    break
        unlisted_faces=int(np.count_nonzero(~np.isin(labels,list(info_labels))))
    part_rows=[]
    total_area=float(np.sum(tri_area))
    for label in unique:
        label=int(label)
        fid=np.flatnonzero(labels==label)
        a=tri_area[fid]
        area=float(a.sum())
        neighbors={z for u,v in edge_pairs if label in (u,v) for z in (u,v) if z!=label}
        centroid=None
        # Local-coordinate centroid is only geometrically meaningful within one GLB primitive.
        # Even then source node transforms/original authored source parity remain UNKNOWN.
        if len(meshes)==1 and area>0:
            center=(tri_center[fid]*a[:,None]).sum(axis=0)/area
            centroid=[float(v) for v in center]
        part_rows.append(dict(
            source_uid=uid,part_id=label,is_zero_label=(label==0),
            processed_glb_primitive_count=len(meshes),part_triangle_count=len(fid),
            part_local_surface_area=area,part_area_fraction=(area/total_area if total_area else None),
            part_face_id_min=int(fid[0]),part_face_id_max=int(fid[-1]),
            part_face_ids_sha256=sha_bytes(np.asarray(fid,dtype='<i8').tobytes()),
            part_mesh_primitive_count=len(np.unique(mesh_index[fid])),
            part_intra_label_connected_pieces=components.get(label),
            part_neighbor_distinct_ids=len(neighbors),
            part_external_adjacent_face_pairs=ext_face_pairs[label],
            part_local_area_centroid_xyz=json.dumps(centroid) if centroid is not None else '',
            part_original_source_name='UNKNOWN',
            part_anatomical_or_functional_concept='UNKNOWN',
            part_semantic_truth_status='PROCESSED_PART_INSTANCE_ONLY',
            source_face_parity_court='UNKNOWN',training_mask_valid=False))
    edge_rows=[dict(source_uid=uid,part_a=a,part_b=b,
                    processed_adjacent_face_pairs=count,
                    source_surface_causality='NOT_TESTED',training_mask_valid=False)
               for (a,b),count in sorted(edge_pairs.items())]
    asset_row=dict(
        source_uid=uid,provider_split=provider.get('provider_split','UNKNOWN'),
        claimed_original_source_sha256=provider.get('source_sha256_provider_claim',''),
        archive_sha256=archive_sha,processed_glb_sha256=sha_bytes(glb_bytes),
        face_map_sha256=sha_bytes(face_bytes),info_sha256=sha_bytes(info_bytes) if info_bytes else '',
        processed_glb_primitive_count=len(meshes),processed_glb_triangle_count=total_faces,
        processed_glb_vertex_count=total_vertices,
        processed_glb_part_count=len(unique),
        processed_part_id0_face_count=int(np.count_nonzero(labels==0)),
        processed_glb_face_coverage_ratio=1.0,
        processed_part_adjacent_different_label_face_pairs=sum(edge_pairs.values()),
        processed_part_boundary_pair_count=len(edge_pairs),
        processed_part_label_connected_piece_count=sum(components.values()),
        processed_part_label_spanning_primitive_count=sum(p['part_mesh_primitive_count']>1 for p in part_rows),
        processed_info_listed_part_count=len(info_labels) if info_bytes else None,
        processed_info_unlisted_face_count=unlisted_faces,
        processed_info_listed_parts_exact_face_sets_match=info_match,
        processed_vertex_local_transform_authority='PER_PRIMITIVE_LOCAL_ONLY',
        processed_glb_uv_primitive_count=uv_count,
        processed_glb_texture_visual_primitive_count=material_count,
        geometry_observation_court='PROCESSED_GLB_AND_FACE_MAP_LOCAL_ONLY',
        original_source_geometry_parity_court='UNKNOWN',
        source_face_order_authority='PROCESSED_GLB_ORDER_ASSUMED_NOT_ORIGINAL_SHA_PARITY',
        anatomical_semantic_court='UNKNOWN',**{'2d_game_art_court':'UNKNOWN'},
        drawable_edit_court='UNKNOWN',source_rights_court='UNKNOWN',
        training_authorized=False,error='')
    return asset_row,part_rows,edge_rows


def _safe_members(bundle, selected):
    groups=defaultdict(dict)
    with tarfile.open(bundle,mode='r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError('NON_FILE_ARCHIVE_ENTRY')
            parts=PurePosixPath(member.name).parts
            if len(parts)!=3 or parts[0]!='source_annotation':
                raise ValueError('UNSAFE_ARCHIVE_PATH')
            uid,filename=parts[1],parts[2]
            if uid not in selected:raise ValueError('UNSELECTED_UID')
            kind={uid+'_segmented.glb':'glb',uid+'_face2label.json':'faces',uid+'_info.json':'info'}.get(filename)
            if not kind or kind in groups[uid]:raise ValueError('INVALID_OR_DUPLICATE_MEMBER')
            if member.size<=0 or member.size>MAX_SOURCE_BYTES:raise ValueError('UNSAFE_MEMBER_SIZE')
            raw=archive.extractfile(member).read(member.size+1)
            if len(raw)!=member.size:raise ValueError('TRUNCATED_MEMBER')
            groups[uid][kind]=raw
    return groups


def run(bundle_dir,manifest_path,queue_path,out_dir,*,allow_receipted_missing=False):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('schema')!='RealSaS.PartVersePartGeometryInventory.v1':
        raise ValueError('UNTRUSTED_SOURCE_MANIFEST')
    with Path(queue_path).open(newline='',encoding='utf-8') as f:
        roster=list(csv.DictReader(f))
    queue={r['partverse_uid']:r for r in roster}
    if len(queue)!=len(roster) or not queue:
        raise ValueError('DUPLICATE_OR_EMPTY_QUEUE')
    if any(r.get('provider_split')!='PROVIDER_TRAIN_CANDIDATE' for r in roster):
        raise ValueError('PROTECTED_PROVIDER_SPLIT')
    if sha_file(queue_path)!=manifest['input_queue_sha256']:
        raise ValueError('QUEUE_SHA_MISMATCH')
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    seen=set();asset_rows=[];parts=[];edges=[];errors=[]
    for spec in manifest['bundles']:
        file=Path(bundle_dir)/spec['file']
        if sha_file(file)!=spec['sha256']:
            raise ValueError('BUNDLE_SHA_MISMATCH:'+spec['file'])
        groups=_safe_members(file,queue)
        if len(groups)!=spec['source_uids']:
            raise ValueError('BUNDLE_UID_COUNT')
        for uid,data in sorted(groups.items()):
            if uid in seen:raise ValueError('DUPLICATE_SOURCE_UID')
            seen.add(uid)
            if 'glb' not in data or 'faces' not in data:
                errors.append(dict(source_uid=uid,error='MISSING_GLB_OR_FACE_MAP'))
                continue
            try:
                a,p,e=inspect(data['glb'],data['faces'],data.get('info'),uid,queue[uid],spec['sha256'])
                asset_rows.append(a);parts.extend(p);edges.extend(e)
            except Exception as exc:
                errors.append(dict(source_uid=uid,error=type(exc).__name__+':'+str(exc)[:160]))
    missing=set(queue)-seen
    if missing:
        if not allow_receipted_missing or len(missing)!=manifest.get('missing_selected_source_uids'):
            raise ValueError('UNRECEIPTED_MISSING_SOURCES')
    else:
        if manifest.get('missing_selected_source_uids',0)!=0:
            raise ValueError('MANIFEST_MISSING_DISAGREEMENT')
    if len(seen)!=manifest['staged_source_uids']:
        raise ValueError('STAGED_SOURCE_COUNT_MISMATCH')
    def dump_csv(name,fields,values):
        with (out_dir/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
            writer.writeheader();writer.writerows(values)
    dump_csv('ASSET_PROCESSED_SEMANTIC_PART_EVIDENCE_V1.csv',ASSET_FIELDS,asset_rows)
    dump_csv('SOURCE_PART_REGION_EVIDENCE_V1.csv',PART_FIELDS,parts)
    dump_csv('SOURCE_PART_ADJACENCY_EVIDENCE_V1.csv',EDGE_FIELDS,edges)
    dump_csv('SOURCE_PART_PROCESSING_EXCEPTIONS_V1.csv',('source_uid','error'),errors)
    summary={
        'schema':SCHEMA,'upstream_manifest_sha256':sha_file(manifest_path),
        'input_queue_sha256':sha_file(queue_path),
        'candidate_source_records':len(queue),'archive_staged_source_records':len(seen),
        'missing_provider_records':len(missing),
        'processed_mesh_label_geometry_observation_count':len(asset_rows),
        'per_part_source_region_observations':len(parts),
        'per_part_adjacency_pairs_observed':len(edges),
        'source_processing_errors':len(errors),
        'processed_faces_sum':sum(int(r['processed_glb_triangle_count']) for r in asset_rows),
        'source_original_geometry_parity_pass':0,
        'anatomical_semantic_concept_pass':0,'stylized_2d_art_pass':0,
        'drawable_edit_causality_pass':0,'independent_training_eligible':0,
        'training_authorized':False,
        'outputs_sha256':{p.name:sha_file(p) for p in sorted(out_dir.glob('*.csv'))}
    }
    (out_dir/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-receipted-missing',action='store_true')
    a=p.parse_args()
    print(json.dumps(run(a.bundle_dir,a.manifest,a.queue,a.out,allow_receipted_missing=a.allow_receipted_missing),indent=2))

if __name__=='__main__':main()