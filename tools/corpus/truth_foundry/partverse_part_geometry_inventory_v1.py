#!/usr/bin/env python3
"""Processed PartVerse geometry/part coverage inventory with explicit courts OPEN.

Reads ONLY selected archive evidence produced from pinned PartVerse-XL annotation
archive. Does not assert original authored source, subject anatomy, named parts,
2D style, drawables, license or any head's training readiness.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for b in iter(lambda:fp.read(4 * 1024 * 1024),b''):
            h.update(b)
    return h.hexdigest()


def to_int(v):
    if isinstance(v,bool) or not isinstance(v,(int,str)):
        raise ValueError('INVALID_FACE_OR_LABEL')
    return int(v)


def load_faces(path: Path):
    raw=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw,dict) or not raw:raise ValueError('EMPTY_FACE_MAP')
    out={to_int(k):to_int(v) for k,v in raw.items()}
    if any(k<0 or v<0 for k,v in out.items()):raise ValueError('NEGATIVE_FACE_OR_LABEL')
    return out


def one_source(src: Path, provider:dict):
    uid=src.name
    face_path=src/(uid+'_face2label.json')
    glb_path=src/(uid+'_segmented.glb')
    info_path=src/(uid+'_info.json')
    result={
       'uid':uid, 'source_sha_provider_claim':provider.get('source_sha256_provider_claim',''),
       'weak_subject_candidate':provider.get('object_type_candidate','UNKNOWN'),
       'weak_style_candidate':provider.get('style_candidate','UNKNOWN'),
       'provider_split':provider.get('provider_split','SPLIT_UNKNOWN'),
       'face2label_exists':face_path.exists(),
       'segmented_glb_exists':glb_path.exists(),
       'info_exists':info_path.exists(),
       'triangle_face_count':None,'vertex_count':None,'raw_mesh_scene_primitive_count':None,
       'connected_component_count':None,'watertight_source_claim':'NOT_CHECKED',
       'labeled_faces':None,'part_id_count_including_zero':None,
       'zero_label_face_count':None,'provider_info_part_id_count':None,
       'face_count_equal_triangles':False,'part_info_face_ids_subset_consistent':False,
       'face_count_label0_unlisted_by_info':None,
       'geometry_face_index_order_court':'UNKNOWN',
       'original_source_sha_bytes_parity_court':'UNKNOWN',
       'native_texture_uv_2d_style_court':'UNKNOWN',
       'rights_court':'UNKNOWN',
       'anatomical_part_concept_court':'UNKNOWN',
       'drawable_necessity_court':'UNKNOWN',
       'semantic_head_training_admitted':False,
       'processing_errors':'',
    }
    problems=[]
    faces={}
    try:
        if face_path.exists():
            faces=load_faces(face_path)
            vals=Counter(faces.values())
            result['labeled_faces']=len(faces)
            result['part_id_count_including_zero']=len(vals)
            result['zero_label_face_count']=vals.get(0,0)
    except (ValueError,KeyError,TypeError,UnicodeDecodeError,json.JSONDecodeError) as exc:
        problems.append('FACE_MAP:'+str(exc))
    if glb_path.exists():
        try:
            import trimesh
            scene=trimesh.load(glb_path,force='scene',process=False)
            meshes=list(scene.geometry.values())
            result['raw_mesh_scene_primitive_count']=len(meshes)
            result['triangle_face_count']=sum(len(m.faces) for m in meshes)
            result['vertex_count']=sum(len(m.vertices) for m in meshes)
            if faces:
                result['face_count_equal_triangles']=(len(faces)==result['triangle_face_count'] and
                    min(faces)==0 and max(faces)==result['triangle_face_count']-1)
                if result['face_count_equal_triangles']:
                    result['geometry_face_index_order_court']='COUNT_AND_CONTIGUITY_ONLY__ORDER_NOT_VERIFIED'
            if len(meshes)==1 and len(meshes[0].faces)<=250000:
                # Component count must not erase source face-count evidence if
                # trimesh splitting/dependency errors happen on exotic meshes.
                try:
                    result['connected_component_count']=len(meshes[0].split(only_watertight=False))
                    result['watertight_source_claim']='PROCESSED_SEGMENTED_MESH_ONLY:'+str(bool(meshes[0].is_watertight))
                except Exception as exc:
                    problems.append('COMPONENT_COUNT:'+type(exc).__name__+':'+str(exc)[:180])
        except Exception as exc:
            problems.append('SEGMENTED_GLB:'+type(exc).__name__+':'+str(exc)[:180])
    if info_path.exists():
        try:
            info=json.loads(info_path.read_text(encoding='utf-8'))
            labels=[to_int(v) for v in info['ordered_face_label']]
            part_faces=[set(to_int(v) for v in part) for part in info['ordered_faceid']]
            result['provider_info_part_id_count']=len(labels)
            correct=bool(faces and len(labels)==len(part_faces) and len(set(labels))==len(labels))
            if correct:
                by_label = defaultdict(set)
                for face_id, label in faces.items():
                    by_label[label].add(face_id)
                for label, indices in zip(labels,part_faces):
                    if indices != by_label.get(label,set()):
                        correct=False
                        break
            result['part_info_face_ids_subset_consistent']=correct
            if correct:
                represented=set(labels)
                result['face_count_label0_unlisted_by_info']=sum(v not in represented for v in faces.values())
        except (ValueError,KeyError,TypeError,UnicodeDecodeError,json.JSONDecodeError) as exc:
            problems.append('INFO:'+str(exc))
    result['processing_errors']=';'.join(problems)
    return result


def run(archive_root:Path, queue_csv:Path, out:Path, max_bundle_uids=100):
    out.mkdir(parents=True,exist_ok=True)
    with queue_csv.open('r',newline='',encoding='utf-8') as f:
        rdr=csv.DictReader(f)
        queue={r['partverse_uid']:r for r in rdr}
    if not queue or len(queue)>3000:raise ValueError('SELECTION_OUT_OF_RANGE')
    if any(r.get('provider_split')!='PROVIDER_TRAIN_CANDIDATE' for r in queue.values()):
        raise ValueError('HELDOUT_SPLIT_IN_QUEUE')
    data=archive_root/'sampled_annotation_data'
    if not data.is_dir():raise ValueError('STAGED_ANNOTATION_DATA_MISSING')
    sources=sorted(p for p in data.iterdir() if p.is_dir())
    if any(p.name not in queue for p in sources):raise ValueError('STAGED_UNSELECTED_SOURCE')
    assets=[one_source(p,queue[p.name]) for p in sources]
    if not assets:raise ValueError('NO_SOURCE_BYTES_STAGED')
    keys=list(assets[0])
    with (out/'ASSET_PART_GEOMETRY_EVIDENCE_V1.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(assets)
    by_kind=Counter(r['weak_subject_candidate'] for r in assets)
    parities=sum(r['face_count_equal_triangles'] for r in assets)
    info_consistent=sum(r['part_info_face_ids_subset_consistent'] for r in assets)
    unknown_zero=sum((r['face_count_label0_unlisted_by_info'] or 0) for r in assets)
    bundles=[]
    bundle_dir=out/'bundle_shards'
    bundle_dir.mkdir(exist_ok=True)
    for index in range(0,len(sources),max_bundle_uids):
        group=sources[index:index+max_bundle_uids]
        path=bundle_dir/f'PARTVERSE_SOURCE_ANNOTATIONS_{index//max_bundle_uids:03d}.tar.gz'
        with tarfile.open(path,'w:gz',compresslevel=3) as tf:
            for src in group:
                for asset in sorted(src.iterdir()):
                    if not asset.is_file():raise ValueError('UNEXPECTED_ENTRY')
                    arcn=str(Path('source_annotation')/src.name/asset.name)
                    tf.add(asset,arcname=arcn,recursive=False)
        bundles.append({'file':path.name,'source_uids':len(group),'bytes':path.stat().st_size,
                        'sha256':digest(path)})
    result={'schema':'RealSaS.PartVersePartGeometryInventory.v1',
            'input_queue_sha256':digest(queue_csv),
            'selected_candidates':len(queue),'staged_source_uids':len(sources),
            'missing_selected_source_uids':len(set(queue)-{p.name for p in sources}),
            'processed_glb_face_count_contiguity_matches':parities,
            'part_info_consistent_for_listed_labels':info_consistent,
            'face_count_missing_in_provider_info_despite_face2label':unknown_zero,
            'weak_subject_distribution':dict(by_kind),'bundle_count':len(bundles),
            'bundles':bundles,'rights_verified':0,'independent_morphology_visual_pass':0,
            'head_semantic_training_pass':0,'2d_art_pass':0,'drawable_pass':0,
            'truth_status':'SOURCE_EVIDENCE_ONLY_NO_TRAINING_AUTHORITY',
            'results_sha256':digest(out/'ASSET_PART_GEOMETRY_EVIDENCE_V1.csv')}
    (out/'SUMMARY_V1.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--staging-root',type=Path,required=True)
    parser.add_argument('--queue-csv',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-bundle-uids',type=int,default=100)
    args=parser.parse_args()
    if args.max_bundle_uids<=0 or args.max_bundle_uids>200:raise ValueError('INVALID_SHARD_WIDTH')
    print(json.dumps(run(args.staging_root,args.queue_csv,args.out,args.max_bundle_uids),indent=2))