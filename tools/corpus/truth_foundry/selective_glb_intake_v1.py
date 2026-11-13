#!/usr/bin/env python3
"""Select a small, morphology-stratified subset of UniML3D canonical Objaverse GLBs.

Source-derived canonical meshes in quarantine only. Provider-canonicalized GLB
is NOT proof of the original asset's authoring pose, rights, native controls,
face semantics, 2D game art, or source identity parity. Avoid accidental
whole-HF-snapshot downloads and preserve one full SHA per downloaded GLB.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import time
from urllib.parse import quote

import requests

REPO = 'Linzhan/UniML3D'
IDENT = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{9,95}$')
SHA40 = re.compile(r'^[0-9a-f]{40}$')
SHA64 = re.compile(r'^[0-9a-f]{64}$')
RARE = ('serpentine', 'marine', 'insectoid', 'avian', 'quadrupedal', 'articulated_rigid')
HEADER_FIELDS = [
    'object_type', 'category_provider_claim', 'provider_dataset', 'candidate_source_split',
    'repo_revision', 'repo_path', 'download_status', 'downloaded_bytes', 'download_sha256',
    'glb_valid', 'glb_schema_error', 'glb_mesh_count', 'glb_primitive_count',
    'glb_mesh_vertex_accessor_count', 'glb_triangle_indices_count', 'glb_skin_count',
    'glb_skin_joint_count', 'glb_node_count', 'glb_animation_count',
    'glb_material_count', 'glb_image_count', 'glb_embedded_image_count',
    'glb_texture_count', 'source_original_byte_sha_verified',
    'original_rights_court', 'original_pose_parity_court', 'anatomical_morphology_court',
    'source_face_id_parity_court', 'semantics_court', 'graphics_2d_art_court',
    'training_admitted',
]


def sha(path: Path) -> str:
    d=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):
            d.update(b)
    return d.hexdigest()


def candidates_from_queue(path: Path, max_selected: int, expected_sha: str):
    if not SHA64.fullmatch(expected_sha) or sha(path)!=expected_sha:
        raise ValueError('FROZEN_SOURCE_QUEUE_SHA_MISMATCH')
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        r=csv.DictReader(stream)
        needed={'dataset','object_type','category','num_joints','num_clips',
                'source_rights_court','training_authorized'}
        if not needed.issubset(set(r.fieldnames or [])):
            raise ValueError('QUEUE_SCHEMA_MISSING_FIELDS')
        rows=list(r)
    if len(rows)!=2500:
        raise ValueError('QUEUE_COUNT_NOT_2500')
    seen=set();groups=defaultdict(deque)
    for row in rows:
        name=row['object_type']
        if not IDENT.fullmatch(name) or name in seen:
            raise ValueError('INVALID_OR_DUPLICATE_OBJECT_ID')
        if row['dataset']!='objaverse' or row['source_rights_court']!='UNKNOWN' or \
                row['training_authorized'].lower()!='false':
            raise ValueError('SOURCE_ALREADY_ADMITTED_OR_NOT_OBJAVERSE')
        if int(row['num_joints'])<3 or int(row['num_clips'])<1:
            raise ValueError('BAD_MOTION_CANDIDATE')
        seen.add(name)
        groups[row['category']].append(row)
    selected=[]
    # Fair, rare-first round robin; don't let one class saturate the first batch.
    rare_groups=list(RARE)
    while len(selected)<max_selected and any(groups[c] for c in rare_groups):
        for cat in rare_groups:
            if groups[cat] and len(selected)<max_selected:
                selected.append(groups[cat].popleft())
    for cat in ('bipedal', 'uncertain'):
        while len(selected)<max_selected and groups[cat]:
            selected.append(groups[cat].popleft())
    if len(selected)!=max_selected:
        raise ValueError('INSUFFICIENT_ELIGIBLE_SOURCE_QUEUE')
    return selected


def glb_structure(path: Path):
    """Count native glTF object structs without pretending to know its physics."""
    result={k:0 for k in ('glb_mesh_count','glb_primitive_count',
           'glb_mesh_vertex_accessor_count','glb_triangle_indices_count',
           'glb_skin_count','glb_skin_joint_count','glb_node_count',
           'glb_animation_count','glb_material_count','glb_image_count',
           'glb_embedded_image_count','glb_texture_count')}
    result.update(glb_valid=False,glb_schema_error='')
    try:
        with path.open('rb') as f:
            start=f.read(20)
            if len(start)!=20:
                raise ValueError('GLB_TRUNCATED_HEADER')
            magic,version,total,chunk_len,chunk_type=struct.unpack('<4sIIII',start)
            if magic!=b'glTF' or version!=2 or total!=path.stat().st_size or \
                    chunk_type!=0x4e4f534a or chunk_len>64*1024*1024:
                raise ValueError('GLB_BAD_CONTAINER_HEADER')
            doc=json.loads(f.read(chunk_len))
        if doc.get('asset',{}).get('version')!='2.0':
            raise ValueError('GLTF_VERSION')
        meshes=doc.get('meshes',[])
        accessors=doc.get('accessors',[])
        result['glb_mesh_count']=len(meshes)
        result['glb_primitive_count']=sum(len(m.get('primitives',[])) for m in meshes)
        for mesh in meshes:
            for prim in mesh.get('primitives',[]):
                a=prim.get('attributes',{}).get('POSITION')
                if isinstance(a,int) and 0<=a<len(accessors):
                    result['glb_mesh_vertex_accessor_count']+=int(accessors[a].get('count',0))
                i=prim.get('indices')
                if isinstance(i,int) and 0<=i<len(accessors) and prim.get('mode',4)==4:
                    result['glb_triangle_indices_count']+=int(accessors[i].get('count',0))//3
        result['glb_skin_count']=len(doc.get('skins',[]))
        result['glb_skin_joint_count']=sum(len(x.get('joints',[])) for x in doc.get('skins',[]))
        result['glb_node_count']=len(doc.get('nodes',[]))
        result['glb_animation_count']=len(doc.get('animations',[]))
        result['glb_material_count']=len(doc.get('materials',[]))
        images=doc.get('images',[])
        result['glb_image_count']=len(images)
        result['glb_embedded_image_count']=sum('bufferView' in x for x in images)
        result['glb_texture_count']=len(doc.get('textures',[]))
        if not meshes:
            raise ValueError('GLB_NO_MESH')
        result['glb_valid']=True
    except (OSError,ValueError,TypeError,KeyError,OverflowError,IndexError,json.JSONDecodeError) as exc:
        result['glb_schema_error']=type(exc).__name__+':'+str(exc)[:100]
    return result


def stream_asset(url: str, target: Path, per_file_cap: int, session=None):
    """Byte-capped streamed response; never follow arbitrary provider path IDs."""
    if session is None:
        session=requests.Session()
    partial=target.with_suffix('.part')
    partial.unlink(missing_ok=True)
    try:
        with session.get(url,stream=True,timeout=(30,180),headers={'User-Agent':'RealSaS-TruthFoundry/1.0'}) as response:
            if response.status_code==404:
                return 'NOT_PUBLISHED'
            response.raise_for_status()
            announced=response.headers.get('Content-Length')
            if announced and int(announced)>per_file_cap:
                return 'OVERSIZE_HEADER'
            size=0
            with partial.open('wb') as f:
                for block in response.iter_content(1024*1024):
                    if not block:continue
                    size+=len(block)
                    if size>per_file_cap:
                        return 'OVERSIZE_STREAM'
                    f.write(block)
        if size<20:
            return 'EMPTY_OR_TRUNCATED'
        partial.replace(target)
        return 'DOWNLOADED'
    except requests.RequestException as exc:
        return 'NETWORK_'+type(exc).__name__
    finally:
        partial.unlink(missing_ok=True)


def run(queue_csv: Path, out: Path, *, revision: str, expected_sha: str,
        max_selected=96, max_total_mib=768, max_file_mib=25, session=None):
    if not SHA40.fullmatch(revision):
        raise ValueError('UPSTREAM_REVISION_MUST_BE_PINNED')
    if max_selected<1 or max_selected>256 or max_total_mib>2048 or max_file_mib>48:
        raise ValueError('QUOTA_NOT_BOUNDED')
    selected=candidates_from_queue(queue_csv,max_selected,expected_sha)
    out.mkdir(parents=True,exist_ok=True)
    assetdir=out/'canonical_glb_quarantine'
    assetdir.mkdir(exist_ok=True)
    budget=max_total_mib*1024*1024
    maxfile=max_file_mib*1024*1024
    used=0; receipts=[]
    for row in selected:
        object_type=row['object_type']
        filename='canonical_assets/objaverse/'+object_type+'.glb'
        url='https://huggingface.co/datasets/'+REPO+'/resolve/'+revision+'/'+quote(filename,safe='/')
        output=assetdir/(object_type+'.glb')
        status='BUDGET_EXHAUSTED'
        file_size=0; file_sha=''
        if used+maxfile<=budget:
            status=stream_asset(url,output,maxfile,session=session)
        if status=='DOWNLOADED':
            file_size=output.stat().st_size
            file_sha=sha(output)
            used+=file_size
            fields=glb_structure(output)
            if not fields['glb_valid']:
                status='INVALID_GLB_QUARANTINED'
        else:
            fields=glb_structure(output) if output.is_file() else {k:0 for k in HEADER_FIELDS if k.startswith('glb_') and k not in ('glb_valid','glb_schema_error')}
            fields.update(glb_valid=False,glb_schema_error='NOT_INSPECTED')
        record={
            'object_type':object_type,
            'category_provider_claim':row['category'],
            'provider_dataset':'objaverse',
            'candidate_source_split':'UNIML_PROVIDER_NOT_FINAL_REALSAS_FIT',
            'repo_revision':revision,
            'repo_path':filename,
            'download_status':status,
            'downloaded_bytes':file_size,
            'download_sha256':file_sha,
            **fields,
            'source_original_byte_sha_verified':False,
            'original_rights_court':'UNKNOWN',
            'original_pose_parity_court':'UNKNOWN',
            'anatomical_morphology_court':'UNKNOWN',
            'source_face_id_parity_court':'UNKNOWN',
            'semantics_court':'UNKNOWN',
            'graphics_2d_art_court':'UNKNOWN',
            'training_admitted':False,
        }
        receipts.append(record)
    with (out/'UNIML3D_SOURCE_GLB_EVIDENCE_V1.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=HEADER_FIELDS,extrasaction='raise')
        w.writeheader();w.writerows(receipts)
    s={
        'schema':'RealSaS.TruthFoundryUniML3DSelectiveGLBIntake.v1',
        'pool':'UNIML3D', 'pinned_repo_revision':revision,
        'queue_sha256':sha(queue_csv),
        'selected_subjects':len(selected),
        'downloaded_glb_count':sum(r['download_status'] in ('DOWNLOADED','INVALID_GLB_QUARANTINED') for r in receipts),
        'glb_structure_pass_count':sum(r['glb_valid'] for r in receipts),
        'downloaded_bytes_total':used,
        'limit_total_bytes':budget,
        'max_file_bytes':maxfile,
        'download_status_distribution':dict(Counter(r['download_status'] for r in receipts)),
        'selected_morphology_distribution':dict(Counter(r['category_provider_claim'] for r in receipts)),
        'source_original_bytes_verified':0,
        'rights_qualified_count':0,
        'anatomical_morphology_qualified':0,
        'semantic_face_labels_qualified':0,
        'model_training_qualified_count':0,
        'training_authorized':False,
        'receipt_sha256':sha(out/'UNIML3D_SOURCE_GLB_EVIDENCE_V1.csv')
    }
    (out/'SUMMARY_V1.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return s


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--queue-sha256',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-selected',type=int,default=96)
    parser.add_argument('--max-total-mib',type=int,default=768)
    parser.add_argument('--max-file-mib',type=int,default=25)
    a=parser.parse_args()
    print(json.dumps(run(a.queue,a.out,revision=a.revision,expected_sha=a.queue_sha256,
                         max_selected=a.max_selected,max_total_mib=a.max_total_mib,
                         max_file_mib=a.max_file_mib),indent=2))

if __name__=='__main__':main()