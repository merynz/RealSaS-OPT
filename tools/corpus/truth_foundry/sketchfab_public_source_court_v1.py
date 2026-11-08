#!/usr/bin/env python3
"""Public Sketchfab source metadata and license *receipt*, not training rights.

Download only metadata for SHA-pinned, source-UID constrained pilot records.
Returns external publisher/license/animation/material/thumbnail URL but never
approves original-source rights, anatomical morphology or 2D art quality.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from collections import Counter

UID=re.compile(r'^[a-f0-9]{32}$')
SHA=re.compile(r'^[a-f0-9]{64}$')
OUT_FIELDS=['source_uid','source_sha_provider_claim','source_url','weak_morphology','weak_style',
 'api_status','api_http_status','publisher_id','publisher_name','published_at','model_name',
 'license_slug','license_fullname','license_url','commercial_training_license_court',
 'is_downloadable','model_vertices_count','model_faces_count','model_animation_count',
 'model_texture_count','model_material_count','model_category_names','model_tag_names',
 'thumbnail_url_provider','source_original_bytes_sha_court','source_visual_morphology_court',
 'source_stylized_2d_art_court','head_training_authorized','error']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for s in iter(lambda:f.read(1024*1024),b''):h.update(s)
    return h.hexdigest()


def _label(v):
    if isinstance(v,str):return v
    if isinstance(v,dict):return str(v.get('slug') or v.get('name') or v.get('fullName') or '')
    return ''


def parse_public_record(src, status, doc=None, err=''):
    uid=src['partverse_uid']
    row={k:'' for k in OUT_FIELDS}
    row.update(source_uid=uid,source_sha_provider_claim=src['source_sha256_provider_claim'],
      source_url=src['source_url'],weak_morphology=src['object_type_candidate'],
      weak_style=src['style_candidate'],api_status='UNKNOWN',api_http_status=status,
      commercial_training_license_court='UNKNOWN__NEEDS_ORIGINAL_LICENSE_REVIEW',
      source_original_bytes_sha_court='UNKNOWN',source_visual_morphology_court='UNKNOWN',
      source_stylized_2d_art_court='UNKNOWN',head_training_authorized=False,error=err)
    if status==200 and isinstance(doc,dict) and doc.get('uid','')==uid:
        lic=doc.get('license') or {}
        if isinstance(lic,dict):
            row['license_slug']=lic.get('slug','') or ''
            row['license_fullname']=lic.get('fullName','') or lic.get('label','') or ''
            row['license_url']=lic.get('url','') or ''
        else: row['license_slug']=_label(lic)
        user=doc.get('user') or {}
        if isinstance(user,dict):
            row['publisher_id']=user.get('uid') or user.get('username') or ''
            row['publisher_name']=user.get('displayName') or user.get('username') or ''
        for field,attr in [('publishedAt','published_at'),('name','model_name'),
           ('isDownloadable','is_downloadable'),('vertexCount','model_vertices_count'),
           ('faceCount','model_faces_count'),('animationCount','model_animation_count'),
           ('textureCount','model_texture_count'),('materialCount','model_material_count')]:
           row[attr]=doc.get(field,'')
        row['model_category_names']=';'.join(filter(None,map(_label,doc.get('categories') or [])))[:2000]
        row['model_tag_names']=';'.join(filter(None,map(_label,doc.get('tags') or [])))[:2000]
        thumbs=(doc.get('thumbnails') or {}).get('images',[]) if isinstance(doc.get('thumbnails'),dict) else []
        if thumbs:
            safe=sorted((x for x in thumbs if isinstance(x,dict)),key=lambda x:int(x.get('width') or 0))
            if safe:row['thumbnail_url_provider']=str(safe[-1].get('url',''))[:2048]
        row['api_status']='PROVIDER_PUBLIC_METADATA__NOT_SOURCE_RIGHTS_PASS'
    else:
        row['api_status']='PUBLIC_LOOKUP_FAILED_OR_RECORD_MISMATCH'
    return row


def run(queue_path,out,*,expected_queue_sha=None,max_requests=96,session=None):
    if expected_queue_sha and sha(queue_path)!=expected_queue_sha:
        raise ValueError('SOURCE_QUEUE_SHA_CHANGED')
    with Path(queue_path).open('r',newline='',encoding='utf-8') as f:
        records=list(csv.DictReader(f))
    if not records or len(records)>max_requests:
        raise ValueError('UNSAFE_REQUEST_COUNT')
    seen=set()
    for row in records:
        uid=row['partverse_uid']
        if not UID.fullmatch(uid) or uid in seen or not SHA.fullmatch(row['source_sha256_provider_claim']):
            raise ValueError('UNSAFE_IDENTITY_OR_DUPLICATE')
        if row['provider_split']!='PROVIDER_TRAIN_CANDIDATE':
            raise ValueError('PROTECTED_SPLIT')
        if row['source_url']!='https://sketchfab.com/3d-models/'+uid:
            raise ValueError('SOURCE_IDENTIFIER_URL_MISMATCH')
        seen.add(uid)
    if session is None:
        import requests
        session=requests.Session()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    observations=[];st=Counter()
    for row in records:
        url='https://api.sketchfab.com/v3/models/'+row['partverse_uid']
        try:
            resp=session.get(url,timeout=(10,22),headers={'Accept':'application/json','User-Agent':'RealSaS-TruthFoundry-Source-Inspection/1.0'})
            status=int(resp.status_code)
            payload=resp.json() if status==200 else None
            result=parse_public_record(row,status,payload)
        except Exception as exc:
            status=0
            result=parse_public_record(row,status,err=type(exc).__name__+':'+str(exc)[:120])
        observations.append(result)
        st[result['api_status']]+=1
        # Bound public endpoint; do not hammer on rate limits.
        if status==429:break
        time.sleep(.12)
    with (out/'SOURCE_PUBLIC_LICENSE_METADATA_V1.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=OUT_FIELDS,extrasaction='raise')
        w.writeheader();w.writerows(observations)
    summary=dict(schema='RealSaS.SketchfabPublicSourceMetadata.v1',
        source_queue_sha256=sha(queue_path),selected_sources=len(records),
        requests_attempted=len(observations),source_public_model_metadata_found=st['PROVIDER_PUBLIC_METADATA__NOT_SOURCE_RIGHTS_PASS'],
        input_queue_remaining_unqueried=len(records)-len(observations),
        source_original_file_bytes_downloaded=0,independently_verified_commercial_training_rights=0,
        independently_reviewed_anatomical_classes=0,independently_approved_stylized_2d_art=0,
        semantic_head_training_pass=0,training_authorized=False,
        api_status_counts=dict(st),output_csv_sha256=sha(out/'SOURCE_PUBLIC_LICENSE_METADATA_V1.csv'))
    (out/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return summary


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--queue',type=Path,required=True)
    a.add_argument('--queue-sha256',required=True)
    a.add_argument('--out',type=Path,required=True)
    o=a.parse_args()
    print(json.dumps(run(o.queue,o.out,expected_queue_sha=o.queue_sha256),indent=2))

if __name__=='__main__':main()