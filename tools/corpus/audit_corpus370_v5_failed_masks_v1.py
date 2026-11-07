#!/usr/bin/env python3
"""Diagnostic-only Census of 18 still-failing Corpus370 V5 source silhouettes.

Provenance: original admitted raster PNG + declared source mask + source-projector
reference/failed overlay. This never relabels, thresholds, or edits the 18 FAILs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

V5_SHA = "8ab2ebf526bbf91d78117580b68ed3784e53bcbafdbd326f012e143c924c0ac4"
SCHEMA = "RealSaS.Corpus370V5RemainingMaskDiagnostic.v1"


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def require(value, code):
    if not value:
        raise ValueError(code)


def box(binary):
    coords = np.argwhere(binary)
    if not len(coords):
        return None
    y0,x0 = coords.min(axis=0)
    y1,x1 = coords.max(axis=0)
    return [int(x0),int(y0),int(x1+1),int(y1+1)]


def audit(v5_path, output):
    require(sha(v5_path)==V5_SHA, 'CORPUS370_MASK_DIAG_V5_INDEX_SHA_DRIFT')
    index=json.loads(v5_path.read_text())
    failures=index['failures']
    require(index['schema']=='RealSaS.Corpus370PreparedObservationPackets.v5'
            and index['failure_count']==len(failures)==18
            and index['prepared_subject_count']==352
            and index['training_authorized'] is False,
            'CORPUS370_MASK_DIAG_SCOPE')
    root=v5_path.parent/'v4_new'/'objects'
    rows=[]
    tiles=[]
    for i,failure in enumerate(sorted(failures,key=lambda x:x['subject_id']),1):
        sid=failure['subject_id']
        reason=str(failure.get('reason',''))
        m=re.fullmatch(r'ADMISSION_MASK_DRIFT:[^:]+:V(\d+):([\d\.e+-]+)',reason)
        require(m and sid in reason,
                'CORPUS370_MASK_DIAG_UNEXPECTED_FAILURE:'+sid)
        view=int(m.group(1))
        gap=float(m.group(2))
        require(gap>2.0, 'CORPUS370_MASK_DIAG_FALSE_MASK_FAIL:'+sid)
        found=list(root.glob('*/admission/'+sid+'/V'+str(view)+'.FAILED.png'))
        require(len(found)==1, 'CORPUS370_MASK_DIAG_FAILED_OVERLAY_COUNT:'+sid)
        failed=found[0]
        object_root=failed.parents[2]
        snapshot=object_root/'observations'/sid/'SOURCE_REST_RENDER_CANDIDATE.json'
        raster=object_root/'observations'/sid/f'V{view}.png'
        mask=object_root/'observations'/sid/f'V{view}.mask.bin'
        require(snapshot.is_file() and raster.is_file() and mask.is_file(),
                'CORPUS370_MASK_DIAG_MISSING_RENDER:'+sid)
        candidate=json.loads(snapshot.read_text())
        require(candidate['subject_id']==sid
                and candidate['source_sha256']==failure['source_sha256']
                and candidate['views'][view]['raster']['sha256']==sha(raster)
                and candidate['views'][view]['foreground_mask']['sha256']==sha(mask),
                'CORPUS370_MASK_DIAG_RENDER_PARENT_DRIFT:'+sid)
        overlay=np.asarray(Image.open(failed).convert('RGB'))
        reference=overlay[:,:,0]>0
        actual=overlay[:,:,1]>0
        rgba=np.asarray(Image.open(raster).convert('RGBA'))
        require(actual.shape==rgba.shape[:2] and np.array_equal(actual,rgba[:,:,3]>0),
                'CORPUS370_MASK_DIAG_OVERLAY_RASTER_MISMATCH:'+sid)
        binary=np.frombuffer(mask.read_bytes(),dtype=np.uint8).reshape(actual.shape)>0
        require(np.array_equal(binary,actual),
                'CORPUS370_MASK_DIAG_BINARY_ALPHA_MISMATCH:'+sid)
        reference_only=reference&~actual
        raster_only=actual&~reference
        count_ref=int(np.count_nonzero(reference_only))
        count_ras=int(np.count_nonzero(raster_only))
        connected_ref=int(label(reference_only)[1])
        connected_ras=int(label(raster_only)[1])
        row={
            'subject_id':sid,
            'source_sha256':failure['source_sha256'],
            'source_id':failure.get('source_id'),
            'source_format':failure.get('source_format'),
            'view_index':view,'failure_gap_px':gap,
            'reference_only_pixels':count_ref,
            'raster_only_pixels':count_ras,
            'reference_only_bbox':box(reference_only),
            'raster_only_bbox':box(raster_only),
            'reference_only_components':connected_ref,
            'raster_only_components':connected_ras,
            'actual_foreground_pixels':int(np.count_nonzero(actual)),
            'reference_foreground_pixels':int(np.count_nonzero(reference)),
            'overlap_pixels':int(np.count_nonzero(actual&reference)),
            'original_raster_sha256':sha(raster),
            'original_mask_sha256':sha(mask),
            'diagnostic_overlay_sha256':sha(failed),
            'training_authorized':False,
        }
        rows.append(row)
        # Diagnostic sheet for visual inspection, not an authority source.
        sheet=np.zeros((actual.shape[0],actual.shape[1],3),dtype=np.uint8)
        sheet[actual&reference]=[200,200,200]
        sheet[reference_only]=[245,55,55]
        sheet[raster_only]=[45,170,245]
        sheet_image=Image.fromarray(sheet,'RGB')
        sheet_image.thumbnail((160,160))
        tiles.append((row,sheet_image))
        print(json.dumps({'event':'CORPUS370_MASK_DIAG',
              'done':i,'total':18,'subject_id':sid,'view_index':view,
              'gap_px':gap,'reference_only':count_ref,
              'raster_only':count_ras}),flush=True)
    output.mkdir(parents=True,exist_ok=True)
    cols=3
    w=370;h=215
    canvas=Image.new('RGB',(cols*w,((len(tiles)+cols-1)//cols)*h),(25,25,31))
    draw=ImageDraw.Draw(canvas)
    for i,(row,tile) in enumerate(tiles):
        x=(i%cols)*w;y=(i//cols)*h
        draw.text((x+8,y+7),row['subject_id'],fill=(255,255,255))
        draw.text((x+8,y+25),str(row['source_id'])+' '+str(row['source_format']),fill=(205,205,205))
        draw.text((x+8,y+43),f"V{row['view_index']} gap {row['failure_gap_px']:.2f}px | ref-only {row['reference_only_pixels']} | raster-only {row['raster_only_pixels']}",fill=(220,220,220))
        canvas.paste(tile,(x+6,y+65))
    canvas_path=output/'18_MASK_FAILURE_CONTACT_SHEET.png'
    canvas.save(canvas_path)
    report={
        'schema':SCHEMA,
        'status':'DIAGNOSTIC_ONLY__NO_MASK_ADMISSION',
        'parent_v5_index_sha256':V5_SHA,
        'failure_count':18,'mask_failed_count':18,
        'family_counts':dict(Counter(r['source_id'] for r in rows)),
        'per_subject':rows,
        'legend':'grey=overlap, red=source-projection-only, blue=Blender-raster-only',
        'diagnostic_contact_sheet_sha256':sha(canvas_path),
        'training_authorized':False,
        'new_observation_pass_minted':False,
    }
    (output/'CORPUS370_V5_18_MASK_DIAGNOSTIC.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v5-index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.v5_index,a.output)
    print(json.dumps({'failure_count':result['failure_count'],
                      'family_counts':result['family_counts']},sort_keys=True),flush=True)


if __name__=='__main__':
    main()
