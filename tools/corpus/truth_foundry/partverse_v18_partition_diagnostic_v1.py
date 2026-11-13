#!/usr/bin/env python3
"""Truth Foundry V18 partition diagnostic over *processed PartVerse-XL GLB*.

Compare connected mesh faces and provider face2label partition under the
released GLB's observed face ordering. No original-source face-ID parity,
anatomical region, painter/drawable ownership, native control or V18 response
truth is inferred. This is a prior diagnostic, not a V18 source-court replay.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import tarfile

MAX_SOURCE_MEMBER=80*1024*1024

class UF:
    def __init__(self,n):
        self.p=list(range(n));self.r=[0]*n
    def find(self,x):
        while x!=self.p[x]:
            self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.r[a]<self.r[b]:a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]:self.r[a]+=1


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def analyze(meshes,face_map):
    """GLB face order *assumed* for exploring annotation agreement only."""
    import numpy as np
    if not meshes:raise ValueError('NO_MESH')
    n=sum(len(m.faces) for m in meshes)
    if n>1_000_000 or n==0:raise ValueError('INVALID_FACES')
    ids=[int(k) for k in face_map]
    if len(ids)!=n or set(ids)!=set(range(n)):
        raise ValueError('FACE_COUNT_OR_CONTIGUITY_MISMATCH')
    values=[face_map[str(i)] for i in range(n)]
    if any(isinstance(v,bool) or str(v).startswith('-') or not str(v).isdigit() for v in values):
        raise ValueError('INVALID_PART_ID')
    labels=np.asarray([int(v) for v in values],dtype=np.int32)
    topology=UF(n);label_conn=UF(n)
    boundary=0;adj_total=0;offset=0
    for mesh in meshes:
        e=mesh.face_adjacency
        for a,b in e:
            a=int(a)+offset;b=int(b)+offset
            topology.union(a,b)
            if labels[a]==labels[b]:label_conn.union(a,b)
            else:boundary+=1
            adj_total+=1
        offset+=len(mesh.faces)
    comp=[topology.find(i) for i in range(n)]
    conn=[label_conn.find(i) for i in range(n)]
    lab_to_comp=defaultdict(set);comp_to_lab=defaultdict(set)
    lab_to_piece=defaultdict(set)
    for l,c,lc in zip(labels,comp,conn):
        l=int(l);lab_to_comp[l].add(c);comp_to_lab[c].add(l);lab_to_piece[l].add(lc)
    return {
      'glb_faces':n,
      'glb_primitive_meshes':len(meshes),
      'processed_topology_islands':len(set(comp)),
      'partverse_part_ids_including_zero':len(lab_to_comp),
      'face_adjacency_pairs':adj_total,
      'adjacent_pairs_different_part_label':boundary,
      'part_labels_spanning_multiple_topo_islands':sum(len(v)>1 for v in lab_to_comp.values()),
      'topo_islands_split_across_part_labels':sum(len(v)>1 for v in comp_to_lab.values()),
      'within_label_connected_pieces_total':sum(map(len,lab_to_piece.values())),
      'max_disconnected_pieces_for_one_part':max(map(len,lab_to_piece.values())),
      'processed_glb_face_count_and_contiguity':True,
      'processed_glb_to_original_source_face_order':'UNKNOWN',
      'native_artist_drawable_ownership':'UNKNOWN',
      'control_edit_consequence_court':'NOT_RUN',
      'semantic_concept_court':'NOT_RUN',
      'training_authorized':False,
    }


def analyze_bytes(glb:bytes,labels:bytes):
    import trimesh
    src=json.loads(labels)
    if not isinstance(src,dict):raise ValueError('FACE_MAP_NOT_OBJECT')
    s=trimesh.load(io.BytesIO(glb),file_type='glb',force='scene',process=False)
    return analyze(list(s.geometry.values()),src)


def scan_bundle(path, sha_expected, selected_uids, seen):
    if sha(path)!=sha_expected:raise ValueError('BUNDLE_SHA_MISMATCH')
    entries=defaultdict(dict)
    with tarfile.open(path,mode='r:gz') as tf:
        for m in tf:
            if not m.isfile() or m.issym() or m.islnk():raise ValueError('UNSAFE_ARCHIVE_MEMBER')
            seg=PurePosixPath(m.name).parts
            if len(seg)!=3 or seg[0]!='source_annotation':raise ValueError('BAD_ARCHIVE_PATH')
            uid=seg[1];fname=seg[2]
            if uid not in selected_uids or uid in seen:raise ValueError('UNSELECTED_OR_DUPLICATE_UID')
            if fname not in {uid+'_segmented.glb',uid+'_face2label.json',uid+'_info.json'}:raise ValueError('UNEXPECTED_MEMBER')
            if m.size<=0 or m.size>MAX_SOURCE_MEMBER:raise ValueError('BAD_MEMBER_SIZE')
            if fname.endswith('_info.json'):continue
            if fname in entries[uid]:raise ValueError('DUPLICATE_FILENAME')
            b=tf.extractfile(m).read(m.size+1)
            if len(b)!=m.size:raise ValueError('SIZE_MISMATCH')
            entries[uid][fname]=b
    records=[]
    for uid, data in entries.items():
        if uid in seen:raise ValueError('DUPLICATE_UID')
        seen.add(uid)
        glb=data.get(uid+'_segmented.glb');lab=data.get(uid+'_face2label.json')
        if not glb or not lab:
            records.append({'uid':uid,'error':'MISSING_SEGMENTED_GLB_OR_LABEL_MAP','training_authorized':False})
            continue
        try:
            record=analyze_bytes(glb,lab)
            record.update(uid=uid,glb_sha256=hashlib.sha256(glb).hexdigest(),
                          face_map_sha256=hashlib.sha256(lab).hexdigest(),error='')
        except Exception as ex:
            record={'uid':uid,'error':type(ex).__name__+':'+str(ex)[:150],
                    'training_authorized':False}
        records.append(record)
    return records


def run(bundle_dir:Path,manifest:Path,queue:Path,out:Path):
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if m.get('schema')!='RealSaS.PartVersePartGeometryInventory.v1':
        raise ValueError('BAD_MANIFEST')
    with queue.open(newline='',encoding='utf-8') as f:
        q=list(csv.DictReader(f))
    selected={r['partverse_uid'] for r in q}
    if len(selected)!=len(q) or not selected:raise ValueError('INVALID_SELECTION')
    if any(r.get('provider_split')!='PROVIDER_TRAIN_CANDIDATE' for r in q):
        raise ValueError('PROTECTED_SPLIT_IN_SELECTION')
    seen=set();rows=[]
    for spec in m['bundles']:
        path=bundle_dir/spec['file']
        rows.extend(scan_bundle(path,spec['sha256'],selected,seen))
    if seen!=selected:raise ValueError('NOT_ALL_SELECTED_ASSETS_SCANNED')
    out.mkdir(parents=True,exist_ok=True)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'PARTVERSE_V18_PARTITION_DIAGNOSTIC_V1.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    ok=[r for r in rows if not r.get('error')]
    total=lambda key:sum(int(r[key]) for r in ok)
    report={
      'schema':'RealSaS.PartVerseV18PartitionDiagnostic.v1',
      'status':'PROCESSED_PARTITION_COMPARISON_ONLY_NOT_SOURCE_TRUTH',
      'source_candidates':len(selected),
      'structurally_compared_assets':len(ok),
      'failed_assets':len(rows)-len(ok),
      'total_processed_connected_topology_islands':total('processed_topology_islands'),
      'total_partverse_part_ids':total('partverse_part_ids_including_zero'),
      'total_cross_part_mesh_face_adjacencies':total('adjacent_pairs_different_part_label'),
      'asset_count_topology_and_part_count_different':sum(r['processed_topology_islands']!=r['partverse_part_ids_including_zero'] for r in ok),
      'asset_count_connected_topo_island_split_by_part':sum(r['topo_islands_split_across_part_labels']>0 for r in ok),
      'asset_count_part_spans_multiple_islands':sum(r['part_labels_spanning_multiple_topo_islands']>0 for r in ok),
      'v18_mechanical_causal_comparison_run':False,
      'native_drawable_pass':0,'original_source_parity_pass':0,
      'semantic_concept_pass':0,'graphics_stylized_art_pass':0,
      'training_authorized':False,
      'output_csv_sha256':sha(out/'PARTVERSE_V18_PARTITION_DIAGNOSTIC_V1.csv'),
    }
    (out/'SUMMARY_V1.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--glb',type=Path);p.add_argument('--face2label',type=Path)
    p.add_argument('--bundle-dir',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--queue',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.glb and a.face2label:
        r=analyze_bytes(a.glb.read_bytes(),a.face2label.read_bytes())
        a.out.mkdir(parents=True,exist_ok=True)
        (a.out/'SUMMARY_V1.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    elif a.bundle_dir and a.manifest and a.queue:
        r=run(a.bundle_dir,a.manifest,a.queue,a.out)
    else:raise SystemExit('Provide (--glb and --face2label) OR (--bundle-dir --manifest --queue)')
    print(json.dumps(r,indent=2,sort_keys=True))

if __name__=='__main__':main()