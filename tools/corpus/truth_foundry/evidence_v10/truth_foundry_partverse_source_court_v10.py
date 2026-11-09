#!/usr/bin/env python3
"""V10 independent processed-source PartVerse graph and rights-evidence census.

All inputs are source-bound *derived* receipts, not creator-original GLB bytes.
No ontology, drawable, global SDF, ML rights or training authority is minted.
No student-visible semantic label is taken from part IDs, names, or weak metadata.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1<<20), b''):
            h.update(buf)
    return h.hexdigest()


def source_lines(paths):
    for p in paths:
        with p.open('r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                yield r


def flag_for(r, key, accept):
    if r[key] not in accept:
        raise AssertionError(('BAD_BOOLEAN_OR_POLICY_STATE', key, r[key]))


def connected(n, ed):
    parent=list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a,b in ed:
        parent[root(a)] = root(b)
    size=collections.Counter(root(i) for i in range(n))
    return len(size),max(size.values()) if size else 0


def report(args):
    input_paths=[Path(x).resolve() for x in [args.region_first,args.region_second,args.adjacency_first,args.adjacency_second,args.license]]
    for p in input_paths:
        if not p.is_file():raise FileNotFoundError(p)
    input_hashes={p.name+'@'+str(i):digest(p) for i,p in enumerate(input_paths)}
    if args.sha_manifest:
        ref=json.loads(Path(args.sha_manifest).read_text())
        for i,p in enumerate(input_paths):
            key=['region_first','region_second','adjacency_first','adjacency_second','license'][i]
            assert input_hashes[p.name+'@'+str(i)]==ref[key], ('INPUT_SHA_MISMATCH',key)

    rights={}
    for r in source_lines([input_paths[-1]]):
        uid=r['uid'];assert uid not in rights,('LICENSE_DUP_UID',uid)
        assert r['provider_original_sha_claim']==r['objaverse_sha256_recorded'] and len(r['provider_original_sha_claim'])==64
        assert r['source_metadata_match_state']=='EXACT_SHA_AND_SOURCE_UID_METADATA_CLAIM'
        assert r['head_training_authorized']=='False' and r['head_training_eligible']=='False'
        assert r['original_source_bytes_verified']=='False'
        rights[uid]=r
    assert len(rights)==7034, ('LICENSE_ROSTER_DRIFT',len(rights))

    regions=collections.defaultdict(dict)
    tri_count=0; part_ct=0; local_errors=collections.Counter()
    for r in source_lines(input_paths[:2]):
        uid=r['source_uid'];pid=r['part_id']; assert uid in rights,('REGION_LICENSE_MISSING',uid)
        assert pid not in regions[uid],('DUP_PART_ID',uid,pid)
        flag_for(r,'training_mask_valid',{'False'})
        assert r['part_semantic_truth_status']=='PROCESSED_PART_INSTANCE_ONLY'
        assert r['part_anatomical_or_functional_concept']=='UNKNOWN'
        assert r['source_face_parity_court']=='UNKNOWN'
        n=int(r['part_triangle_count']);lo=int(r['part_face_id_min']);hi=int(r['part_face_id_max']);
        pieces=int(r['part_intra_label_connected_pieces']); ar=float(r['part_local_surface_area']);frac=float(r['part_area_fraction'])
        assert n>0 and lo>=0 and hi>=lo and hi-lo+1==n,('FACE_RANGE_COUNT_BAD',uid,pid)
        assert pieces>0 and ar>=0 and math.isfinite(ar) and frac>=0 and math.isfinite(frac)
        assert int(r['part_mesh_primitive_count'])>=1
        assert len(r['part_face_ids_sha256'])==64
        regions[uid][pid]={
            'triangles':n,'begin':lo,'end':hi,'area':ar,'fraction':frac,'pieces':pieces,
            'neighbors_claimed':int(r['part_neighbor_distinct_ids']),
            'face_pairs_claimed':int(r['part_external_adjacent_face_pairs']),
            'zero':r['is_zero_label']=='True',
            'face_ids_digest':r['part_face_ids_sha256']}
        tri_count+=n;part_ct+=1
    assert len(regions)==7028 and part_ct==69742, ('REGION_ROSTER_OR_PART_COUNT_DRIFT',len(regions),part_ct)

    graph=collections.defaultdict(list);seen_edge=set();pair_ct=0
    for r in source_lines(input_paths[2:4]):
        uid=r['source_uid'];a=r['part_a'];b=r['part_b'];assert uid in regions
        assert a in regions[uid] and b in regions[uid] and a!=b,('ADJ_ORPHAN_OR_SELF',uid,a,b)
        key=(uid,*sorted((a,b)));assert key not in seen_edge,('DUP_ADJ',key)
        seen_edge.add(key)
        assert r['training_mask_valid']=='False' and r['source_surface_causality']=='NOT_TESTED'
        count=int(r['processed_adjacent_face_pairs']);assert count>0
        graph[uid].append((a,b,count));pair_ct+=1
    assert pair_ct==9881

    dest=Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    out=dest/'PARTVERSE_SOURCE_BOUND_PART_GRAPH_V10_7034.csv.gz'
    partout=dest/'PARTVERSE_SOURCE_BOUND_PART_DETAIL_V10_69742.csv.gz'
    fields=['source_key','source_uid','provider_original_sha_claim','license_class_claim','license_policy_triage',
    'processed_parts_available','processed_numeric_court','processed_face_count','processed_part_label_count',
    'processed_part_area_fraction_sum','processed_part_area_largest_fraction','processed_part_area_smallest_fraction',
    'processed_part_area_gini','processed_intralabel_piece_count','processed_labels_with_multiple_islands',
    'processed_interpart_edges','processed_external_adjacent_face_pairs','processed_graph_components',
    'processed_graph_isolated_labels','processed_graph_cycle_rank','processed_graph_max_degree',
    'processed_graph_max_component_size','processed_first_face_id','processed_last_face_id',
    'processed_face_partition_gap_count','processed_face_partition_overlap_count',
    'processed_part_neighbor_parity','processed_part_face_pair_parity','processed_region_exception',
    'weak_provider_morphology_NOT_TRUTH','weak_provider_style_NOT_TRUTH',
    'original_creator_bytes_rehashed','original_mesh_vs_processed_parity','anatomical_semantic_truth','drawable_layer_truth',
    'global_sdf_sign_truth','training_authorized']
    partfields=['source_key','part_id','triangles','face_id_start','face_id_end','face_ids_sha256','area_fraction',
        'connected_pieces','degree','boundary_adjacent_face_pairs','part_role','original_source_part_parity','training_authorized']
    stats=collections.Counter();candidates=[]
    with gzip.open(partout,'wt',encoding='utf8',newline='') as pf:
        pw=csv.DictWriter(pf,fieldnames=partfields);pw.writeheader()
        for uid in sorted(rights):
            src=rights[uid];parts=regions.get(uid,{})
            edge=graph.get(uid,[])
            if not parts:
                status='NO_PROCESSED_PART_SOURCE_RECEIPT';stats['missing_processed']+=1
                p_rows=[];metrics={}
            else:
                ids=sorted(parts,key=lambda p: int(p) if p.isdecimal() else p)
                idx={x:i for i,x in enumerate(ids)}
                degree=collections.Counter();neighbors=collections.Counter();pcount=collections.Counter()
                for a,b,n in edge:
                    degree[a]+=1;degree[b]+=1
                    pcount[a]+=n;pcount[b]+=n
                nr=sorted(parts.values(),key=lambda x:x['begin']);gaps=0;overlaps=0;expected=0
                for x in nr:
                    if x['begin']>expected:gaps+=x['begin']-expected
                    if x['begin']<expected:overlaps+=expected-x['begin']
                    expected=max(expected,x['end']+1)
                fractions=[x['fraction'] for x in nr];S=sum(fractions)
                n=len(fractions)
                gini=(sum((2*i-n-1)*a for i,a in enumerate(sorted(fractions),1))/(n*S)) if S else 0
                errors=[]
                if gaps:errors.append('FACE_GAPS')
                if overlaps:errors.append('FACE_OVERLAPS')
                if abs(S-1)>0.001:errors.append('PART_FRACTION_NOT_CLOSED')
                wrong_neighbors=sum(degree[pid]!=parts[pid]['neighbors_claimed'] for pid in parts)
                wrong_facepairs=sum(pcount[pid]!=parts[pid]['face_pairs_claimed'] for pid in parts)
                if wrong_neighbors:errors.append('NEIGHBOR_PARITY')
                if wrong_facepairs:errors.append('FACE_PAIR_PARITY')
                comps,largest=connected(n,[(idx[a],idx[b]) for a,b,_ in edge])
                cycle_rank=len(edge)-n+comps
                assert cycle_rank>=0
                status='PASS_PROCESSED_PART_GEOMETRY_NUMERIC_ONLY' if not errors else 'FAIL_PROCESSED_PART_RECEIPT_'+','.join(errors)
                stats['processed_'+('pass' if not errors else 'fail')]+=1
                for err in errors:stats['error_'+err]+=1
                metrics={
                    'processed_face_count':sum(x['triangles'] for x in nr),
                    'processed_part_label_count':n,
                    'processed_part_area_fraction_sum':S,
                    'processed_part_area_largest_fraction':max(fractions),
                    'processed_part_area_smallest_fraction':min(fractions),
                    'processed_part_area_gini':gini,
                    'processed_intralabel_piece_count':sum(x['pieces'] for x in nr),
                    'processed_labels_with_multiple_islands':sum(x['pieces']>1 for x in nr),
                    'processed_interpart_edges':len(edge),
                    'processed_external_adjacent_face_pairs':sum(v for _,_,v in edge),
                    'processed_graph_components':comps,
                    'processed_graph_isolated_labels':sum(not degree[p] for p in ids),
                    'processed_graph_cycle_rank':cycle_rank,
                    'processed_graph_max_degree':max(degree.values(),default=0),
                    'processed_graph_max_component_size':largest,
                    'processed_first_face_id':min(x['begin'] for x in nr),
                    'processed_last_face_id':max(x['end'] for x in nr),
                    'processed_face_partition_gap_count':gaps,
                    'processed_face_partition_overlap_count':overlaps,
                    'processed_part_neighbor_parity': 'PASS' if not wrong_neighbors else 'FAIL',
                    'processed_part_face_pair_parity':'PASS' if not wrong_facepairs else 'FAIL',
                    'processed_region_exception':'NONE' if not errors else '|'.join(errors),
                }
                if metrics['processed_labels_with_multiple_islands']>0:stats['multicomponent_labels']+=1
                if metrics['processed_graph_isolated_labels']>0:stats['isolated_labels']+=1
                if edge:stats['any_adjacency']+=1
                for pid in ids:
                    p=parts[pid]
                    pw.writerow({'source_key':'PARTVERSE_XL:'+uid,'part_id':pid,'triangles':p['triangles'],
                        'face_id_start':p['begin'],'face_id_end':p['end'],'face_ids_sha256':p['face_ids_digest'],
                        'area_fraction':p['fraction'],'connected_pieces':p['pieces'],'degree':degree[pid],
                        'boundary_adjacent_face_pairs':pcount[pid],'part_role':'UNKNOWN',
                        'original_source_part_parity':'NOT_COURTED','training_authorized':'FALSE'})
            row={'source_key':'PARTVERSE_XL:'+uid,'source_uid':uid,
                'provider_original_sha_claim':src['provider_original_sha_claim'],
                'license_class_claim':src['corrected_license_class_v2'],
                'license_policy_triage':src['source_license_policy_triage'],
                'processed_parts_available':bool(parts),'processed_numeric_court':status,
                **metrics,
                'weak_provider_morphology_NOT_TRUTH':src['provider_weak_morphology'],
                'weak_provider_style_NOT_TRUTH':src['provider_weak_style'],
                'original_creator_bytes_rehashed':'FALSE','original_mesh_vs_processed_parity':'UNKNOWN',
                'anatomical_semantic_truth':'UNKNOWN','drawable_layer_truth':'UNKNOWN',
                'global_sdf_sign_truth':'UNKNOWN','training_authorized':'FALSE'}
            candidates.append(row)
            stats['licenses_'+src['corrected_license_class_v2']]+=1
    assert len(candidates)==7034
    with gzip.open(out,'wt',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(candidates)
    summary={'schema':'RealSaS.PartVerseProcessedSourceBoundCourt.v10',
             'scope':'PROCESSED_MESH_PART_GRAPH_PLUS_PROVIDER_METADATA__NO_ORIGINAL_GLTF_PARITY',
             'original_creator_bytes_rehashed':0,'independent_morphology_court_pass':0,
             'semantic_ontology_pass':0,'drawable_ownership_pass':0,
             'global_sdf_sign_pass':0,'training_admission':0,
             'provider_license_metadata_uid_sha_crosswalk':7034,
             'source_rows':len(candidates),'processed_source_rows':len(regions),
             'processed_part_rows':part_ct,'processed_adjacency_rows':pair_ct,
             'processed_triangles':tri_count,
             'counts':dict(stats),
             'source_input_sha256':input_hashes,
             'output_sha256':{out.name:digest(out),partout.name:digest(partout)}}
    p=dest/'SUMMARY_PARTVERSE_V10.json';p.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))
    assert stats['processed_fail']==0 and stats['processed_pass']==7028 and stats['missing_processed']==6
    assert stats['licenses_NONCOMMERCIAL_OR_RESTRICTED']==301
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for x in ('region_first','region_second','adjacency_first','adjacency_second','license','out'):
        ap.add_argument('--'+x.replace('_','-'),required=True)
    ap.add_argument('--sha-manifest')
    report(ap.parse_args())