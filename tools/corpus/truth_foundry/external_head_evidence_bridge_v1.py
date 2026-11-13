#!/usr/bin/env python3
"""Bridge existing 4363 source census and 7500 external candidate records.

Produces one source/head evidence stage row. *Not* truth certification, source
identity merging or training admission. Avoids turning provider label counts
and same Sketchfab IDs into licensed characters.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

HEADS=('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')
FIELDS=['source_key','corpus','provider','source_identifier','claimed_source_sha256',
        'provider_class_weak','head','evidence_stage','evidence_origin','observed_value',
        'original_rights_court','source_original_byte_court',
        'independent_head_target_court','morphology_visual_court',
        'semantic_concept_court','2d_art_court','training_authorized','materialization_authorized']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def rows(p):
    with p.open('r',newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def needed(valid,why):
    if not valid:raise ValueError('EXTERNAL_HEAD_BRIDGE_'+why)


def form(source_key,corpus,provider,source_identifier,source_sha,weak,head,stage,origin,value):
    return dict(source_key=source_key,corpus=corpus,provider=provider,
      source_identifier=source_identifier,claimed_source_sha256=source_sha,
      provider_class_weak=weak,head=head,evidence_stage=stage,
      evidence_origin=origin,observed_value=value,
      original_rights_court='UNKNOWN',source_original_byte_court='UNKNOWN',
      independent_head_target_court='UNKNOWN',morphology_visual_court='UNKNOWN',
      semantic_concept_court='UNKNOWN',**{'2d_art_court':'UNKNOWN'},
      training_authorized=False,materialization_authorized=False)


def run(base_csv,partverse_csv,artic_csv,uniml_csv,out):
    base=rows(base_csv);parts=rows(partverse_csv);art=rows(artic_csv);uniml=rows(uniml_csv)
    needed(len(base)==4363 and len(parts)==2500 and len(art)==2500 and len(uniml)==2500,'FROZEN_COUNTS')
    needed(sum(r.get('corpus')=='CORE370' for r in base)==370,'CORE_COUNT')
    needed(all(p['face_count_equal_triangles']=='True' and p['face2label_exists']=='True' for p in parts),'PART_FACES_UNQUALIFIED')
    needed(all(x.get('provider_split')=='PROVIDER_TRAIN_CANDIDATE' for x in parts),'PART_HELDOUT')
    needed(all(x.get('dataset')=='objaverse' for x in uniml),'UNIML_SOURCE')
    source_keys=[];out_rows=[];observed=Counter()
    def write_item(key,cohort,provider,source_id,source_sha,weak,observations):
        source_keys.append(key)
        for head in HEADS:
            stage, origin, v=observations.get(head,('UNKNOWN','NONE',''))
            out_rows.append(form(key,cohort,provider,source_id,source_sha,weak,head,stage,origin,v))
            observed[(cohort,head,stage)]+=1

    for b in base:
        cohort=b['corpus'];key=cohort+':'+b['asset_id']
        physical=b.get('physical_census_loaded','').lower()=='true'
        obs={}
        if physical:
            obs['IRIS_GEOMETRY']=('IMPORTED_PHYSICAL_COUNTS_ONLY','MASTER_SCENE_IMPORT',b.get('vertex_count',''))
            obs['TESSA']=('IMPORTED_PHYSICAL_COUNTS_ONLY','MASTER_SCENE_IMPORT',b.get('mesh_disconnected_components',''))
            obs['AXIS']=('IMPORTED_BONE_COUNTS_ONLY','MASTER_SCENE_IMPORT',b.get('observed_bone_count',''))
            obs['MIRA']=('IMPORTED_SKIN_ROW_COUNTS_ONLY','MASTER_SCENE_IMPORT',b.get('skin_rows_sampled',''))
            obs['IRIS_GRAPHICS']=('IMPORTED_MATERIAL_COUNTS_ONLY','MASTER_SCENE_IMPORT',b.get('image_count',''))
            if int(b.get('mesh_disconnected_components','') or 0)>1:
                obs['IRIS_SEMANTIC']=('MODULAR_GEOMETRY_HINT_ONLY','MASTER_SCENE_IMPORT',b['mesh_disconnected_components'])
        write_item(key,cohort,b['provider'],b['asset_id'],b['source_sha256'],b.get('morphology','UNKNOWN'),obs)

    for p in parts:
        uid=p['uid'];key='PARTVERSE_XL:'+uid
        write_item(key,'EXTERNAL_CANDIDATE','PARTVERSE_XL',uid,p['source_sha_provider_claim'],p['weak_subject_candidate'],{
            'IRIS_GEOMETRY':('PROCESSED_SEGMENTED_GLBFACE_COUNT_PARITY','PARTVERSE_FACE_MAP_AND_SEGMENTED_GLB',p['triangle_face_count']),
            'IRIS_SEMANTIC':('PROCESSED_3D_FACE_PART_INSTANCE_PROPOSAL','PARTVERSE_FACE2LABEL_AND_INFO',p['part_id_count_including_zero']),
            'IRIS_GRAPHICS':('PART_GEOMETRY_VISIBILITY_CANDIDATE_ONLY','PARTVERSE_PROCESSED_GLB',''),
            'TESSA':('PROCESSED_SEGMENTED_MESH_TOPOLOGY_CANDIDATE','PARTVERSE_PROCESSED_GLB',p['raw_mesh_scene_primitive_count']),
        })

    for a in art:
        uid=a['uuid'];key='ARTICULATION_XL2:'+uid
        write_item(key,'EXTERNAL_CANDIDATE','ARTICULATION_XL2',uid,'',a['candidate_family'],{
            'IRIS_GEOMETRY':('METADATA_VERTEX_FACE_COUNTS_ONLY','ARTICULATION_XL2_CSV',a['face_count']),
            'TESSA':('METADATA_TOPOLOGY_COUNTS_ONLY','ARTICULATION_XL2_CSV',a['face_count']),
            'AXIS':('METADATA_JOINT_BONE_COUNTS_ONLY','ARTICULATION_XL2_CSV',a['joint_count']),
            'MIRA':('METADATA_RIG_HAS_SKIN_CANDIDATE_ONLY','ARTICULATION_XL2_CSV',a['bone_count']),
        })
    for u in uniml:
        uid=u['object_type'];key='UNIML3D:'+uid
        write_item(key,'EXTERNAL_CANDIDATE','UNIML3D',uid,'',u['category'],{
            'AXIS':('PROVIDER_SKELETON_CLIP_COUNTS_ONLY','UNIML3D_SKELETONS_CSV',u['num_joints']),
        })

    needed(len(source_keys)==11863 and len(set(source_keys))==11863,'DUPLICATE_KEYS')
    needed(len(out_rows)==71178,'SIX_HEAD_ROWS')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'FOUNDRY_UNIFIED_SOURCE_HEAD_OBSERVATIONS_V1.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='raise');w.writeheader();w.writerows(out_rows)

    part_ids={p['uid'] for p in parts}; art_ids={a['uuid'] for a in art};uniml_ids={u['object_type'] for u in uniml}
    memberships=Counter()
    for uid in (part_ids|art_ids|uniml_ids):
        memberships[tuple(name for name,idset in [('PARTVERSE_XL',part_ids),('ARTICULATION_XL2',art_ids),('UNIML3D',uniml_ids)] if uid in idset)]+=1
    overlaps={
        'partverse_articulation_exact_provider_uid':len(part_ids&art_ids),
        'partverse_uniml_exact_provider_uid':len(part_ids&uniml_ids),
        'articulation_uniml_exact_provider_uid':len(art_ids&uniml_ids),
        'all_three_exact_provider_uid':len(part_ids&art_ids&uniml_ids),
        'unique_external_provider_id_strings_not_source_clusters':len(part_ids|art_ids|uniml_ids),
    }
    all_shas={r['source_sha256'] for r in base if r.get('source_sha256')}
    part_cross_sha=sum(p['source_sha_provider_claim'] in all_shas for p in parts)
    summary={
        'schema':'RealSaS.TruthFoundryExternalHeadEvidenceBridge.v1',
        'source_record_count':len(source_keys),'source_head_rows':len(out_rows),
        'source_records_base':len(base),'source_records_external_candidates':len(parts)+len(art)+len(uniml),
        'partverse_structural_face_region_candidates':len(parts),
        'partverse_labeled_triangle_count':sum(int(p['labeled_faces']) for p in parts),
        'partverse_part_id_count_source_local_sum':sum(int(p['part_id_count_including_zero']) for p in parts),
        'partverse_zero_label_face_count':sum(int(p['zero_label_face_count']) for p in parts),
        'partverse_info_unlisted_faces':sum(int(p['face_count_label0_unlisted_by_info'] or 0) for p in parts),
        'partverse_unmeasured_component_count_due_graph_dependency':sum(bool(p['processing_errors'].startswith('COMPONENT_COUNT:')) for p in parts),
        'partverse_count_contiguity_glb_match':sum(p['face_count_equal_triangles']=='True' for p in parts),
        'core370_source_record_count':370,
        'cross_external_provider_uid_duplicate_candidates':overlaps,
        'base_source_sha_present_count':sum(bool(b.get('source_sha256')) for b in base),
        'partverse_provider_sha_matches_existing_claim':part_cross_sha,
        'new_independent_qualified_head_truth_from_bridge':0,
        'new_independent_qualified_semantic_concepts_from_bridge':0,
        'per_head_evidence_stage_counts':{h:dict(Counter({stage:count for (co,head,stage),count in observed.items() if head==h for stage,count in [(stage,count)]})) for h in HEADS},
        'head_training_authorized':{h:0 for h in HEADS},
        'copyright_and_source_rights_court':'REQUIRED_PER_SOURCE__NOT_GRANTED',
        'training_authorized':False,
        'input_sha256':{label:sha(path) for label,path in [('baseline',base_csv),('partverse',partverse_csv),('articulation',artic_csv),('uniml',uniml_csv)]},
        'output_sha256':sha(out/'FOUNDRY_UNIFIED_SOURCE_HEAD_OBSERVATIONS_V1.csv'),
    }
    # Safer aggregate grouped by head and evidence: do not overwrite stages from different cohorts.
    stage_totals=defaultdict(Counter)
    for row in out_rows:stage_totals[row['head']][row['evidence_stage']]+=1
    summary['per_head_evidence_stage_counts']={h:dict(stage_totals[h]) for h in HEADS}
    (out/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--partverse',type=Path,required=True)
    p.add_argument('--articulation',type=Path,required=True)
    p.add_argument('--uniml',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.base,a.partverse,a.articulation,a.uniml,a.out),indent=2))

if __name__=='__main__':main()