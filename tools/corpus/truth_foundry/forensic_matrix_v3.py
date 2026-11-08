#!/usr/bin/env python3
"""RealSaS Truth Foundry: evidence-rich 11,863 source forensic inventory.

Bridge all currently observed, SHA-receipted source evidence (not training
admission) into the existing 264-facet + 37-target contracts. Originals are
represented by an already frozen Master3993/Core370 facet ledger; external
sources contribute typed metadata/processed-mesh/canonical-GLB observations.
No consumer can interpret a COUNT, a claimed anatomy or a face-part proposal
as independently reviewed semantic, graphics, or mechanical truth.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HEADS = ('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')
SCHEMA='RealSaS.TruthFoundryForensicSourceMatrix.v3'
TARGET_COLUMNS=['source_key','head','target','proposed_owner','resolved_consumer_or_court',
                'required_independent_court','provenance_basis','value_observed',
                'observation_state','reviewed_truth_status','training_mask_valid',
                'source_rights_court','source_lineage_court','source_original_sha_verified']
PRIOR_SOURCE_HASH='b03fc7b378ab5e8f70bd83f55822372eb5516746cc3d0b96286a1e903d0d2d1f'
OBS_PROCESSED='PROCESSED_MESH_OBSERVATION__NOT_ORIGINAL_PARITY'
OBS_CANONICAL='CANONICAL_GLB_STRUCTURE__NOT_ORIGINAL_PARITY'
OBS_META='PROVIDER_METADATA_CLAIM__NOT_SOURCE_TRUTH'
UNKNOWN='UNKNOWN_UNREVIEWED'

# For fields outside the canonical 264 facets, keep ALL provider columns in
# separate per-asset raw evidence. Never silently collapse rig/texture data.
BASE_FACET_MAP = {}  # Existing 4363 baseline has its separately validated observations.
PART_FACET_MAP = {
  'source_vertex_face_count':('triangle_face_count',OBS_PROCESSED),
  'mesh_object_vs_connected_island':('raw_mesh_scene_primitive_count',OBS_PROCESSED),
  'per_surface_instance_identity':('part_id_count_including_zero',OBS_PROCESSED),
  'part_to_component_many_to_many':('part_id_count_including_zero',OBS_PROCESSED),
  'source_face_concept_review':('part_id_count_including_zero',
      'SEGMENTATION_PROPOSAL_ONLY__CONCEPT_NOT_REVIEWED'),
}
ART_FACET_MAP = {
  'source_vertex_face_count':('face_count',OBS_META),
  'joint_parent_hierarchy_and_cycles':('joint_count',OBS_META),
  'skin_sparse_influence_index_weight':('bone_count',
      'PROVIDER_BONE_COUNT_ONLY__SKIN_NOT_READ'),
  'joint_rest_and_bind_world_local_matrices':('joint_count',
      'PROVIDER_JOINT_COUNT_ONLY__MATRICES_NOT_READ'),
}
UNIML_FACET_MAP = {
  'joint_parent_hierarchy_and_cycles':('num_joints',OBS_META),
  'source_action_curve_inventory':('num_clips',
      'PROVIDER_CLIP_COUNT_ONLY__CURVES_NOT_READ'),
  'anatomical_front_axis':('face_source',
      'PROVIDER_FACING_METADATA__NATIVE_FRAME_NOT_COURTED'),
}
UNIML_GLB_FACET_MAP = {
  'source_vertex_face_count':('glb_triangle_indices_count',OBS_CANONICAL),
  'source_material_slot_inventory':('glb_material_count',OBS_CANONICAL),
  'native_texture_image_references':('glb_image_count',OBS_CANONICAL),
  'source_action_curve_inventory':('glb_animation_count',OBS_CANONICAL),
  'joint_parent_hierarchy_and_cycles':('glb_skin_joint_count',
      'CANONICAL_GLB_SKIN_JOINT_COUNT__HIERARCHY_NOT_COURTED'),
}

TARGET_FIELD_HINTS = {
  'source_face_region_support': 'part_id_count_including_zero',
  'instance_component_membership': 'part_id_count_including_zero',
  'source_face_component_graph': 'triangle_face_count',
  'mesh_connectivity_and_charts': 'raw_mesh_scene_primitive_count',
  'mesh_surface_xyz': 'vertex_count',
  'source_native_rgba_evidence': 'image_count',
  'source_bone_hierarchy': 'observed_bone_count',
  'joint_rest_bind': 'observed_bone_count',
  'skin_influences': 'skin_rows_sampled',
}


def require(flag,code):
    if not flag: raise ValueError('FORENSIC_V3_'+code)


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):d.update(chunk)
    return d.hexdigest()


def csv_rows(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def initial_facets(keys):
    return {k:{'observation_state':UNKNOWN,'observed_value':None,
               'independent_court_state':'UNKNOWN','court_receipt_sha256':None,
               'supervision_status':'NOT_ADMITTED','training_mask_valid':False}
            for k in keys}


def obs(facets,key,raw,stage):
    require(key in facets,'UNKNOWN_FACET_'+key)
    if raw in (None,'','UNKNOWN','None'):
        return False
    if stage not in (OBS_PROCESSED,OBS_CANONICAL,OBS_META) and not (
        stage.startswith(('PROVIDER_','CANONICAL_','SEGMENTATION_'))):
        raise ValueError('UNRECOGNIZED_OBSERVATION_STAGE')
    # All externally observed values are explicitly NOT an independent PASS.
    facets[key].update(observation_state=stage,observed_value=raw)
    return True


def csv_write(path,fields,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
        w.writeheader()
        w.writerows(rows)


def run(root:Path,out:Path,*,strict=True,expected_source_n=11863):
    root,out=Path(root),Path(out)
    source=root/'SOURCE_RECORD_CENSUS_4363.csv'
    baseline=root/'ASSET_FACET_EVIDENCE_MATRIX_V1.jsonl.gz'
    part=root/'ASSET_PART_GEOMETRY_EVIDENCE_V1.csv'
    art=root/'truth_foundry_parallel_20261108/ARTICULATION_2500_SOURCE_CANDIDATE_QUEUE.csv'
    uniml=root/'truth_foundry_parallel_20261108/UNIML3D_2500_DIVERSITY_SOURCE_CANDIDATE_QUEUE.csv'
    glb=root/'UNIML3D_SOURCE_GLB_EVIDENCE_V1.csv'
    head_bridge=root/'foundry_external_evidence_bridge_v1/FOUNDRY_UNIFIED_SOURCE_HEAD_OBSERVATIONS_V1.csv'
    catalog=root/'RealSaS_Truth_Foundry_Inventory_2026-11-08/DETAIL_MATRIX_V1/TARGET_CATALOG_37.csv'
    facet_catalog=root/'FACET_COVERAGE_BY_COHORT_V1.csv'
    req=[source,baseline,part,art,uniml,glb,head_bridge,catalog,facet_catalog]
    require(all(p.is_file() for p in req),'MISSING_INPUT')
    if strict:
        require(sha(source)==PRIOR_SOURCE_HASH,'FROZEN_SOURCE_CENSUS_SHA')
        frozen={
          part:'305e17b217e40ca46135dea6fac9aadd15c1b26dd13f140aaffe482201c9eabd',
          art:'e1dd85380aa09bb55181cf6e96e6db01f5d7d45a55a3d09031c60de7eccc75d4',
          uniml:'653ab31e1e25d20c1737f71c3af080a1f4562ebf204311353afc8fb6e609de43',
          head_bridge:'3aaababc94d2eb551a3d36c4b52ac82cc2cb406d85191031540e86adfea2a35d',
        }
        for p,expected in frozen.items():
            require(sha(p)==expected,'FROZEN_INPUT_SHA_'+p.name)

    facets_raw=csv_rows(facet_catalog)
    facet_by_id={f['facet_id']:f for f in facets_raw}
    require(len(facets_raw)==2*264 and len(facet_by_id)==264,'FACET_CATALOG')
    keys=sorted(facet_by_id)
    groups={f['group'] for f in facet_by_id.values()}
    require(len(groups)==14,'FACET_GROUPS')
    targets=csv_rows(catalog)
    require(len(targets)==37 and len({(t['proposed_head'],t['target']) for t in targets})==37,'TARGET_CATALOG')
    require(set(t['proposed_head'] for t in targets)==set(HEADS),'SIX_TARGET_HEADS')
    src=csv_rows(source)
    part_rows=csv_rows(part)
    art_rows=csv_rows(art)
    uniml_rows=csv_rows(uniml)
    glb_rows=csv_rows(glb)
    bridge_rows=csv_rows(head_bridge)
    if strict:
        require([len(src),len(part_rows),len(art_rows),len(uniml_rows),len(bridge_rows)] == [4363,2500,2500,2500,11863*6],'SOURCE_COUNTS')
    source_by_key={r['corpus']+':'+r['asset_id']:r for r in src}
    require(len(source_by_key)==len(src),'DUPLICATE_BASE_SOURCE')
    gb={r['object_type']:r for r in glb_rows}
    require(len(gb)==len(glb_rows),'DUPLICATE_UNIML_GLB')
    heads={}
    for r in bridge_rows:
        key=(r['source_key'],r['head'])
        require(key not in heads,'DUPLICATE_BRIDGE_HEAD')
        heads[key]=r
    original_sha=(sha(source),sha(baseline))
    out.mkdir(parents=True,exist_ok=True)
    status_count=Counter();head_evidence=Counter();facet_progress=defaultdict(Counter)
    source_count=0;target_slots=0;facet_slots=0;group_tasks=0
    observed_facet_slots=0;case_mix=Counter();source_shape_counts=Counter()
    head_targets=out/'HEAD_TARGET_EVIDENCE_11863x37_V3.csv'
    reviews=out/'GROUP_COURT_REVIEW_QUEUE_11863x14_V3.csv'
    ledger=out/'FORENSIC_ASSET_RECORDS_11863x264_V3.jsonl.gz'
    target_f=['source_key','head','target','proposed_owner','resolved_consumer_or_court',
              'required_independent_court','source_facet','training_head_target',
              'priority','provenance_basis','value_observed','observation_state','reviewed_truth_status','training_mask_valid',
              'source_rights_court','source_lineage_court','source_original_sha_verified']
    review_f=['source_key','corpus','group','facet_count','observed_not_courted_facets',
              'unknown_facets','independent_pass','task_status','evidence_needed',
              'training_authorized']

    # Keep original per-asset counts and complete 264 facets; external data is
    # encoded with the same keyspace but never promoted to qualified truth.
    def items():
        seen=set()
        with gzip.open(baseline,'rt',encoding='utf-8') as f:
            for line in f:
                b=json.loads(line)
                key=b['corpus']+':'+b['asset_id']
                require(key in source_by_key,'BASE_FACET_UNKNOWN_KEY')
                require(key not in seen,'BASE_FACET_DUPLICATE')
                seen.add(key)
                srcrow=source_by_key[key]
                require(set(b['facet_evidence'])==set(keys),'BASE_FACET_KEY_PARITY')
                yield key,b['corpus'],srcrow['provider'],srcrow,b['facet_evidence'],None
        require(len(seen)==len(src),'BASE_COVERAGE')
        for p in part_rows:
            uid=p['uid'];key='PARTVERSE_XL:'+uid
            require(key not in seen,'EXTERNAL_DUPLICATE')
            seen.add(key)
            require(p['face2label_exists']=='True' and p['segmented_glb_exists']=='True' and
                    p['face_count_equal_triangles']=='True','PART_SOURCE_PART_STRUCTURE_NOT_OBSERVED')
            require(p['semantic_head_training_admitted']=='False','PART_FALSE_TRAINING')
            fc=initial_facets(keys)
            for k,(field,stage) in PART_FACET_MAP.items():obs(fc,k,p.get(field),stage)
            yield key,'EXTERNAL_CANDIDATE','PARTVERSE_XL',p,fc,None
        for p in art_rows:
            uid=p['uuid'];key='ARTICULATION_XL2:'+uid
            require(key not in seen,'EXTERNAL_DUPLICATE')
            seen.add(key)
            fc=initial_facets(keys)
            for k,(field,stage) in ART_FACET_MAP.items():obs(fc,k,p.get(field),stage)
            yield key,'EXTERNAL_CANDIDATE','ARTICULATION_XL2',p,fc,None
        for p in uniml_rows:
            uid=p['object_type'];key='UNIML3D:'+uid
            require(key not in seen,'EXTERNAL_DUPLICATE')
            seen.add(key)
            fc=initial_facets(keys)
            for k,(field,stage) in UNIML_FACET_MAP.items():obs(fc,k,p.get(field),stage)
            detail=gb.get(uid)
            if detail and detail.get('download_status')=='DOWNLOADED':
                require(detail['glb_valid']=='True' and len(detail['download_sha256'])==64,'BAD_DOWNLOADED_GLB_RECEIPT')
                for k,(field,stage) in UNIML_GLB_FACET_MAP.items():obs(fc,k,detail.get(field),stage)
            yield key,'EXTERNAL_CANDIDATE','UNIML3D',p,fc,detail
        require(len(seen)==expected_source_n,'TOTAL_SOURCE_COUNT')

    import io
    with (head_targets.open('w',encoding='utf-8',newline='') as ht,
          reviews.open('w',encoding='utf-8',newline='') as rv,
          ledger.open('wb') as raw):
        wt=csv.DictWriter(ht,fieldnames=target_f);wt.writeheader()
        wr=csv.DictWriter(rv,fieldnames=review_f);wr.writeheader()
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=5) as zf:
            with io.TextIOWrapper(zf,encoding='utf-8',newline='\n') as js:
                for source_key,cohort,provider,details,fc,glb_detail in items():
                    source_count+=1
                    facet_slots+=len(fc)
                    kind='ORIGINAL' if provider not in ('PARTVERSE_XL','ARTICULATION_XL2','UNIML3D') else provider
                    case_mix[kind]+=1
                    observed_here=sum(v['observation_state']!=UNKNOWN for v in fc.values())
                    observed_facet_slots+=observed_here
                    status_count['non_authoritative_observation']+=observed_here
                    status_count['unknown']+=len(fc)-observed_here
                    for k,v in fc.items():
                        grp=facet_by_id[k]['group']
                        facet_progress[grp]['observed_not_courted']+=v['observation_state']!=UNKNOWN
                        facet_progress[grp]['unknown']+=v['observation_state']==UNKNOWN
                        require(v['independent_court_state']=='UNKNOWN' and not v['training_mask_valid'],
                                'PRETEND_INDEPENDENT_COURT')
                    # Cross-head source route stage measured by previous SHA-frozen ledger.
                    head_info={}
                    target_info=[]
                    for head in HEADS:
                        h=heads[(source_key,head)]
                        require(h['training_authorized']=='False','BRIDGE_ALREADY_TRAINED')
                        head_info[head]={
                          'evidence_stage':h['evidence_stage'],
                          'evidence_origin':h['evidence_origin'],
                          'observed_value':h['observed_value'],
                          'qualified_fit_court':'UNKNOWN',
                          'valid_supervision_loss_mask':False,
                        }
                        if h['evidence_stage']!='UNKNOWN':head_evidence[head]+=1
                    for t in targets:
                        head=t['proposed_head'];h=heads[(source_key,head)]
                        evidence_ref=h['evidence_origin']
                        value=h['observed_value']
                        # Each target preserves its own court, instead of falsely
                        # broadcasting a known mesh count as many training labels.
                        target_obs='HEAD_LEVEL_HINT_ONLY__NOT_TARGET_COURTED' if h['evidence_stage']!='UNKNOWN' else UNKNOWN
                        if provider=='PARTVERSE_XL' and t['target']=='source_face_region_support':
                            evidence_ref='PARTVERSE_FACE2LABEL_PROCESSED_SURFACE_ONLY'
                            value=details['part_id_count_including_zero']
                            target_obs=OBS_PROCESSED
                        row={
                          'source_key':source_key,'head':head,'target':t['target'],
                          'proposed_owner':t['proposed_head'],
                          'resolved_consumer_or_court':t['resolved_consumer_or_court'],
                          'required_independent_court':t['required_independent_court'],
                          'source_facet':t['source_facet'],
                          'training_head_target':t['training_head_target'],
                          'priority':t['priority'],
                          'provenance_basis':evidence_ref,'value_observed':value,
                          'observation_state':target_obs,'reviewed_truth_status':'NOT_ADMITTED',
                          'training_mask_valid':False,'source_rights_court':'UNKNOWN',
                          'source_lineage_court':'UNKNOWN','source_original_sha_verified':False,
                        }
                        wt.writerow(row);target_slots+=1
                    for g in sorted(groups):
                        fks=[k for k in keys if facet_by_id[k]['group']==g]
                        measured=sum(fc[k]['observation_state']!=UNKNOWN for k in fks)
                        remaining=len(fks)-measured
                        wr.writerow({
                            'source_key':source_key,'corpus':cohort,'group':g,
                            'facet_count':len(fks),'observed_not_courted_facets':measured,
                            'unknown_facets':remaining,'independent_pass':0,
                            'task_status':'OPEN__NOT_INDEPENDENTLY_COURTED',
                            'evidence_needed':'SOURCE_RIGHTS+ORIGINAL_SHA+INDEPENDENT_'+g.upper(),
                            'training_authorized':False,
                        })
                        group_tasks+=1
                    sha_claim=details.get('source_sha256') or details.get('source_sha_provider_claim') or details.get('source_sha256_provider_claim') or ''
                    if kind=='UNIML3D':sha_claim=None
                    if glb_detail and glb_detail.get('download_status')=='DOWNLOADED':
                        source_shape_counts['UNIML3D_CANONICAL_GLB_OBSERVED']+=1
                    item={
                        'schema':SCHEMA,'source_key':source_key,'corpus':cohort,'provider':provider,
                        'source_sha_claim':sha_claim or None,
                        'source_original_bytes_sha_courted':False,
                        'lineage_dedup_verified':False,
                        'full_source_fields':details,
                        'glb_canonical_observation_fields':glb_detail,
                        'source_facet_count':len(fc),
                        'source_facets_observed_not_courted':observed_here,
                        'source_facet_evidence':fc,
                        'head_route_observations':head_info,
                        'head_specific_training_pass':{h:False for h in HEADS},
                        'source_character_identity_approved':False,
                        'source_anatomy_approved':False,
                        'two_d_appearance_approved':False,
                        'training_authorized':False,'materialization_authorized':False,
                    }
                    js.write(json.dumps(item,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n')
    require(source_count==expected_source_n,'FINAL_SOURCE_COUNT')
    require(facet_slots==source_count*264,'FINAL_FACET_SLOTS')
    require(target_slots==source_count*37,'FINAL_HEAD_TARGET_SLOTS')
    require(group_tasks==source_count*14,'FINAL_GROUP_REVIEW_TASKS')
    summary={
      'schema':SCHEMA,'source_record_candidates':source_count,
      'unique_character_identities_verified':0,
      'original_source_sha_bytes_qualified':0,
      'per_head_qualified_sources':{h:0 for h in HEADS},
      'min_qualified_sources_goal_per_head':8000,
      'per_head_shortfall_verified':{h:8000 for h in HEADS},
      'head_target_definitions':len(targets),'facet_definitions':len(keys),
      'review_group_definitions':len(groups),'head_target_slots':target_slots,
      'facet_slots':facet_slots,'review_group_tasks':group_tasks,
      'facet_non_authoritative_observation_slots':observed_facet_slots,
      'facet_unknown_slots':facet_slots-observed_facet_slots,
      'sources_with_head_level_observation_not_truth':dict(head_evidence),
      'source_cohorts':dict(case_mix),
      'uniml_canonical_glb_count':source_shape_counts['UNIML3D_CANONICAL_GLB_OBSERVED'],
      'partverse_processed_face_mesh_parity_count':len(part_rows),
      'articulation_metadata_candidate_count':len(art_rows),
      'source_identity_dedup':'PENDING_PER_ORIGINAL_SHA_AND_LINEAGE',
      'training_authorized':False,'materialization_authorized':False,
      'source_input_sha256':{p.name:sha(p) for p in req},
      'output_sha256':{p.name:sha(p) for p in [ledger,head_targets,reviews]},
      'facets_by_group_observation_not_courted':{g:dict(v) for g,v in facet_progress.items()},
    }
    (out/'SUMMARY_V3.json').write_text(json.dumps(summary,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    v=run(a.root,a.out)
    print(json.dumps({k:v[k] for k in ('source_record_candidates','head_target_slots','facet_slots','review_group_tasks','facet_non_authoritative_observation_slots','facet_unknown_slots','per_head_qualified_sources','uniml_canonical_glb_count')},indent=2))