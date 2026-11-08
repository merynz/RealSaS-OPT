#!/usr/bin/env python3
"""Truth Foundry source-specific graphics/one-view eligibility audit, not training admission.

Joins six existing, already observed source/processed-GLB cohorts. No new asset
identity is minted: all keys are provider-scoped; original-byte lineage, source
license, source RGB/pixel authority, style and unseen-back labels remain OPEN.

Does NOT claim material count, texture reference, processed segmentation or
canonical-GLB image count is an approved 2D target. The 8-view cameras and
student predicted six views are separate contracts. See companion JSON spec.
"""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path
from collections import Counter, defaultdict

SCHEMA = 'RealSaS.TruthFoundry.GraphicsSingleViewEvidence.v1'
HEADS = ('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')
F = ['source_key','provider','provider_source_id','dataset_row_type','provider_sha_claim',
     'processed_or_canonical_sha_observed','original_source_sha_rehashed','source_rights_court',
     'lineage_split_court','morphology_weak_category','morphology_visual_court',
     'physical_scene_census_observed','segmented_mesh_source_observed',
     'numeric_material_count','numeric_image_count','numeric_texture_count',
     'image_dependencies_missing_count','material_image_observation',
     'candidate_pixel_authority','eligible_source_pixel_court_pass',
     'texture_image_bytes_decoded','uv_source_parity_pass','alpha_source_parity_pass',
     'appearance_native_rgba_render_pass','approved_stylized_2d_art_pass',
     'source_teacher_eight_exact_views_pass','front_teacher_view_pass','back_teacher_view_pass',
     'teacher_face_id_depth_normal_pass','hidden_source_surface_visibility_pass',
     'single_view_front_real_observation_pass','student_predicted_six_views_pass',
     'teacher_target_camera_leakage_court','front_back_identity_court',
     'graphics_prior_training_allowed','graphics_actual_target_ready',
     'source_scene_structural_eligibility','source_recovery_next_action',
     'graphics_priority_bucket','rigged_source_hint','geometry_head_court','semantic_head_court',
     'graphics_head_court','tessa_head_court','axis_head_court','mira_head_court']


def require(v, why):
    if not v: raise ValueError('GRAPHICS_SINGLEVIEW_'+why)


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''): h.update(part)
    return h.hexdigest()


def integer(value):
    if value is None or str(value).strip() in ('','nan','None'):
        return None
    s=str(value).strip()
    if s.endswith('.0'): s=s[:-2]
    require(s.isdigit(),'INVALID_COUNT')
    return int(s)


def record(provider, uid, rowtype, orig_sha='', processed_sha=''):
    require(provider and uid,'UNKEYED_SOURCE')
    p={k:'' for k in F}
    p.update(source_key=provider+':'+uid,provider=provider,provider_source_id=uid,
             dataset_row_type=rowtype,provider_sha_claim=orig_sha,
             processed_or_canonical_sha_observed=processed_sha,
             original_source_sha_rehashed='False',source_rights_court='UNKNOWN',
             lineage_split_court='UNKNOWN',morphology_weak_category='UNKNOWN',
             morphology_visual_court='UNKNOWN',physical_scene_census_observed='False',
             segmented_mesh_source_observed='False',material_image_observation='UNKNOWN',
             candidate_pixel_authority='UNQUALIFIED',eligible_source_pixel_court_pass='False',
             texture_image_bytes_decoded='False',uv_source_parity_pass='False',
             alpha_source_parity_pass='False',appearance_native_rgba_render_pass='False',
             approved_stylized_2d_art_pass='False',source_teacher_eight_exact_views_pass='False',
             front_teacher_view_pass='False',back_teacher_view_pass='False',
             teacher_face_id_depth_normal_pass='False',hidden_source_surface_visibility_pass='False',
             single_view_front_real_observation_pass='False',student_predicted_six_views_pass='False',
             teacher_target_camera_leakage_court='UNKNOWN',front_back_identity_court='UNKNOWN',
             graphics_prior_training_allowed='False',graphics_actual_target_ready='False',
             source_scene_structural_eligibility='UNKNOWN',source_recovery_next_action='RECOVER_ORIGINAL_SOURCE',
             graphics_priority_bucket='DEFERRED_NOT_YET_SOURCE_CHECKED',rigged_source_hint='UNKNOWN')
    for h in HEADS:
        key={'IRIS_GEOMETRY':'geometry_head_court','IRIS_SEMANTIC':'semantic_head_court',
             'IRIS_GRAPHICS':'graphics_head_court','TESSA':'tessa_head_court',
             'AXIS':'axis_head_court','MIRA':'mira_head_court'}[h]
        p[key]='UNKNOWN_NOT_ADMITTED'
    return p


def run(inputs, out, expected=None):
    """inputs: six CSVs (source, part1, part2, artic, uniml, uniml_glb)."""
    source,p1,p2,art,uniml,glbs = [Path(x) for x in inputs]
    require(all(x.is_file() for x in (source,p1,p2,art,uniml,glbs)),'MISSING_INPUT')
    src=read_csv(source); first=read_csv(p1);second=read_csv(p2)
    artt=read_csv(art);uni=read_csv(uniml);gr=read_csv(glbs)
    require(len(gr)==len({x['object_type'] for x in gr}),'DUPLICATE_GLB_INVENTORY')
    gidx={x['object_type']:x for x in gr}
    counts=Counter(); rows=[];seen=set(); heads=defaultdict(Counter)

    def add(r):
        k=r['source_key'];require(k not in seen,'DUPLICATE_SOURCE_KEY');seen.add(k)
        require(all(r.get(f,'')!='True' for f in (
            'original_source_sha_rehashed','eligible_source_pixel_court_pass',
            'approved_stylized_2d_art_pass','graphics_prior_training_allowed',
            'graphics_actual_target_ready','source_teacher_eight_exact_views_pass')),
            'FALSE_POSITIVE_ADMISSION')
        rows.append(r);counts[r['provider']]+=1
        heads['IRIS_GRAPHICS'][r['graphics_priority_bucket']]+=1
    for r in src:
        p=record(r['corpus'],r['asset_id'],'BASE_4363',r['source_sha256'])
        p['lineage_split_court']='HISTORICAL_SPLIT_ONLY__LINEAGE_UNCOURTED'
        p['morphology_weak_category']=r.get('morphology','UNKNOWN')
        phys=r['physical_census_loaded'].lower()=='true'
        p['physical_scene_census_observed']=str(phys)
        if phys:
            mats,images,miss=map(integer,[r['material_count'],r['image_count'],r['image_dependency_missing_count']])
            require(all(x is not None for x in (mats,images,miss)),'BASE_MISSING_IMPORT_FIELDS')
            p.update(numeric_material_count=str(mats),numeric_image_count=str(images),
                     image_dependencies_missing_count=str(miss),
                     rigged_source_hint='IMPORTED_SKIN_ROWS_NOT_RIG_AUTHORITY' if integer(r['skin_rows_sampled']) else 'UNKNOWN')
            if images>0 and miss==0:
                p.update(material_image_observation='IMPORTER_IMAGE_REFERENCES_WITH_ZERO_REPORTED_MISSING_DEPENDENCIES',
                         candidate_pixel_authority='IMAGE_REFERENCE_ONLY__PIXELS_NOT_CHECKED',
                         graphics_priority_bucket='P0_PIXEL_RECOVERY_CANDIDATE',
                         source_recovery_next_action='HASH_AND_DECODE_SOURCE_IMAGE_BYTES_VERIFY_UV_ALPHA')
            elif images>0:
                p.update(material_image_observation='IMAGE_REFERENCES_WITH_MISSING_DEPENDENCIES',
                         graphics_priority_bucket='P1_RELINK_TEXTURE_DEPENDENCIES',
                         source_recovery_next_action='RELINK_IMAGE_DEPENDENCIES_CHECK_SOURCE_BYTE_SHA')
            elif mats>0:
                p.update(material_image_observation='MATERIALS_NO_IMAGES_OBSERVED',
                         graphics_priority_bucket='P2_MATERIAL_COLOR_RENDER_CANDIDATE',
                         source_recovery_next_action='SOURCE_MATERIAL_SHADER_AND_PALETTE_REVIEW')
            else:
                p.update(material_image_observation='NO_MATERIAL_OR_IMAGE_IMPORTER_OBSERVATION',
                         graphics_priority_bucket='P3_SOURCE_APPEARANCE_RECOVERY',
                         source_recovery_next_action='SOURCE_APPEARANCE_MISSING_OR_PROCEDURAL_CHECK')
            p['source_scene_structural_eligibility']='IMPORTED_SCENE_ONLY__NOT_RENDER_COURTED'
        else:
            p.update(graphics_priority_bucket='P4_NO_PHYSICAL_SCENE_EVIDENCE',
                     source_recovery_next_action='RECOVER_ORIGINAL_SCENE_AND_TEXTURE_MANIFEST')
        add(p)
    for wave, dataset in [('WAVE1_2500',first),('WAVE2_4534',second)]:
        for r in dataset:
            p=record('PARTVERSE_XL',r['uid'],wave,r['source_sha_provider_claim'])
            p.update(morphology_weak_category=r.get('weak_subject_candidate') or 'UNKNOWN',
                     segmented_mesh_source_observed=str(r.get('segmented_glb_exists')=='True'),
                     material_image_observation='PROCESSED_PARTVERSE_MESH_ONLY__NATIVE_TEXTURE_NOT_COURTED',
                     graphics_priority_bucket='P5_FACE_LABEL_SOURCE_NEEDS_ORIGINAL_TEXTURE',
                     source_recovery_next_action='RECOVER_ORIGINAL_OBJAVERSE_TEXTURE_UV_ALPHA_AND_SOURCE_FACE_PARITY',
                     source_scene_structural_eligibility='PROCESSED_FACE_COUNT_PARITY_ONLY' if r['face_count_equal_triangles']=='True' else 'PROCESSED_FACE_COUNT_PARITY_MISMATCH')
            add(p)
    for r in artt:
        p=record('ARTICULATION_XL2',r['uuid'],'PROVIDER_METADATA_ONLY',r.get('source_sha256_provider_claim',''))
        p.update(morphology_weak_category=r.get('candidate_family') or 'UNKNOWN',
                 graphics_priority_bucket='P6_ARTICULATION_METADATA_NEEDS_ORIGINAL_TEXTURE',
                 source_recovery_next_action='RECOVER_PROVIDER_ORIGINAL_OBJAVERSE_AND_RENDER_TEACHER')
        add(p)
    for r in uni:
        uid=r['object_type']; g=gidx.get(uid)
        glb_downloaded=bool(g and g['download_status']=='DOWNLOADED' and g['glb_valid']=='True')
        p=record('UNIML3D',uid,'SELECTED_CANONICAL_SOURCE' if glb_downloaded else 'PROVIDER_METADATA_ONLY',
                 r.get('original_asset_sha256',''),g['download_sha256'] if glb_downloaded else '')
        p['morphology_weak_category']=r.get('category') or 'UNKNOWN'
        p['rigged_source_hint']='PROVIDER_SKELETON_METADATA_ONLY'
        if glb_downloaded:
            m=integer(g['glb_material_count']);images=integer(g['glb_image_count']);textures=integer(g['glb_texture_count'])
            require(all(x is not None for x in (m,images,textures)),'UNIML_INVALID_GLB_COUNTS')
            p.update(numeric_material_count=str(m),numeric_image_count=str(images),numeric_texture_count=str(textures),
                     rigged_source_hint='CANONICAL_GLB_SKIN_STRUCTURE__NO_CONTROL_PROOF',
                     source_scene_structural_eligibility='CANONICAL_GLB_VALID__ORIGINAL_SOURCE_NOT_COURTED')
            if images>0:
                p.update(material_image_observation='CANONICAL_GLB_EMBEDDED_IMAGES__DECODE_NOT_VERIFIED',
                         candidate_pixel_authority='CANONICAL_GLB_IMAGE_REFERENCE_ONLY__SOURCE_ART_NOT_COURTED',
                         graphics_priority_bucket='P0_CANONICAL_PIXEL_INSPECTION_CANDIDATE',
                         source_recovery_next_action='DECODE_EMBEDDED_IMAGE_MATCH_GLTF_TEXTURE_UV_ALPHA_AND_SOURCE_RIGHTS')
            else:
                p.update(material_image_observation='CANONICAL_GLB_NO_IMAGE',
                         graphics_priority_bucket='P2_CANONICAL_MATERIAL_ONLY',
                         source_recovery_next_action='RECOVER_NATIVE_TEXTURE_OR_APPROVED_SOLID_COLOR_STYLE')
        else:
            p.update(material_image_observation='NO_DOWNLOADED_GLB',
                     graphics_priority_bucket='P4_UNIML_CANONICAL_GLB_NOT_DOWNLOADED',
                     source_recovery_next_action='SELECTED_GLB_OR_ORIGINAL_SOURCE_DOWNLOAD_WITH_QUOTA_GUARD')
        add(p)
    if expected is not None: require(len(rows)==expected,'EXPECTED_SOURCE_COUNT')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    coverage=out/'GRAPHICS_SINGLEVIEW_SOURCE_COVERAGE_V1.csv'
    with coverage.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=F);w.writeheader();w.writerows(rows)
    primary=[x for x in rows if x['graphics_priority_bucket'].startswith('P0_')]
    queue=out/'GRAPHICS_PIXEL_SOURCE_RECOVERY_QUEUE_V1.csv'
    with queue.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=F);w.writeheader();w.writerows(sorted(primary,key=lambda x:(x['provider'],x['source_key'])))
    providers=dict(sorted(counts.items()))
    bucket=Counter(x['graphics_priority_bucket'] for x in rows)
    c={
       'schema':SCHEMA,'total_provider_scoped_source_records':len(rows),
       'provider_record_counts':providers,'source_identity_lineage_dedup_complete':False,
       'graphics_priority_buckets':dict(sorted(bucket.items())),
       'pixel_source_recovery_candidates_not_approved':len(primary),
       'teacher_front_back_exact_multiview_pairs_pass':0,'approved_stylized_2d_art_target_count':0,
       'student_singleview_backside_prior_train_pairs_admitted':0,
       'native_graphics_source_pixel_court_pass':0,'source_original_rights_qualified':0,
       'teacher_camera_student_leakage_court_pass':0,'six_head_fit_identity_admitted':{h:0 for h in HEADS},
       'input_shas':{x.name:sha(x) for x in (source,p1,p2,art,uniml,glbs)},
       'output_sha256':{coverage.name:sha(coverage),queue.name:sha(queue)},
       'truth_status':'PROVENANCE_TYPED_IMPORT_AND_DATASET_OBSERVATIONS_ONLY',
       'training_authorized':False,
    }
    (out/'SUMMARY_V1.json').write_text(json.dumps(c,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return c


def main():
    p=argparse.ArgumentParser()
    for n in ('source','partverse_wave1','partverse_wave2','articulation_queue','uniml_queue','uniml_glb'):
        p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-sources',type=int)
    a=p.parse_args()
    files=[getattr(a,n) for n in ('source','partverse_wave1','partverse_wave2','articulation_queue','uniml_queue','uniml_glb')]
    print(json.dumps(run(files,a.out,a.expected_sources),indent=2))

if __name__=='__main__': main()