#!/usr/bin/env python3
"""Exact Master3993 physical source-census reader of four immutable import waves.

This script only reads SHA-bound import observations. It never infers character
morphology, semantic labels, original author controls or training eligibility.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

SCHEMA = 'RealSaS.Master3993PhysicalSourceCensus.v1'
RECEIPT = 'RealSaS.Corpus4008Wave1SourceRigSceneObservation.v1'
COUNTS = (
    'source_scene_node_count', 'mesh_object_count', 'mesh_disconnected_components',
    'mesh_boundary_edges', 'mesh_nonmanifold_edges', 'polygon_count', 'vertex_count',
    'armature_count', 'observed_bone_count', 'bone_parented_mesh_candidates',
    'skin_rows_sampled', 'skin_rows_bad_sum', 'skin_rows_zero_weights',
    'material_count', 'image_count', 'image_dependency_missing_count',
    'observed_action_count', 'observed_action_curve_count',
    'observed_driver_channel_count', 'observed_ik_constraint_count',
    'observed_nla_strip_count', 'observed_nondeform_bone_count',
)

def fail(message):
    raise ValueError('PHYSICAL_CENSUS_' + message)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for data in iter(lambda: fp.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()

def read_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as fp:
        return list(csv.DictReader(fp))

def write_csv(path: Path, data):
    if not data: fail('NO_OUTPUT')
    with path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def bucket(number, bounds):
    for boundary in bounds:
        if number <= boundary:
            return str(boundary)
    return f'>{bounds[-1]}'

def run(master: Path, waves: tuple[Path, ...], output: Path):
    roster = read_csv(master)
    if len(roster) != 3993: fail('MASTER_COUNT')
    unique = {r['canonical_asset_id']: r for r in roster}
    if len(unique) != len(roster): fail('MASTER_DUPLICATE')
    if any(r['training_authorized'] != 'False' or r['materialization_authorized'] != 'False' for r in roster):
        fail('PREAUTHORIZED_MASTER_FORBIDDEN')
    receipts = {}
    original_receipts = collections.Counter()
    by_wave = collections.Counter()
    for wave_idx, root in enumerate(waves, 1):
        if not root.is_dir(): fail(f'MISSING_WAVE_{wave_idx}')
        files = sorted(root.rglob('asset_*.summary.json'))
        if not files: fail(f'EMPTY_WAVE_{wave_idx}')
        for path in files:
            x = json.loads(path.read_text(encoding='utf-8'))
            aid = x.get('asset_id')
            if aid not in unique: fail('SUMMARY_ID_NOT_IN_MASTER:' + str(aid))
            if aid in receipts: fail('DUPLICATE_RECEIPT:' + aid)
            row = unique[aid]
            if x.get('schema') != RECEIPT or x.get('status') != 'FULL_IMPORTED_SOURCE_SCENE_AND_CONTROL_CENSUS__NO_SOURCE_TRUTH_COURTS':
                fail('SUMMARY_SCHEMA_OR_STATUS:' + aid)
            if x.get('training_authorized') is not False or x.get('commercial_ml_rights_courted') is not False:
                fail('SUMMARY_MISLABELED_RIGHTS:' + aid)
            if x.get('source_character_vs_prop') != 'UNKNOWN' or x.get('functional_native_ik_fk') != 'UNKNOWN':
                fail('SUMMARY_UNSUPPORTED_AUTHORITY:' + aid)
            if x.get('candidate_id') != row['candidate_id'] or x.get('source_sha256') != row['source_import_sha256']:
                fail('SUMMARY_SOURCE_IDENTITY:' + aid)
            if x.get('source_format') != row['source_import_format']:
                fail('SUMMARY_FORMAT:' + aid)
            if row['historical_split'] == 'EXTERNAL_HOLDOUT': fail('HOLDOUT_SOURCE_IMPORT:' + aid)
            expected_wave = 4 if wave_idx == 4 else 3
            if str(row['source_import_wave']) != str(expected_wave): fail('SUMMARY_IMPORT_WAVE:' + aid)
            if wave_idx == 4 and sha(path) != row['wave4_receipt_summary_sha256']:
                fail('SUMMARY_SHA_WAVE4:' + aid)
            c = x.get('numeric_census')
            if not isinstance(c, dict): fail('MISSING_NUMERIC_CENSUS:' + aid)
            if any(type(c.get(k)) is not int or c[k] < 0 for k in COUNTS):
                fail('INVALID_NUMERIC_COUNTS:' + aid)
            receipts[aid] = (c, sha(path), wave_idx)
            by_wave[str(wave_idx)] += 1
            original_receipts[x['source_sha256']] += 1
    if max(original_receipts.values(), default=0) != 1: fail('DUPLICATE_SOURCE_SHA')
    observed = {a for a, r in unique.items() if r['source_import_observed'] == 'True'}
    if len(observed) != 2466 or observed != set(receipts):
        fail('INCOMPLETE_OBSERVED_COHORT:' + str((len(observed), len(receipts), len(observed - set(receipts)), len(set(receipts) - observed))))
    output.mkdir(parents=True, exist_ok=True)
    data = []
    for r in roster:
        aid = r['canonical_asset_id']
        known = aid in receipts
        c, receipt_sha, wave_idx = receipts[aid] if known else ({}, '', None)
        row = {
            'asset_id': aid, 'candidate_id': r['candidate_id'], 'provider': r['provider'],
            'historical_split': r['historical_split'], 'source_sha256': r['source_import_sha256'] or r['expected_source_sha256'],
            'source_format': r['source_import_format'], 'source_import_observed': known,
            'source_import_wave': r['source_import_wave'], 'census_summary_sha256': receipt_sha,
            'census_state': 'IMPORTED_NUMERIC_OBSERVATION_ONLY' if known else 'SOURCE_IMPORT_NOT_OBSERVED',
            'source_author_orientation': 'UNKNOWN', 'source_initial_anatomical_pose': 'UNKNOWN',
            'character_vs_prop': 'UNKNOWN', 'morphology': 'UNKNOWN',
            'semantic_region_source_truth': 'UNKNOWN', 'texture_pixel_parity': 'UNKNOWN',
            'approved_stylized_2d_truth': 'UNKNOWN', 'native_control_functionality': 'UNKNOWN',
            'deformation_quality': 'UNKNOWN', 'commercial_ml_training_rights': 'UNKNOWN',
            'materialization_authorized': False, 'training_authorized': False,
        }
        row.update({k: c.get(k, '') for k in COUNTS})
        data.append(row)
    write_csv(output / 'MASTER3993_SOURCE_PHYSICAL_CENSUS_V1.csv', data)
    flags=[]
    for row in data:
        if row['census_state'] != 'IMPORTED_NUMERIC_OBSERVATION_ONLY':
            flags.append({'asset_id': row['asset_id'], 'flag': 'SOURCE_IMPORT_OBSERVATION_MISSING', 'severity': 'OPEN'})
            continue
        for key,flag in (
            ('image_dependency_missing_count','EXTERNAL_TEXTURE_DEPENDENCY_MISSING'),
            ('mesh_boundary_edges','OPEN_OR_MODULAR_MESH_REVIEW'),
            ('mesh_nonmanifold_edges','NONMANIFOLD_REVIEW'),
            ('skin_rows_bad_sum','SKIN_ROW_SUM_REVIEW'),
            ('skin_rows_zero_weights','UNWEIGHTED_VERTICES_REVIEW'),
        ):
            if row[key]: flags.append({'asset_id':row['asset_id'],'flag':flag,'severity':'REVIEW_NOT_REJECT'})
        if not row['observed_ik_constraint_count']:
            flags.append({'asset_id':row['asset_id'],'flag':'IK_NOT_OBSERVED_SOURCE_UNKNOWN','severity':'NOT_NEGATIVE_LABEL'})
        if not row['image_count']:
            flags.append({'asset_id':row['asset_id'],'flag':'NO_IMPORT_IMAGE_OBSERVED_NOT_TEXTURE_ABSENCE','severity':'REVIEW_NOT_REJECT'})
    write_csv(output / 'MASTER3993_PHYSICAL_REVIEW_FLAGS_V1.csv', flags)
    scopes=('provider','source_format','historical_split')
    roll=[]
    for scope in scopes:
        for v in sorted({row[scope] for row in data}):
            group = [row for row in data if row[scope] == v]
            good = [row for row in group if row['source_import_observed']]
            entry={'facet':scope,'value':v,'candidate_records':len(group),'import_observed':len(good)}
            for col in COUNTS:
                entry[col+'_sum'] = sum(row[col] for row in good)
                entry[col+'_positive_assets'] = sum(row[col]>0 for row in good)
            roll.append(entry)
    write_csv(output / 'MASTER3993_PHYSICAL_CENSUS_ROLLUP_V1.csv', roll)
    hist=[]
    for col,edges in [('mesh_object_count',[0,1,2,4,9,19]), ('mesh_disconnected_components',[0,1,2,4,9,49]), ('armature_count',[0,1,2,4]), ('observed_bone_count',[0,1,16,32,64,128]), ('material_count',[0,1,2,4,9]), ('image_count',[0,1,2,4,9]), ('observed_action_count',[0,1,2,4,9])]:
        c = collections.Counter(bucket(row[col],edges) for row in data if row['source_import_observed'])
        for value,n in sorted(c.items()): hist.append({'metric':col,'bucket_upper_bound':value,'assets':n})
    write_csv(output / 'MASTER3993_PHYSICAL_CENSUS_HISTOGRAM_V1.csv', hist)
    info = {
        'schema':SCHEMA,'source_count':3993,'imported_source_observations':len(receipts),
        'import_unobserved':3993-len(receipts),'wave_receipts':dict(sorted(by_wave.items())),
        'import_source_summary_sha_full_join':True,'raw_original_mesh_bytes_rehashed_by_this_process':False,
        'source_identity_sha256':sha(master),
        'files':{file.name:sha(file) for file in sorted(output.glob('*.csv'))},
        'semantic_annotation_admitted':0,'graphics_2d_approved_admitted':0,
        'commercial_ml_rights_courted':False,'character_morphology_courted':False,
        'core370_not_in_this_roster':370,'cross_corpus_identity_overlay_not_consumed':True,
        'training_authorized':False,'materialization_authorized':False,
    }
    (output / 'MASTER3993_PHYSICAL_CENSUS_SUMMARY_V1.json').write_text(json.dumps(info,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return info

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--master-v2',type=Path,required=True)
    for i in range(1,5):p.add_argument(f'--wave{i}',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=run(a.master_v2,tuple(getattr(a,f'wave{i}') for i in range(1,5)),a.output)
    print(json.dumps(result,indent=2))
if __name__ == '__main__':main()