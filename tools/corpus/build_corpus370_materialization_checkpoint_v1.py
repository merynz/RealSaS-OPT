#!/usr/bin/env python3
"""Report exact supervision coverage before any notebook is generated.

An open diagnostic matrix produces an explicitly blocked checkpoint. Final
materialization requires bound rosters and a current whole-lineage split. This
tool copies court decisions, never converts UNKNOWN to MASKED or FAIL to PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from models.materialization_contract_v1 import sha256, require
from tools.corpus.derive_corpus370_model_rosters_v1 import derive

SCHEMA = 'RealSaS.Corpus370MaterializationCheckpoint.v1'
HEADS = {
    'IRIS': {
        'surface_unsigned_distance': 'SURFACE_GEOMETRY_UDF',
        'global_sdf_sign': 'GLOBAL_SOLID_SDF',
        'global_occupancy': 'GLOBAL_SOLID_SDF',
        'oriented_normals': 'SURFACE_NORMALS',
        'appearance': 'APPEARANCE_CAA',
        'visibility': 'VISIBILITY_OCCLUSION',
        'ray_compositing': 'RAY_COMPOSITING',
        'causal_response': 'CAUSAL_RESPONSE',
    },
    'TESSA': {'mechanical_topology': 'MECHANICAL_SURFACE_TOPOLOGY', 'mechanical_teacher': 'TESSA_TEACHER'},
}


def ref(path):
    return {'path': str(path.resolve()), 'sha256': sha256(path)}


def check_split(split, roster, matrix_ids):
    require(split.get('schema') == 'RealSaS.Corpus370LineageSplit.v1'
            and split.get('status') == 'PASS_GLOBAL_PRODUCT_UNSEEN_AND_MODEL_TRAIN_VAL_SPLIT_SEALED'
            and split.get('split_authorized') is True and split.get('training_authorized') is False,
            'CHECKPOINT_SPLIT_NOT_SEALED')
    global_ids = set(split['global_unseen']['subject_ids'])
    require(global_ids.issubset(matrix_ids), 'CHECKPOINT_UNKNOWN_GLOBAL_SUBJECT')
    result = {}
    for model, payload in roster['models'].items():
        row = split['models'][model]
        sets = {key: set(row[key + '_subject_ids']) for key in ('train', 'val', 'global_unseen')}
        require(all(len(sets[key]) == len(row[key + '_subject_ids']) for key in sets), 'CHECKPOINT_DUPLICATE_SPLIT_SUBJECT')
        eligible = {s['subject_id'] for s in payload['eligible_subjects']}
        require(set.union(*sets.values()) == eligible, 'CHECKPOINT_SPLIT_ELIGIBILITY_DRIFT:' + model)
        require(sets['train'].isdisjoint(sets['val']) and (sets['train'] | sets['val']).isdisjoint(global_ids)
                and sets['global_unseen'] == eligible & global_ids, 'CHECKPOINT_GLOBAL_UNSEEN_LEAKAGE:' + model)
        result[model] = {key: sorted(values) for key, values in sets.items()}
    return result


def build(matrix_path, split_path=None):
    matrix = json.loads(matrix_path.read_text())
    require(matrix.get('schema') == 'RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1'
            and matrix.get('training_authorized') is False, 'CHECKPOINT_MATRIX_CONTRACT')
    subjects = matrix['subjects']
    ids = {s['subject_id'] for s in subjects}
    require(len(ids) == len(subjects) == matrix['subject_count'] == 370, 'CHECKPOINT_EXACT370_REQUIRED')
    roster = derive(matrix_path)
    blockers = []
    diagnostic = bool(matrix.get('readback_scope'))
    if diagnostic:
        blockers.append('MATRIX_IS_PARTIAL_DIAGNOSTIC')
    else:
        from tools.corpus.build_corpus370_subject_family_authority_matrix_v1 import build as replay_matrix
        for parent in [matrix['policy'], matrix['candidate_manifest'], *matrix['evidence_indexes']]:
            path = Path(parent['path'])
            require(path.is_file() and sha256(path) == parent['sha256'], 'CHECKPOINT_MATRIX_PARENT_DRIFT')
        replay = replay_matrix(Path(matrix['policy']['path']), Path(matrix['candidate_manifest']['path']),
                               [Path(r['path']) for r in matrix['evidence_indexes']])
        require(replay == matrix, 'CHECKPOINT_MATRIX_REPLAY_DRIFT')
    partitions = None
    split = None
    if split_path:
        split = json.loads(split_path.read_text())
        # The split must have been sealed from these exact matrix-derived rosters.
        parent = split['parents']['model_rosters']
        path = Path(parent['path'])
        require(path.is_file() and sha256(path) == parent['sha256'], 'CHECKPOINT_SPLIT_ROSTER_BYTES_DRIFT')
        require(json.loads(path.read_text()) == roster, 'CHECKPOINT_SPLIT_MATRIX_ROSTER_DRIFT')
        partitions = check_split(split, roster, ids)
    else:
        blockers.append('CURRENT_LINEAGE_SAFE_SPLIT_REQUIRED')
    models = {}
    for model, heads in HEADS.items():
        payload = roster['models'][model]
        eligible = {r['subject_id'] for r in payload['eligible_subjects']}
        required = set(next(iter(subjects))['model_eligibility'][model]['required_pass'])
        families = required | set(heads.values())
        subject_rows = []
        fully = partially = 0
        unknown = []
        reason_counts = Counter()
        coverage = {h: Counter() for h in heads}
        for row in subjects:
            sid = row['subject_id']
            statuses = {f: row['families'][f]['status'] for f in sorted(families)}
            admitted = sid in eligible
            enabled = {h: admitted and statuses[f] == 'PASS' for h, f in heads.items()}
            split_name = next((k.upper() for k, values in (partitions or {}).get(model, {}).items() if sid in values), None)
            if admitted:
                if all(v == 'PASS' for v in statuses.values()): fully += 1
                else: partially += 1
                unknown += [{'subject_id': sid, 'family': f} for f, value in statuses.items() if value == 'UNKNOWN']
                for head, on in enabled.items():
                    if on:
                        coverage[head]['all_eligible_subjects'] += 1
                        if split_name: coverage[head][split_name.lower() + '_subjects'] += 1
            reasons = {f: row['families'][f]['reason_codes'] for f in families if statuses[f] != 'PASS'}
            hard = row['model_eligibility'][model]['blocking_families']
            for family in hard:
                for reason in reasons[family]: reason_counts[family + ':' + reason] += 1
            subject_rows.append({'subject_id': sid, 'source_sha256': row['source_sha256'],
                'disposition': ('ADMIT_FULL' if all(v == 'PASS' for v in statuses.values()) else 'ADMIT_PARTIAL') if admitted else 'BLOCKED',
                'family_status': statuses, 'supervision_enabled': enabled, 'split': split_name,
                'hard_blocking_families': hard, 'nonpass_reasons': reasons})
        if unknown:
            blockers.append(model + ':CURRENT_OPTIONAL_FAMILY_COURTS_REQUIRED')
        if not eligible:
            blockers.append(model + ':NO_ELIGIBLE_SUBJECTS')
        models[model] = {'eligible_subject_count': len(eligible), 'blocked_subject_count': 370 - len(eligible),
            'source_qualified_subject_count': sum(s['families']['SOURCE_IMPORT']['status'] == 'PASS' for s in subjects),
            'observation_qualified_subject_count': sum(s['families']['CAMERA_OBSERVATION']['status'] == 'PASS' for s in subjects),
            'eligible_global_solid_subject_count': sum(s['subject_id'] in eligible and s['families']['GLOBAL_SOLID_SDF']['status'] == 'PASS' for s in subjects),
            'full_supervision_subject_count': fully, 'partial_supervision_subject_count': partially,
            'split_counts': {k: len(v) for k, v in (partitions or {}).get(model, {}).items()} if partitions else None,
            'head_subject_coverage': {h: {k: c[k] for k in ('all_eligible_subjects', 'train_subjects', 'val_subjects', 'global_unseen_subjects')}
                                      for h, c in coverage.items()},
            'hard_blocker_reason_counts': dict(sorted(reason_counts.items())), 'unknown_eligible_cells': unknown,
            'subjects': subject_rows}
    a = {s['subject_id'] for s in roster['models']['IRIS']['eligible_subjects']}
    b = {s['subject_id'] for s in roster['models']['TESSA']['eligible_subjects']}
    return {'schema': SCHEMA, 'status': 'BLOCKED_PRE_MATERIALIZATION' if blockers else 'PASS_PRE_MATERIALIZATION_CHECKPOINT',
        'counts_are_final': not blockers, 'candidate_subject_count': 370,
        'parents': {'authority_matrix': ref(matrix_path), 'lineage_split': ref(split_path) if split_path else None},
        'models': models, 'overlap': {'iris_and_tessa': len(a & b), 'iris_only': len(a - b), 'tessa_only': len(b - a), 'neither': 370 - len(a | b)},
        'blockers': blockers, 'materialized_numeric_sample_counts': None,
        'counts_are_subjects_not_training_samples': True,
        'component_sign_promoted_to_global_sdf': False,
        'tessa_training_requires_frozen_iris_z_and_g': True,
        'materialization_authorized': not blockers, 'training_authorized': False}


def render(value):
    lines = ['# Corpus370 materialization checkpoint', '', '**' + value['status'] + '**', '',
             'Counts below are subjects; sample counts require completed materialization.', '',
             '| Model | Eligible / 370 | Full | Partial | Blocked | Train / val / unseen |',
             '|---|---:|---:|---:|---:|---|']
    for name, row in value['models'].items():
        split = row['split_counts']
        count = 'PENDING' if split is None else ' / '.join(str(split[k]) for k in ('train', 'val', 'global_unseen'))
        lines.append(f"| {name} | {row['eligible_subject_count']} | {row['full_supervision_subject_count']} | {row['partial_supervision_subject_count']} | {row['blocked_subject_count']} | {count} |")
    lines += ['', '| Model / head | All eligible | Train | Val | Global unseen |', '|---|---:|---:|---:|---:|']
    for name, row in value['models'].items():
        for head, coverage in row['head_subject_coverage'].items():
            values = [str(coverage['all_eligible_subjects'])] + [str(coverage[k]) if row['split_counts'] is not None else 'PENDING' for k in ('train_subjects', 'val_subjects', 'global_unseen_subjects')]
            lines.append('| ' + name + ' / ' + head + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'IRIS/TESSA overlap: ' + json.dumps(value['overlap'], sort_keys=True), '',
              'Materialized numeric targets: PENDING. Training authorization: false.', '']
    if value['blockers']:
        lines += ['Outstanding evidence:', ''] + ['- ' + b for b in value['blockers']] + ['']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--split', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    value = build(a.matrix, a.split)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'CORPUS370_MATERIALIZATION_CHECKPOINT_V1.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    (a.output / 'CORPUS370_MATERIALIZATION_CHECKPOINT_V1.md').write_text(render(value))
    print(json.dumps({k: value[k] for k in ('status', 'overlap', 'blockers', 'materialization_authorized')}))


if __name__ == '__main__': main()
