#!/usr/bin/env python3
"""Independent finite-edit algebra court (does not import producer code).

Checks complete/partial source task semantics, every producer atom and selected
handle. Small fully observed problems use exhaustive partition enumeration as
a separate minimal-K oracle. This does not establish native Blender source truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INPUT_SCHEMA = 'RealSaS.OperationalEditWorkloadEvidence.v1'
OUTPUT_SCHEMA = 'RealSaS.OperationalDrawableDecisionIR.v1'
DENIED = ('graphics_owner_authorized', 'artist_drawable_owner_authorized',
          'iris_training_authorized', 'tessa_training_authorized',
          'materialization_authorized', 'training_authorized')


def check(cond, code):
    if not cond:
        raise ValueError(code)


def sha_obj(obj):
    data = (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def exhaustive_minimum(faces, tasks):
    """Enumerate ALL set partitions for <=7 fully observed faces."""
    n = len(faces)
    if n > 7:
        return None
    valid = []

    def visit(labels, maximum):
        if len(labels) == n:
            blocks = {}
            for i, grp in enumerate(labels):
                blocks.setdefault(grp, []).append(i)
            for task in tasks:
                for positions in blocks.values():
                    if len({task['support'][faces[k]] for k in positions}) != 1:
                        break
                else:
                    continue
                break
            else:
                valid.append(tuple(tuple(x) for x in blocks.values()))
            return
        for group in range(maximum + 2):
            visit(labels + [group], max(maximum, group))

    visit([0], 0)
    return min(map(len, valid)) if valid else None


def court(evidence, report):
    check(evidence.get('schema') == INPUT_SCHEMA, 'OD_COURT_INPUT_SCHEMA')
    check(report.get('schema') == OUTPUT_SCHEMA, 'OD_COURT_OUTPUT_SCHEMA')
    check(report.get('source_input_sha256') == sha_obj(evidence), 'OD_COURT_INPUT_HASH')
    x = dict(report)
    stated_hash = x.pop('report_sha256', None)
    check(stated_hash == sha_obj(x), 'OD_COURT_REPORT_HASH')
    for k in DENIED:
        check(report.get(k) is False, 'OD_COURT_FALSE_AUTHORITY:' + k)
    n = evidence['source_face_count']
    check(type(n) is int and 1 <= n <= 2_000_000, 'OD_COURT_FACE_COUNT')
    for k in ('source_original_sha256', 'source_rest_sha256', 'source_face_order_sha256'):
        check(report.get(k) == evidence.get(k), 'OD_COURT_SOURCE_IDENTITY:' + k)
    check(report.get('workload_policy') == evidence.get('workload_policy'),
          'OD_COURT_WORKLOAD_POLICY')
    tasks = sorted(evidence['operations'], key=lambda t: t['operation_id'])
    check(bool(tasks) and len({t['operation_id'] for t in tasks}) == len(tasks),
          'OD_COURT_TASK_IDS')
    check(report.get('sorted_operation_ids') == [t['operation_id'] for t in tasks],
          'OD_COURT_SORTED_TASKS')
    for t in tasks:
        check(len(t['support']) == len(t['validity']) == n and
              all(type(a) is bool for a in t['support'] + t['validity']),
              'OD_COURT_TASK_BOOLEAN')

    atoms = report.get('atoms', [])
    assignment = report.get('face_atom_index', [])
    check(isinstance(atoms, list) and len(atoms) > 0 and len(assignment) == n,
          'OD_COURT_ATOM_DOMAIN')
    groups = []
    accounted = []
    missing = set(i for i in range(n) if any(not t['validity'][i] for t in tasks))
    for idx, item in enumerate(atoms):
        faces = item.get('face_indices')
        check(isinstance(faces, list) and faces == sorted(set(faces)) and
              bool(faces) and all(type(f) is int and 0 <= f < n for f in faces),
              'OD_COURT_FACE_SET')
        accounted += faces
        check(all(assignment[f] == idx for f in faces), 'OD_COURT_ATOM_ADDRESS')
        missing_here = bool(set(faces) & missing)
        if missing_here:
            check(len(faces) == 1 and not item['qualified_complete_signature'] and
                  item['operation_signature'] is None and item['authority'] == 'UNKNOWN',
                  'OD_COURT_UNKNOWN_MERGE')
        else:
            profile = [t['support'][faces[0]] for t in tasks]
            check(item['qualified_complete_signature'] is True and
                  item['operation_signature'] == profile and
                  item['authority'] == 'POLICY_DERIVED', 'OD_COURT_PROFILE')
            for f in faces:
                check(all(t['support'][f] == profile[i] for i, t in enumerate(tasks)),
                      'OD_COURT_ILLEGAL_MERGE')
        expected_id = 'OD:' + sha_obj({'source_original_sha256': evidence['source_original_sha256'],
                                       'source_face_order_sha256': evidence['source_face_order_sha256'],
                                       'operation_ids': [t['operation_id'] for t in tasks],
                                       'face_indices': faces,
                                       'signature': item['operation_signature'],
                                       'workload_policy': evidence['workload_policy']})[:24]
        check(item.get('atom_id') == expected_id, 'OD_COURT_ATOM_ID')
        groups.append(faces)
    check(sorted(accounted) == list(range(n)), 'OD_COURT_INCOMPLETE_OR_OVERLAPPING')
    check(groups == sorted(groups, key=lambda g: g[0]), 'OD_COURT_UNSTABLE_ORDER')

    # Independent quotient reconstruction: equal full task profiles form blocks,
    # unobserved addresses remain singleton placeholders.
    complete_by_signature = {}
    for f in range(n):
        if f in missing:
            continue
        key = tuple(t['support'][f] for t in tasks)
        complete_by_signature.setdefault(key, set()).add(f)
    expected = list(complete_by_signature.values()) + [{f} for f in missing]
    check({frozenset(a) for a in groups} == {frozenset(a) for a in expected},
          'OD_COURT_NOT_UNIQUE_COARSEST_QUOTIENT')
    check(len(groups) == len(expected), 'OD_COURT_NOT_MINIMAL')

    recon = report.get('operation_reconstruction')
    check(len(recon) == len(tasks), 'OD_COURT_TASK_RECON_COUNT')
    for task, result in zip(tasks, recon):
        check(result.get('operation_id') == task['operation_id'] and
              result.get('exact_on_observed_support') is True,
              'OD_COURT_TASK_RECON_ID')
        select = result['selected_atom_indices']
        check(select == sorted(set(select)) and
              all(type(z) is int and 0 <= z < len(groups) for z in select),
              'OD_COURT_TASK_RECON_SELECTOR')
        guessed = [idx in select for idx in assignment]
        for f in range(n):
            if task['validity'][f]:
                check(guessed[f] == task['support'][f], 'OD_COURT_TASK_MISMATCH')
        check(result['unobserved_face_count'] == n - sum(task['validity']),
              'OD_COURT_UNKNOWN_DENOMINATOR')

    summary = report.get('audit')
    check(isinstance(summary, dict) and
          summary.get('atom_count') == len(groups) and
          summary.get('unknown_face_count') == len(missing) and
          summary.get('unknown_placeholder_count') == len(missing) and
          summary.get('complete_profile_atom_count') == len(complete_by_signature) and
          summary.get('all_tasks_fully_observed') is (not missing) and
          summary.get('same_drawable_training_pairs_qualified') == 0 and
          summary.get('different_drawable_training_pairs_qualified') == 0 and
          summary.get('heldout_source_edit_court') == 'NOT_EVALUATED' and
          summary.get('source_shader_replay_qualified') is False and
          summary.get('original_artist_owner') == 'UNKNOWN',
          'OD_COURT_SUMMARY_AUTHORITY')

    oracle = None
    if not missing and n <= 7:
        oracle = exhaustive_minimum(list(range(n)), tasks)
        check(oracle == len(groups), 'OD_COURT_EXHAUSTIVE_MINIMUM_DISAGREES')
    return {'status': 'PASS_ALGEBRA_ONLY__NO_REAL_SOURCE_OWNER_AUTHORITY',
            'source_input_sha256': report['source_input_sha256'],
            'report_sha256': report['report_sha256'],
            'face_count': n, 'atom_count': len(groups),
            'unknown_faces': len(missing), 'exhaustive_oracle_minimum_K': oracle,
            'training_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--decision', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    data = court(json.loads(args.evidence.read_text(encoding='utf-8')),
                 json.loads(args.decision.read_text(encoding='utf-8')))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n',
                            encoding='utf-8')
    print(json.dumps(data, sort_keys=True))


if __name__ == '__main__':
    main()
