#!/usr/bin/env python3
"""Finite workload-conditioned drawable edit algebra; research-only.

An atomic drawable handle applies one Boolean hide/tint selector uniformly
to a fixed source-face set. The unique coarsest fully observable partition is
the equivalence quotient of task support columns. Missing measurements never
become negative labels; incomplete columns are unqualified singleton placeholders.

This is NOT source artist ownership, native edit validation or training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

INPUT_SCHEMA = 'RealSaS.OperationalEditWorkloadEvidence.v1'
OUTPUT_SCHEMA = 'RealSaS.OperationalDrawableDecisionIR.v1'
DECISION_POLICY = 'COARSEST_EXACT_UNIFORM_BOOLEAN_ATOMS__UNKNOWN_SINGLETON_V1'
FALSE_AUTHORITY = (
    'graphics_owner_authorized',
    'artist_drawable_owner_authorized',
    'iris_training_authorized',
    'tessa_training_authorized',
    'materialization_authorized',
    'training_authorized',
)


def canonical_bytes(v: Any) -> bytes:
    return (json.dumps(v, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def digest(v: Any) -> str:
    return hashlib.sha256(canonical_bytes(v)).hexdigest()


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _sha(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(x in '0123456789abcdef' for x in v)


def validate_evidence(v: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    require(isinstance(v, dict) and v.get('schema') == INPUT_SCHEMA,
            'OD_INPUT_SCHEMA_INVALID')
    n = v.get('source_face_count')
    require(type(n) is int and 1 <= n <= 2_000_000, 'OD_FACE_COUNT_INVALID')
    for key in ('source_original_sha256', 'source_rest_sha256', 'source_face_order_sha256'):
        require(_sha(v.get(key)), 'OD_SOURCE_SHA_INVALID:' + key)
    tasks = v.get('operations')
    require(isinstance(tasks, list) and 1 <= len(tasks) <= 512,
            'OD_OPERATION_SET_INVALID')
    require(len(tasks) * n <= 50_000_000, 'OD_OPERATION_MEMORY_BUDGET_EXCEEDED')
    names: set[str] = set()
    for task in tasks:
        require(isinstance(task, dict), 'OD_OPERATION_ROW_INVALID')
        name = task.get('operation_id')
        require(isinstance(name, str) and bool(name.strip()) and len(name) <= 256
                and name not in names, 'OD_OPERATION_ID_INVALID')
        names.add(name)
        require(_sha(task.get('evidence_sha256')), 'OD_OPERATION_EVIDENCE_SHA_INVALID')
        s, a = task.get('support'), task.get('validity')
        require(isinstance(s, list) and isinstance(a, list)
                and len(s) == n and len(a) == n, 'OD_OPERATION_DOMAIN_INVALID')
        require(all(type(z) is bool for z in s) and all(type(z) is bool for z in a),
                'OD_OPERATION_BOOLEAN_EXPECTED')
    require(isinstance(v.get('workload_policy'), str)
            and bool(v['workload_policy'].strip()), 'OD_WORKLOAD_POLICY_MISSING')
    return n, sorted(tasks, key=lambda row: row['operation_id'])


def classify_pair(tasks: list[dict[str, Any]], a: int, b: int) -> dict[str, Any]:
    """Workload relation, not intrinsic artist ownership."""
    distinguishing = []
    complete = True
    for task in tasks:
        both = task['validity'][a] and task['validity'][b]
        if both and task['support'][a] != task['support'][b]:
            distinguishing.append(task['operation_id'])
        if not both:
            complete = False
    if distinguishing:
        status = 'MUST_SEPARATE_FOR_WORKLOAD'
    elif complete:
        status = 'OPERATION_EQUIVALENT_FOR_WORKLOAD'
    else:
        status = 'UNKNOWN'
    return {'relation': status, 'distinguishing_operations': distinguishing,
            'fully_observed': complete}


def compile_evidence(v: dict[str, Any]) -> dict[str, Any]:
    n, tasks = validate_evidence(v)
    signatures: list[tuple[bool, ...] | None] = []
    for fi in range(n):
        known = all(task['validity'][fi] for task in tasks)
        signatures.append(tuple(task['support'][fi] for task in tasks) if known else None)

    complete_groups: dict[tuple[bool, ...], list[int]] = defaultdict(list)
    groups: list[tuple[list[int], tuple[bool, ...] | None]] = []
    for face, signature in enumerate(signatures):
        if signature is None:
            groups.append(([face], None))
        else:
            complete_groups[signature].append(face)
    groups += [(faces, sig) for sig, faces in complete_groups.items()]
    groups.sort(key=lambda row: row[0][0])

    atoms = []
    face_to_atom = [-1] * n
    for i, (faces, sig) in enumerate(groups):
        for f in faces:
            face_to_atom[f] = i
        key = {'source_original_sha256': v['source_original_sha256'],
               'source_face_order_sha256': v['source_face_order_sha256'],
               'operation_ids': [r['operation_id'] for r in tasks],
               'face_indices': faces, 'signature': sig,
               'workload_policy': v['workload_policy']}
        atoms.append({'atom_id': 'OD:' + digest(key)[:24], 'face_indices': faces,
                      'qualified_complete_signature': sig is not None,
                      'operation_signature': list(sig) if sig is not None else None,
                      'authority': 'POLICY_DERIVED' if sig is not None else 'UNKNOWN'})
    require(all(z >= 0 for z in face_to_atom), 'OD_INTERNAL_FACE_ACCOUNTING')

    operation_audit = []
    for task in tasks:
        selected, unobserved = [], 0
        for i, atom in enumerate(atoms):
            f = atom['face_indices']
            valid_values = [task['support'][x] for x in f if task['validity'][x]]
            if len(valid_values) != len(f):
                unobserved += len(f) - len(valid_values)
            require(len(set(valid_values)) <= 1, 'OD_INTERNAL_OPERATION_CONFLICT')
            if valid_values and valid_values[0]:
                selected.append(i)
        operation_audit.append({'operation_id': task['operation_id'],
                                'exact_on_observed_support': True,
                                'unobserved_face_count': unobserved,
                                'selected_atom_indices': selected})

    unknown_faces = [i for i, sig in enumerate(signatures) if sig is None]
    report = {
        'schema': OUTPUT_SCHEMA,
        'decision_policy': DECISION_POLICY,
        'source_original_sha256': v['source_original_sha256'],
        'source_rest_sha256': v['source_rest_sha256'],
        'source_face_order_sha256': v['source_face_order_sha256'],
        'source_input_sha256': digest(v),
        'workload_policy': v['workload_policy'],
        'source_face_count': n,
        'sorted_operation_ids': [row['operation_id'] for row in tasks],
        'face_atom_index': face_to_atom,
        'atoms': atoms,
        'operation_reconstruction': operation_audit,
        'audit': {
            'atom_count': len(atoms),
            'complete_profile_atom_count': len(complete_groups),
            'unknown_placeholder_count': len(unknown_faces),
            'unknown_face_count': len(unknown_faces),
            'all_tasks_fully_observed': not unknown_faces,
            'coarsest_quotient_proven_only_on_complete_domain': True,
            'same_drawable_training_pairs_qualified': 0,
            'different_drawable_training_pairs_qualified': 0,
            'heldout_source_edit_court': 'NOT_EVALUATED',
            'original_artist_owner': 'UNKNOWN',
            'source_shader_replay_qualified': False,
        },
        **{key: False for key in FALSE_AUTHORITY},
    }
    report['report_sha256'] = digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    outcome = compile_evidence(json.loads(args.evidence.read_text(encoding='utf-8')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(outcome))
    print(json.dumps({'report_sha256': outcome['report_sha256'],
                      'atom_count': outcome['audit']['atom_count'],
                      'unknown_face_count': outcome['audit']['unknown_face_count'],
                      'authority': 'RESEARCH_ONLY'}, sort_keys=True))


if __name__ == '__main__':
    main()
