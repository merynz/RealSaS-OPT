from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_summary(state: dict) -> dict:
    prefixes = Counter()
    shapes = {}
    for key, value in state.items():
        prefix = str(key).split('.', 1)[0]
        prefixes[prefix] += 1
        if hasattr(value, 'shape'):
            shapes[str(key)] = list(value.shape)
    return {
        'parameter_tensor_count': len(shapes),
        'top_level_prefix_counts': dict(sorted(prefixes.items())),
        'sample_shapes': {k: shapes[k] for k in sorted(shapes)[:80]},
    }


def inspect_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    out = {
        'path_name': path.name,
        'sha256': sha256(path),
        'bytes': path.stat().st_size,
        'payload_type': type(payload).__name__,
    }
    if not isinstance(payload, dict):
        return out
    out['top_level_keys'] = sorted(map(str, payload.keys()))
    for key in ('schema', 'architecture_id', 'config_hash', 'step', 'optimizer_steps'):
        if key in payload and isinstance(payload[key], (str, int, float, bool, type(None))):
            out[key] = payload[key]
    if isinstance(payload.get('authority'), dict):
        authority = dict(payload['authority'])
        out['authority'] = {
            k: authority.get(k)
            for k in sorted(authority)
            if k not in {'metadata'}
        }
        if isinstance(authority.get('metadata'), dict):
            out['authority_metadata'] = authority['metadata']
    state = None
    if isinstance(payload.get('state_dict'), dict):
        state = payload['state_dict']
        out['state_container'] = 'state_dict'
    elif isinstance(payload.get('model'), dict):
        state = payload['model']
        out['state_container'] = 'model'
    elif all(isinstance(k, str) for k in payload) and any(hasattr(v, 'shape') for v in payload.values()):
        state = payload
        out['state_container'] = 'top_level'
    if isinstance(state, dict):
        out['state'] = tensor_summary(state)
    if isinstance(payload.get('codec_a0_token'), dict):
        out['codec_a0_token'] = payload['codec_a0_token']
    if isinstance(payload.get('config'), dict):
        out['config'] = payload['config']
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas-checkpoint', required=True)
    ap.add_argument('--mira-checkpoint', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    atlas = Path(args.atlas_checkpoint).resolve()
    mira = Path(args.mira_checkpoint).resolve()
    result = {
        'schema': 'RealSaS.ATLASMIRASubstitutionCandidateInventory.v1',
        'scope': 'RESEARCH_SUBSTITUTION_READINESS__NOT_PRODUCT_AUTHORITY',
        'atlas': inspect_checkpoint(atlas),
        'mira': inspect_checkpoint(mira),
        'authority_rules': {
            'fbx_object_packaging_authority': False,
            'carrier_identity': 'GEOMETRY_COMPONENT_SUPPORT_SEMANTICS',
            'raw_teacher_rig_truth_bone_count': 41,
            'teacher_deform_motion_target_count': 23,
            'atlas_historical_28_is_derived_projection_not_raw_truth': True,
            'mira_must_be_requeried_on_exact_test_skeleton_for_single_variable_court': True,
        },
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print('ATLAS_MIRA_SUBSTITUTION_CANDIDATE_INVENTORY_PASS', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
