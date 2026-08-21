from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = 'RealSaS.N1D.Rank2G.G2.ReciprocalCycleEvidence.v1'
QUESTION_COMMIT = '973107d67992fad17ad83bb4d822a13913482b47'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def array_digest(arrays) -> str:
    h = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(a)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(np.asarray(a.shape, np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def base_evidence_digest(coords, scores, valid, VA, XYA, output_idx) -> str:
    # Exact V3-A / G1 digest contract. Extra G2 arrays are intentionally excluded.
    return array_digest([
        ('coords', np.asarray(coords)),
        ('scores', np.asarray(scores)),
        ('valid', np.asarray(valid)),
        ('V_A', np.asarray(VA)),
        ('XY_A', np.asarray(XYA)),
        ('output_idx', np.asarray(output_idx)),
    ])


def g2_evidence_digest(coords, scores, valid, VA, XYA, output_idx, reverse_xy, reverse_score, cycle_px) -> str:
    return array_digest([
        ('coords', np.asarray(coords)),
        ('scores', np.asarray(scores)),
        ('valid', np.asarray(valid)),
        ('V_A', np.asarray(VA)),
        ('XY_A', np.asarray(XYA)),
        ('output_idx', np.asarray(output_idx)),
        ('reverse_xy', np.asarray(reverse_xy)),
        ('reverse_score', np.asarray(reverse_score)),
        ('cycle_px', np.asarray(cycle_px)),
    ])


def load_obs_module(science_dir: Path):
    sys.path.insert(0, str(science_dir))
    spec = importlib.util.spec_from_file_location('g2_obs_exact', science_dir / 'observable_phase.py')
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError('observable_phase loader missing')
    spec.loader.exec_module(mod)
    return mod


def _single_descriptor(m, desc_view, xy):
    q = np.asarray(xy, np.float32).reshape(1, 2)
    z = m.sample_desc_single_view(desc_view, q)
    z = m.normrows(z)
    if z.shape[0] != 1:
        raise RuntimeError(('single descriptor shape', z.shape))
    return z[0]


def _reverse_cycle(m, desc_A_view, desc_B_view, mask_A, cache_A, q_B, q_A):
    """One controlled B->A descriptor return from a frozen forward B proposal.

    MASt3R lineage: reciprocal NN matching alternates nearest-neighbour associations
    between descriptor fields. G2 deliberately performs only the return leg from each
    already-frozen B proposal so proposal support remains byte/semantically unchanged.
    """
    if not np.any(mask_A):
        return np.array([np.nan, np.nan], np.float32), np.float32(np.nan), np.float32(np.inf)
    z_B = _single_descriptor(m, desc_B_view, q_B)
    q_back, s_back = m.top4_with_scores(desc_A_view, z_B, mask_A, cache_A)
    if len(q_back) == 0:
        return np.array([np.nan, np.nan], np.float32), np.float32(np.nan), np.float32(np.inf)
    q_hat = np.asarray(q_back[0], np.float32)
    score = np.float32(s_back[0])
    cyc = np.float32(np.linalg.norm(q_hat.astype(np.float64) - np.asarray(q_A, np.float64)))
    return q_hat, score, cyc


def extract(root: Path, science_dir: Path, state_path: Path, family: int, episode: str):
    obs = load_obs_module(science_dir)
    meta_path = state_path.with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    if meta.get('truth_access') != 'NONE':
        raise RuntimeError('observable state truth contract invalid')

    state = np.load(state_path, allow_pickle=False)
    m, A, B, PA, VA, seedB, output_idx, hb, prb, diag, field, xyA = obs._model_context(
        root, family, episode, meta['route']
    )

    P = np.asarray(PA[output_idx], np.float32)
    V = np.asarray(VA[:, output_idx], np.uint8)
    if not np.array_equal(P, np.asarray(state['P_A'], np.float32)):
        raise RuntimeError('P_A replay mismatch')
    if not np.array_equal(V, np.asarray(state['V_A'], np.uint8)):
        raise RuntimeError('V_A replay mismatch')

    XYA = np.asarray(state['XY_A'], np.float32)
    replay_xy = np.stack([m.project_points(PA[output_idx], v) for v in range(8)]).astype(np.float32)
    if not np.array_equal(replay_xy, XYA):
        raise RuntimeError('XY_A replay mismatch')

    # Exact V3-A forward proposal construction.
    zA = m.descriptor_consensus(m.sample_field_np(field['descriptor'][0, 0], xyA), VA)
    masks_B = [m.foreground_mask(p) for p in B]
    cache_B = [m.prepare_b_search(field['descriptor'][0, 1, v], masks_B[v]) for v in range(8)]

    # G2-only reverse-search substrate. Same coarse->refine search semantics, opposite pose.
    masks_A = [m.foreground_mask(p) for p in A]
    cache_A = [m.prepare_b_search(field['descriptor'][0, 0, v], masks_A[v]) for v in range(8)]

    n_out = len(output_idx)
    coords = np.full((n_out, 8, 4, 2), np.nan, np.float32)
    scores = np.full((n_out, 8, 4), np.nan, np.float32)
    valid = np.zeros((n_out, 8, 4), np.uint8)
    reverse_xy = np.full((n_out, 8, 4, 2), np.nan, np.float32)
    reverse_score = np.full((n_out, 8, 4), np.nan, np.float32)
    cycle_px = np.full((n_out, 8, 4), np.inf, np.float32)

    for oi, gi0 in enumerate(output_idx):
        gi = int(gi0)
        for v in range(8):
            if not bool(VA[v, gi]):
                continue
            q, s = m.top4_with_scores(field['descriptor'][0, 1, v], zA[gi], masks_B[v], cache_B[v])
            n = min(4, len(q))
            coords[oi, v, :n] = q[:n]
            scores[oi, v, :n] = s[:n]
            valid[oi, v, :n] = 1
            q_A = XYA[v, oi]
            for k in range(n):
                q_hat, rs, cyc = _reverse_cycle(
                    m,
                    field['descriptor'][0, 0, v],
                    field['descriptor'][0, 1, v],
                    masks_A[v],
                    cache_A[v],
                    q[k],
                    q_A,
                )
                reverse_xy[oi, v, k] = q_hat
                reverse_score[oi, v, k] = rs
                cycle_px[oi, v, k] = cyc

    output_idx_arr = np.asarray(output_idx, np.int32)
    base_digest = base_evidence_digest(coords, scores, valid, V, XYA, output_idx_arr)
    full_digest = g2_evidence_digest(
        coords, scores, valid, V, XYA, output_idx_arr, reverse_xy, reverse_score, cycle_px
    )

    valid_cycles = cycle_px[valid.astype(bool)]
    finite_cycles = valid_cycles[np.isfinite(valid_cycles)]
    audit = {
        'proposal_valid_n': int(valid.sum()),
        'reverse_finite_n': int(len(finite_cycles)),
        'reverse_nonfinite_n': int(len(valid_cycles) - len(finite_cycles)),
        'cycle_min_px': float(np.min(finite_cycles)) if len(finite_cycles) else None,
        'cycle_median_px': float(np.median(finite_cycles)) if len(finite_cycles) else None,
        'cycle_max_px': float(np.max(finite_cycles)) if len(finite_cycles) else None,
    }

    return {
        'arrays': {
            'coords': coords,
            'scores': scores,
            'valid': valid,
            'V_A': V,
            'XY_A': XYA,
            'output_idx': output_idx_arr,
            'reverse_xy': reverse_xy,
            'reverse_score': reverse_score,
            'cycle_px': cycle_px,
        },
        'meta': {
            'schema': SCHEMA,
            'question_commit': QUESTION_COMMIT,
            'family': int(family),
            'episode': episode,
            'truth_access': 'NONE',
            'observable_state_sha256': sha256_file(state_path),
            'observable_state_meta_sha256': sha256_file(meta_path),
            'base_proposal_evidence_content_sha256': base_digest,
            'g2_evidence_content_sha256': full_digest,
            'audit': audit,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--science-dir', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--episode', default='e01')
    ap.add_argument('--out-evidence', type=Path, required=True)
    ap.add_argument('--out-meta', type=Path, required=True)
    a = ap.parse_args()

    if a.out_evidence.exists() or a.out_meta.exists():
        raise RuntimeError('refusing overwrite')

    pack = extract(a.root, a.science_dir, a.state, a.family, a.episode)
    a.out_evidence.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out_evidence, **pack['arrays'])
    pack['meta']['evidence_file_sha256'] = sha256_file(a.out_evidence)
    a.out_meta.parent.mkdir(parents=True, exist_ok=True)
    a.out_meta.write_text(json.dumps(pack['meta'], indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(pack['meta'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
