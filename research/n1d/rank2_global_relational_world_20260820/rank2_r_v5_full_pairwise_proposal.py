from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA = 'RealSaS.N1D.Rank2R.V5.FullPairwiseProposal.v1'
CONTRACT_BLOB_SHA = 'db98ddf3b885ad28598592f8e30350dab198b1a5'
HUBER_DELTA = 0.25
N_RESTARTS = 16
STRICT_EPS = 1e-12
FORBIDDEN_COST = 1e9
SEED_TAG = 0x5205

Site = Tuple[int, int]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical_sha(obj) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(b).hexdigest()


def rank01(vals, descending=False):
    x = np.asarray(vals, float)
    o = np.argsort(-x if descending else x, kind='stable')
    r = np.empty(len(x), float)
    r[o] = np.arange(len(x), dtype=float)
    return r / max(len(x) - 1, 1)


def huber_abs(x, delta=HUBER_DELTA):
    x = np.abs(np.asarray(x, float))
    return np.where(x <= delta, .5 * x * x, delta * (x - .5 * delta))


def scene_scale(P):
    P = np.asarray(P, float)
    if len(P) < 2:
        return 1.0
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    v = D[np.triu_indices(len(P), 1)]
    v = v[v > 1e-9]
    return float(np.median(v)) if len(v) else 1.0


def row_sites(coords, scores, valid, i, v):
    best = {}
    for k in range(4):
        if not valid[i, v, k]:
            continue
        key = tuple(map(int, np.rint(coords[i, v, k]).astype(int)))
        sc = float(scores[i, v, k])
        if key not in best or sc > best[key]:
            best[key] = sc
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0]))


def array_digest(arrays) -> str:
    h = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(a)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(np.asarray(a.shape, np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def evidence_content_digest(z) -> str:
    return array_digest([
        ('coords', np.asarray(z['coords'])),
        ('scores', np.asarray(z['scores'])),
        ('valid', np.asarray(z['valid'])),
        ('V_A', np.asarray(z['V_A'])),
        ('XY_A', np.asarray(z['XY_A'])),
        ('output_idx', np.asarray(z['output_idx'])),
    ])


@dataclass
class ViewProblem:
    family: int
    view: int
    carriers: List[int]
    sites: List[Site]
    feasible: Dict[int, List[int]]
    g_cost: Dict[Tuple[int, int], float]
    pair_r: Dict[Tuple[int, int], Dict[Tuple[int, int], float]]
    scale_A: float
    scale_B: float
    rows_serial: Dict[str, dict]

    @property
    def n(self):
        return len(self.carriers)


def _unit(x):
    n = float(np.linalg.norm(x))
    return np.asarray(x, float) / max(n, 1e-9)


def build_view_problem(family: int, coords, scores, valid, VA, XYA, v: int) -> ViewProblem:
    carriers = [int(i) for i in np.where(VA[v] > 0)[0]]
    row_items = {i: row_sites(coords, scores, valid, i, v) for i in carriers}
    if any(len(row_items[i]) == 0 for i in carriers):
        raise RuntimeError(('visible carrier has zero proposal', family, v))

    sites = sorted({site for i in carriers for site, _ in row_items[i]}, key=lambda p: (p[1], p[0]))
    site_to_id = {s: j for j, s in enumerate(sites)}
    feasible = {}
    g_cost = {}
    rows_serial = {}
    for i in carriers:
        items = row_items[i]
        sc = np.asarray([x[1] for x in items], float)
        gr = rank01(sc, descending=True)
        ids = [site_to_id[x[0]] for x in items]
        feasible[i] = ids
        for sid, gc in zip(ids, gr):
            g_cost[(i, sid)] = float(gc)
        rows_serial[str(i)] = {
            'sites': [list(x[0]) for x in items],
            'site_ids': ids,
            'descriptor_scores': sc.tolist(),
            'G_rank': gr.tolist(),
            'g_top1': list(items[0][0]),
        }

    scale_A = scene_scale(XYA[v, carriers])
    scale_B = scene_scale(np.asarray(sites, float))
    pair_r = {}
    for aa in range(len(carriers)):
        i = carriers[aa]
        for bb in range(aa + 1, len(carriers)):
            j = carriers[bb]
            combos = []
            raws = []
            da = (np.asarray(XYA[v, i], float) - np.asarray(XYA[v, j], float)) / max(scale_A, 1e-9)
            for si in feasible[i]:
                for sj in feasible[j]:
                    if si == sj:
                        continue
                    db = (np.asarray(sites[si], float) - np.asarray(sites[sj], float)) / max(scale_B, 1e-9)
                    feat = np.r_[db - da, np.linalg.norm(db) - np.linalg.norm(da), _unit(db) - _unit(da)]
                    raw = float(np.mean(huber_abs(feat, HUBER_DELTA)))
                    combos.append((si, sj))
                    raws.append(raw)
            if not combos:
                continue
            rr = rank01(np.asarray(raws), descending=False)
            pair_r[(i, j)] = {c: float(r) for c, r in zip(combos, rr)}

    return ViewProblem(
        family=family, view=v, carriers=carriers, sites=sites, feasible=feasible,
        g_cost=g_cost, pair_r=pair_r, scale_A=scale_A, scale_B=scale_B,
        rows_serial=rows_serial,
    )


def pair_cost(prob: ViewProblem, i: int, si: int, j: int, sj: int) -> float:
    if i == j or si == sj:
        raise RuntimeError(('illegal pair query', i, si, j, sj))
    if i < j:
        table = prob.pair_r.get((i, j))
        key = (si, sj)
    else:
        table = prob.pair_r.get((j, i))
        key = (sj, si)
    if table is None or key not in table:
        raise RuntimeError(('missing pair factor', prob.family, prob.view, i, si, j, sj))
    return table[key]


def kuhn_max_cardinality(prob: ViewProblem) -> int:
    owner = {}
    def aug(i, seen):
        for s in prob.feasible[i]:
            if s in seen:
                continue
            seen.add(s)
            if s not in owner or aug(owner[s], seen):
                owner[s] = i
                return True
        return False
    return int(sum(bool(aug(i, set())) for i in prob.carriers))


def hungarian_state(prob: ViewProblem, edge_mode: str, restart: int = 0):
    n, ns = prob.n, len(prob.sites)
    # Lexicographic implementation: one abstention costs more than the maximum possible
    # total real-edge secondary cost. This encodes max-cardinality first, not a scientific penalty.
    abstain_lex = float(n + 1)
    C = np.full((n, ns + n), FORBIDDEN_COST, float)
    rng = None
    if edge_mode == 'random':
        rng = np.random.default_rng(np.random.SeedSequence([SEED_TAG, prob.family, prob.view, restart]))
    for rr, i in enumerate(prob.carriers):
        for s in prob.feasible[i]:
            if edge_mode == 'g':
                c = prob.g_cost[(i, s)]
            elif edge_mode == 'zero':
                c = 0.0
            elif edge_mode == 'random':
                c = float(rng.random())
            else:
                raise ValueError(edge_mode)
            C[rr, s] = c
        C[rr, ns + rr] = abstain_lex
    ri, ci = linear_sum_assignment(C)
    if len(ri) != n:
        raise RuntimeError(('hungarian row coverage failure', prob.family, prob.view))
    state = {i: -1 for i in prob.carriers}
    for r, c in zip(ri, ci):
        i = prob.carriers[int(r)]
        val = float(C[r, c])
        if c < ns and val < FORBIDDEN_COST / 2:
            state[i] = int(c)
        elif c == ns + int(r):
            state[i] = -1
        else:
            raise RuntimeError(('forbidden hungarian assignment', prob.family, prob.view, i, int(c), val))
    return state


def state_real_count(state) -> int:
    return int(sum(s >= 0 for s in state.values()))


def validate_state(prob: ViewProblem, state, K: int):
    if set(state) != set(prob.carriers):
        raise RuntimeError(('carrier coverage', prob.family, prob.view))
    reals = [(i, s) for i, s in state.items() if s >= 0]
    if len(reals) != K:
        raise RuntimeError(('cardinality', prob.family, prob.view, len(reals), K))
    sites = [s for _, s in reals]
    if len(set(sites)) != len(sites):
        raise RuntimeError(('injectivity', prob.family, prob.view))
    for i, s in reals:
        if s not in prob.feasible[i]:
            raise RuntimeError(('infeasible real edge', prob.family, prob.view, i, s))


def objective(prob: ViewProblem, state):
    reals = sorted((i, s) for i, s in state.items() if s >= 0)
    K = len(reals)
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    sum_g = float(sum(prob.g_cost[(i, s)] for i, s in reals))
    sum_r = 0.0
    for a in range(K):
        i, si = reals[a]
        for b in range(a + 1, K):
            j, sj = reals[b]
            sum_r += pair_cost(prob, i, si, j, sj)
    eg = sum_g / K
    npairs = K * (K - 1) // 2
    er = sum_r / npairs if npairs else 0.0
    return float(eg + er), float(eg), float(er), float(sum_g), float(sum_r)


def state_key(prob: ViewProblem, state):
    return tuple((i, int(state[i])) for i in prob.carriers)


def _candidate_update(best, delta_e, move_key, payload):
    if delta_e >= -STRICT_EPS:
        return best
    if best is None or delta_e < best[0] - 1e-15 or (abs(delta_e - best[0]) <= 1e-15 and move_key < best[1]):
        return (float(delta_e), move_key, payload)
    return best


def descend(prob: ViewProblem, init_state, K: int):
    state = dict(init_state)
    validate_state(prob, state, K)
    E, EG, ER, sum_g, sum_r = objective(prob, state)
    npairs = K * (K - 1) // 2
    iterations = 0
    while True:
        iterations += 1
        if iterations > 10000:
            raise RuntimeError(('local descent fail-safe exceeded', prob.family, prob.view))
        real = sorted(i for i in prob.carriers if state[i] >= 0)
        abst = sorted(i for i in prob.carriers if state[i] < 0)
        used = {state[i] for i in real}
        unused = set(range(len(prob.sites))) - used
        best = None

        # 1. one carrier to an unused feasible site
        for i in real:
            old = state[i]
            for s in prob.feasible[i]:
                if s == old or s not in unused:
                    continue
                dg = prob.g_cost[(i, s)] - prob.g_cost[(i, old)]
                dr = 0.0
                for j in real:
                    if j == i:
                        continue
                    dr += pair_cost(prob, i, s, j, state[j]) - pair_cost(prob, i, old, j, state[j])
                de = dg / K + (dr / npairs if npairs else 0.0)
                best = _candidate_update(best, de, ('move', i, s), ('move', i, s))

        # 2. feasible pair-site swaps
        for ai in range(len(real)):
            i = real[ai]; si = state[i]
            for aj in range(ai + 1, len(real)):
                j = real[aj]; sj = state[j]
                if sj not in prob.feasible[i] or si not in prob.feasible[j]:
                    continue
                dg = (prob.g_cost[(i, sj)] + prob.g_cost[(j, si)] -
                      prob.g_cost[(i, si)] - prob.g_cost[(j, sj)])
                dr = pair_cost(prob, i, sj, j, si) - pair_cost(prob, i, si, j, sj)
                for k in real:
                    if k == i or k == j:
                        continue
                    sk = state[k]
                    dr += pair_cost(prob, i, sj, k, sk) - pair_cost(prob, i, si, k, sk)
                    dr += pair_cost(prob, j, si, k, sk) - pair_cost(prob, j, sj, k, sk)
                de = dg / K + (dr / npairs if npairs else 0.0)
                best = _candidate_update(best, de, ('swap', i, j), ('swap', i, j))

        # 3. assigned/ABSTAIN exchange preserving K; replacement may use vacated or already-unused site.
        if abst:
            for i in real:
                si = state[i]
                candidate_free = unused | {si}
                for j in abst:
                    for s in prob.feasible[j]:
                        if s not in candidate_free:
                            continue
                        dg = prob.g_cost[(j, s)] - prob.g_cost[(i, si)]
                        dr = 0.0
                        for k in real:
                            if k == i:
                                continue
                            sk = state[k]
                            dr += pair_cost(prob, j, s, k, sk) - pair_cost(prob, i, si, k, sk)
                        de = dg / K + (dr / npairs if npairs else 0.0)
                        best = _candidate_update(best, de, ('exchange', i, j, s), ('exchange', i, j, s))

        if best is None:
            break
        oldE = E
        kind, *args = best[2]
        if kind == 'move':
            i, s = args; state[i] = s
        elif kind == 'swap':
            i, j = args; state[i], state[j] = state[j], state[i]
        elif kind == 'exchange':
            i, j, s = args; state[i] = -1; state[j] = s
        else:
            raise RuntimeError(kind)
        validate_state(prob, state, K)
        E, EG, ER, sum_g, sum_r = objective(prob, state)
        if not E < oldE - STRICT_EPS:
            raise RuntimeError(('non-strict accepted step', prob.family, prob.view, oldE, E, best))

    return state, {'objective': E, 'E_G': EG, 'E_R': ER, 'sum_G': sum_g, 'sum_R': sum_r, 'descent_steps': iterations - 1}


def serialize_state(prob: ViewProblem, state):
    assigned = {}
    abstained = []
    for i in prob.carriers:
        s = state[i]
        if s < 0:
            abstained.append(i)
        else:
            assigned[str(i)] = list(prob.sites[s])
    return assigned, abstained


def solve_view(prob: ViewProblem):
    K_kuhn = kuhn_max_cardinality(prob)
    zero_state = hungarian_state(prob, 'zero', 0)
    K_hung = state_real_count(zero_state)
    if K_kuhn != K_hung:
        raise RuntimeError(('independent max matching disagreement', prob.family, prob.view, K_kuhn, K_hung))
    K = K_kuhn

    restarts = []
    best_pack = None
    for r in range(N_RESTARTS):
        init = hungarian_state(prob, 'g' if r == 0 else 'random', r)
        if state_real_count(init) != K:
            raise RuntimeError(('restart cardinality failure', prob.family, prob.view, r))
        final, metrics = descend(prob, init, K)
        validate_state(prob, final, K)
        replay = objective(prob, final)
        replay_delta = max(abs(replay[0] - metrics['objective']), abs(replay[1] - metrics['E_G']), abs(replay[2] - metrics['E_R']))
        if replay_delta > 1e-12:
            raise RuntimeError(('objective replay mismatch', prob.family, prob.view, r, replay_delta))
        key = state_key(prob, final)
        h = hashlib.sha256(repr(key).encode()).hexdigest()
        rec = {'restart': r, **metrics, 'objective_replay_max_abs': float(replay_delta), 'assignment_sha256': h}
        restarts.append(rec)
        pack = (metrics['objective'], key, r, final, metrics)
        if best_pack is None or pack[:2] < best_pack[:2]:
            best_pack = pack

    _, best_key, best_restart, best_state, best_metrics = best_pack
    same_best = int(sum(state_key(prob, descend(prob, hungarian_state(prob, 'g' if r == 0 else 'random', r), K)[0]) == best_key for r in []))
    # agreement derived from already recorded assignment hashes, no rerun
    best_hash = hashlib.sha256(repr(best_key).encode()).hexdigest()
    same_best = int(sum(x['assignment_sha256'] == best_hash for x in restarts))
    assigned, abstained = serialize_state(prob, best_state)
    vals = np.asarray([x['objective'] for x in restarts], float)
    return {
        'view': prob.view,
        'visible_n': prob.n,
        'unique_site_n': len(prob.sites),
        'max_real_matching_n_kuhn': K_kuhn,
        'max_real_matching_n_hungarian': K_hung,
        'K': K,
        'assigned': assigned,
        'abstained': abstained,
        'abstain_n': len(abstained),
        'objective': best_metrics['objective'],
        'E_G': best_metrics['E_G'],
        'E_R': best_metrics['E_R'],
        'best_restart': best_restart,
        'best_assignment_sha256': best_hash,
        'restart_best_agreement_n': same_best,
        'restart_unique_assignment_n': len({x['assignment_sha256'] for x in restarts}),
        'restart_objective_min': float(vals.min()),
        'restart_objective_max': float(vals.max()),
        'restart_objective_spread': float(vals.max() - vals.min()),
        'scale_A': prob.scale_A,
        'scale_B': prob.scale_B,
        'objective_recompute_max_abs': float(max(x['objective_replay_max_abs'] for x in restarts)),
        'rows': prob.rows_serial,
        'restarts': restarts,
    }


def solve_family(evidence: Path, family: int, expected_content_sha: str):
    z = np.load(evidence, allow_pickle=False)
    digest = evidence_content_digest(z)
    if digest != expected_content_sha:
        raise RuntimeError(('V3A evidence content digest mismatch', family, digest, expected_content_sha))
    coords = np.asarray(z['coords']); scores = np.asarray(z['scores']); valid = np.asarray(z['valid'])
    VA = np.asarray(z['V_A']); XYA = np.asarray(z['XY_A'])
    views = []
    for v in range(8):
        prob = build_view_problem(family, coords, scores, valid, VA, XYA, v)
        views.append(solve_view(prob))
    return {
        'schema': SCHEMA,
        'contract_blob_sha': CONTRACT_BLOB_SHA,
        'family': family,
        'episode': 'e01',
        'truth_access': 'NONE',
        'proposal_evidence_content_sha256': digest,
        'proposal_evidence_file_sha256': sha256_file(evidence),
        'views': views,
        'audit': {
            'visible_total': int(sum(x['visible_n'] for x in views)),
            'assigned_real_total': int(sum(len(x['assigned']) for x in views)),
            'abstain_total': int(sum(x['abstain_n'] for x in views)),
            'full_real_matching_views': int(sum(x['K'] == x['visible_n'] for x in views)),
            'matching_independent_parity': bool(all(x['max_real_matching_n_kuhn'] == x['max_real_matching_n_hungarian'] for x in views)),
            'objective_recompute_max_abs': float(max(x['objective_recompute_max_abs'] for x in views)),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--expected-content-sha', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise RuntimeError('refusing overwrite')
    result = solve_family(a.evidence, a.family, a.expected_content_sha)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({'family': a.family, 'audit': result['audit'], 'out_sha256': sha256_file(a.out)}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
