from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = 'RealSaS.N1D.Rank2G.G2.ReciprocalCycleProposal.v2'
QUESTION_COMMIT = '973107d67992fad17ad83bb4d822a13913482b47'
EXPECTED_V5_SHA256 = 'ea7ab996f881a08538c3cc049ca799e98783919255ed2c5214c01feada3161d7'

YAW_DEG = (0., 45., 90., 135., 180., 225., 270., 315.)
IMAGE_NATIVE = 256
G_DESC_WEIGHT = 1.0 / 3.0
G_GEOM_WEIGHT = 1.0 / 3.0
G_CYCLE_WEIGHT = 1.0 / 3.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_v5(path: Path):
    got = sha256_file(path)
    if got != EXPECTED_V5_SHA256:
        raise RuntimeError(('V5 source sha256 mismatch', got, EXPECTED_V5_SHA256))
    spec = importlib.util.spec_from_file_location('rank2_v5_exact_g2_parent', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    if spec.loader is None:
        raise RuntimeError('V5 loader missing')
    spec.loader.exec_module(mod)
    return mod


def camera_basis():
    rr, uu = [], []
    for deg in YAW_DEG:
        t = np.deg2rad(deg)
        radial = np.array([np.sin(t), -np.cos(t), 0.], float)
        fw = -radial
        up = np.array([0., 0., 1.], float)
        right = np.cross(fw, up)
        right /= max(np.linalg.norm(right), 1e-12)
        rr.append(right)
        uu.append(up)
    return np.stack(rr), np.stack(uu)


CAM_R, CAM_U = camera_basis()


def project_points_np(P, view):
    P = np.asarray(P, float)
    x = P @ CAM_R[int(view)]
    y = P @ CAM_U[int(view)]
    return np.stack([
        (2 * x + 1) * .5 * (IMAGE_NATIVE - 1),
        (1 - 2 * y) * .5 * (IMAGE_NATIVE - 1),
    ], axis=-1)


def linear_triangulate_np(obs):
    AA, bb = [], []
    for v, x, y in obs:
        sx = float(x) / (IMAGE_NATIVE - 1) - .5
        sy = float(y) / (IMAGE_NATIVE - 1) - .5
        AA += [CAM_R[int(v)], -CAM_U[int(v)]]
        bb += [sx, sy]
    AA = np.asarray(AA, float)
    bb = np.asarray(bb, float)
    if np.linalg.matrix_rank(AA) < 3:
        return None
    p, _, rank, _ = np.linalg.lstsq(AA, bb, rcond=None)
    return p if rank >= 3 else None


def _eval_world_point(p, usable, row_items_all):
    errs, desc, nearest = [], [], []
    support2 = support4 = 0
    for w in usable:
        items = row_items_all[w]
        C = np.asarray([x[0] for x in items], float)
        q = project_points_np(np.asarray(p)[None], w)[0]
        d = np.linalg.norm(C - q[None], axis=1)
        j = int(np.argmin(d))
        e = float(d[j])
        errs.append(e)
        desc.append(float(items[j][1]))
        support2 += int(e <= 2.)
        support4 += int(e <= 4.)
        nearest.append((w, j))
    return float(np.mean(errs)), float(np.mean(desc)), support2, support4, nearest


def proposal_geometry_raw(i, v, site, row_items_by_carrier):
    row_items_all = row_items_by_carrier[i]
    usable = [w for w in range(8) if len(row_items_all[w])]
    best = None
    for u in usable:
        if u == v:
            continue
        for site2, _ in row_items_all[u]:
            p = linear_triangulate_np([(v, *site), (u, *site2)])
            if p is None or not np.isfinite(p).all() or not np.all(np.abs(p) <= .75):
                continue
            e, d, s2, s4, nearest = _eval_world_point(p, usable, row_items_all)
            key = (e, -s4, -s2, -d, 0, tuple(np.round(p, 12)))
            if best is None or key < best[0]:
                best = (key, e, s4, s2, d, 'seed')
            pref = linear_triangulate_np([(w, *row_items_all[w][j][0]) for w, j in nearest])
            if pref is not None and np.isfinite(pref).all() and np.all(np.abs(pref) <= .75):
                e2, d2, a2, a4, _ = _eval_world_point(pref, usable, row_items_all)
                key2 = (e2, -a4, -a2, -d2, 1, tuple(np.round(pref, 12)))
                if best is None or key2 < best[0]:
                    best = (key2, e2, a4, a2, d2, 'refined')
    if best is None:
        return float('inf'), 0, 0, None, 'none'
    return float(best[1]), int(best[2]), int(best[3]), float(best[4]), best[5]


def row_cycle_by_site(coords, scores, valid, cycle_px, reverse_xy, reverse_score, i, v):
    # Same duplicate winner rule as V5.row_sites: highest descriptor score wins.
    best = {}
    for k in range(4):
        if not valid[i, v, k]:
            continue
        key = tuple(map(int, np.rint(coords[i, v, k]).astype(int)))
        sc = float(scores[i, v, k])
        rec = (
            sc,
            float(cycle_px[i, v, k]),
            np.asarray(reverse_xy[i, v, k], float),
            float(reverse_score[i, v, k]),
        )
        if key not in best or sc > best[key][0]:
            best[key] = rec
    return best


def array_digest(arrays) -> str:
    h = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(a)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(np.asarray(a.shape, np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def base_evidence_content_digest(z) -> str:
    return array_digest([
        ('coords', np.asarray(z['coords'])),
        ('scores', np.asarray(z['scores'])),
        ('valid', np.asarray(z['valid'])),
        ('V_A', np.asarray(z['V_A'])),
        ('XY_A', np.asarray(z['XY_A'])),
        ('output_idx', np.asarray(z['output_idx'])),
    ])


def evidence_content_digest(z) -> str:
    return array_digest([
        ('coords', np.asarray(z['coords'])),
        ('scores', np.asarray(z['scores'])),
        ('valid', np.asarray(z['valid'])),
        ('V_A', np.asarray(z['V_A'])),
        ('XY_A', np.asarray(z['XY_A'])),
        ('output_idx', np.asarray(z['output_idx'])),
        ('reverse_xy', np.asarray(z['reverse_xy'])),
        ('reverse_score', np.asarray(z['reverse_score'])),
        ('cycle_px', np.asarray(z['cycle_px'])),
    ])


def finite_rank(v5, x):
    x = np.asarray(x, float)
    if np.any(~np.isfinite(x)):
        finite = x[np.isfinite(x)]
        worst = float(finite.max()) + 1e6 if len(finite) else 1e9
        x = np.where(np.isfinite(x), x, worst)
    return x, v5.rank01(x, descending=False)


def build_view_problem(v5, family, coords, scores, valid, VA, XYA, cycle_px, reverse_xy, reverse_score, v):
    carriers = [int(i) for i in np.where(VA[v] > 0)[0]]
    rows_all = {i: [v5.row_sites(coords, scores, valid, i, w) for w in range(8)] for i in carriers}
    rows = {i: rows_all[i][v] for i in carriers}
    cycles = {i: row_cycle_by_site(coords, scores, valid, cycle_px, reverse_xy, reverse_score, i, v) for i in carriers}
    if any(len(rows[i]) == 0 for i in carriers):
        raise RuntimeError(('visible carrier has zero proposal', family, v))

    sites = sorted({site for i in carriers for site, _ in rows[i]}, key=lambda p: (p[1], p[0]))
    site_to_id = {s: j for j, s in enumerate(sites)}
    feasible, g_cost, rows_serial = {}, {}, {}

    for i in carriers:
        items = rows[i]
        sc = np.asarray([x[1] for x in items], float)
        desc_rank = v5.rank01(sc, descending=True)

        geo_raw, geo_debug = [], []
        for site, _ in items:
            gr, s4, s2, ds, mode = proposal_geometry_raw(i, v, site, rows_all)
            geo_raw.append(gr)
            geo_debug.append({
                'reproj_px': None if not np.isfinite(gr) else float(gr),
                'support4': s4,
                'support2': s2,
                'descriptor_support': ds,
                'mode': mode,
            })
        geo_raw, geo_rank = finite_rank(v5, geo_raw)

        cyc_raw = np.asarray([cycles[i][x[0]][1] for x in items], float)
        cyc_raw, cycle_rank = finite_rank(v5, cyc_raw)

        combined = (
            G_DESC_WEIGHT * desc_rank
            + G_GEOM_WEIGHT * geo_rank
            + G_CYCLE_WEIGHT * cycle_rank
        )
        ids = [site_to_id[x[0]] for x in items]
        feasible[i] = ids
        for sid, gc in zip(ids, combined):
            g_cost[(i, sid)] = float(gc)

        rows_serial[str(i)] = {
            'sites': [list(x[0]) for x in items],
            'site_ids': ids,
            'descriptor_scores': sc.tolist(),
            'G_descriptor_rank': desc_rank.tolist(),
            'G_geometry_raw_reproj_px': geo_raw.tolist(),
            'G_geometry_rank': geo_rank.tolist(),
            'G_reciprocal_cycle_raw_px': cyc_raw.tolist(),
            'G_reciprocal_cycle_rank': cycle_rank.tolist(),
            'G_rank': combined.tolist(),
            'G_weights': {
                'descriptor': G_DESC_WEIGHT,
                'geometry': G_GEOM_WEIGHT,
                'reciprocal_cycle': G_CYCLE_WEIGHT,
            },
            'geometry_debug': geo_debug,
            'reverse_xy': [cycles[i][x[0]][2].tolist() for x in items],
            'reverse_descriptor_score': [cycles[i][x[0]][3] for x in items],
            'g_top1_descriptor': list(items[int(np.argmin(desc_rank))][0]),
            'g_top1_geometry': list(items[int(np.argmin(geo_rank))][0]),
            'g_top1_cycle': list(items[int(np.argmin(cycle_rank))][0]),
            'g_top1_combined': list(items[int(np.argmin(combined))][0]),
        }

    # Exact V5 R construction, using V5 functions/constants directly.
    scale_A = v5.scene_scale(XYA[v, carriers])
    scale_B = v5.scene_scale(np.asarray(sites, float))
    pair_r = {}
    for aa in range(len(carriers)):
        i = carriers[aa]
        for bb in range(aa + 1, len(carriers)):
            j = carriers[bb]
            combos, raws = [], []
            da = (np.asarray(XYA[v, i], float) - np.asarray(XYA[v, j], float)) / max(scale_A, 1e-9)
            for si in feasible[i]:
                for sj in feasible[j]:
                    if si == sj:
                        continue
                    db = (np.asarray(sites[si], float) - np.asarray(sites[sj], float)) / max(scale_B, 1e-9)
                    feat = np.r_[db - da, np.linalg.norm(db) - np.linalg.norm(da), v5._unit(db) - v5._unit(da)]
                    raw = float(np.mean(v5.huber_abs(feat, v5.HUBER_DELTA)))
                    combos.append((si, sj))
                    raws.append(raw)
            if combos:
                rr = v5.rank01(np.asarray(raws), descending=False)
                pair_r[(i, j)] = {c: float(r) for c, r in zip(combos, rr)}

    return v5.ViewProblem(
        family=family,
        view=v,
        carriers=carriers,
        sites=sites,
        feasible=feasible,
        g_cost=g_cost,
        pair_r=pair_r,
        scale_A=scale_A,
        scale_B=scale_B,
        rows_serial=rows_serial,
    )


def solve_family(v5, evidence: Path, family: int, expected_base_sha: str, expected_g2_sha: str, episode: str):
    z = np.load(evidence, allow_pickle=False)
    base_sha = base_evidence_content_digest(z)
    g2_sha = evidence_content_digest(z)
    if base_sha != expected_base_sha:
        raise RuntimeError(('base V3A/G1 evidence content digest mismatch', family, base_sha, expected_base_sha))
    if g2_sha != expected_g2_sha:
        raise RuntimeError(('G2 evidence content digest mismatch', family, g2_sha, expected_g2_sha))

    coords = np.asarray(z['coords'])
    scores = np.asarray(z['scores'])
    valid = np.asarray(z['valid'])
    VA = np.asarray(z['V_A'])
    XYA = np.asarray(z['XY_A'])
    cycle_px = np.asarray(z['cycle_px'])
    reverse_xy = np.asarray(z['reverse_xy'])
    reverse_score = np.asarray(z['reverse_score'])

    views = []
    for v in range(8):
        prob = build_view_problem(v5, family, coords, scores, valid, VA, XYA, cycle_px, reverse_xy, reverse_score, v)
        views.append(v5.solve_view(prob))

    return {
        'schema': SCHEMA,
        'question_commit': QUESTION_COMMIT,
        'family': family,
        'episode': episode,
        'truth_access': 'NONE',
        'v5_source_sha256': EXPECTED_V5_SHA256,
        'base_proposal_evidence_content_sha256': base_sha,
        'g2_evidence_content_sha256': g2_sha,
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
    ap.add_argument('--v5-source', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    ap.add_argument('--family', type=int, required=True)
    ap.add_argument('--episode', default='e01')
    ap.add_argument('--expected-base-content-sha', required=True)
    ap.add_argument('--expected-g2-content-sha', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise RuntimeError('refusing overwrite')
    v5 = load_v5(a.v5_source)
    result = solve_family(v5, a.evidence, a.family, a.expected_base_content_sha, a.expected_g2_content_sha, a.episode)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps({'family': a.family, 'audit': result['audit'], 'out_sha256': sha256_file(a.out)}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
