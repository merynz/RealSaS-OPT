from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

ROW_TOL = 0.030
COARSE_KEEP = 16
P_KEEP = 4
RRF_K = 60.0
TOPK_MAX = 8
CYCLE_TOL = 0.030
UNCERTAIN_Q = 0.75
VERY_UNCERTAIN_Q = 0.90


def _sample_nodes(field: torch.Tensor, view: int, xy: np.ndarray) -> np.ndarray:
    """field [1,V,C,H,W], xy [N,2] normalized -> [N,C] float32."""
    if len(xy) == 0:
        return np.zeros((0, field.shape[2]), np.float32)
    g = torch.as_tensor(xy, dtype=field.dtype, device=field.device)[None, :, None, :]
    f = field[:, view]
    s = F.grid_sample(f, g, mode='bilinear', padding_mode='zeros', align_corners=False)
    return s[0, :, :, 0].T.detach().float().cpu().numpy()


def _percentile_rank_high(values: np.ndarray) -> np.ndarray:
    """0=lowest risk, 1=highest risk; deterministic stable ranking."""
    v = np.asarray(values, np.float64).reshape(-1)
    if len(v) <= 1:
        return np.zeros(len(v), np.float32)
    order = np.argsort(v, kind='stable')
    ranks = np.empty(len(v), np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    return (ranks / float(len(v) - 1)).astype(np.float32)


def _rank_positions(order: np.ndarray, n: int) -> np.ndarray:
    r = np.empty(n, np.int32)
    r[order] = np.arange(1, n + 1, dtype=np.int32)
    return r


@dataclass
class ViewNodes:
    track_ids: np.ndarray
    xy: np.ndarray
    P: np.ndarray
    N: np.ndarray
    log_sigma: np.ndarray
    risk_q: np.ndarray
    Zc: np.ndarray
    Zf: np.ndarray
    row_by_track: Dict[int, int]


@dataclass
class PairResult:
    src_view: int
    tgt_view: int
    src_track_ids: np.ndarray
    tgt_track_ids: np.ndarray
    top_track_ids: np.ndarray
    top_scores: np.ndarray
    coarse_top_track_ids: np.ndarray
    fine_top_track_ids: np.ndarray
    p_top_track_ids: np.ndarray
    coarse_top1: np.ndarray
    fine_top1: np.ndarray
    p_top1: np.ndarray
    p_top1_dist: np.ndarray
    row_by_track: Dict[int, int]


def build_view_nodes(outputs: Dict[str, torch.Tensor], z) -> List[ViewNodes]:
    vis = z['track_visible'].astype(bool)
    xy_all = z['track_xy'].astype(np.float32)
    out: List[ViewNodes] = []
    for v in range(8):
        ids = np.flatnonzero(vis[:, v]).astype(np.int32)
        xy = xy_all[ids, v].astype(np.float32)
        P = _sample_nodes(outputs['P'], v, xy)
        N = _sample_nodes(outputs['N'], v, xy)
        U = _sample_nodes(outputs['log_sigma'], v, xy)[:, 0]
        Zc = _sample_nodes(outputs['Z_coarse'], v, xy)
        Zf = _sample_nodes(outputs['Z_fine'], v, xy)
        Zc /= np.maximum(np.linalg.norm(Zc, axis=1, keepdims=True), 1e-8)
        Zf /= np.maximum(np.linalg.norm(Zf, axis=1, keepdims=True), 1e-8)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)
        out.append(ViewNodes(ids, xy, P, N, U, _percentile_rank_high(U), Zc, Zf,
                             {int(t): i for i, t in enumerate(ids.tolist())}))
    return out


def _top_track_ids(order_local: np.ndarray, tgt_track_ids: np.ndarray, k: int = TOPK_MAX) -> np.ndarray:
    ans = np.full(k, -1, np.int32)
    take = min(k, len(order_local))
    if take:
        ans[:take] = tgt_track_ids[order_local[:take]]
    return ans


def match_pair(src: ViewNodes, tgt: ViewNodes, src_view: int, tgt_view: int) -> PairResult:
    ns, nt = len(src.track_ids), len(tgt.track_ids)
    if ns == 0 or nt == 0:
        raise RuntimeError(f'empty view nodes for pair {src_view}->{tgt_view}')
    coarse = src.Zc @ tgt.Zc.T
    fine = src.Zf @ tgt.Zf.T
    pd = np.linalg.norm(src.P[:, None, :] - tgt.P[None, :, :], axis=-1)
    row_ok = np.abs(src.xy[:, None, 1] - tgt.xy[None, :, 1]) <= ROW_TOL

    top = np.full((ns, TOPK_MAX), -1, np.int32)
    scores = np.full((ns, TOPK_MAX), -np.inf, np.float32)
    ct = np.full_like(top, -1); ft = np.full_like(top, -1); pt = np.full_like(top, -1)
    c1 = np.full(ns, -1, np.int32); f1 = np.full(ns, -1, np.int32); p1 = np.full(ns, -1, np.int32)
    p1d = np.full(ns, np.inf, np.float32)

    all_t = np.arange(nt, dtype=np.int32)
    for qi in range(ns):
        valid = np.flatnonzero(row_ok[qi]).astype(np.int32)
        if len(valid) == 0:
            valid = all_t
        c_order = valid[np.argsort(-coarse[qi, valid], kind='stable')]
        f_order = valid[np.argsort(-fine[qi, valid], kind='stable')]
        p_order = valid[np.argsort(pd[qi, valid], kind='stable')]
        ct[qi] = _top_track_ids(c_order, tgt.track_ids)
        ft[qi] = _top_track_ids(f_order, tgt.track_ids)
        pt[qi] = _top_track_ids(p_order, tgt.track_ids)
        c1[qi] = tgt.track_ids[c_order[0]]; f1[qi] = tgt.track_ids[f_order[0]]; p1[qi] = tgt.track_ids[p_order[0]]
        p1d[qi] = float(pd[qi, p_order[0]])

        keep = np.unique(np.concatenate([c_order[:min(COARSE_KEEP, len(c_order))],
                                         p_order[:min(P_KEEP, len(p_order))]])).astype(np.int32)
        rc_full = _rank_positions(np.argsort(-coarse[qi, keep], kind='stable'), len(keep))
        rf_full = _rank_positions(np.argsort(-fine[qi, keep], kind='stable'), len(keep))
        rp_full = _rank_positions(np.argsort(pd[qi, keep], kind='stable'), len(keep))
        rrf = 1.0/(RRF_K + rc_full) + 1.0/(RRF_K + rf_full) + 1.0/(RRF_K + rp_full)
        ord_keep = np.argsort(-rrf, kind='stable')
        take = min(TOPK_MAX, len(ord_keep))
        local = keep[ord_keep[:take]]
        top[qi, :take] = tgt.track_ids[local]
        scores[qi, :take] = rrf[ord_keep[:take]].astype(np.float32)

    return PairResult(src_view, tgt_view, src.track_ids, tgt.track_ids, top, scores, ct, ft, pt,
                      c1, f1, p1, p1d, {int(t): i for i, t in enumerate(src.track_ids.tolist())})


def build_all_pairs(nodes: List[ViewNodes]) -> Dict[Tuple[int, int], PairResult]:
    return {(s, t): match_pair(nodes[s], nodes[t], s, t) for s in range(8) for t in range(8) if s != t}


def _third_views(s: int, t: int) -> List[int]:
    out=[]
    for delta in (2, -2, 3, -3, 1, -1):
        u=(s+delta)%8
        if u not in (s,t) and u not in out:
            out.append(u)
        if len(out)==2: break
    return out


def _contains(arr: np.ndarray, value: int, k: int) -> bool:
    return bool(np.any(arr[:k] == int(value)))


def evaluate_asset_matcher(nodes: List[ViewNodes], pairs: Dict[Tuple[int,int], PairResult], z) -> Dict:
    vis = z['track_visible'].astype(bool)
    counters = {
        'queries':0,
        'coarse_top1':0,'coarse_top4':0,'coarse_top8':0,
        'fine_top1':0,'fine_top4':0,'fine_top8':0,
        'p_top1':0,'p_top4':0,'p_top8':0,
        'composite_top1':0,'composite_top4':0,'composite_top8':0,
        'reciprocal_top1':0,'reciprocal_top4':0,'cycle_any':0,
        'singleton_count':0,'singleton_correct':0,
        'set4_count':0,'set8_count':0,'output_truth':0,
        'set_size_sum':0,
        'truth_row_domain':0,
    }
    singleton_wrong=[]; hard_tail=[]
    for s in range(8):
        for t in range(8):
            if s==t: continue
            pr=pairs[(s,t)]; rev=pairs[(t,s)]
            tgt_node=nodes[t]; src_node=nodes[s]
            common=np.flatnonzero(vis[:,s] & vis[:,t]).astype(np.int32)
            for truth in common.tolist():
                qi=pr.row_by_track.get(int(truth))
                ti=tgt_node.row_by_track.get(int(truth))
                if qi is None or ti is None: continue
                counters['queries'] += 1
                if abs(float(src_node.xy[qi,1]-tgt_node.xy[ti,1])) <= ROW_TOL:
                    counters['truth_row_domain'] += 1
                for name,arr in [('coarse',pr.coarse_top_track_ids),('fine',pr.fine_top_track_ids),('p',pr.p_top_track_ids),('composite',pr.top_track_ids)]:
                    for k in (1,4,8):
                        if _contains(arr[qi],truth,k): counters[f'{name}_top{k}'] += 1
                pred=int(pr.top_track_ids[qi,0])
                if pred < 0: continue
                rqi=rev.row_by_track.get(pred)
                rtop = rev.top_track_ids[rqi] if rqi is not None else np.full(TOPK_MAX,-1,np.int32)
                recip1=_contains(rtop,truth,1); recip4=_contains(rtop,truth,4)
                counters['reciprocal_top1'] += int(recip1); counters['reciprocal_top4'] += int(recip4)

                cycle_support=0
                for u in _third_views(s,t):
                    direct=pairs[(s,u)]; indirect=pairs[(t,u)]
                    dqi=direct.row_by_track.get(truth); iqi=indirect.row_by_track.get(pred)
                    if dqi is None or iqi is None: continue
                    a=int(direct.top_track_ids[dqi,0]); b=int(indirect.top_track_ids[iqi,0])
                    if a<0 or b<0: continue
                    au=nodes[u].row_by_track.get(a); bu=nodes[u].row_by_track.get(b)
                    if au is None or bu is None: continue
                    if float(np.linalg.norm(nodes[u].xy[au]-nodes[u].xy[bu])) <= CYCLE_TOL:
                        cycle_support += 1
                counters['cycle_any'] += int(cycle_support>=1)

                votes=int(pr.coarse_top1[qi]==pred)+int(pr.fine_top1[qi]==pred)+int(pr.p_top1[qi]==pred)
                tgt_q=tgt_node.row_by_track[pred]
                risk=max(float(src_node.risk_q[qi]),float(tgt_node.risk_q[tgt_q]))
                singleton = bool(recip1 and cycle_support>=1 and votes>=2 and risk<UNCERTAIN_Q)
                if singleton:
                    set_size=1; counters['singleton_count']+=1
                    ok=(pred==truth); counters['singleton_correct']+=int(ok)
                    if not ok: singleton_wrong.append({'s':s,'t':t,'truth':truth,'pred':pred,'votes':votes,'cycle':cycle_support,'risk_q':risk})
                elif risk>=VERY_UNCERTAIN_Q or not recip4:
                    set_size=8; counters['set8_count']+=1
                else:
                    set_size=4; counters['set4_count']+=1
                counters['set_size_sum'] += set_size
                ok=_contains(pr.top_track_ids[qi],truth,set_size)
                counters['output_truth'] += int(ok)
                if not _contains(pr.top_track_ids[qi],truth,8):
                    hard_tail.append({'s':s,'t':t,'truth':truth,'pred_top8':pr.top_track_ids[qi].tolist()})

    q=max(counters['queries'],1)
    sc=max(counters['singleton_count'],1)
    metrics={
        'queries':counters['queries'],
        'truth_row_domain_recall':counters['truth_row_domain']/q,
        'raw_coarse':{f'top{k}':counters[f'coarse_top{k}']/q for k in (1,4,8)},
        'raw_fine':{f'top{k}':counters[f'fine_top{k}']/q for k in (1,4,8)},
        'raw_P':{f'top{k}':counters[f'p_top{k}']/q for k in (1,4,8)},
        'composite':{f'top{k}':counters[f'composite_top{k}']/q for k in (1,4,8)},
        'reciprocal_top1_rate':counters['reciprocal_top1']/q,
        'reciprocal_top4_rate':counters['reciprocal_top4']/q,
        'cycle_any_rate':counters['cycle_any']/q,
        'qualification':{
            'singleton_coverage':counters['singleton_count']/q,
            'singleton_precision':counters['singleton_correct']/sc if counters['singleton_count'] else None,
            'set4_rate':counters['set4_count']/q,
            'set8_rate':counters['set8_count']/q,
            'mean_set_size':counters['set_size_sum']/q,
            'output_truth_coverage':counters['output_truth']/q,
        },
        'hard_tail_count':len(hard_tail),
        'singleton_wrong_examples':singleton_wrong[:20],
        'hard_tail_examples':hard_tail[:20],
    }
    return metrics


def aggregate_asset_metrics(rows: List[Dict]) -> Dict:
    if not rows: return {'assets':0,'queries':0}
    q=np.array([max(1,r['queries']) for r in rows],np.float64); Q=float(q.sum())
    def wav(path):
        vals=[]
        for r in rows:
            x=r
            for p in path: x=x[p]
            vals.append(float(x) if x is not None else np.nan)
        a=np.asarray(vals,np.float64); m=np.isfinite(a)
        if not np.any(m): return None
        return float(np.sum(a[m]*q[m])/np.sum(q[m]))
    out={'assets':len(rows),'queries':int(Q)}
    out['truth_row_domain_recall']=wav(['truth_row_domain_recall'])
    for sec in ('raw_coarse','raw_fine','raw_P','composite'):
        out[sec]={k:wav([sec,k]) for k in ('top1','top4','top8')}
    out['reciprocal_top1_rate']=wav(['reciprocal_top1_rate']); out['reciprocal_top4_rate']=wav(['reciprocal_top4_rate']); out['cycle_any_rate']=wav(['cycle_any_rate'])
    out['qualification']={k:wav(['qualification',k]) for k in ('singleton_coverage','singleton_precision','set4_rate','set8_rate','mean_set_size','output_truth_coverage')}
    out['family_tail']={
        'composite_top4_p10':float(np.percentile([r['composite']['top4'] for r in rows],10)),
        'composite_top8_p10':float(np.percentile([r['composite']['top8'] for r in rows],10)),
        'output_truth_coverage_p10':float(np.percentile([r['qualification']['output_truth_coverage'] for r in rows],10)),
    }
    out['hard_tail_count']=int(sum(r['hard_tail_count'] for r in rows))
    return out
