from __future__ import annotations

"""X6 accelerator-screen / CPU-verify C0 traversal.

Accelerator arithmetic never owns a pruning decision. It may only propose a
strict-sign node. Every proposal is re-evaluated by the authoritative CPU
float64 X5 batched engine; only a matching CPU strict sign becomes terminal.

An accelerator UNKNOWN remains active. An accelerator false decisive proposal
is rejected and also remains active. Therefore accelerator numerical behavior
can reduce efficiency but cannot create an EMPTY certificate by itself.
"""

from dataclasses import dataclass
from collections import Counter
import numpy as np
import torch

from batched_c0_x5 import (
    _bound_batch_chunked_prepared,
    _split_octants_batch,
)
from range_engine_c0_v3 import PreparedField


@dataclass(frozen=True)
class X6Terminal:
    root_id: int
    depth: int
    sign: int
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    screen_lower: float
    screen_upper: float
    cpu_lower: float
    cpu_upper: float


@dataclass(frozen=True)
class X6Result:
    root_states: tuple[str, ...]
    root_positive_terminal_counts: tuple[int, ...]
    root_negative_terminal_counts: tuple[int, ...]
    root_unresolved_leaf_counts: tuple[int, ...]
    terminal_count: int
    final_unresolved_leaf_count: int
    screen_decisive_proposal_count: int
    cpu_confirmed_count: int
    cpu_rejected_count: int
    accelerator_box_eval_count: int
    cpu_verify_box_eval_count: int
    terminals: tuple[X6Terminal, ...]


def strict_signs(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower=np.asarray(lower,dtype=np.float64)
    upper=np.asarray(upper,dtype=np.float64)
    if lower.shape!=upper.shape:
        raise ValueError("BOUND_SHAPE_MISMATCH")
    return np.where(lower>0.0,1,np.where(upper<0.0,-1,0)).astype(np.int8)


def confirm_candidate_signs(
    screen_lower: np.ndarray,
    screen_upper: np.ndarray,
    cpu_lower: np.ndarray,
    cpu_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return screen signs, CPU signs, and confirmed signs.

    Confirmed sign is nonzero only when both backends are strict and identical.
    """
    ss=strict_signs(screen_lower,screen_upper)
    cs=strict_signs(cpu_lower,cpu_upper)
    confirmed=np.where((ss!=0)&(ss==cs),ss,0).astype(np.int8)
    return ss,cs,confirmed


def _root_state(pos:int,neg:int,unk:int)->str:
    if pos and neg:
        return "PROVEN_ZERO_EXISTS"
    if unk:
        return "UNKNOWN"
    if pos and not neg:
        return "PROVEN_EMPTY_POSITIVE"
    if neg and not pos:
        return "PROVEN_EMPTY_NEGATIVE"
    return "UNKNOWN"


def screen_verify_single_regime_roots(
    cpu_p: PreparedField,
    cpu_planes: torch.Tensor,
    accelerator_p: PreparedField,
    accelerator_planes: torch.Tensor,
    los: np.ndarray,
    his: np.ndarray,
    *,
    max_micro_depth: int=2,
    accelerator_node_batch_size: int=1024,
    cpu_verify_batch_size: int=32,
) -> X6Result:
    if max_micro_depth<0:
        raise ValueError("NEGATIVE_MICRO_DEPTH")
    root_los=np.asarray(los,dtype=np.float64).reshape(-1,3)
    root_his=np.asarray(his,dtype=np.float64).reshape(-1,3)
    if root_los.shape!=root_his.shape or np.any(root_los>=root_his):
        raise ValueError("INVALID_ROOT_BOXES")
    nroots=len(root_los)
    if nroots==0:
        return X6Result(tuple(),tuple(),tuple(),tuple(),0,0,0,0,0,0,0,tuple())

    frontier_lo=root_los
    frontier_hi=root_his
    frontier_root=np.arange(nroots,dtype=np.int64)

    terminals=[]
    screen_prop=0
    cpu_confirm=0
    cpu_reject=0
    accel_evals=0
    cpu_evals=0

    for depth in range(max_micro_depth+1):
        s_lo,s_hi=_bound_batch_chunked_prepared(
            accelerator_p,
            accelerator_planes,
            frontier_lo,
            frontier_hi,
            node_batch_size=accelerator_node_batch_size,
        )
        accel_evals+=len(frontier_lo)
        s_sign=strict_signs(s_lo,s_hi)
        proposal_idx=np.where(s_sign!=0)[0]
        screen_prop+=len(proposal_idx)

        confirmed=np.zeros(len(frontier_lo),dtype=np.int8)
        cpu_lo_all=np.full(len(frontier_lo),np.nan,dtype=np.float64)
        cpu_hi_all=np.full(len(frontier_lo),np.nan,dtype=np.float64)

        if len(proposal_idx):
            c_lo,c_hi=_bound_batch_chunked_prepared(
                cpu_p,
                cpu_planes,
                frontier_lo[proposal_idx],
                frontier_hi[proposal_idx],
                node_batch_size=cpu_verify_batch_size,
            )
            cpu_evals+=len(proposal_idx)
            cpu_lo_all[proposal_idx]=c_lo
            cpu_hi_all[proposal_idx]=c_hi
            _ss,cs,conf=confirm_candidate_signs(
                s_lo[proposal_idx],s_hi[proposal_idx],c_lo,c_hi
            )
            confirmed[proposal_idx]=conf
            cpu_confirm+=int(np.count_nonzero(conf))
            cpu_reject+=int(np.count_nonzero((s_sign[proposal_idx]!=0)&(conf==0)))

        term_idx=np.where(confirmed!=0)[0]
        for j in term_idx:
            terminals.append(
                X6Terminal(
                    root_id=int(frontier_root[j]),
                    depth=int(depth),
                    sign=int(confirmed[j]),
                    lo=tuple(float(v) for v in frontier_lo[j]),
                    hi=tuple(float(v) for v in frontier_hi[j]),
                    screen_lower=float(s_lo[j]),
                    screen_upper=float(s_hi[j]),
                    cpu_lower=float(cpu_lo_all[j]),
                    cpu_upper=float(cpu_hi_all[j]),
                )
            )

        active=np.where(confirmed==0)[0]
        if depth>=max_micro_depth:
            final_lo=frontier_lo[active]
            final_hi=frontier_hi[active]
            final_root=frontier_root[active]
            break
        if len(active)==0:
            final_lo=np.empty((0,3),dtype=np.float64)
            final_hi=np.empty((0,3),dtype=np.float64)
            final_root=np.empty((0,),dtype=np.int64)
            break

        alo=frontier_lo[active]
        ahi=frontier_hi[active]
        aroot=frontier_root[active]
        clo,chi,pidx=_split_octants_batch(alo,ahi)
        frontier_lo=clo
        frontier_hi=chi
        frontier_root=aroot[pidx]
    else:
        raise RuntimeError("UNREACHABLE")

    pos=Counter(t.root_id for t in terminals if t.sign>0)
    neg=Counter(t.root_id for t in terminals if t.sign<0)
    unk=Counter(int(r) for r in final_root)
    states=[]
    pos_counts=[]
    neg_counts=[]
    unk_counts=[]
    for rid in range(nroots):
        p=int(pos[rid]); n=int(neg[rid]); u=int(unk[rid])
        states.append(_root_state(p,n,u))
        pos_counts.append(p);neg_counts.append(n);unk_counts.append(u)

    return X6Result(
        root_states=tuple(states),
        root_positive_terminal_counts=tuple(pos_counts),
        root_negative_terminal_counts=tuple(neg_counts),
        root_unresolved_leaf_counts=tuple(unk_counts),
        terminal_count=len(terminals),
        final_unresolved_leaf_count=len(final_root),
        screen_decisive_proposal_count=int(screen_prop),
        cpu_confirmed_count=int(cpu_confirm),
        cpu_rejected_count=int(cpu_reject),
        accelerator_box_eval_count=int(accel_evals),
        cpu_verify_box_eval_count=int(cpu_evals),
        terminals=tuple(terminals),
    )
