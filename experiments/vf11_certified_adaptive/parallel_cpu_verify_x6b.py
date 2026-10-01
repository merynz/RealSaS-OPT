from __future__ import annotations

"""X6B parallel CPU confirmation for accelerator-screened C0 proposals.

The proof semantics are identical to X6. Only the CPU confirmation execution
is parallelized across independent proposal boxes.

By default workers use the spawn start method so an already-initialized
CUDA context in the parent process is never inherited through fork.
"""

from collections import Counter
from contextlib import nullcontext
import multiprocessing as mp
import time

import numpy as np
import torch

from accelerator_screen_x6 import (
    X6Result,
    X6Terminal,
    strict_signs,
    confirm_candidate_signs,
    _root_state,
)
from batched_c0_x5 import (
    _bound_batch_chunked_prepared,
    _split_octants_batch,
)
from range_engine_c0_v3 import PreparedField


_WORKER_P: PreparedField | None = None
_WORKER_PLANES: torch.Tensor | None = None
_WORKER_NODE_BATCH_SIZE: int = 32
_WORKER_TORCH_THREADS: int = 1


def _worker_init(
    p: PreparedField,
    planes: torch.Tensor,
    node_batch_size: int,
    torch_threads: int,
):
    global _WORKER_P, _WORKER_PLANES, _WORKER_NODE_BATCH_SIZE, _WORKER_TORCH_THREADS
    _WORKER_P = p
    _WORKER_PLANES = planes
    _WORKER_NODE_BATCH_SIZE = int(node_batch_size)
    _WORKER_TORCH_THREADS = int(torch_threads)
    torch.set_num_threads(_WORKER_TORCH_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _worker_bound(task):
    order, los, his = task
    if _WORKER_P is None or _WORKER_PLANES is None:
        raise RuntimeError("X6B_WORKER_NOT_INITIALIZED")
    lo, hi = _bound_batch_chunked_prepared(
        _WORKER_P,
        _WORKER_PLANES,
        np.asarray(los, dtype=np.float64),
        np.asarray(his, dtype=np.float64),
        node_batch_size=_WORKER_NODE_BATCH_SIZE,
    )
    return int(order), lo, hi


class ParallelCPUVerifier:
    def __init__(
        self,
        p: PreparedField,
        planes: torch.Tensor,
        *,
        workers: int,
        node_batch_size: int = 32,
        torch_threads_per_worker: int = 1,
        start_method: str = "spawn",
    ):
        self.p = p
        self.planes = planes
        self.workers = max(1, int(workers))
        self.node_batch_size = int(node_batch_size)
        self.torch_threads_per_worker = int(torch_threads_per_worker)
        self.start_method = str(start_method)
        self._pool = None

    def __enter__(self):
        if self.workers == 1:
            return self
        ctx = mp.get_context(self.start_method)
        self._pool = ctx.Pool(
            self.workers,
            initializer=_worker_init,
            initargs=(
                self.p,
                self.planes,
                self.node_batch_size,
                self.torch_threads_per_worker,
            ),
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def bound(
        self,
        los: np.ndarray,
        his: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        los = np.asarray(los, dtype=np.float64).reshape(-1, 3)
        his = np.asarray(his, dtype=np.float64).reshape(-1, 3)
        if los.shape != his.shape:
            raise ValueError("X6B_BOUND_SHAPE_MISMATCH")
        n = len(los)
        if n == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        if self.workers == 1:
            return _bound_batch_chunked_prepared(
                self.p,
                self.planes,
                los,
                his,
                node_batch_size=self.node_batch_size,
            )
        if self._pool is None:
            raise RuntimeError("X6B_VERIFIER_CONTEXT_NOT_ENTERED")

        index_chunks = [
            idx for idx in np.array_split(np.arange(n), self.workers) if len(idx)
        ]
        tasks = [
            (order, los[idx], his[idx])
            for order, idx in enumerate(index_chunks)
        ]
        rows = self._pool.map(_worker_bound, tasks)
        rows.sort(key=lambda x: x[0])
        lower = np.concatenate([r[1] for r in rows])
        upper = np.concatenate([r[2] for r in rows])
        if len(lower) != n or len(upper) != n:
            raise RuntimeError("X6B_PARALLEL_RESULT_LENGTH_MISMATCH")
        return lower, upper


def screen_verify_single_regime_roots_parallel_cpu(
    cpu_p: PreparedField,
    cpu_planes: torch.Tensor,
    accelerator_p: PreparedField,
    accelerator_planes: torch.Tensor,
    los: np.ndarray,
    his: np.ndarray,
    *,
    max_micro_depth: int = 2,
    accelerator_node_batch_size: int = 1024,
    cpu_verify_batch_size: int = 32,
    cpu_workers: int = 4,
    cpu_torch_threads_per_worker: int = 1,
    cpu_start_method: str = "spawn",
    store_terminals: bool = True,
    cpu_verifier: ParallelCPUVerifier | None = None,
) -> X6Result:
    if max_micro_depth < 0:
        raise ValueError("NEGATIVE_MICRO_DEPTH")
    root_los = np.asarray(los, dtype=np.float64).reshape(-1, 3)
    root_his = np.asarray(his, dtype=np.float64).reshape(-1, 3)
    if root_los.shape != root_his.shape or np.any(root_los >= root_his):
        raise ValueError("INVALID_ROOT_BOXES")
    nroots = len(root_los)
    if nroots == 0:
        return X6Result(
            tuple(), tuple(), tuple(), tuple(), 0, 0, 0, 0, 0, 0, 0,
            0.0, 0.0, 0.0, tuple()
        )

    frontier_lo = root_los
    frontier_hi = root_his
    frontier_root = np.arange(nroots, dtype=np.int64)

    terminals = []
    terminal_count = 0
    pos_counts_arr = np.zeros(nroots, dtype=np.int64)
    neg_counts_arr = np.zeros(nroots, dtype=np.int64)
    screen_prop = 0
    cpu_confirm = 0
    cpu_reject = 0
    accel_evals = 0
    cpu_evals = 0
    accel_seconds = 0.0
    cpu_seconds = 0.0
    overhead_seconds = 0.0

    verifier_context = (
        nullcontext(cpu_verifier)
        if cpu_verifier is not None
        else ParallelCPUVerifier(
            cpu_p,
            cpu_planes,
            workers=cpu_workers,
            node_batch_size=cpu_verify_batch_size,
            torch_threads_per_worker=cpu_torch_threads_per_worker,
            start_method=cpu_start_method,
        )
    )
    with verifier_context as verifier:
        for depth in range(max_micro_depth + 1):
            t0 = time.perf_counter()
            s_lo, s_hi = _bound_batch_chunked_prepared(
                accelerator_p,
                accelerator_planes,
                frontier_lo,
                frontier_hi,
                node_batch_size=accelerator_node_batch_size,
            )
            accel_seconds += time.perf_counter() - t0
            accel_evals += len(frontier_lo)
            s_sign = strict_signs(s_lo, s_hi)
            proposal_idx = np.where(s_sign != 0)[0]
            screen_prop += len(proposal_idx)

            confirmed = np.zeros(len(frontier_lo), dtype=np.int8)
            cpu_lo_all = np.full(len(frontier_lo), np.nan, dtype=np.float64)
            cpu_hi_all = np.full(len(frontier_lo), np.nan, dtype=np.float64)

            if len(proposal_idx):
                t0 = time.perf_counter()
                c_lo, c_hi = verifier.bound(
                    frontier_lo[proposal_idx],
                    frontier_hi[proposal_idx],
                )
                cpu_seconds += time.perf_counter() - t0
                cpu_evals += len(proposal_idx)
                cpu_lo_all[proposal_idx] = c_lo
                cpu_hi_all[proposal_idx] = c_hi
                _ss, _cs, conf = confirm_candidate_signs(
                    s_lo[proposal_idx],
                    s_hi[proposal_idx],
                    c_lo,
                    c_hi,
                )
                confirmed[proposal_idx] = conf
                cpu_confirm += int(np.count_nonzero(conf))
                cpu_reject += int(
                    np.count_nonzero((s_sign[proposal_idx] != 0) & (conf == 0))
                )

            term_idx = np.where(confirmed != 0)[0]
            terminal_count += len(term_idx)
            if len(term_idx):
                term_roots = frontier_root[term_idx]
                term_signs = confirmed[term_idx]
                np.add.at(pos_counts_arr, term_roots[term_signs > 0], 1)
                np.add.at(neg_counts_arr, term_roots[term_signs < 0], 1)

            if store_terminals:
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

            active = np.where(confirmed == 0)[0]
            if depth >= max_micro_depth:
                final_root = frontier_root[active]
                break
            if len(active) == 0:
                final_root = np.empty((0,), dtype=np.int64)
                break

            t0 = time.perf_counter()
            alo = frontier_lo[active]
            ahi = frontier_hi[active]
            aroot = frontier_root[active]
            clo, chi, pidx = _split_octants_batch(alo, ahi)
            overhead_seconds += time.perf_counter() - t0
            frontier_lo = clo
            frontier_hi = chi
            frontier_root = aroot[pidx]
        else:
            raise RuntimeError("UNREACHABLE")

    unk = Counter(int(r) for r in final_root)
    states = []
    pos_counts = []
    neg_counts = []
    unk_counts = []
    for rid in range(nroots):
        p = int(pos_counts_arr[rid])
        n = int(neg_counts_arr[rid])
        u = int(unk[rid])
        states.append(_root_state(p, n, u))
        pos_counts.append(p)
        neg_counts.append(n)
        unk_counts.append(u)

    return X6Result(
        root_states=tuple(states),
        root_positive_terminal_counts=tuple(pos_counts),
        root_negative_terminal_counts=tuple(neg_counts),
        root_unresolved_leaf_counts=tuple(unk_counts),
        terminal_count=int(terminal_count),
        final_unresolved_leaf_count=len(final_root),
        screen_decisive_proposal_count=int(screen_prop),
        cpu_confirmed_count=int(cpu_confirm),
        cpu_rejected_count=int(cpu_reject),
        accelerator_box_eval_count=int(accel_evals),
        cpu_verify_box_eval_count=int(cpu_evals),
        accelerator_seconds=float(accel_seconds),
        cpu_verify_seconds=float(cpu_seconds),
        refinement_overhead_seconds=float(overhead_seconds),
        terminals=tuple(terminals),
    )
