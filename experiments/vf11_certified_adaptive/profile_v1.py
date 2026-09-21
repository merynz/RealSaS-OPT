from __future__ import annotations

"""Deterministic research profiler for VF-11 cell certifiability."""

from collections import Counter
import time
from typing import Callable, Iterable

import torch

from range_engine_v1 import certify_cell


def deterministic_cell_indices(depth: int, max_cells: int | None = None, seed: int = 26092111):
    if depth < 0:
        raise ValueError("DEPTH_NEGATIVE")
    n = 1 << int(depth)
    total = n*n*n
    take = total if max_cells is None else min(total, int(max_cells))
    if take <= 0:
        raise ValueError("MAX_CELLS_NONPOSITIVE")
    if take == total:
        for code in range(total):
            yield code % n, (code//n) % n, code//(n*n)
        return
    step = 2654435761 % total
    if step % 2 == 0:
        step = (step + 1) % total
    start = int(seed) % total
    for j in range(take):
        code = (start + j*step) % total
        yield code % n, (code//n) % n, code//(n*n)


def cell_bounds(index_xyz, depth: int, domain_lo: float, domain_hi: float, *, device, dtype):
    n = 1 << int(depth)
    width = (float(domain_hi)-float(domain_lo))/float(n)
    idx = torch.tensor(index_xyz,dtype=dtype,device=device)
    lo = float(domain_lo) + idx*width
    return lo, lo+width


def profile_depth(field, planes: torch.Tensor, *, depth: int, max_cells: int | None,
                  domain_lo: float, domain_hi: float, seed: int = 26092111,
                  log_every: int = 0, logger: Callable[[str],None] | None = None) -> dict:
    if not (-1.0 <= domain_lo < domain_hi <= 1.0):
        raise ValueError("INVALID_PROFILE_DOMAIN")
    indices = list(deterministic_cell_indices(depth,max_cells=max_cells,seed=seed))
    counts, reasons = Counter(), Counter()
    start = time.perf_counter()
    for i,idx in enumerate(indices,1):
        lo,hi = cell_bounds(idx,depth,domain_lo,domain_hi,device=planes.device,dtype=planes.dtype)
        cert = certify_cell(field,planes,lo,hi)
        counts[cert.state] += 1
        reasons[cert.reason] += 1
        if log_every and logger and i % int(log_every) == 0:
            logger(f"depth={depth} processed={i}/{len(indices)} counts={dict(counts)}")
    elapsed = time.perf_counter()-start
    visited = len(indices)
    surface_candidate = visited-counts["PROVEN_EMPTY"]
    total = (1 << depth)**3
    return {
        "depth":int(depth),
        "grid_axis_cells":int(1 << depth),
        "full_grid_cell_count":int(total),
        "sampled_cell_count":int(visited),
        "sample_is_exhaustive":bool(visited==total),
        "domain_lo":float(domain_lo),
        "domain_hi":float(domain_hi),
        "states":{
            "PROVEN_EMPTY":int(counts["PROVEN_EMPTY"]),
            "PROVEN_REGULAR":int(counts["PROVEN_REGULAR"]),
            "UNKNOWN":int(counts["UNKNOWN"]),
        },
        "fractions":{
            "PROVEN_EMPTY":float(counts["PROVEN_EMPTY"]/max(visited,1)),
            "PROVEN_REGULAR":float(counts["PROVEN_REGULAR"]/max(visited,1)),
            "UNKNOWN":float(counts["UNKNOWN"]/max(visited,1)),
            "REGULAR_GIVEN_NOT_PROVEN_EMPTY":float(counts["PROVEN_REGULAR"]/max(surface_candidate,1)),
        },
        "reason_histogram":dict(sorted(reasons.items())),
        "elapsed_seconds":float(elapsed),
        "cells_per_second":float(visited/max(elapsed,1e-12)),
    }


def profile_depths(field, planes: torch.Tensor, *, depths: Iterable[int],
                   max_cells_per_depth: int | None, domain_lo: float, domain_hi: float,
                   seed: int = 26092111, log_every: int = 0,
                   logger: Callable[[str],None] | None = print) -> dict:
    rows=[]
    for depth in depths:
        if logger:
            logger(f"VF11_CERT_PROFILE_START depth={int(depth)}")
        row=profile_depth(
            field,planes,depth=int(depth),max_cells=max_cells_per_depth,
            domain_lo=domain_lo,domain_hi=domain_hi,seed=seed+int(depth),
            log_every=log_every,logger=logger,
        )
        rows.append(row)
        if logger:
            logger(f"VF11_CERT_PROFILE_DONE depth={row['depth']} states={row['states']} elapsed={row['elapsed_seconds']:.3f}s")
    return {
        "schema":"RealSaS.VF11CertifiabilityProfile.v1",
        "status":"RESEARCH_MEASUREMENT_ONLY__NO_PRODUCT_AUTHORITY",
        "numerically_rigorous":False,
        "depths":rows,
    }
