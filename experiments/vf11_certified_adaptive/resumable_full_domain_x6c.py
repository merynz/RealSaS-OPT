from __future__ import annotations

"""X6C resumable chunk artifact helpers.

Each X6B call operates on independent base-regime roots. X6C persists one
self-describing NPZ per contiguous root range and merges them only after exact
coverage/non-overlap checks.

This module changes no proof mathematics.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import tempfile
from typing import Iterable

import numpy as np


STATE_TO_CODE = {
    "UNKNOWN": 0,
    "PROVEN_EMPTY_POSITIVE": 1,
    "PROVEN_EMPTY_NEGATIVE": 2,
    "PROVEN_ZERO_EXISTS": 3,
}
CODE_TO_STATE = {v: k for k, v in STATE_TO_CODE.items()}


@dataclass(frozen=True)
class ChunkPayload:
    start: int
    stop: int
    state_codes: np.ndarray
    positive_terminal_counts: np.ndarray
    negative_terminal_counts: np.ndarray
    unresolved_leaf_counts: np.ndarray
    metadata: dict


def chunk_ranges(total_roots: int, chunk_roots: int) -> list[tuple[int, int]]:
    total = int(total_roots)
    step = int(chunk_roots)
    if total <= 0 or step <= 0:
        raise ValueError("INVALID_CHUNK_SHAPE")
    return [(s, min(s + step, total)) for s in range(0, total, step)]


def chunk_filename(start: int, stop: int) -> str:
    return f"chunk_{int(start):06d}_{int(stop):06d}.npz"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _as_1d(name: str, values, dtype, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"{name}_LENGTH_MISMATCH:{len(arr)}!={n}")
    return arr


def write_chunk_atomic(
    directory: str | Path,
    *,
    start: int,
    stop: int,
    root_states: Iterable[str],
    positive_terminal_counts,
    negative_terminal_counts,
    unresolved_leaf_counts,
    metadata: dict,
) -> tuple[Path, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start = int(start)
    stop = int(stop)
    n = stop - start
    if start < 0 or stop <= start:
        raise ValueError("INVALID_CHUNK_RANGE")

    states = list(root_states)
    if len(states) != n:
        raise ValueError("ROOT_STATE_LENGTH_MISMATCH")
    try:
        state_codes = np.asarray([STATE_TO_CODE[s] for s in states], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"UNKNOWN_ROOT_STATE:{exc}") from exc

    pos = _as_1d("positive_terminal_counts", positive_terminal_counts, np.int32, n)
    neg = _as_1d("negative_terminal_counts", negative_terminal_counts, np.int32, n)
    unk = _as_1d("unresolved_leaf_counts", unresolved_leaf_counts, np.int32, n)

    meta = dict(metadata)
    meta.update({
        "schema": "RealSaS.VF11X6CChunk.v1",
        "start": start,
        "stop": stop,
        "root_count": n,
    })
    meta_json = json.dumps(meta, sort_keys=True, separators=(",", ":"))

    final_path = directory / chunk_filename(start, stop)
    fd, tmp_name = tempfile.mkstemp(
        prefix=final_path.name + ".",
        suffix=".tmp",
        dir=str(directory),
    )
    os.close(fd)
    try:
        with open(tmp_name, "wb") as f:
            np.savez_compressed(
                f,
                state_codes=state_codes,
                positive_terminal_counts=pos,
                negative_terminal_counts=neg,
                unresolved_leaf_counts=unk,
                metadata_json=np.asarray(meta_json),
            )
        os.replace(tmp_name, final_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    return final_path, sha256_file(final_path)


def read_chunk(path: str | Path) -> ChunkPayload:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        start = int(meta["start"])
        stop = int(meta["stop"])
        n = stop - start
        state = _as_1d("state_codes", z["state_codes"], np.uint8, n)
        pos = _as_1d("positive_terminal_counts", z["positive_terminal_counts"], np.int32, n)
        neg = _as_1d("negative_terminal_counts", z["negative_terminal_counts"], np.int32, n)
        unk = _as_1d("unresolved_leaf_counts", z["unresolved_leaf_counts"], np.int32, n)

    if int(meta.get("root_count", -1)) != n:
        raise ValueError("CHUNK_METADATA_ROOT_COUNT_MISMATCH")
    if not np.all(np.isin(state, np.asarray(list(CODE_TO_STATE), dtype=np.uint8))):
        raise ValueError("INVALID_STATE_CODE")
    return ChunkPayload(start, stop, state, pos, neg, unk, meta)


def validate_complete_chunk_set(
    chunk_paths: Iterable[str | Path],
    *,
    total_roots: int,
    expected_ranges: Iterable[tuple[int, int]] | None = None,
) -> list[ChunkPayload]:
    payloads = [read_chunk(p) for p in chunk_paths]
    payloads.sort(key=lambda x: x.start)
    if not payloads:
        raise ValueError("NO_CHUNKS")

    ranges = [(p.start, p.stop) for p in payloads]
    if expected_ranges is not None:
        exp = [(int(a), int(b)) for a, b in expected_ranges]
        if ranges != exp:
            raise ValueError(f"CHUNK_RANGE_SET_MISMATCH:{ranges}!={exp}")

    cursor = 0
    for p in payloads:
        if p.start != cursor:
            raise ValueError(f"CHUNK_GAP_OR_OVERLAP_AT:{cursor}:{p.start}")
        cursor = p.stop
    if cursor != int(total_roots):
        raise ValueError(f"INCOMPLETE_CHUNK_COVERAGE:{cursor}!={total_roots}")
    return payloads


def merge_complete_chunks(
    chunk_paths: Iterable[str | Path],
    *,
    total_roots: int,
    expected_ranges: Iterable[tuple[int, int]] | None = None,
) -> dict:
    payloads = validate_complete_chunk_set(
        chunk_paths,
        total_roots=total_roots,
        expected_ranges=expected_ranges,
    )
    cpu_rejected_total = sum(int(p.metadata.get("cpu_rejected_count", 0)) for p in payloads)
    if cpu_rejected_total != 0:
        raise ValueError(f"CPU_REJECTION_PRESENT:{cpu_rejected_total}")

    state_codes = np.concatenate([p.state_codes for p in payloads])
    pos = np.concatenate([p.positive_terminal_counts for p in payloads])
    neg = np.concatenate([p.negative_terminal_counts for p in payloads])
    unk = np.concatenate([p.unresolved_leaf_counts for p in payloads])

    state_unique, state_counts_raw = np.unique(state_codes, return_counts=True)
    state_counts = {
        CODE_TO_STATE[int(code)]: int(count)
        for code, count in zip(state_unique, state_counts_raw)
    }

    aggregate_keys = [
        "terminal_count",
        "final_unresolved_leaf_count",
        "screen_decisive_proposal_count",
        "cpu_confirmed_count",
        "cpu_rejected_count",
        "accelerator_box_eval_count",
        "cpu_verify_box_eval_count",
    ]
    aggregate = {
        key: int(sum(int(p.metadata.get(key, 0)) for p in payloads))
        for key in aggregate_keys
    }
    timing_keys = [
        "wall_seconds",
        "accelerator_seconds",
        "parallel_cpu_verify_seconds",
        "refinement_overhead_seconds",
    ]
    aggregate.update({
        key: float(sum(float(p.metadata.get(key, 0.0)) for p in payloads))
        for key in timing_keys
    })

    return {
        "state_codes": state_codes,
        "positive_terminal_counts": pos,
        "negative_terminal_counts": neg,
        "unresolved_leaf_counts": unk,
        "state_counts": state_counts,
        "aggregate": aggregate,
        "chunk_count": len(payloads),
    }
