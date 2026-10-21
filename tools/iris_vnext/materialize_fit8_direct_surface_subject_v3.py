#!/usr/bin/env python3
from __future__ import annotations

"""Exact-containment isolated direct-surface worker for one FIT8 subject.

This is a gate-correction wrapper around the already-resumable V2 subject lane.
It preserves the exact V8 model/checkpoint, scene encoding, scalar-field query,
SparseRegularTetraPolicyV5 T512/T1024 carrier, source bytes and write format.
Only the shell-truncation gate changes: the canonical exact containment decoder
must pass before extraction; the historical boundary-crossing proxy is telemetry.
"""

import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any

from models.iris.v5.contained_sparse_tetra_decoder_v5 import CONTAINED_DECODER_ID_V5
from models.iris.vnext.sparse_surface_v4 import decode_vnext_sparse_surface_v4
from tools.iris_vnext import materialize_fit8_direct_surface_subject_v2 as v2


PROGRESS_SCHEMA = "RealSaS.Fit8DirectIRISSurfaceSubjectProgress.v2"


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".staged",
        delete=False,
    ) as handle:
        staged = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _progress_exact_containment(
    subject_root: Path,
    *,
    subject_id: str,
    execution_source_sha: str,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    value: dict[str, Any] = {
        "schema": PROGRESS_SCHEMA,
        "subject_id": subject_id,
        "execution_source_sha": execution_source_sha,
        "stage": stage,
        "pid": int(os.getpid()),
        "unix_time": float(time.time()),
        "decoder_id": CONTAINED_DECODER_ID_V5,
        "base_cells": 512,
        "fine_cells": 1024,
        "exact_shell_containment_required": True,
        "legacy_boundary_crossing_tets_role": "TELEMETRY_ONLY",
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "teacher_truth_used": False,
        "tessa_invoked": False,
        "scientific_pass_claimed": False,
    }
    if extra:
        value.update(extra)
    _atomic_json(value, subject_root / "DIRECT_IRIS_SURFACE_PROGRESS_V2.json")


def _install_exact_containment_bridge() -> None:
    # materialize_subject() resolves these names from its defining module globals.
    # Rebinding only these two already-reviewed hooks keeps the V2 source/hydration/
    # scene/checkpoint logic unchanged while replacing the falsified shell gate.
    v2.decode_vnext_sparse_surface_v3 = decode_vnext_sparse_surface_v4
    v2._progress = _progress_exact_containment


def main() -> int:
    args = v2.parser().parse_args()
    _install_exact_containment_bridge()
    try:
        result = v2.materialize_subject(args)
    except Exception as exc:
        path = v2._write_failure(args, exc)
        print(f"DIRECT_SURFACE_SUBJECT_FAILURE receipt={path}", flush=True)
        traceback.print_exc()
        return 3

    if result.get("decoder_id") != CONTAINED_DECODER_ID_V5:
        raise RuntimeError("DIRECT_SURFACE_V3_SUBJECT_DECODER_ID_DRIFT")
    diagnostics = dict(result.get("decoder_diagnostics") or {})
    if diagnostics.get("exact_shell_containment_required") is not True:
        raise RuntimeError("DIRECT_SURFACE_V3_SUBJECT_CONTAINMENT_PROOF_MISSING")
    if diagnostics.get("passed") is not True:
        raise RuntimeError("DIRECT_SURFACE_V3_SUBJECT_CONTAINMENT_NOT_PASS")
    if diagnostics.get("legacy_boundary_crossing_tets_is_truncation_gate") is not False:
        raise RuntimeError("DIRECT_SURFACE_V3_SUBJECT_LEGACY_PROXY_REACTIVATED")

    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
