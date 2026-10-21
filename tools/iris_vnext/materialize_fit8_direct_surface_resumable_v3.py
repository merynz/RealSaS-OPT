#!/usr/bin/env python3
from __future__ import annotations

"""Resumable FIT8 direct-surface materializer with corrected shell containment.

The V2 parent orchestration already supplies process isolation, hash-bound resume,
return-code/signal classification, and durable progress/failure receipts.  V3
reuses that orchestration while binding each child to the corrected exact-shell-
containment subject worker.  Model bytes, source bytes, scalar field, coordinates,
T512/T1024 carrier, indexed MT geometry construction, optimizer protocol and court
thresholds are unchanged.
"""

import json
from pathlib import Path
from typing import Any

from models.iris.v5.contained_sparse_tetra_decoder_v5 import CONTAINED_DECODER_ID_V5
from tools.iris_vnext import materialize_fit8_direct_surface_resumable_v2 as v2


CHILD_MODULE_V2 = "tools.iris_vnext.materialize_fit8_direct_surface_subject_v2"
CHILD_MODULE_V3 = "tools.iris_vnext.materialize_fit8_direct_surface_subject_v3"
RUNTIME_SCHEMA_V3 = "RealSaS.Fit8DirectIRISSurfaceRuntimeHardening.v3"


def _child_redirect_run(cmd: Any, *args: Any, **kwargs: Any):
    if isinstance(cmd, (list, tuple)):
        rewritten = list(cmd)
        rewritten = [CHILD_MODULE_V3 if str(x) == CHILD_MODULE_V2 else x for x in rewritten]
        cmd = rewritten
    return _REAL_SUBPROCESS_RUN(cmd, *args, **kwargs)


_REAL_SUBPROCESS_RUN = v2.subprocess.run


def materialize(args):
    previous_decoder_id = v2.DECODER_ID
    previous_run = v2.subprocess.run
    v2.DECODER_ID = CONTAINED_DECODER_ID_V5
    v2.subprocess.run = _child_redirect_run
    try:
        batch = v2.materialize(args)
    finally:
        v2.DECODER_ID = previous_decoder_id
        v2.subprocess.run = previous_run

    output_root = Path(args.output_dir).resolve()
    subjects = list(batch.get("subjects") or ())
    if len(subjects) != len(v2.SUBJECT_IDS):
        raise RuntimeError("DIRECT_SURFACE_V3_BATCH_SUBJECT_COUNT_DRIFT")

    for subject_id in v2.SUBJECT_IDS:
        subject_root = output_root / "subjects" / subject_id
        direct = v2.load_json(subject_root / "DIRECT_IRIS_SURFACE_RECEIPT.json")
        execution = v2.load_json(subject_root / "DIRECT_IRIS_SURFACE_SUBJECT_EXECUTION_V2.json")
        if direct.get("decoder_id") != CONTAINED_DECODER_ID_V5:
            raise RuntimeError("DIRECT_SURFACE_V3_DIRECT_RECEIPT_DECODER_DRIFT:" + subject_id)
        if execution.get("decoder_id") != CONTAINED_DECODER_ID_V5:
            raise RuntimeError("DIRECT_SURFACE_V3_EXECUTION_RECEIPT_DECODER_DRIFT:" + subject_id)
        diagnostics = dict(execution.get("decoder_diagnostics") or {})
        if diagnostics.get("exact_shell_containment_required") is not True:
            raise RuntimeError("DIRECT_SURFACE_V3_CONTAINMENT_PROOF_MISSING:" + subject_id)
        if diagnostics.get("passed") is not True:
            raise RuntimeError("DIRECT_SURFACE_V3_CONTAINMENT_NOT_PASS:" + subject_id)
        if diagnostics.get("legacy_boundary_crossing_tets_is_truncation_gate") is not False:
            raise RuntimeError("DIRECT_SURFACE_V3_LEGACY_PROXY_REACTIVATED:" + subject_id)

    batch["sparse_surface_policy"] = {
        **dict(batch.get("sparse_surface_policy") or {}),
        "decoder_id": CONTAINED_DECODER_ID_V5,
        "policy": "EXACT_SELECTED_T512_TO_SPARSE_T1024_EXACT_SHELL_CONTAINMENT_V5_1",
        "containment_gate": "EXACT_REFINED_CARRIER_BOUNDARY_FACE_ZERO_CONTACT_V1",
        "exact_shell_containment_required": True,
        "legacy_boundary_crossing_tets_role": "TELEMETRY_ONLY",
    }
    batch["runtime_hardening"] = {
        **dict(batch.get("runtime_hardening") or {}),
        "schema": RUNTIME_SCHEMA_V3,
        "orchestration_only": False,
        "gate_correction_only": True,
        "subject_process_isolation": True,
        "completed_subject_resume": True,
        "returncode_and_signal_classification": True,
        "field_semantics_changed": False,
        "geometry_extraction_semantics_changed": False,
        "shell_gate_semantics_corrected": True,
        "exact_shell_containment_required": True,
        "legacy_boundary_proxy_is_truncation_gate": False,
        "surface_policy_changed": True,
        "model_parameter_bytes_changed": False,
        "optimizer_or_schedule_changed": False,
        "court_thresholds_changed": False,
    }
    batch["binding_sha256"] = v2.value_sha256(batch)
    v2._atomic_json(batch, output_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json")
    return batch


def parser():
    return v2.parser()


def main() -> int:
    result = materialize(parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
