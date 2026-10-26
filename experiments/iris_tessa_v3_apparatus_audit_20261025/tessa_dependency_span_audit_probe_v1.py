#!/usr/bin/env python3
"""Execute TESSA_DEPENDENCY_SPAN_AUDIT_PREREG_V1 exactly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
from models.tessa.v2 import TESSAGrammarStateV2, TESSAStructuralizerConfigV2
from models.tessa.v2.windowed_incremental_generation_v2 import (
    assert_windowed_incremental_reference_parity_v2,
)

SCHEMA = "RealSaS.TESSADependencySpanAudit.v1"


def strip_mesh(face_count: int):
    n = face_count + 2
    x = np.linspace(-0.45, 0.45, n, dtype=np.float64)
    y = np.where(np.arange(n) % 2 == 0, -0.12, 0.12).astype(np.float64)
    z = 0.04 * np.sin(np.linspace(0.0, 6.0 * np.pi, n, dtype=np.float64))
    vertices = np.stack((x, y, z), axis=-1)
    faces = np.asarray([[i, i + 1, i + 2] for i in range(face_count)], dtype=np.int64)
    return vertices, faces


def disconnected_mesh(face_count: int):
    vertices = []
    faces = []
    for i in range(face_count):
        cx = -0.42 + 0.84 * ((i % 20) / 19.0)
        cy = -0.42 + 0.84 * (((i // 20) % 20) / 19.0)
        cz = -0.30 + 0.60 * ((i % 7) / 6.0)
        base = len(vertices)
        eps = 0.004
        vertices.extend(((cx - eps, cy - eps, cz), (cx + eps, cy - eps, cz), (cx, cy + eps, cz)))
        faces.append((base, base + 1, base + 2))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def mixed_mesh(strip_faces: int = 900, isolated_faces: int = 160):
    v0, f0 = strip_mesh(strip_faces)
    v1, f1 = disconnected_mesh(isolated_faces)
    # Shrink/translate the disconnected bank so all coordinates remain normalized.
    v0 = v0.copy()
    v0[:, 1] *= 0.7
    v1 = v1.copy()
    v1[:, 0] *= 0.45
    v1[:, 1] = 0.24 + 0.45 * v1[:, 1]
    v1[:, 2] *= 0.45
    f1 = f1 + len(v0)
    return np.concatenate((v0, v1), axis=0), np.concatenate((f0, f1), axis=0)


def audit_sequence(name: str, vertices: np.ndarray, faces: np.ndarray, codec_cfg, grammar_cfg):
    seq = encode_adjacent_teacher_mesh_v1(
        vertices,
        faces,
        component_id=name,
        chart_id="global",
        cfg=codec_cfg,
    )
    state = TESSAGrammarStateV2.from_prefix(seq.token_ids, config=grammar_cfg)
    if not state.complete:
        raise AssertionError(f"GRAMMAR_NOT_COMPLETE:{name}")
    if state.face_count != int(faces.shape[0]):
        raise AssertionError(f"FACE_COUNT_DRIFT:{name}:{state.face_count}:{faces.shape[0]}")
    if state.restart_count != seq.restart_count:
        raise AssertionError(f"RESTART_COUNT_DRIFT:{name}:{state.restart_count}:{seq.restart_count}")
    return {
        "name": name,
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "tokens": len(seq.token_ids),
        "restart_count": int(seq.restart_count),
        "exceeds_local_window": len(seq.token_ids) > grammar_cfg.local_attention_window,
        "grammar_complete": bool(state.complete),
        "grammar_face_count": int(state.face_count),
    }


def run() -> dict:
    cfg = TESSAStructuralizerConfigV2()
    cfg.validate()
    codec = TESSAConfigV1(
        coordinate_bins=cfg.coordinate_bins,
        max_faces=cfg.max_faces,
        max_vertices=cfg.max_vertices,
    )

    cases = []
    cases.append(audit_sequence("LONG_CONNECTED_STRIP", *strip_mesh(1500), codec, cfg))
    cases.append(audit_sequence("RESTART_HEAVY", *disconnected_mesh(520), codec, cfg))
    cases.append(audit_sequence("MIXED", *mixed_mesh(), codec, cfg))

    parity = assert_windowed_incremental_reference_parity_v2(device="cpu")
    parity_ok = (
        parity.get("status") == "PASS__WINDOWED_INCREMENTAL_REFERENCE_PARITY"
        and int(parity.get("post_window_prefixes", 0)) > 0
    )
    grammar_ok = all(c["grammar_complete"] and c["grammar_face_count"] == c["faces"] for c in cases)
    long_case_ok = any(c["exceeds_local_window"] for c in cases)

    serialization_local_structural_span = 6
    minimum_post_window_direct_history = cfg.local_attention_window - cfg.query_chunk_size + 1
    margin = minimum_post_window_direct_history / float(serialization_local_structural_span)
    margin_ok = minimum_post_window_direct_history >= 32 * serialization_local_structural_span

    if not (grammar_ok and long_case_ok and parity_ok):
        status = "TESSA_DEPENDENCY_SPAN_AUDIT_INVALID"
    elif margin_ok:
        status = "TESSA_4096_WINDOW_RETAIN_PRETRUTH__NO_INCREASE_EVIDENCE_V1"
    else:
        status = "TESSA_LOCAL_WINDOW_REOPEN_REQUIRED_V1"

    return {
        "schema": SCHEMA,
        "status": status,
        "config": {
            "local_attention_window": cfg.local_attention_window,
            "query_chunk_size": cfg.query_chunk_size,
            "minimum_post_window_direct_history": minimum_post_window_direct_history,
            "serialization_local_structural_span": serialization_local_structural_span,
            "history_margin_x": margin,
        },
        "cases": cases,
        "incremental_reference_parity": parity,
        "gates": {
            "grammar_replay": grammar_ok,
            "sequence_exceeding_window_present": long_case_ok,
            "incremental_reference_post_window_parity": parity_ok,
            "serialization_margin": margin_ok,
        },
        "selection": {
            "retain_local_attention_window_pretruth": cfg.local_attention_window if status.startswith("TESSA_4096") else None,
            "increase_authorized": False,
            "decrease_authorized": False,
            "reason": "NO_PRETRUTH_MECHANISM_EVIDENCE_TO_INCREASE; REAL_LEARNED_LONG_RANGE_DEPENDENCY_REQUIRED_BEFORE_DECREASE",
        },
        "authority": {
            "product_authority_changed": False,
            "truth_builder_training_authorized": False,
            "fit8_training_authorized": False,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = run()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "config": result["config"], "cases": result["cases"], "parity": result["incremental_reference_parity"], "out": str(a.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
