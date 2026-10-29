#!/usr/bin/env python3
"""Materialize one exact source mesh into the canonical TESSA teacher domain.

Input mesh bytes are an admitted rest-mesh extraction from the sealed source asset, not an IRIS
prediction.  The output is target-only truth: normalized vertices/faces plus a token-sequence
representability receipt.  It must never be mounted into the student conditioning cache builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.training_data_v1 import build_teacher_asset_sequence_v1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2


OUTPUT_SCHEMA = "RealSaS.Truth139TESSATeacherMesh.v1"


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalization(value: dict) -> tuple[np.ndarray, float]:
    schema = value.get("schema_version") or value.get("schema")
    if schema != "RealSaS.NormalizationDomainIR.v1":
        raise ValueError("TRUTH139_TESSA_TEACHER_NORMALIZATION_SCHEMA")
    center = np.asarray(value.get("center_xyz"), dtype=np.float64)
    half = float(value.get("half_extent", 0.0))
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(half) or half <= 0.0:
        raise ValueError("TRUTH139_TESSA_TEACHER_NORMALIZATION_INVALID")
    return center, half


def _codec_config_v1(v2: TESSAStructuralizerConfigV2) -> TESSAConfigV1:
    """V2 deliberately retains the V1 grammar; carry V2 scale/window controls into the codec."""
    v2.validate()
    return TESSAConfigV1(
        coordinate_bins=v2.coordinate_bins,
        max_faces=v2.max_faces,
        max_vertices=v2.max_vertices,
        local_attention_window=v2.local_attention_window,
        query_chunk_size=v2.query_chunk_size,
        surface_latent_count=v2.surface_memory_tokens,
    )


def build_teacher_mesh(
    *,
    subject_id: str,
    source_sha256: str,
    rest_vertices: np.ndarray,
    faces: np.ndarray,
    normalization: dict,
    tessa_config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
) -> tuple[dict, dict]:
    if not subject_id or len(str(source_sha256)) != 64:
        raise ValueError("TRUTH139_TESSA_TEACHER_IDENTITY_INVALID")
    vertices = np.asarray(rest_vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("TRUTH139_TESSA_TEACHER_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.min(tri) < 0 or np.max(tri) >= len(vertices):
        raise ValueError("TRUTH139_TESSA_TEACHER_FACES_INVALID")
    center, half = _normalization(normalization)
    iris_vertices = (vertices - center[None, :]) / half
    if np.any(iris_vertices < -1.000001) or np.any(iris_vertices > 1.000001):
        raise ValueError("TRUTH139_TESSA_TEACHER_OUTSIDE_IRIS_NORMALIZATION")
    tessa_vertices = (iris_vertices * 0.5).astype(np.float32)
    faces_out = tri.astype(np.int64)

    sequence = build_teacher_asset_sequence_v1(
        tessa_vertices,
        faces_out,
        cfg=_codec_config_v1(tessa_config),
    )
    arrays = {
        "vertices_normalized": tessa_vertices,
        "faces": faces_out,
    }
    receipt = {
        "schema": OUTPUT_SCHEMA,
        "subject_id": str(subject_id),
        "source_sha256": str(source_sha256),
        "normalization_schema": "RealSaS.NormalizationDomainIR.v1",
        "iris_to_tessa_coordinate_scale": 0.5,
        "vertex_count": int(len(tessa_vertices)),
        "face_count": int(len(faces_out)),
        "component_count": int(sequence.component_count),
        "restart_count": int(sequence.restart_count),
        "token_count": int(len(sequence.token_ids)),
        "coordinate_bins": int(tessa_config.coordinate_bins),
        "local_attention_window": int(tessa_config.local_attention_window),
        "teacher_topology_target_only": True,
        "student_conditioning_allowed": False,
        "representability_court": "PASS_FULL_SOURCE_TOPOLOGY_SEQUENCE",
    }
    return arrays, receipt


def _write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial.npz")
    np.savez(staged, **arrays)
    staged.replace(path)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-mesh-npz", required=True)
    parser.add_argument("--normalization-json", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--receipt-json", required=True)
    args = parser.parse_args()
    source = Path(args.source_mesh_npz).resolve()
    with np.load(source, allow_pickle=False) as z:
        vertices = np.asarray(z["rest_vertices"])
        faces = np.asarray(z["faces"])
    normalization = json.loads(Path(args.normalization_json).resolve().read_text())
    arrays, receipt = build_teacher_mesh(
        subject_id=args.subject_id,
        source_sha256=args.source_sha256,
        rest_vertices=vertices,
        faces=faces,
        normalization=normalization,
    )
    output = Path(args.output_npz).resolve()
    _write_npz(output, arrays)
    receipt = {**receipt, "teacher_mesh_sha256": _hash(output)}
    _write_json(Path(args.receipt_json).resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
