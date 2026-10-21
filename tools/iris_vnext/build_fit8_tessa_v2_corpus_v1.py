#!/usr/bin/env python3
from __future__ import annotations

"""Build the first serious Fit8 TESSA V2 corpus from model-produced evidence.

Conditioning authority:
  exact sealed V8 scene planes
  direct high-resolution IRIS surface witness
  V8 local H + explicitly trained mechanical outputs on that witness

Teacher topology is used only as the autoregressive target. It is never copied
into scene/surface conditioning. Contact persistence remains invalid/abstained.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v2.training_corpus_v2 import (
    TESSATrainingWindowPolicyV2,
    build_global_history_training_windows_v2,
    reconstruct_global_sequence_v2,
)
from tools.iris_vnext.materialize_fit8_direct_surface_v1 import (
    git_head,
    load_exact_v8_model,
    load_json,
    load_subject_source,
    sha256_file,
    tensor_sha256,
)


SCHEMA = "RealSaS.Fit8TESSAV2TrainingCorpus.v1"
SUBJECT_SCHEMA = "RealSaS.TESSAV2TrainingSubject.v1"
WITNESS_COUNT = 8192
QUERY_CHUNK = 8192


def value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sample_surface_witness(
    vertices: np.ndarray,
    faces: np.ndarray,
    implicit_normals: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = np.asarray(implicit_normals, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or n.shape != v.shape or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("TESSA_V2_CORPUS_SURFACE_ARRAY_SHAPE_INVALID")
    if not np.isfinite(v).all() or not np.isfinite(n).all() or len(v) == 0 or len(f) == 0:
        raise ValueError("TESSA_V2_CORPUS_SURFACE_ARRAY_INVALID")
    tri = v[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = area2 > 2.0e-12
    if not bool(valid.any()):
        raise ValueError("TESSA_V2_CORPUS_SURFACE_NO_NONDEGENERATE_FACE")
    valid_faces = np.flatnonzero(valid)
    weights = area2[valid]
    weights = weights / weights.sum()
    rng = np.random.default_rng(int(seed))
    chosen = valid_faces[rng.choice(len(valid_faces), size=int(count), replace=True, p=weights)]
    selected = tri[chosen]
    uv = rng.random((int(count), 2))
    reflect = uv.sum(axis=1) > 1.0
    uv[reflect] = 1.0 - uv[reflect]
    b1 = uv[:, 0:1]
    b2 = uv[:, 1:2]
    b0 = 1.0 - b1 - b2
    points = b0 * selected[:, 0] + b1 * selected[:, 1] + b2 * selected[:, 2]
    vertex_n = n[f[chosen]]
    normals = b0 * vertex_n[:, 0] + b1 * vertex_n[:, 1] + b2 * vertex_n[:, 2]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norm <= 1.0e-12):
        raise ValueError("TESSA_V2_CORPUS_WITNESS_NORMAL_UNRESOLVED")
    normals = normals / norm
    if np.min(points) < -1.000001 or np.max(points) > 1.000001:
        raise ValueError("TESSA_V2_CORPUS_IRIS_WITNESS_DOMAIN_DRIFT")
    return points.astype(np.float32), normals.astype(np.float32)


def _query_v8_evidence(model, scene_planes: torch.Tensor, points_iris: np.ndarray) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    h_rows: list[torch.Tensor] = []
    mechanics: dict[str, list[torch.Tensor]] = {
        "stretch_major": [],
        "stretch_minor": [],
        "tangent_direction_2theta": [],
        "bend_demand": [],
        "rigidity_probability": [],
    }
    for start in range(0, len(points_iris), QUERY_CHUNK):
        pts = torch.from_numpy(points_iris[start : start + QUERY_CHUNK]).to(
            device=scene_planes.device, dtype=torch.float32
        )[None]
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            out = model.query_evidence(scene_planes, pts)
        h_rows.append(out["H"][0].detach().float().cpu())
        demand = out["mechanical_demand"]
        for key in mechanics:
            mechanics[key].append(demand[key][0].detach().float().cpu())
    h = torch.cat(h_rows, dim=0)
    merged = {key: torch.cat(rows, dim=0) for key, rows in mechanics.items()}
    if h.shape[0] != len(points_iris) or any(value.shape[0] != len(points_iris) for value in merged.values()):
        raise AssertionError("TESSA_V2_CORPUS_EVIDENCE_CARDINALITY_DRIFT")
    return h, merged


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_sha = str(args.expected_source_sha).lower()
    if git_head(repo_root) != expected_sha:
        raise RuntimeError("TESSA_V2_CORPUS_CHECKOUT_SHA_DRIFT")
    if not torch.cuda.is_available():
        raise RuntimeError("TESSA_V2_CORPUS_CUDA_REQUIRED")
    device = torch.device(args.device)

    prepared_root = Path(args.prepared_root).resolve()
    manifest_path = Path(args.prepared_manifest).resolve()
    prepared_manifest = load_json(manifest_path)
    if prepared_manifest.get("schema") != "RealSaS.IRISVNextFit8Manifest.v4":
        raise ValueError("TESSA_V2_CORPUS_PREPARED_SCHEMA_INVALID")
    rows = list(prepared_manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in rows) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_CORPUS_SUBJECT_ORDER_DRIFT")

    direct_root = Path(args.direct_surface_root).resolve()
    batch_path = direct_root / "FIT8_DIRECT_IRIS_SURFACE_BATCH_RECEIPT.json"
    direct_batch = load_json(batch_path)
    if (
        direct_batch.get("schema") != "RealSaS.Fit8DirectIRISSurfaceMaterialization.v1"
        or direct_batch.get("tessa_invoked") is not False
        or direct_batch.get("optimizer_steps_executed") != 0
    ):
        raise ValueError("TESSA_V2_CORPUS_DIRECT_SURFACE_AUTHORITY_INVALID")
    batch_rows = {str(row["subject_id"]): row for row in direct_batch.get("subjects") or ()}
    if tuple(batch_rows) != SUBJECT_IDS:
        raise ValueError("TESSA_V2_CORPUS_DIRECT_SUBJECT_ORDER_DRIFT")

    model = load_exact_v8_model(Path(args.v8_checkpoint), device)
    tessa_cfg = TESSAStructuralizerConfigV2(); tessa_cfg.validate()
    window_policy = TESSATrainingWindowPolicyV2(); window_policy.validate(tessa_cfg)
    iris_cfg = IRISVNextConfigV1(); iris_cfg.validate()

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("TESSA_V2_CORPUS_OUTPUT_MUST_BE_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)
    subject_receipts: list[dict[str, Any]] = []

    for subject_index, row in enumerate(rows):
        subject_id = SUBJECT_IDS[subject_index]
        shard_path = prepared_root / str(row["shard_file"])
        source = load_subject_source(shard_path, str(row["shard_sha256"]), device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            scene_planes = model.encode_scene(
                source["image_tokens"], source["token_xy"], source["camera_features"]
            ).float().detach()
        stage_path = direct_root / "subjects" / subject_id / "DIRECT_IRIS_SURFACE.stage12.npz"
        receipt_path = direct_root / "subjects" / subject_id / "DIRECT_IRIS_SURFACE_RECEIPT.json"
        direct_receipt = load_json(receipt_path)
        if sha256_file(stage_path) != batch_rows[subject_id]["stage12_npz_sha256"]:
            raise ValueError("TESSA_V2_CORPUS_DIRECT_SURFACE_SHA_DRIFT:" + subject_id)
        if direct_receipt.get("candidate_binding_hash") != batch_rows[subject_id]["candidate_binding_hash"]:
            raise ValueError("TESSA_V2_CORPUS_DIRECT_BINDING_DRIFT:" + subject_id)
        with np.load(stage_path, allow_pickle=False) as data:
            points_iris, normals = _sample_surface_witness(
                data["vertices_normalized"], data["faces"], data["implicit_normals"],
                count=WITNESS_COUNT, seed=26091991 + subject_index,
            )
        h, mechanics = _query_v8_evidence(model, scene_planes, points_iris)

        raw = torch.load(shard_path, map_location="cpu", weights_only=True)
        sequence = reconstruct_global_sequence_v2(
            raw["topology_input_ids"], raw["topology_labels"], raw["topology_window_offsets"],
            raw["topology_window_count"], config=tessa_cfg,
        )
        windows, labels, offsets = build_global_history_training_windows_v2(
            sequence, policy=window_policy, config=tessa_cfg,
        )
        payload = {
            "schema": SUBJECT_SCHEMA,
            "subject_id": subject_id,
            "scene_planes": scene_planes[0].cpu(),
            # TESSA codec domain is [-0.5,0.5]; H/mechanics were queried at the
            # corresponding IRIS point in [-1,1].
            "witness_xyz": torch.from_numpy(points_iris * 0.5),
            "witness_normal": torch.from_numpy(normals),
            "witness_h": h,
            "mechanical_stretch_major": mechanics["stretch_major"],
            "mechanical_stretch_minor": mechanics["stretch_minor"],
            "mechanical_tangent_direction_2theta": mechanics["tangent_direction_2theta"],
            "mechanical_bend_demand": mechanics["bend_demand"],
            "mechanical_rigidity_probability": mechanics["rigidity_probability"],
            "contact_persistence_valid": False,
            "topology_input_ids": windows,
            "topology_labels": labels,
            "topology_window_offsets": offsets,
            "global_topology_sequence": sequence,
            "teacher_topology_in_conditioning": False,
            "teacher_rig_skin_in_conditioning": False,
            "direct_surface_candidate_binding_hash": direct_receipt["candidate_binding_hash"],
            "prepared_shard_sha256": str(row["shard_sha256"]),
            "v8_scene_planes_sha256": tensor_sha256(scene_planes),
        }
        file_name = subject_id.lower().replace("-", "").replace(" ", "") + ".tessa_v2.pt"
        out_path = output_root / file_name
        torch.save(payload, out_path)
        topology_meta = (row.get("metadata") or {}).get("topology_teacher") or {}
        subject_receipts.append({
            "subject_id": subject_id,
            "file": file_name,
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
            "direct_surface_candidate_binding_hash": direct_receipt["candidate_binding_hash"],
            "global_token_count": int(sequence.numel()),
            "training_window_count": int(windows.shape[0]),
            "training_context_tokens": int(window_policy.context_tokens),
            "scored_tokens_per_full_window": int(window_policy.score_tokens),
            "teacher_component_count": int(topology_meta.get("component_count", -1)),
            "teacher_face_count": int(topology_meta.get("face_count", -1)),
            "teacher_vertex_count": int(topology_meta.get("vertex_count", -1)),
            "witness_count": WITNESS_COUNT,
        })
        del source, scene_planes, raw, payload, h, mechanics
        torch.cuda.empty_cache()

    manifest = {
        "schema": SCHEMA,
        "status": "MATERIALIZED__TESSA_V2_TARGET_ONLY_TEACHER_TOPOLOGY__NO_TRAINING",
        "execution_source_sha": expected_sha,
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "direct_surface_batch_sha256": sha256_file(batch_path),
        "v8_checkpoint_sha256": sha256_file(Path(args.v8_checkpoint)),
        "subject_order": list(SUBJECT_IDS),
        "subjects": subject_receipts,
        "conditioning": {
            "scene": "SEALED_V8_SCENE_PLANES",
            "surface": "DIRECT_IRIS_T512_T1024_SURFACE_WITNESS",
            "local_features": "SEALED_V8_H_AT_DIRECT_SURFACE_WITNESS",
            "mechanics": "SEALED_V8_TRAINED_MECHANICAL_DEMAND_AT_WITNESS",
            "contact_persistence": "ABSTAIN_INVALID",
            "teacher_topology": "TARGET_ONLY__FORBIDDEN_FROM_CONDITIONING",
        },
        "training_window_policy": {
            "context_tokens": window_policy.context_tokens,
            "score_tokens": window_policy.score_tokens,
            "stride_tokens": window_policy.stride_tokens,
            "global_history_exercised": True,
        },
        "optimizer_steps_executed": 0,
        "training_authorized": False,
        "scientific_pass_claimed": False,
    }
    manifest["value_sha256"] = value_sha256(manifest)
    (output_root / "TESSA_V2_CORPUS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--prepared-manifest", required=True)
    p.add_argument("--v8-checkpoint", required=True)
    p.add_argument("--direct-surface-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--device", default="cuda")
    result = build(p.parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
