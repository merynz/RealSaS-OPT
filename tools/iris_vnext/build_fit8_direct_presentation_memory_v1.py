#!/usr/bin/env python3
from __future__ import annotations

"""Build inference-admissible TESSA presentation memory from Direct IRIS queries.

The input query corpus is produced from the model surface and admitted source
views only. This stage deterministically types three role-free evidence families:

1. source RGBA/alpha/appearance boundary at a unique model-surface front hit;
2. pairwise front order for non-tied model-surface hits on one source ray;
3. complete-stack global visibility with the same tie-safe ABSTAIN law as V2.

No teacher artifact or V2 payload is accepted. V2 defines/evaluates semantics;
it is not inference conditioning. Zero optimizer/backward steps are executed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from models.iris.vnext.cohort_v1 import SUBJECT_IDS
from models.iris.vnext.presentation_evidence_v1 import (
    PRESENTATION_FEATURE_NAMES_V1,
    PRESENTATION_KIND_PAIR,
    PRESENTATION_KIND_RAY_HIT,
    RoleFreePresentationEvidenceV1,
    build_role_free_presentation_evidence_v1,
)
from models.iris.vnext.presentation_observations_v1 import (
    surface_presentation_from_source_observations_v1,
)
from tools.iris_vnext.build_fit8_vnext_shards_v1 import _seed


SCHEMA = "RealSaS.DirectIRISPresentationMemorySubject.v1"
MANIFEST_SCHEMA = "RealSaS.DirectIRISPresentationMemoryManifest.v1"
STATUS = "MATERIALIZED_INFERENCE_ADMISSIBLE_ROLE_FREE_PRESENTATION__NO_TRAINING"
SURFACE_TOKENS_PER_VIEW = 256
PAIR_TOKENS_PER_VIEW = 384
RAY_HIT_TOKENS_PER_VIEW = 384
MAX_PRESENTATION_TOKENS_PER_SUBJECT = 8192


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _value_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DIRECT_PRESENTATION_MEMORY_JSON_OBJECT_REQUIRED")
    return value


def _choice(count: int, limit: int, *, key: str) -> np.ndarray:
    if count <= 0 or limit <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(_seed(key))
    return rng.permutation(count)[: min(count, limit)].astype(np.int64)


def _stack_visibility(depth: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth, dtype=np.float64).reshape(-1)
    if not len(d):
        return np.zeros(0, np.float32), np.zeros(0, np.bool_)
    if not np.isfinite(d).all() or np.any(np.diff(d) < -epsilon):
        raise ValueError("DIRECT_PRESENTATION_MEMORY_STACK_DEPTH_INVALID")
    p = np.zeros(len(d), dtype=np.float32)
    valid = np.ones(len(d), dtype=np.bool_)
    minimum = np.abs(d - d[0]) <= float(epsilon)
    if int(np.count_nonzero(minimum)) == 1:
        p[0] = 1.0
    else:
        valid[minimum] = False
    return p, valid


def _concat(parts: list[RoleFreePresentationEvidenceV1]) -> RoleFreePresentationEvidenceV1:
    if not parts:
        raise ValueError("DIRECT_PRESENTATION_MEMORY_EMPTY_PARTS")
    return RoleFreePresentationEvidenceV1(
        values=torch.cat([part.values for part in parts], dim=1),
        valid_mask=torch.cat([part.valid_mask for part in parts], dim=1),
        kind_id=torch.cat([part.kind_id for part in parts], dim=1),
    )


def _camera_rows(camera: np.ndarray, count: int) -> torch.Tensor:
    c = np.asarray(camera, dtype=np.float32).reshape(13)
    return torch.from_numpy(np.repeat(c[None, :], int(count), axis=0))[None]


def _surface_part(
    data: dict[str, np.ndarray], *, subject_id: str, view_index: int, epsilon: float
) -> RoleFreePresentationEvidenceV1 | None:
    offsets = np.asarray(data["stack_offsets"], np.int64)
    depth = np.asarray(data["stack_depth"], np.float64)
    point = np.asarray(data["stack_point_iris"], np.float32)
    candidates: list[int] = []
    for ray in range(len(offsets) - 1):
        start, end = int(offsets[ray]), int(offsets[ray + 1])
        if end <= start:
            continue
        minimum = np.abs(depth[start:end] - depth[start]) <= float(epsilon)
        if int(np.count_nonzero(minimum)) == 1:
            candidates.append(ray)
    chosen_local = _choice(
        len(candidates),
        SURFACE_TOKENS_PER_VIEW,
        key=f"direct-presentation-memory-v1:{subject_id}:v{view_index}:surface",
    )
    if not len(chosen_local):
        return None
    rays = np.asarray(candidates, np.int64)[chosen_local]
    starts = offsets[rays]
    xyz = torch.from_numpy((point[starts] * 0.5).astype(np.float32))[None]
    rgba = torch.from_numpy(np.asarray(data["source_rgba"], np.float32)[rays])[None]
    alpha = torch.from_numpy(np.asarray(data["source_alpha_boundary"], np.float32)[rays])[None]
    appearance = torch.from_numpy(np.asarray(data["source_appearance_boundary"], np.float32)[rays])[None]
    support = torch.from_numpy(np.asarray(data["source_support"], np.float32)[rays])[None]
    valid = torch.ones((1, len(rays)), dtype=torch.bool)
    return surface_presentation_from_source_observations_v1(
        xyz=xyz,
        camera_features=_camera_rows(data["camera_features"], len(rays)),
        token_valid_mask=valid,
        source_rgba=rgba,
        source_rgba_valid=valid,
        alpha_boundary=alpha,
        appearance_boundary=appearance,
        boundary_valid=valid,
        source_support=support,
        source_support_valid=valid,
    )


def _pair_part(
    data: dict[str, np.ndarray], *, subject_id: str, view_index: int, epsilon: float
) -> RoleFreePresentationEvidenceV1 | None:
    base = np.asarray(data["pair_base_pair_id"], np.int64)
    if not len(base):
        return None
    ray = np.asarray(data["pair_ray_index"], np.int64)
    rank_a = np.asarray(data["pair_rank_a"], np.int64)
    rank_b = np.asarray(data["pair_rank_b"], np.int64)
    offsets = np.asarray(data["stack_offsets"], np.int64)
    depth = np.asarray(data["stack_depth"], np.float64)
    da = depth[offsets[ray] + rank_a]
    db = depth[offsets[ray] + rank_b]
    front_valid = np.abs(da - db) > float(epsilon)
    eligible_groups = np.unique(base[front_valid])
    group_budget = PAIR_TOKENS_PER_VIEW // 2
    selected_group_indices = _choice(
        len(eligible_groups),
        group_budget,
        key=f"direct-presentation-memory-v1:{subject_id}:v{view_index}:pair",
    )
    if not len(selected_group_indices):
        return None
    selected_groups = eligible_groups[selected_group_indices]
    selected = np.flatnonzero(front_valid & np.isin(base, selected_groups))
    # Every selected unordered pair must carry both orientations.
    if len(selected) > PAIR_TOKENS_PER_VIEW or len(selected) % 2:
        raise ValueError("DIRECT_PRESENTATION_MEMORY_PAIR_ORIENTATION_DRIFT")
    if any(int(np.count_nonzero(base[selected] == group)) != 2 for group in selected_groups):
        raise ValueError("DIRECT_PRESENTATION_MEMORY_PAIR_GROUP_CARDINALITY_DRIFT")
    xyz_a = torch.from_numpy(np.asarray(data["pair_point_a_iris"], np.float32)[selected] * 0.5)[None]
    xyz_b = torch.from_numpy(np.asarray(data["pair_point_b_iris"], np.float32)[selected] * 0.5)[None]
    token_valid = torch.ones((1, len(selected)), dtype=torch.bool)
    front = torch.from_numpy((da[selected] < db[selected]).astype(np.float32))[None]
    return build_role_free_presentation_evidence_v1(
        xyz=xyz_a,
        paired_xyz=xyz_b,
        paired_xyz_valid=token_valid,
        kind_id=torch.full((1, len(selected)), PRESENTATION_KIND_PAIR, dtype=torch.long),
        token_valid_mask=token_valid,
        camera_features=_camera_rows(data["camera_features"], len(selected)),
        camera_valid=token_valid,
        front_probability=front,
        front_valid=token_valid,
    )


def _ray_part(
    data: dict[str, np.ndarray], *, subject_id: str, view_index: int, epsilon: float
) -> RoleFreePresentationEvidenceV1 | None:
    offsets = np.asarray(data["stack_offsets"], np.int64)
    depth = np.asarray(data["stack_depth"], np.float64)
    point = np.asarray(data["stack_point_iris"], np.float32)
    ray_order = _choice(
        len(offsets) - 1,
        len(offsets) - 1,
        key=f"direct-presentation-memory-v1:{subject_id}:v{view_index}:ray",
    )
    chosen_rays: list[int] = []
    used = 0
    for ray in ray_order.tolist():
        count = int(offsets[ray + 1] - offsets[ray])
        if count <= 0 or count > RAY_HIT_TOKENS_PER_VIEW:
            continue
        if used + count > RAY_HIT_TOKENS_PER_VIEW:
            continue
        chosen_rays.append(int(ray))
        used += count
        if used == RAY_HIT_TOKENS_PER_VIEW:
            break
    if not chosen_rays:
        return None

    xyz_rows: list[np.ndarray] = []
    vis_rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    for ray in chosen_rays:
        start, end = int(offsets[ray]), int(offsets[ray + 1])
        p, valid = _stack_visibility(depth[start:end], epsilon)
        xyz_rows.append(point[start:end] * 0.5)
        vis_rows.append(p)
        valid_rows.append(valid)
    xyz = torch.from_numpy(np.concatenate(xyz_rows, axis=0).astype(np.float32))[None]
    visibility = torch.from_numpy(np.concatenate(vis_rows).astype(np.float32))[None]
    visibility_valid = torch.from_numpy(np.concatenate(valid_rows).astype(np.bool_))[None]
    token_valid = torch.ones((1, xyz.shape[1]), dtype=torch.bool)
    return build_role_free_presentation_evidence_v1(
        xyz=xyz,
        kind_id=torch.full((1, xyz.shape[1]), PRESENTATION_KIND_RAY_HIT, dtype=torch.long),
        token_valid_mask=token_valid,
        camera_features=_camera_rows(data["camera_features"], xyz.shape[1]),
        camera_valid=token_valid,
        global_visibility=visibility,
        global_visibility_valid=visibility_valid,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    expected_source = str(args.expected_source_sha).strip().lower()
    if _git_head(repo_root) != expected_source:
        raise RuntimeError("DIRECT_PRESENTATION_MEMORY_CHECKOUT_SHA_DRIFT")
    query_root = Path(args.query_root).resolve()
    query_manifest_path = query_root / "DIRECT_IRIS_PRESENTATION_QUERIES_V1_MANIFEST.json"
    query_manifest = _json(query_manifest_path)
    if query_manifest.get("schema") != "RealSaS.DirectIRISPresentationQueriesManifest.v1":
        raise ValueError("DIRECT_PRESENTATION_MEMORY_QUERY_SCHEMA_INVALID")
    for forbidden in (
        "teacher_mesh_input",
        "teacher_topology_input",
        "teacher_rig_skin_input",
        "materialized_v2_truth_input",
    ):
        if query_manifest.get(forbidden) is not False:
            raise ValueError("DIRECT_PRESENTATION_MEMORY_QUERY_PROVENANCE_INVALID:" + forbidden)
    if int(query_manifest.get("optimizer_steps_executed", -1)) != 0:
        raise ValueError("DIRECT_PRESENTATION_MEMORY_QUERY_OPTIMIZER_DRIFT")
    subjects = list(query_manifest.get("subjects") or ())
    if tuple(row.get("subject_id") for row in subjects) != SUBJECT_IDS:
        raise ValueError("DIRECT_PRESENTATION_MEMORY_SUBJECT_ORDER_DRIFT")

    output = Path(args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("DIRECT_PRESENTATION_MEMORY_OUTPUT_MUST_BE_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []

    for subject in subjects:
        subject_id = str(subject["subject_id"])
        epsilon = float(subject["direct_surface_depth_tie_epsilon"])
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("DIRECT_PRESENTATION_MEMORY_TIE_EPS_INVALID:" + subject_id)
        parts: list[RoleFreePresentationEvidenceV1] = []
        view_counts: list[dict[str, int]] = []
        for view_row in subject.get("views") or ():
            view_index = int(view_row["view_index"])
            path = query_root / str(view_row["file"])
            if _sha(path) != str(view_row["sha256"]):
                raise ValueError("DIRECT_PRESENTATION_MEMORY_QUERY_SHA_DRIFT")
            with np.load(path, allow_pickle=False) as npz:
                data = {name: np.asarray(npz[name]) for name in npz.files}
            local_parts = [
                _surface_part(data, subject_id=subject_id, view_index=view_index, epsilon=epsilon),
                _pair_part(data, subject_id=subject_id, view_index=view_index, epsilon=epsilon),
                _ray_part(data, subject_id=subject_id, view_index=view_index, epsilon=epsilon),
            ]
            counts = {"surface": 0, "pair": 0, "ray_hit": 0}
            for name, part in zip(tuple(counts), local_parts):
                if part is not None:
                    counts[name] = int(part.values.shape[1])
                    parts.append(part)
            counts["view_index"] = view_index
            view_counts.append(counts)
        evidence = _concat(parts)
        if evidence.values.shape[1] > MAX_PRESENTATION_TOKENS_PER_SUBJECT:
            raise AssertionError("DIRECT_PRESENTATION_MEMORY_TOKEN_BUDGET_DRIFT")
        payload = {
            "schema": SCHEMA,
            "subject_id": subject_id,
            "values": evidence.values[0].contiguous(),
            "valid_mask": evidence.valid_mask[0].contiguous(),
            "kind_id": evidence.kind_id[0].contiguous(),
            "feature_names": PRESENTATION_FEATURE_NAMES_V1,
            "source_query_subject_binding": {
                "direct_surface_sha256": subject["direct_surface_sha256"],
                "direct_surface_candidate_binding_hash": subject[
                    "direct_surface_candidate_binding_hash"
                ],
                "camera_bundle_sha256": subject["camera_bundle_sha256"],
            },
            "teacher_geometry_in_conditioning": False,
            "teacher_topology_in_conditioning": False,
            "materialized_v2_truth_in_conditioning": False,
        }
        name = subject_id.lower().replace("-", "").replace(" ", "") + ".presentation_v1.pt"
        path = output / name
        torch.save(payload, path)
        receipts.append(
            {
                "subject_id": subject_id,
                "file": name,
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "token_count": int(evidence.values.shape[1]),
                "view_counts": view_counts,
                "direct_surface_sha256": subject["direct_surface_sha256"],
            }
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": STATUS,
        "execution_source_sha": expected_source,
        "query_manifest_sha256": _sha(query_manifest_path),
        "subject_order": list(SUBJECT_IDS),
        "subjects": receipts,
        "feature_names": list(PRESENTATION_FEATURE_NAMES_V1),
        "sampling_policy": {
            "surface_tokens_per_view": SURFACE_TOKENS_PER_VIEW,
            "pair_tokens_per_view": PAIR_TOKENS_PER_VIEW,
            "ray_hit_tokens_per_view": RAY_HIT_TOKENS_PER_VIEW,
            "maximum_tokens_per_subject": MAX_PRESENTATION_TOKENS_PER_SUBJECT,
            "pair_orientation_policy": "SAMPLE_BASE_PAIR_GROUPS_AND_KEEP_BOTH_ORIENTATIONS",
            "ray_policy": "SAMPLE_WHOLE_COMPLETE_STACKS_ONLY",
            "surface_observation_policy": "UNIQUE_MODEL_SURFACE_MINIMUM_ONLY",
        },
        "evidence_authority": {
            "surface_rgba_boundary": "ADMITTED_SOURCE_OBSERVATION_ON_UNIQUE_DIRECT_SURFACE_FRONT_HIT",
            "pair_front": "DETERMINISTIC_DIRECT_SURFACE_DEPTH_ORDER__TIES_ABSTAIN",
            "global_visibility": "DETERMINISTIC_COMPLETE_DIRECT_SURFACE_STACK__TIED_MINIMA_ABSTAIN",
            "occlusion": "ABSTAIN",
            "deformation_coherence": "ABSTAIN",
            "attachment_persistence": "ABSTAIN",
        },
        "teacher_geometry_input": False,
        "teacher_topology_input": False,
        "materialized_v2_truth_input": False,
        "optimizer_steps_executed": 0,
        "backward_steps_executed": 0,
        "tessa_training_executed": False,
        "scientific_pass_claimed": False,
        "training_authorized": False,
        "next_action": "BIND_AS_B_ARM_PRESENTATION_CONDITIONING_AND_RUN_ZERO_STEP_PREFLIGHT",
    }
    manifest["value_sha256"] = _value_sha(manifest)
    manifest_path = output / "DIRECT_IRIS_PRESENTATION_MEMORY_V1_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--expected-source-sha", required=True)
    p.add_argument("--query-root", required=True)
    p.add_argument("--output-root", required=True)
    return p


def main() -> int:
    result = build(parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
