from __future__ import annotations

"""Measure marginal articulation gain for Geppetto K1\K0 source controls.

Training/evaluation-only court. For every source control admitted by K1 but not
K0, apply generic rest-local +/-10 degree probes around all three local axes and
measure whether that control changes the downstream skinning affine field of
skin-supported deform descendants.

This court does not use source names, requested control counts, animation clips,
or product inference. It constructs the K2 target only from independently
measured non-zero articulation effect and never mints product authority.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.geppetto_reference_strength_fullstack_v1.mechanical_capacity_target_v2 import (
    POLICY_K0,
    POLICY_K1,
    build_mechanical_capacity_target_v2,
)
from experiments.geppetto_reference_strength_fullstack_v1.mechanical_core_target_v1 import (
    world_heads_from_rest_world_source_v1,
)


ANGLE_DEG = 10.0
SUPPORT_EPS = 1e-8
EFFECT_EPS = 1e-9


def _load_source(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {"parents", "deform_mask", "skin", "rest_world_source"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(
                "GEPPETTO_MARGINAL_SOURCE_ARRAYS_MISSING:"
                + ",".join(sorted(missing))
            )
        parents = np.asarray(z["parents"], np.int64)
        deform = np.asarray(z["deform_mask"], np.uint8).astype(bool)
        skin = np.asarray(z["skin"], np.float64)
        rest = np.asarray(z["rest_world_source"], np.float64)
    j = len(parents)
    if deform.shape != (j,) or rest.shape != (j, 4, 4):
        raise RuntimeError("GEPPETTO_MARGINAL_SOURCE_SHAPE_DRIFT")
    if skin.ndim != 2 or skin.shape[1] != j:
        raise RuntimeError("GEPPETTO_MARGINAL_SKIN_SHAPE_DRIFT")
    if not np.isfinite(rest).all() or not np.isfinite(skin).all():
        raise RuntimeError("GEPPETTO_MARGINAL_NONFINITE_SOURCE")
    if np.any(skin < 0.0):
        raise RuntimeError("GEPPETTO_MARGINAL_NEGATIVE_SKIN")
    if np.any((parents < -1) | (parents >= j)) or np.any(parents == np.arange(j)):
        raise RuntimeError("GEPPETTO_MARGINAL_PARENT_INVALID")
    return parents, deform, skin, rest


def _rotation4(axis: int, degrees: float) -> np.ndarray:
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0,1,2")
    a = math.radians(float(degrees))
    c, s = math.cos(a), math.sin(a)
    R = np.eye(4, dtype=np.float64)
    if axis == 0:
        R[:3, :3] = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64
        )
    elif axis == 1:
        R[:3, :3] = np.asarray(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
        )
    else:
        R[:3, :3] = np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    return R


def _rest_locals(parents: np.ndarray, rest_world: np.ndarray) -> np.ndarray:
    out = np.empty_like(rest_world, dtype=np.float64)
    for i, p in enumerate(parents.tolist()):
        if p < 0:
            out[i] = rest_world[i]
        else:
            out[i] = np.linalg.inv(rest_world[int(p)]) @ rest_world[i]
    return out


def _worlds_from_locals(
    parents: np.ndarray, locals_: np.ndarray, active: int, axis: int, degrees: float
) -> np.ndarray:
    j = len(parents)
    memo: list[np.ndarray | None] = [None] * j
    active_rot = _rotation4(axis, degrees)

    def solve(i: int) -> np.ndarray:
        hit = memo[i]
        if hit is not None:
            return hit
        local = locals_[i] @ active_rot if i == active else locals_[i]
        p = int(parents[i])
        world = local if p < 0 else solve(p) @ local
        memo[i] = world
        return world

    for i in range(j):
        solve(i)
    return np.stack([x for x in memo if x is not None], axis=0)


def _descendants(parents: np.ndarray, root: int) -> np.ndarray:
    out = np.zeros(len(parents), dtype=bool)
    for node in range(len(parents)):
        cur = node
        while cur >= 0:
            if cur == root:
                out[node] = True
                break
            cur = int(parents[cur])
    return out


def _target_indices(target) -> set[int]:
    return set(map(int, target.source_indices_provenance_only.tolist()))


def _quantile(a: np.ndarray, p: float) -> float:
    a = np.asarray(a, np.float64)
    return float(np.quantile(a, p)) if a.size else 0.0


def _measure_control(
    *,
    control: int,
    parents: np.ndarray,
    deform: np.ndarray,
    skin: np.ndarray,
    rest_world: np.ndarray,
    rest_local: np.ndarray,
    body_span: float,
) -> dict:
    descendants = _descendants(parents, control)
    effective_cols = descendants & deform
    effective_skin = np.asarray(skin[:, effective_cols], np.float64)
    descendant_skin_mass = float(effective_skin.sum())
    descendant_supported_bone_count = int(
        np.count_nonzero(
            np.asarray(skin, np.float64).sum(axis=0)[effective_cols] > SUPPORT_EPS
        )
    )
    direct_mass = float(np.asarray(skin[:, control], np.float64).sum())

    inv_rest = np.linalg.inv(rest_world)
    eye = np.eye(4, dtype=np.float64)
    supported_rows = np.asarray(skin.sum(axis=1) > SUPPORT_EPS, bool)
    probe_rows = []

    for axis in range(3):
        for sign in (-1.0, 1.0):
            degrees = sign * ANGLE_DEG
            posed = _worlds_from_locals(
                parents, rest_local, control, axis, degrees
            )
            skin_mats = posed @ inv_rest
            delta = skin_mats - eye[None, :, :]
            # Non-deform bones are not allowed to create deformation authority.
            delta[:, :, :] *= deform[:, None, None]
            blended = np.einsum("vj,jab->vab", skin, delta, optimize=True)
            rot = np.linalg.norm(blended[:, :3, :3], axis=(1, 2))
            trans = np.linalg.norm(blended[:, :3, 3], axis=1) / body_span
            affine = np.sqrt(rot * rot + trans * trans)
            active_vals = affine[supported_rows]
            affected = int(np.count_nonzero(active_vals > EFFECT_EPS))
            probe_rows.append(
                {
                    "axis": int(axis),
                    "degrees": float(degrees),
                    "affected_vertex_count": affected,
                    "affine_delta_rms_supported": float(
                        np.sqrt(np.mean(active_vals * active_vals))
                    )
                    if active_vals.size
                    else 0.0,
                    "affine_delta_p95_supported": _quantile(active_vals, 0.95),
                    "affine_delta_max_supported": float(
                        active_vals.max(initial=0.0)
                    ),
                }
            )

    max_effect = max(row["affine_delta_max_supported"] for row in probe_rows)
    max_affected = max(row["affected_vertex_count"] for row in probe_rows)

    if descendant_skin_mass <= SUPPORT_EPS:
        verdict = "PROVEN_ZERO_SKIN_DEFORMATION_EFFECT"
        necessary = False
    elif max_effect > EFFECT_EPS and max_affected > 0:
        verdict = "PROVEN_NONZERO_MARGINAL_ARTICULATION_EFFECT"
        necessary = True
    else:
        verdict = "INCONCLUSIVE_NUMERIC_EFFECT"
        necessary = False

    return {
        "source_index_provenance_only": int(control),
        "source_deform": bool(deform[control]),
        "direct_skin_mass": direct_mass,
        "descendant_effective_skin_mass": descendant_skin_mass,
        "descendant_supported_deform_bone_count": descendant_supported_bone_count,
        "verdict": verdict,
        "necessary_for_k2": bool(necessary),
        "max_affected_vertex_count": int(max_affected),
        "max_affine_delta_supported": float(max_effect),
        "probes": probe_rows,
    }


def main(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parents, deform, skin, rest = _load_source(args.teacher_source_npz)
    heads = world_heads_from_rest_world_source_v1(rest)

    k0 = build_mechanical_capacity_target_v2(
        parents=parents,
        deform_mask=deform,
        skin=skin,
        bone_heads_world=heads,
        policy=POLICY_K0,
    )
    k1 = build_mechanical_capacity_target_v2(
        parents=parents,
        deform_mask=deform,
        skin=skin,
        bone_heads_world=heads,
        policy=POLICY_K1,
    )
    k0_set = _target_indices(k0)
    k1_set = _target_indices(k1)
    marginal = sorted(k1_set - k0_set)
    if not marginal:
        raise RuntimeError("GEPPETTO_MARGINAL_NO_K1_MINUS_K0_CONTROLS")

    span_vec = heads.max(axis=0) - heads.min(axis=0)
    body_span = float(np.linalg.norm(span_vec))
    if not np.isfinite(body_span) or body_span <= 1e-12:
        raise RuntimeError("GEPPETTO_MARGINAL_BODY_SPAN_INVALID")

    rest_local = _rest_locals(parents, rest)
    reconstructed = _worlds_from_locals(parents, rest_local, -1, 0, 0.0)
    rest_reconstruction_error = float(np.max(np.abs(reconstructed - rest)))
    if rest_reconstruction_error > 1e-7:
        raise RuntimeError(
            f"GEPPETTO_MARGINAL_REST_RECONSTRUCTION_DRIFT:{rest_reconstruction_error}"
        )

    controls = [
        _measure_control(
            control=i,
            parents=parents,
            deform=deform,
            skin=skin,
            rest_world=rest,
            rest_local=rest_local,
            body_span=body_span,
        )
        for i in marginal
    ]
    inconclusive = [
        row for row in controls if row["verdict"] == "INCONCLUSIVE_NUMERIC_EFFECT"
    ]
    if inconclusive:
        raise RuntimeError(
            "GEPPETTO_MARGINAL_INCONCLUSIVE_CONTROLS:"
            + ",".join(str(x["source_index_provenance_only"]) for x in inconclusive)
        )

    necessary = np.zeros(len(parents), dtype=bool)
    for row in controls:
        if row["necessary_for_k2"]:
            necessary[int(row["source_index_provenance_only"])] = True

    report = {
        "schema": "RealSaS.GeppettoMarginalArticulationGainCourt.v1",
        "status": "MEASURED__K1_MARGINAL_EFFECT_CLASSIFIED__NO_PROMOTION_CLAIM",
        "probe_contract": {
            "type": "REST_LOCAL_AXIS_MICRO_ROTATION",
            "angle_degrees": ANGLE_DEG,
            "axes": 3,
            "signed_probes_per_axis": 2,
            "effect_metric": (
                "SKIN_WEIGHTED_DESCENDANT_AFFINE_DELTA_3x4;"
                "TRANSLATION_NORMALIZED_BY_SOURCE_HEAD_SPAN"
            ),
            "support_epsilon": SUPPORT_EPS,
            "effect_epsilon": EFFECT_EPS,
            "source_names_used": False,
            "animation_clips_used": False,
        },
        "source_joint_count": int(len(parents)),
        "source_deform_count": int(np.count_nonzero(deform)),
        "source_skin_supported_deform_count": int(
            np.count_nonzero(deform & (skin.sum(axis=0) > SUPPORT_EPS))
        ),
        "body_span": body_span,
        "rest_reconstruction_max_abs_error": rest_reconstruction_error,
        "target_counts": {
            "K0_CURRENT_CORE": int(k0.count),
            "K1_ALL_DEFORM": int(k1.count),
        },
        "set_deltas": {
            "K1_minus_K0": int(len(k1_set - k0_set)),
        },
        "marginal_nonzero_effect_control_count": int(np.count_nonzero(necessary)),
        "marginal_nonzero_effect_source_indices_provenance_only": [
            int(x) for x in np.flatnonzero(necessary).tolist()
        ],
        "controls": controls,
        "training_used": False,
        "teacher_source_used_for_evaluation_only": True,
        "product_authority_minted": False,
        "generalization_claimed": False,
        "claim_boundary": [
            "This court measures whether omitted source controls create a non-zero hierarchical LBS affine-field degree of freedom under generic micro-poses.",
            "It does not claim perceptual importance, animation quality, or cross-character generalization.",
            "This court classifies the extra K1 all-deform controls relative to K0; it does not redefine the existing K2 helper-extension contract.",
            "Source row indices are emitted only as provenance and are not learner inputs or product IDs.",
        ],
    }
    (args.out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.out_dir / "K1_MARGINAL_EFFECT_MASK.npz",
        marginal_nonzero_effect_mask=necessary.astype(np.uint8),
    )
    print(
        "GEPPETTO_MARGINAL_ARTICULATION_GAIN_RESULT="
        + json.dumps(
            {
                "status": report["status"],
                "target_counts": report["target_counts"],
                "set_deltas": report["set_deltas"],
                "marginal_nonzero_effect_control_count": report["marginal_nonzero_effect_control_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-source-npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    try:
        main(a)
    except Exception as exc:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "ERROR.json").write_text(
            json.dumps(
                {"type": type(exc).__name__, "message": str(exc)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
