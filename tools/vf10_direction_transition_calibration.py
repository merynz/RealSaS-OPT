from __future__ import annotations

"""Subject-free VF-10 adjacent-direction transition calibration.

The calibration deliberately searches for controls that ALREADY satisfy the frozen
adjacent-pair compatibility policy. The new metric is allowed to mint a threshold
only when smooth cyclic view-dependent appearance and isolated single-view shimmer
separate under the second-order triplet residual.
"""

import json
import os
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_quality_v2 import (
    adjacent_direction_transition_metrics,
    cross_view_source_compatibility_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "canonical" / "CAA_V2_SUBJECT_FREE_NUMERICAL_POLICY_20260920.json"
OUT = Path(os.environ.get("REALSAS_VF10_OUT", "vf10_out"))
OUT.mkdir(parents=True, exist_ok=True)


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))[
        "completion_quality_policy"
    ]


def _pair_pass(metrics: dict, policy: dict) -> bool:
    pair_ok = all(
        int(row["shared_direct_sample_count"])
        >= int(policy["cross_view_min_shared_direct_samples_per_pair"])
        and float(row["p95_premultiplied_rgba_l1"])
        <= float(policy["cross_view_max_pair_p95_rgba_l1"])
        and float(row["color_conflict_fraction"])
        <= float(policy["cross_view_max_pair_color_conflict_fraction"])
        and float(row["p95_alpha_abs"])
        <= float(policy["cross_view_max_pair_p95_alpha_abs"])
        and float(row["alpha_conflict_fraction"])
        <= float(policy["cross_view_max_pair_alpha_conflict_fraction"])
        for row in metrics["per_pair"]
    )
    component_min = int(policy["cross_view_min_component_samples_for_gate"])
    component_rows = [
        row
        for row in metrics["per_pair_component"]
        if int(row["shared_direct_sample_count"]) >= component_min
    ]
    component_ok = all(
        float(row["color_conflict_fraction"])
        <= float(policy["cross_view_max_component_color_conflict_fraction"])
        and float(row["alpha_conflict_fraction"])
        <= float(policy["cross_view_max_component_alpha_conflict_fraction"])
        for row in component_rows
    )
    return bool(pair_ok and component_ok)


def _base_rgba(*, rgb_amplitude: float = 0.0, alpha_amplitude: float = 0.0) -> np.ndarray:
    count = 64
    out = np.zeros((8, count, 4), dtype=np.uint8)
    for view in range(8):
        phase = 2.0 * np.pi * float(view) / 8.0
        rgb = int(round(128.0 + float(rgb_amplitude) * np.sin(phase)))
        alpha = int(round(205.0 + float(alpha_amplitude) * np.sin(phase)))
        out[view, :, :3] = np.uint8(np.clip(rgb, 0, 255))
        out[view, :, 3] = np.uint8(np.clip(alpha, 0, 255))
    return out


def _measure(name: str, rgba: np.ndarray, valid: np.ndarray, component: np.ndarray, policy: dict) -> dict:
    pair = cross_view_source_compatibility_metrics(
        direct_valid=valid,
        direct_rgba=rgba,
        sample_component_index=component,
        color_conflict_cut_rgba_l1=float(
            policy["cross_view_color_conflict_cut_rgba_l1"]
        ),
        alpha_conflict_cut=float(policy["cross_view_alpha_conflict_cut"]),
    )
    transition = adjacent_direction_transition_metrics(
        direct_valid=valid,
        direct_rgba=rgba,
        sample_component_index=component,
    )
    return {
        "name": name,
        "existing_pair_policy_passed": _pair_pass(pair, policy),
        "pair_max_p95_pm_l1": max(
            float(row["p95_premultiplied_rgba_l1"]) for row in pair["per_pair"]
        ),
        "pair_max_p95_alpha_abs": max(
            float(row["p95_alpha_abs"]) for row in pair["per_pair"]
        ),
        "triplet_max_p95_pm_l1": max(
            float(row["p95_center_residual_pm_l1"])
            for row in transition["per_triplet"]
        ),
        "triplet_max_p95_alpha_abs": max(
            float(row["p95_center_alpha_residual"])
            for row in transition["per_triplet"]
        ),
        "component_max_p95_pm_l1": max(
            (
                float(row["p95_center_residual_pm_l1"])
                for row in transition["per_triplet_component"]
            ),
            default=0.0,
        ),
        "component_max_p95_alpha_abs": max(
            (
                float(row["p95_center_alpha_residual"])
                for row in transition["per_triplet_component"]
            ),
            default=0.0,
        ),
    }


def main() -> int:
    policy = _policy()
    valid = np.ones((8, 64), dtype=bool)
    component = np.asarray([0] * 32 + [1] * 32, dtype=np.int32)

    benign = [
        _measure(
            "CONSTANT",
            _base_rgba(rgb_amplitude=0.0, alpha_amplitude=0.0),
            valid,
            component,
            policy,
        ),
        _measure(
            "SMOOTH_RGB_NEAR_PAIR_ENVELOPE",
            _base_rgba(rgb_amplitude=100.0, alpha_amplitude=0.0),
            valid,
            component,
            policy,
        ),
        _measure(
            "SMOOTH_ALPHA_NEAR_PAIR_ENVELOPE",
            _base_rgba(rgb_amplitude=0.0, alpha_amplitude=48.0),
            valid,
            component,
            policy,
        ),
        _measure(
            "SMOOTH_COMBINED",
            _base_rgba(rgb_amplitude=72.0, alpha_amplitude=32.0),
            valid,
            component,
            policy,
        ),
    ]

    rgb_spike = _base_rgba(rgb_amplitude=36.0, alpha_amplitude=0.0)
    rgb_spike[2, :, :3] = np.clip(
        rgb_spike[2, :, :3].astype(np.int16) + 50,
        0,
        255,
    ).astype(np.uint8)

    component_rgb_spike = _base_rgba(rgb_amplitude=36.0, alpha_amplitude=0.0)
    component_rgb_spike[6, 32:, :3] = np.clip(
        component_rgb_spike[6, 32:, :3].astype(np.int16) + 50,
        0,
        255,
    ).astype(np.uint8)

    alpha_spike = _base_rgba(rgb_amplitude=0.0, alpha_amplitude=0.0)
    alpha_spike[5, :, 3] = np.uint8(175)

    component_alpha_spike = _base_rgba(rgb_amplitude=0.0, alpha_amplitude=0.0)
    component_alpha_spike[1, 32:, 3] = np.uint8(175)

    adversarial_rgb = [
        _measure("SINGLE_VIEW_RGB_SPIKE", rgb_spike, valid, component, policy),
        _measure(
            "COMPONENT_LOCAL_RGB_SPIKE",
            component_rgb_spike,
            valid,
            component,
            policy,
        ),
    ]
    adversarial_alpha = [
        _measure("SINGLE_VIEW_ALPHA_SPIKE", alpha_spike, valid, component, policy),
        _measure(
            "COMPONENT_LOCAL_ALPHA_SPIKE",
            component_alpha_spike,
            valid,
            component,
            policy,
        ),
    ]

    if not all(row["existing_pair_policy_passed"] for row in benign):
        raise RuntimeError("VF10_BENIGN_CONTROL_FAILS_EXISTING_PAIR_POLICY")
    if not all(
        row["existing_pair_policy_passed"]
        for row in adversarial_rgb + adversarial_alpha
    ):
        raise RuntimeError("VF10_ADVERSARY_MUST_BE_BLIND_TO_EXISTING_PAIR_POLICY")

    benign_pm = max(
        max(row["triplet_max_p95_pm_l1"], row["component_max_p95_pm_l1"])
        for row in benign
    )
    adversarial_pm = min(
        max(row["triplet_max_p95_pm_l1"], row["component_max_p95_pm_l1"])
        for row in adversarial_rgb
    )
    benign_alpha = max(
        max(
            row["triplet_max_p95_alpha_abs"],
            row["component_max_p95_alpha_abs"],
        )
        for row in benign
    )
    adversarial_alpha_floor = min(
        max(
            row["triplet_max_p95_alpha_abs"],
            row["component_max_p95_alpha_abs"],
        )
        for row in adversarial_alpha
    )

    if not benign_pm < adversarial_pm:
        raise RuntimeError("VF10_PM_CONTROL_CLASSES_NOT_SEPARABLE")
    if not benign_alpha < adversarial_alpha_floor:
        raise RuntimeError("VF10_ALPHA_CONTROL_CLASSES_NOT_SEPARABLE")

    selected_pm = 0.5 * (benign_pm + adversarial_pm)
    selected_alpha = 0.5 * (benign_alpha + adversarial_alpha_floor)

    out = {
        "schema": "RealSaS.VF10DirectionTransitionCalibration.v1",
        "status": "PASS_SUBJECT_FREE_SEPARATION",
        "subject_inputs_used": False,
        "knight_result_used": False,
        "mage_result_used": False,
        "raw_rgb_equality_required": False,
        "existing_pair_policy_held_fixed": True,
        "benign": benign,
        "adversarial_rgb": adversarial_rgb,
        "adversarial_alpha": adversarial_alpha,
        "selection": {
            "rule": "MIDPOINT(MAX_BENIGN,MIN_ADVERSARIAL)",
            "max_benign_pm_l1": benign_pm,
            "min_adversarial_pm_l1": adversarial_pm,
            "selected_max_triplet_p95_pm_l1": selected_pm,
            "max_benign_alpha_abs": benign_alpha,
            "min_adversarial_alpha_abs": adversarial_alpha_floor,
            "selected_max_triplet_p95_alpha_abs": selected_alpha,
            "minimum_shared_triplet_samples": int(
                policy["cross_view_min_shared_direct_samples_per_pair"]
            ),
            "minimum_component_samples": int(
                policy["cross_view_min_component_samples_for_gate"]
            ),
        },
    }
    path = OUT / "VF10_DIRECTION_TRANSITION_CALIBRATION.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
