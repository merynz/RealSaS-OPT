"""Independent edit pixel-consequence predicate; NO source renderer import.

Observed RGBA images and independently source-ray-qualified interior pixel samples.
This verifies a declared edit workload only, not intrinsic drawable ownership.
"""
from __future__ import annotations
import numpy as np


def judge_edit_rgba(
    baseline, edited, samples, edit_color_index, *,
    alpha_tolerance=2.0 / 255.0,
    collateral_rgb_tolerance=3.0 / 255.0,
    min_selected_rgb_delta=8.0 / 255.0,
    min_selected_samples=8,
    min_unselected_samples=8,
):
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(edited, dtype=np.float64)
    if (a.ndim != 3 or a.shape[-1] != 4 or a.shape != b.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("PHYSICAL_PIXEL_CONSEQUENCE_LAYOUT_INVALID")
    if type(edit_color_index) is not int or edit_color_index < 0:
        raise ValueError("PHYSICAL_PIXEL_CONSEQUENCE_EDIT_SELECTOR_INVALID")
    max_alpha = float(np.max(np.abs(a[:, :, 3] - b[:, :, 3])))
    if max_alpha > alpha_tolerance:
        raise ValueError("PHYSICAL_PIXEL_CONSEQUENCE_ALPHA_CHANGED")
    selected = unselected = selected_unchanged = collateral = 0
    for s in samples:
        x, y, idx = int(s["x"]), int(s["y"]), int(s["color_index"])
        if not (0 <= x < a.shape[1] and 0 <= y < a.shape[0]):
            raise ValueError("PHYSICAL_PIXEL_CONSEQUENCE_SAMPLE_OUTSIDE")
        if idx < 0:
            raise ValueError("PHYSICAL_PIXEL_CONSEQUENCE_UNKNOWN_SAMPLE_ADMITTED")
        diff = float(np.max(np.abs(a[y, x, :3] - b[y, x, :3])))
        if idx == edit_color_index:
            selected += 1
            if diff < min_selected_rgb_delta:
                selected_unchanged += 1
        else:
            unselected += 1
            if diff > collateral_rgb_tolerance:
                collateral += 1
    enough = selected >= min_selected_samples and unselected >= min_unselected_samples
    return {
        "status": "INSUFFICIENT_INTERIOR_SAMPLES" if not enough else (
            "PASS_PIXEL_LOCALITY" if collateral == 0 and selected_unchanged == 0
            else "FAIL_PIXEL_LOCALITY"),
        "selected_count": selected,
        "unselected_count": unselected,
        "collateral_count": collateral,
        "selected_unchanged_count": selected_unchanged,
        "alpha_max_difference": max_alpha,
        "qualified": bool(enough and collateral == 0 and selected_unchanged == 0),
    }
