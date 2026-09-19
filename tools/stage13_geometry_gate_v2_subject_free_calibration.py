from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.rest_preservation_v1 import (
    _boundary,
    _silhouette_distance,
)

OUT = Path(os.environ.get("REALSAS_CALIBRATION_OUT", "calibration_out"))
OUT.mkdir(parents=True, exist_ok=True)
H = W = 1024


def _write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mask_bytes(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.uint8).reshape(-1).tobytes()


def _ellipse(cx, cy, rx, ry):
    yy, xx = np.mgrid[0:H, 0:W]
    X = xx + 0.5
    Y = yy + 0.5
    return (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2) <= 1.0


def _disk(cx, cy, r):
    yy, xx = np.mgrid[0:H, 0:W]
    X = xx + 0.5
    Y = yy + 0.5
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r * r


def _rect(x0, y0, x1, y1):
    yy, xx = np.mgrid[0:H, 0:W]
    X = xx + 0.5
    Y = yy + 0.5
    return (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)


def broad_shape():
    return _ellipse(496.0, 535.0, 236.0, 352.0) | _disk(496.0, 205.0, 98.0)


def concave_shape():
    body = _ellipse(500.0, 540.0, 220.0, 300.0)
    left_arm = _rect(235.0, 365.0, 500.0, 415.0)
    right_arm = _rect(500.0, 365.0, 785.0, 415.0)
    cut = _rect(435.0, 420.0, 565.0, 620.0)
    return (body | left_arm | right_arm) & ~cut


def multicomponent_shape():
    main = broad_shape()
    accessory = _disk(835.0, 515.0, 20.0)
    thin = _rect(795.0, 300.0, 803.0, 450.0)
    return main | accessory | thin


def sparse_boundary_jitter(mask: np.ndarray, stride: int) -> np.ndarray:
    pred = mask.copy()
    b = _boundary(mask)
    coords = np.argwhere(b)
    chosen = coords[::stride]
    # Alternating isolated remove/add perturbations. Additions choose the first 4-neighbor background pixel.
    for k, (y, x) in enumerate(chosen):
        y = int(y); x = int(x)
        if k % 2 == 0:
            pred[y, x] = False
        else:
            for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                yy = y + dy; xx = x + dx
                if 0 <= yy < H and 0 <= xx < W and not mask[yy, xx]:
                    pred[yy, xx] = True
                    break
    return pred


def shift_x(mask: np.ndarray, px: int) -> np.ndarray:
    out = np.zeros_like(mask)
    if px > 0:
        out[:, px:] = mask[:, :-px]
    elif px < 0:
        out[:, :px] = mask[:, -px:]
    else:
        out[:] = mask
    return out


def metrics(authority: np.ndarray, predicted: np.ndarray) -> dict:
    cov = coverage_metrics(
        _mask_bytes(authority), _mask_bytes(predicted), width=W, height=H
    )
    comp = source_connected_component_recall_metrics(
        _mask_bytes(authority), _mask_bytes(predicted),
        width=W, height=H, minimum_foreground_fraction=0.0,
    )
    edge_mean, edge_p95, edge_max = _silhouette_distance(authority, predicted)
    return {
        "recall": float(cov["recall"]),
        "precision": float(cov["precision"]),
        "largest_coherent_hole_fraction": float(cov["largest_coherent_hole_fraction"]),
        "interior_uncovered_fraction": float(cov["interior_uncovered_fraction"]),
        "minimum_component_recall": float(comp["minimum_eligible_component_recall"]),
        "source_component_count": int(comp["source_component_count"]),
        "silhouette_edge_mean_px": float(edge_mean),
        "silhouette_edge_p95_px": float(edge_p95),
        "silhouette_edge_max_px": float(edge_max),
        "foreground_pixel_count": int(cov["foreground_pixel_count"]),
    }


def case_set():
    broad = broad_shape()
    concave = concave_shape()
    multi = multicomponent_shape()

    # Interior corruptions are deliberately source-internal, not boundary perturbations.
    hole = _disk(496.0, 535.0, 8.0)
    pepper = np.zeros_like(broad)
    interior = binary_erosion(broad, structure=np.ones((3,3), dtype=bool), iterations=2)
    coords = np.argwhere(interior)
    pepper_coords = coords[::500]  # ~0.2% sparse interior loss
    pepper[pepper_coords[:,0], pepper_coords[:,1]] = True

    missing_accessory = broad | _rect(795.0, 300.0, 803.0, 450.0)  # drops detached disk
    missing_thin = broad | _disk(835.0, 515.0, 20.0)               # drops thin detached bar

    notch = broad.copy()
    notch[470:500, 725:745] = False

    return [
        ("BROAD_IDENTITY", broad, broad, "ACCEPT"),
        ("CONCAVE_IDENTITY", concave, concave, "ACCEPT"),
        ("MULTI_IDENTITY", multi, multi, "ACCEPT"),
        ("BROAD_SPARSE_BOUNDARY_JITTER_1PCT", broad, sparse_boundary_jitter(broad, 100), "ACCEPT"),
        ("CONCAVE_SPARSE_BOUNDARY_JITTER_1PCT", concave, sparse_boundary_jitter(concave, 100), "ACCEPT"),

        ("BROAD_SHIFT_X_1PX", broad, shift_x(broad, 1), "REJECT"),
        ("BROAD_ERODE_CROSS_1PX", broad, binary_erosion(broad, iterations=1), "REJECT"),
        ("BROAD_DILATE_CROSS_1PX", broad, binary_dilation(broad, iterations=1), "REJECT"),
        ("BROAD_INTERIOR_HOLE_R8", broad, broad & ~hole, "REJECT"),
        ("BROAD_INTERIOR_PEPPER_0P2PCT", broad, broad & ~pepper, "REJECT"),
        ("BROAD_LOCAL_NOTCH_30X20", broad, notch, "REJECT"),
        ("MULTI_MISSING_ACCESSORY", multi, missing_accessory, "REJECT"),
        ("MULTI_MISSING_THIN_FEATURE", multi, missing_thin, "REJECT"),
    ]


PROFILES = [
    {
        "profile_id": "P999",
        "min_recall": 0.999,
        "min_precision": 0.999,
        "max_largest_coherent_hole_fraction": 0.00025,
        "max_interior_uncovered_fraction": 0.0005,
        "min_component_recall": 0.999,
        "max_silhouette_edge_p95_px": 0.5,
    },
    {
        "profile_id": "P9975",
        "min_recall": 0.9975,
        "min_precision": 0.9975,
        "max_largest_coherent_hole_fraction": 0.0005,
        "max_interior_uncovered_fraction": 0.001,
        "min_component_recall": 0.9975,
        "max_silhouette_edge_p95_px": 0.5,
    },
    {
        "profile_id": "P995",
        "min_recall": 0.995,
        "min_precision": 0.995,
        "max_largest_coherent_hole_fraction": 0.001,
        "max_interior_uncovered_fraction": 0.001,
        "min_component_recall": 0.995,
        "max_silhouette_edge_p95_px": 0.5,
    },
    {
        "profile_id": "P990",
        "min_recall": 0.99,
        "min_precision": 0.99,
        "max_largest_coherent_hole_fraction": 0.0025,
        "max_interior_uncovered_fraction": 0.0025,
        "min_component_recall": 0.99,
        "max_silhouette_edge_p95_px": 0.5,
    },
]


def passes(m: dict, p: dict) -> bool:
    return (
        m["recall"] >= p["min_recall"]
        and m["precision"] >= p["min_precision"]
        and m["largest_coherent_hole_fraction"] <= p["max_largest_coherent_hole_fraction"]
        and m["interior_uncovered_fraction"] <= p["max_interior_uncovered_fraction"]
        and m["minimum_component_recall"] >= p["min_component_recall"]
        and m["silhouette_edge_p95_px"] <= p["max_silhouette_edge_p95_px"]
    )


def main() -> int:
    rows = {}
    cases = case_set()
    for cid, authority, predicted, expectation in cases:
        rows[cid] = {
            "expectation": expectation,
            "metrics": metrics(authority, predicted),
        }

    evals = []
    for profile in PROFILES:
        ok = True
        classified = {}
        for cid, row in rows.items():
            passed = passes(row["metrics"], profile)
            exp = row["expectation"]
            if exp == "ACCEPT" and not passed:
                ok = False
            if exp == "REJECT" and passed:
                ok = False
            classified[cid] = passed
        evals.append({
            "profile": profile,
            "expected_classification_pass": bool(ok),
            "case_pass": classified,
        })

    selected = next((x["profile"] for x in evals if x["expected_classification_pass"]), None)
    result = {
        "schema": "RealSaS.Stage13GeometryGateV2SubjectFreeCalibration.v1",
        "status": "PASS" if selected else "FAIL_NO_PROFILE_SEPARATES_DECLARED_CONTROLS",
        "subject_inputs_used": False,
        "knight_result_used": False,
        "mage_result_used": False,
        "resolution": [W, H],
        "accept_semantics": "identity plus sparse isolated <=1px raster-boundary disagreement; no systematic silhouette displacement",
        "reject_semantics": "systematic 1px silhouette bias, visible interior loss, disconnected component loss, or thin-feature loss",
        "candidate_profiles_frozen_before_run": PROFILES,
        "selection_rule": "FIRST_STRICT_TO_LOOSE_PROFILE_THAT_ACCEPTS_ALL_ACCEPT_CASES_AND_REJECTS_ALL_REJECT_CASES",
        "selected_profile": selected,
        "cases": rows,
        "profile_evaluation": evals,
    }
    _write("STAGE13_GEOMETRY_GATE_V2_1024_CALIBRATION.json", result)
    print(json.dumps({
        "status": result["status"],
        "selected_profile": selected,
        "case_metrics": rows,
    }, indent=2, sort_keys=True))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
