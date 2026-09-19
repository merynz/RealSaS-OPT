from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.substrate.adequacy_v1 import (
    select_adequate_rigging_surface_v1,
)

OUT = Path(os.environ.get("REALSAS_CALIBRATION_OUT", "calibration_out"))
OUT.mkdir(parents=True, exist_ok=True)


def _write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mask_bytes(a: np.ndarray) -> bytes:
    return np.asarray(a, dtype=np.uint8).reshape(-1).tobytes()


def _ellipse(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0


def _disk(h: int, w: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _stage13_case_metrics(authority: np.ndarray, predicted: np.ndarray, floor: float) -> dict:
    h, w = authority.shape
    a = _mask_bytes(authority)
    p = _mask_bytes(predicted)
    cov = coverage_metrics(a, p, width=w, height=h)
    comp = source_connected_component_recall_metrics(
        a, p, width=w, height=h, minimum_foreground_fraction=float(floor)
    )
    return {
        **cov,
        "source_component_count": int(comp["source_component_count"]),
        "eligible_component_count": int(comp["eligible_component_count"]),
        "minimum_eligible_component_recall": float(comp["minimum_eligible_component_recall"]),
        "components": comp["components"],
    }


def _stage13_calibration() -> dict:
    h = w = 512
    body = _ellipse(h, w, 248.0, 267.0, 118.0, 176.0)
    head = _disk(h, w, 248.0, 103.0, 49.0)
    authority_base = body | head

    accessory = _disk(h, w, 405.0, 250.0, 16.0)
    authority_accessory = authority_base | accessory

    speck = np.zeros((h, w), dtype=bool)
    speck[32, 32] = True
    authority_speck = authority_base | speck

    hole = _disk(h, w, 248.0, 267.0, 13.0)

    cases = {
        "EXACT": (authority_base, authority_base, "ACCEPT"),
        "BOUNDARY_ERODE_1PX": (authority_base, binary_erosion(authority_base, iterations=1), "ACCEPT"),
        "BOUNDARY_DILATE_1PX": (authority_base, binary_dilation(authority_base, iterations=1), "ACCEPT"),
        "MISSING_DISCONNECTED_ACCESSORY": (authority_accessory, authority_base, "REJECT"),
        "COHERENT_INTERIOR_HOLE": (authority_base, authority_base & ~hole, "REJECT"),
        "ONE_PIXEL_SOURCE_SPECKLE": (authority_speck, authority_base, "DIAGNOSTIC"),
    }

    profiles = [
        {
            "profile_id": "PRODUCT_STRICT_INHERIT_G5",
            "provenance": "QUALIFIED_MESH_PRODUCT_POLICY_V1_20260918.MESH",
            "min_recall": 0.97,
            "min_precision": 0.995,
            "max_largest_coherent_hole_fraction": 0.005,
            "max_interior_uncovered_fraction": 0.005,
            "min_component_recall": 0.97,
            "component_min_foreground_fraction": 0.0,
        },
        {
            "profile_id": "IRIS_FIT_VISIBLE_GATE_PLUS_PRODUCT_ANTI_HOLE",
            "provenance": "IRIS prereg visible P/R + frozen G5 anti-hole; all source components eligible",
            "min_recall": 0.96,
            "min_precision": 0.94,
            "max_largest_coherent_hole_fraction": 0.005,
            "max_interior_uncovered_fraction": 0.005,
            "min_component_recall": 0.96,
            "component_min_foreground_fraction": 0.0,
        },
        {
            "profile_id": "IRIS_FIT_VISIBLE_GATE_PLUS_PRODUCT_ANTI_HOLE_NOISE_FLOOR",
            "provenance": "same as prior, with 0.1% source-foreground speckle floor",
            "min_recall": 0.96,
            "min_precision": 0.94,
            "max_largest_coherent_hole_fraction": 0.005,
            "max_interior_uncovered_fraction": 0.005,
            "min_component_recall": 0.96,
            "component_min_foreground_fraction": 0.001,
        },
    ]

    profile_eval = []
    for profile in profiles:
        per_case = {}
        expected_ok = True
        for name, (authority, predicted, expectation) in cases.items():
            m = _stage13_case_metrics(authority, predicted, profile["component_min_foreground_fraction"])
            passed = (
                m["recall"] >= profile["min_recall"]
                and m["precision"] >= profile["min_precision"]
                and m["largest_coherent_hole_fraction"] <= profile["max_largest_coherent_hole_fraction"]
                and m["interior_uncovered_fraction"] <= profile["max_interior_uncovered_fraction"]
                and m["minimum_eligible_component_recall"] >= profile["min_component_recall"]
            )
            if expectation == "ACCEPT" and not passed:
                expected_ok = False
            if expectation == "REJECT" and passed:
                expected_ok = False
            per_case[name] = {"expectation": expectation, "passed": bool(passed), "metrics": m}
        profile_eval.append({"profile": profile, "expected_classification_pass": bool(expected_ok), "cases": per_case})

    selected = next((row["profile"] for row in profile_eval if row["expected_classification_pass"]), None)
    return {
        "schema": "RealSaS.Stage13SubjectFreeCalibration.v1",
        "status": "PASS" if selected is not None else "FAIL_NO_PROFILE_SEPARATES_DECLARED_CASES",
        "subject_inputs_used": False,
        "knight_result_used": False,
        "resolution": [w, h],
        "selection_rule": "FIRST_STRICT_TO_LOOSE_PREREGISTERED_PROFILE_THAT_ACCEPTS_ALL_ACCEPT_CASES_AND_REJECTS_ALL_REJECT_CASES",
        "profiles": profile_eval,
        "selected_policy": selected,
    }


def _uv_sphere(n_lat: int, n_lon: int, scale=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0)):
    sx, sy, sz = map(float, scale)
    ox, oy, oz = map(float, offset)
    verts = [(ox, oy, oz + sz)]
    for i in range(1, n_lat):
        phi = math.pi * i / n_lat
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(n_lon):
            th = 2.0 * math.pi * j / n_lon
            verts.append((ox + sx * sp * math.cos(th), oy + sy * sp * math.sin(th), oz + sz * cp))
    south = len(verts)
    verts.append((ox, oy, oz - sz))
    faces = []
    for j in range(n_lon):
        a = 1 + j
        b = 1 + (j + 1) % n_lon
        faces.append((0, a, b))
    for i in range(1, n_lat - 1):
        r0 = 1 + (i - 1) * n_lon
        r1 = 1 + i * n_lon
        for j in range(n_lon):
            a = r0 + j
            b = r0 + (j + 1) % n_lon
            c = r1 + j
            d = r1 + (j + 1) % n_lon
            faces.append((a, c, b))
            faces.append((b, c, d))
    r = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        a = r + j
        b = r + (j + 1) % n_lon
        faces.append((a, south, b))
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    q = v - np.asarray(offset, dtype=np.float32)[None, :]
    n = np.column_stack((q[:, 0] / max(sx * sx, 1e-12), q[:, 1] / max(sy * sy, 1e-12), q[:, 2] / max(sz * sz, 1e-12)))
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(min=1e-12)
    return v, f, n.astype(np.float32)


def _combine_meshes(meshes):
    vs, fs, ns = [], [], []
    off = 0
    for v, f, n in meshes:
        vs.append(v)
        fs.append(f + off)
        ns.append(n)
        off += len(v)
    return np.concatenate(vs), np.concatenate(fs), np.concatenate(ns)


def _cams(res=512):
    rows = []
    for view in range(8):
        yaw = math.radians(45.0 * view)
        forward = np.asarray([-math.sin(yaw), -math.cos(yaw), 0.0])
        rows.append({
            "view_index": view,
            "origin": (-4.0 * forward).tolist(),
            "right": [-math.cos(yaw), math.sin(yaw), 0.0],
            "screen_up": [0.0, 0.0, 1.0],
            "forward": forward.tolist(),
            "half_extent": 1.25,
            "resolution": int(res),
        })
    return rows


def _stage14_calibration() -> dict:
    shapes = {
        "SPHERE": _uv_sphere(32, 64, scale=(0.72, 0.72, 0.72)),
        "TALL_THIN_ELLIPSOID": _uv_sphere(40, 64, scale=(0.22, 0.28, 0.92)),
        "FLAT_WIDE_ELLIPSOID": _uv_sphere(32, 72, scale=(0.92, 0.58, 0.18)),
        "TWO_COMPONENTS": _combine_meshes([
            _uv_sphere(30, 56, scale=(0.58, 0.48, 0.72), offset=(-0.18, 0.0, 0.0)),
            _uv_sphere(20, 40, scale=(0.17, 0.14, 0.22), offset=(0.72, 0.0, 0.15)),
        ]),
    }

    measurement_policy = {
        "min_candidate_nodes": 128,
        "max_candidate_nodes": 8192,
        "candidate_growth_factor": 2.0,
        "refinement_rounds": 0,
        "max_dense_to_surface_p95_norm": 10.0,
        "max_dense_to_surface_max_norm": 10.0,
        "max_normal_p95_deg": 180.0,
        "max_projected_p95_px": 1e9,
        "max_projected_max_px": 1e9,
        "component_min_dense_fraction": 0.001,
        "min_nodes_per_component": 1,
        "max_component_alias_nodes": 10**9,
    }

    profiles = [
        {
            "profile_id": "STRICT",
            "max_dense_to_surface_p95_norm": 0.03,
            "max_dense_to_surface_max_norm": 0.08,
            "max_normal_p95_deg": 30.0,
            "max_projected_p95_px": 6.0,
            "max_projected_max_px": 16.0,
            "component_min_dense_fraction": 0.001,
            "min_nodes_per_component": 8,
            "max_component_alias_nodes": 0,
        },
        {
            "profile_id": "BALANCED",
            "max_dense_to_surface_p95_norm": 0.05,
            "max_dense_to_surface_max_norm": 0.12,
            "max_normal_p95_deg": 45.0,
            "max_projected_p95_px": 8.0,
            "max_projected_max_px": 24.0,
            "component_min_dense_fraction": 0.001,
            "min_nodes_per_component": 8,
            "max_component_alias_nodes": 0,
        },
        {
            "profile_id": "RELAXED",
            "max_dense_to_surface_p95_norm": 0.08,
            "max_dense_to_surface_max_norm": 0.18,
            "max_normal_p95_deg": 60.0,
            "max_projected_p95_px": 12.0,
            "max_projected_max_px": 32.0,
            "component_min_dense_fraction": 0.001,
            "min_nodes_per_component": 8,
            "max_component_alias_nodes": 0,
        },
    ]

    raw = {}
    for name, mesh in shapes.items():
        v, f, n = mesh
        _, report = select_adequate_rigging_surface_v1(
            v, f, n, _cams(),
            normalization_center=(0.0, 0.0, 0.0),
            normalization_half_extent=1.0,
            authority_label="SUBJECT_FREE_CALIBRATION",
            source_run_id="SUBJECT_FREE_CALIBRATION",
            source_checkpoint_sha256="a" * 64,
            source_zero_surface_sha256=(name.encode().hex() + "0" * 64)[:64],
            normal_k=64,
            visibility_depth_tolerance_norm=0.02,
            adequacy_policy=measurement_policy,
            metadata={"calibration_shape": name, "subject_free": True},
        )
        raw[name] = report

    def metric_pass(m: dict, p: dict) -> bool:
        return (
            m["dense_to_surface_p95_norm"] <= p["max_dense_to_surface_p95_norm"]
            and m["dense_to_surface_max_norm"] <= p["max_dense_to_surface_max_norm"]
            and m["normal_p95_deg"] <= p["max_normal_p95_deg"]
            and m["projected_p95_px"] <= p["max_projected_p95_px"]
            and m["projected_max_px"] <= p["max_projected_max_px"]
            and m["minimum_nodes_per_eligible_component"] >= p["min_nodes_per_component"]
            and m["component_alias_node_count"] <= p["max_component_alias_nodes"]
        )

    profile_eval = []
    for p in profiles:
        shape_rows = {}
        all_shapes_have_pass = True
        for name, report in raw.items():
            passing = [m for m in report["evaluated_candidates"] if metric_pass(m, p)]
            if not passing:
                all_shapes_have_pass = False
                selected = None
            else:
                selected = min(passing, key=lambda m: (int(m["actual_node_count"]), int(m["candidate_target_node_cap"])))
            shape_rows[name] = {
                "has_passing_candidate": bool(passing),
                "selected_candidate": selected,
            }
        profile_eval.append({"profile": p, "all_shapes_have_passing_candidate": bool(all_shapes_have_pass), "shapes": shape_rows})

    selected_profile = next((row["profile"] for row in profile_eval if row["all_shapes_have_passing_candidate"]), None)
    selected_policy = None
    if selected_profile is not None:
        selected_policy = {
            "min_candidate_nodes": 128,
            "max_candidate_nodes": 8192,
            "candidate_growth_factor": 2.0,
            "refinement_rounds": 3,
            **{k: v for k, v in selected_profile.items() if k != "profile_id"},
        }

    return {
        "schema": "RealSaS.Stage14SubjectFreeCalibration.v1",
        "status": "PASS" if selected_policy is not None else "FAIL_NO_PROFILE_COVERS_ALL_SUBJECT_FREE_SHAPES",
        "subject_inputs_used": False,
        "knight_result_used": False,
        "normal_k": 64,
        "visibility_depth_tolerance_norm": 0.02,
        "search_caps": [128, 256, 512, 1024, 2048, 4096, 8192],
        "selection_rule": "FIRST_STRICT_TO_LOOSE_PREREGISTERED_TOLERANCE_PROFILE_WITH_AT_LEAST_ONE_PASSING_CANDIDATE_FOR_EVERY_SUBJECT_FREE_SHAPE",
        "profiles": profile_eval,
        "raw_measurement_reports": raw,
        "selected_profile": selected_profile,
        "selected_adequacy_policy": selected_policy,
    }


def main() -> int:
    s13 = _stage13_calibration()
    _write("STAGE13_SUBJECT_FREE_CALIBRATION_RESULT.json", s13)
    s14 = _stage14_calibration()
    _write("STAGE14_SUBJECT_FREE_CALIBRATION_RESULT.json", s14)
    summary = {
        "schema": "RealSaS.Stage13_14SubjectFreeCalibrationSummary.v1",
        "status": "PASS" if s13["status"] == "PASS" and s14["status"] == "PASS" else "FAIL",
        "github_sha": os.environ.get("GITHUB_SHA", ""),
        "subject_inputs_used": False,
        "knight_result_used": False,
        "stage13_selected_policy": s13.get("selected_policy"),
        "stage14_normal_k": s14.get("normal_k"),
        "stage14_visibility_depth_tolerance_norm": s14.get("visibility_depth_tolerance_norm"),
        "stage14_selected_adequacy_policy": s14.get("selected_adequacy_policy"),
    }
    _write("STAGE13_14_SUBJECT_FREE_CALIBRATION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
