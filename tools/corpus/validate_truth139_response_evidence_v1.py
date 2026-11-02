#!/usr/bin/env python3
"""Independent numeric court for Truth139 IRIS response evidence.

The court does not call the response producer.  It binds the raw Stage47 tensors,
checks reset/sham/action/gauge integrity, reconstructs the declared control-balanced
action measure, and independently recomputes sampled face response using a metric
formulation based on the rest Gram matrix.  Renderer/appearance evidence is not
part of this court.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PARENT_SCHEMA = "RealSaS.Truth139IrisResponseEvidence.v1"
PARENT_STATUS = "PASS_IRIS_RESPONSE_EVIDENCE_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
SCHEMA = "RealSaS.Truth139IrisResponseEvidenceCourt.v1"
PASS_STATUS = "PASS_IRIS_RESPONSE_EVIDENCE_INDEPENDENT_NUMERIC_COURT"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bound(ref: dict, code: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    if not path.is_file() or len(expected) != 64 or sha256(path) != expected:
        raise ValueError(code)
    return path


def action_weights(actions: list[dict]) -> tuple[np.ndarray, dict[int, list[int]]]:
    if not actions:
        raise ValueError("TRUTH139_RESPONSE_COURT_ACTIONS_REQUIRED")
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(actions):
        control = row.get("control_index_privileged")
        amplitude = row.get("amplitude")
        sign = row.get("sign")
        if type(control) is not int or control < 0:
            raise ValueError("TRUTH139_RESPONSE_COURT_CONTROL_INDEX")
        if not isinstance(amplitude, (int, float)) or not math.isfinite(float(amplitude)) or float(amplitude) <= 0:
            raise ValueError("TRUTH139_RESPONSE_COURT_AMPLITUDE")
        if sign not in (-1, 1):
            raise ValueError("TRUTH139_RESPONSE_COURT_SIGN")
        groups.setdefault(control, []).append(index)
    controls = sorted(groups)
    if controls != list(range(len(controls))):
        raise ValueError("TRUTH139_RESPONSE_COURT_CONTROL_SET")

    for control, indices in groups.items():
        by_amplitude: dict[float, set[int]] = {}
        for index in indices:
            row = actions[index]
            by_amplitude.setdefault(float(row["amplitude"]), set()).add(int(row["sign"]))
        if len(by_amplitude) < 2 or any(signs != {-1, 1} for signs in by_amplitude.values()):
            raise ValueError("TRUTH139_RESPONSE_COURT_PLUS_MINUS_COMPLETENESS:" + str(control))

    weights = np.zeros(len(actions), dtype=np.float64)
    control_mass = 1.0 / len(groups)
    for indices in groups.values():
        weights[indices] = control_mass / len(indices)
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-12, rtol=0.0):
        raise ValueError("TRUTH139_RESPONSE_COURT_ACTION_MASS")
    return weights, groups


def sample_indices(mask: np.ndarray, maximum: int) -> np.ndarray:
    valid = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
    if len(valid) == 0:
        raise ValueError("TRUTH139_RESPONSE_COURT_NO_VALID_FACES")
    if len(valid) <= maximum:
        return valid
    positions = np.linspace(0, len(valid) - 1, maximum, dtype=np.int64)
    return valid[positions]


def independent_face_response(
    rest: np.ndarray,
    faces: np.ndarray,
    isolated: np.ndarray,
    weights: np.ndarray,
    face_index: int,
    degeneracy_tolerance: float,
) -> tuple[float, float, bool]:
    tri = faces[int(face_index)]
    x = rest[tri]
    e0 = x[1] - x[0]
    e1 = x[2] - x[0]
    area2 = float(np.linalg.norm(np.cross(e0, e1)))
    if not math.isfinite(area2) or area2 <= degeneracy_tolerance:
        return 0.0, 0.0, False

    edge = np.stack((e0, e1), axis=1)
    gram0 = edge.T @ edge
    if not np.isfinite(gram0).all() or np.linalg.det(gram0) <= 0.0:
        return 0.0, 0.0, False
    chol = np.linalg.cholesky(gram0)
    inv_chol = np.linalg.inv(chol)

    strain_sq = 0.0
    valid_mass = 0.0
    centroid_sq = 0.0
    rest_centroid = x.mean(axis=0)
    for action_index, mass in enumerate(weights):
        y = isolated[action_index, tri]
        dy = np.stack((y[1] - y[0], y[2] - y[0]), axis=1)
        gram = dy.T @ dy
        # Generalized metric operator for G y = lambda G0 x, expressed in a
        # Cholesky-whitened rest basis. Its eigenvalues are squared principal stretches.
        c = inv_chol @ gram @ inv_chol.T
        eig = np.linalg.eigvalsh(0.5 * (c + c.T))
        if not np.isfinite(eig).all():
            raise ValueError("TRUTH139_RESPONSE_COURT_EIGEN_NONFINITE")
        eig = np.clip(eig, 0.0, None)
        stretch = np.sqrt(eig)
        strain_sq += float(mass) * float(np.sum((stretch - 1.0) ** 2))
        valid_mass += float(mass)
        delta = y.mean(axis=0) - rest_centroid
        centroid_sq += float(mass) * float(np.dot(delta, delta))
    if valid_mass <= 0.0:
        return 0.0, 0.0, False
    return math.sqrt(strain_sq / valid_mass), math.sqrt(centroid_sq), True


def central_derivative_diagnostic(actions: list[dict], isolated: np.ndarray, groups: dict[int, list[int]]) -> dict:
    relative = []
    support = []
    for control, indices in groups.items():
        pairs: dict[float, dict[int, int]] = {}
        for index in indices:
            row = actions[index]
            pairs.setdefault(float(row["amplitude"]), {})[int(row["sign"])] = index
        derivatives = []
        for amplitude in sorted(pairs):
            row = pairs[amplitude]
            plus = isolated[row[1]]
            minus = isolated[row[-1]]
            derivative = (plus - minus) / (2.0 * amplitude)
            derivatives.append(derivative)
        norms = [float(np.linalg.norm(value)) for value in derivatives]
        support.append({"control": int(control), "derivative_norms": norms})
        for a, b in zip(derivatives[:-1], derivatives[1:]):
            denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0e-12)
            relative.append(float(np.linalg.norm(a - b) / denom))
    return {
        "control_count": int(len(groups)),
        "adjacent_amplitude_pair_count": int(len(relative)),
        "median_adjacent_amplitude_derivative_relative_delta": float(np.median(relative)) if relative else 0.0,
        "maximum_adjacent_amplitude_derivative_relative_delta": float(max(relative, default=0.0)),
        "zero_derivative_control_count": int(sum(all(value <= 1.0e-12 for value in row["derivative_norms"]) for row in support)),
    }


def validate(manifest: dict, *, maximum_sampled_faces: int = 128) -> dict:
    if manifest.get("schema") != PARENT_SCHEMA or manifest.get("status") != PARENT_STATUS:
        raise ValueError("TRUTH139_RESPONSE_COURT_PARENT")
    subject_id = str(manifest.get("subject_id") or "")
    if not subject_id:
        raise ValueError("TRUTH139_RESPONSE_COURT_SUBJECT")
    source = dict(manifest.get("source") or {})
    scene_path = bound(dict(source.get("teacher_scene") or {}), "TRUTH139_RESPONSE_COURT_SCENE_DRIFT")
    geometry_path = bound(dict(source.get("geometry_and_response") or {}), "TRUTH139_RESPONSE_COURT_GEOMETRY_DRIFT")
    output_path = bound(dict(manifest.get("output") or {}), "TRUTH139_RESPONSE_COURT_OUTPUT_DRIFT")

    scene = json.loads(scene_path.read_text())
    if scene.get("schema") != "RealSaS.TeacherSceneIR.v2" or scene.get("subject_key") != subject_id:
        raise ValueError("TRUTH139_RESPONSE_COURT_SCENE_BINDING")
    if scene.get("rest_contract") != "EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY":
        raise ValueError("TRUTH139_RESPONSE_COURT_REST_CONTRACT")
    if (scene.get("observation_rest_binding") or {}).get("status") != "PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY":
        raise ValueError("TRUTH139_RESPONSE_COURT_REPLAY_BINDING")

    with np.load(geometry_path, allow_pickle=False) as z:
        required = {
            "rest_vertices_source_world", "faces", "pose_identity_baseline",
            "sham_before", "sham_after", "isolated_position", "root_control_position",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_RESPONSE_COURT_RAW_ARRAYS_MISSING")
        rest = np.asarray(z["rest_vertices_source_world"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        baseline = np.asarray(z["pose_identity_baseline"], dtype=np.float64)
        sham_before = np.asarray(z["sham_before"], dtype=np.float64)
        sham_after = np.asarray(z["sham_after"], dtype=np.float64)
        isolated = np.asarray(z["isolated_position"], dtype=np.float64)
        root = np.asarray(z["root_control_position"], dtype=np.float64)

    if rest.ndim != 2 or rest.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("TRUTH139_RESPONSE_COURT_MESH_LAYOUT")
    if not np.array_equal(rest, baseline):
        raise ValueError("TRUTH139_RESPONSE_COURT_POSE_IDENTITY_NOT_EXACT")
    if isolated.ndim != 3 or isolated.shape[1:] != rest.shape or root.shape != rest.shape:
        raise ValueError("TRUTH139_RESPONSE_COURT_RESPONSE_LAYOUT")
    if not all(np.isfinite(value).all() for value in (rest, baseline, sham_before, sham_after, isolated, root)):
        raise ValueError("TRUTH139_RESPONSE_COURT_NONFINITE")

    scale = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("TRUTH139_RESPONSE_COURT_SCALE")
    reset_tolerance = max(scale * 1.0e-10, 1.0e-12)
    sham_before_drift = float(np.max(np.abs(sham_before - baseline)))
    sham_after_drift = float(np.max(np.abs(sham_after - baseline)))
    if sham_before_drift > reset_tolerance or sham_after_drift > reset_tolerance:
        raise ValueError("TRUTH139_RESPONSE_COURT_SHAM_RESET_DRIFT")

    root_delta = root - baseline
    translation = np.mean(root_delta, axis=0)
    root_residual = float(np.max(np.linalg.norm(root_delta - translation[None, :], axis=1)))
    gauge_tolerance = max(scale * 1.0e-9, 1.0e-11)
    if root_residual > gauge_tolerance:
        raise ValueError("TRUTH139_RESPONSE_COURT_ROOT_GAUGE_NOT_COMMON_TRANSLATION")

    actions = list((scene.get("calibration") or {}).get("actions") or ())
    if len(actions) != isolated.shape[0]:
        raise ValueError("TRUTH139_RESPONSE_COURT_ACTION_CARDINALITY")
    weights, groups = action_weights(actions)
    derivative_diag = central_derivative_diagnostic(actions, isolated, groups)

    with np.load(output_path, allow_pickle=False) as z:
        required = {
            "action_weight", "surface_face_valid", "surface_nonrigid_response",
            "surface_response_energy", "response_pair_face", "response_pair_coherence",
            "response_pair_jump", "response_pair_valid",
        }
        if not required.issubset(set(z.files)):
            raise ValueError("TRUTH139_RESPONSE_COURT_OUTPUT_ARRAYS_MISSING")
        produced_weight = np.asarray(z["action_weight"], dtype=np.float64)
        face_valid = np.asarray(z["surface_face_valid"], dtype=np.bool_)
        nonrigid = np.asarray(z["surface_nonrigid_response"], dtype=np.float64)
        energy = np.asarray(z["surface_response_energy"], dtype=np.float64)
        pair = np.asarray(z["response_pair_face"], dtype=np.int64)
        pair_coherence = np.asarray(z["response_pair_coherence"], dtype=np.float64)
        pair_jump = np.asarray(z["response_pair_jump"], dtype=np.float64)
        pair_valid = np.asarray(z["response_pair_valid"], dtype=np.bool_)

    if produced_weight.shape != weights.shape or not np.allclose(produced_weight, weights, atol=1.0e-14, rtol=0.0):
        raise ValueError("TRUTH139_RESPONSE_COURT_ACTION_WEIGHT_DRIFT")
    if face_valid.shape != (len(faces),) or nonrigid.shape != face_valid.shape or energy.shape != face_valid.shape:
        raise ValueError("TRUTH139_RESPONSE_COURT_FACE_OUTPUT_LAYOUT")
    if pair.ndim != 2 or pair.shape[1:] != (2,) or pair_coherence.shape != (len(pair),) or pair_jump.shape != (len(pair),) or pair_valid.shape != (len(pair),):
        raise ValueError("TRUTH139_RESPONSE_COURT_PAIR_OUTPUT_LAYOUT")
    if not np.isfinite(nonrigid).all() or not np.isfinite(energy).all() or not np.isfinite(pair_coherence).all() or not np.isfinite(pair_jump).all():
        raise ValueError("TRUTH139_RESPONSE_COURT_OUTPUT_NONFINITE")

    sample = sample_indices(face_valid, int(maximum_sampled_faces))
    degeneracy_tolerance = 1.0e-10 * scale * scale
    max_nonrigid_error = 0.0
    max_energy_error = 0.0
    for face_index in sample:
        expected_nonrigid, expected_energy, valid = independent_face_response(
            rest, faces, isolated, weights, int(face_index), degeneracy_tolerance
        )
        if not valid:
            raise ValueError("TRUTH139_RESPONSE_COURT_SAMPLED_FACE_UNEXPECTED_INVALID")
        ne = abs(float(nonrigid[face_index]) - expected_nonrigid)
        ee = abs(float(energy[face_index]) - expected_energy)
        max_nonrigid_error = max(max_nonrigid_error, ne)
        max_energy_error = max(max_energy_error, ee)
        if ne > max(1.0e-8, abs(expected_nonrigid) * 1.0e-7):
            raise ValueError("TRUTH139_RESPONSE_COURT_NONRIGID_RECOMPUTE_MISMATCH")
        if ee > max(scale * 1.0e-9, abs(expected_energy) * 1.0e-7, 1.0e-10):
            raise ValueError("TRUTH139_RESPONSE_COURT_ENERGY_RECOMPUTE_MISMATCH")

    displacement = np.linalg.norm(isolated - baseline[None, :, :], axis=-1)
    maximum_displacement = float(np.max(displacement, initial=0.0))
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-30",
        "status": PASS_STATUS,
        "subject_id": subject_id,
        "response_evidence_identity_sha256": str(manifest.get("response_evidence_identity_sha256") or ""),
        "bindings": {
            "manifest_output_sha256": sha256(output_path),
            "teacher_scene_sha256": sha256(scene_path),
            "geometry_and_response_sha256": sha256(geometry_path),
        },
        "numeric_integrity": {
            "pose_identity_exact": True,
            "sham_before_max_abs_drift": sham_before_drift,
            "sham_after_max_abs_drift": sham_after_drift,
            "reset_tolerance": reset_tolerance,
            "root_translation_vector_source_units": translation.tolist(),
            "root_translation_max_residual": root_residual,
            "root_gauge_tolerance": gauge_tolerance,
            "maximum_isolated_vertex_displacement_source_units": maximum_displacement,
            "maximum_displacement_over_source_diagonal": maximum_displacement / scale,
            **derivative_diag,
        },
        "independent_recomputation": {
            "sampled_face_count": int(len(sample)),
            "maximum_nonrigid_abs_error": float(max_nonrigid_error),
            "maximum_response_energy_abs_error": float(max_energy_error),
            "producer_function_called_by_court": False,
            "metric_formulation": "REST_GRAM_CHOLESKY_GENERALIZED_PRINCIPAL_STRETCH",
        },
        "policy": {
            "renderer_measurement_used": False,
            "appearance_measurement_used": False,
            "semantic_part_labels_used": False,
            "student_likelihood_used": False,
            "student_t_is_training_objective_not_truth": True,
        },
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-sampled-faces", type=int, default=128)
    args = parser.parse_args()
    manifest = json.loads(args.response_manifest.resolve().read_text())
    result = validate(manifest, maximum_sampled_faces=args.maximum_sampled_faces)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "subject_id": result["subject_id"],
        "independent_recomputation": result["independent_recomputation"],
        "court_identity_sha256": result["court_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
