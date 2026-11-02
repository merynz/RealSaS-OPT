"""Bound temporary deformation tensors without changing the measurement law.

Full vertex extent, all actions and global face adjacency remain authoritative.
Only independent face calculations are partitioned; no face/action is sampled.
"""
from __future__ import annotations

import numpy as np
from models.iris.truth import presentation_response_v1 as reference


def _face_response(rest, tri, posed, weights, tolerance):
    # The reference producer's face formulas, unchanged. Validation and global
    # scale are evaluated once by the caller, not once per face block.
    rest_tri = rest[tri]
    posed_tri = posed[:, tri, :]
    repeated_rest = np.broadcast_to(rest_tri[None, ...], posed_tri.shape)
    strain, area_ratio, valid = reference.triangle_metric_strain(
        repeated_rest,
        posed_tri,
        degeneracy_tolerance=tolerance,
    )
    # C = 2E + I.  Its eigenvalues are squared principal stretches in the
    # qualified rest tangent metric.
    cauchy_green = 2.0 * strain + np.eye(2, dtype=np.float64)
    eigval, eigvec = np.linalg.eigh(cauchy_green)
    eigval = np.clip(eigval, 0.0, None)
    stretch_minor_action = np.sqrt(eigval[..., 0])
    stretch_major_action = np.sqrt(eigval[..., 1])
    anisotropy_action = np.abs(stretch_major_action - stretch_minor_action)
    # Principal major eigenvector in the rest tangent basis.  2theta removes the
    # sign ambiguity of an unoriented principal axis.
    major_axis = eigvec[..., :, 1]
    theta = np.arctan2(major_axis[..., 1], major_axis[..., 0])
    tangent_2theta_action = np.stack((np.cos(2.0 * theta), np.sin(2.0 * theta)), axis=-1)

    w = weights[:, None]
    valid_f = valid.astype(np.float64)
    mass = np.sum(w * valid_f, axis=0)
    safe_mass = np.where(mass > 0, mass, 1.0)
    major_mean = np.sum(w * np.where(valid, stretch_major_action, 0.0), axis=0) / safe_mass
    minor_mean = np.sum(w * np.where(valid, stretch_minor_action, 0.0), axis=0) / safe_mass
    area_mean = np.sum(w * np.where(valid, area_ratio, 0.0), axis=0) / safe_mass
    anisotropy_mean = np.sum(w * np.where(valid, anisotropy_action, 0.0), axis=0) / safe_mass
    tangent_vector = np.sum(
        (w * valid_f * anisotropy_action)[..., None] * tangent_2theta_action,
        axis=0,
    )
    tangent_norm = np.linalg.norm(tangent_vector, axis=-1)
    tangent_valid = (mass > 0) & (anisotropy_mean > 1.0e-5) & (tangent_norm > 1.0e-8)
    tangent_mean = np.zeros((len(tri), 2), dtype=np.float64)
    tangent_mean[tangent_valid] = tangent_vector[tangent_valid] / tangent_norm[tangent_valid, None]

    strain_delta = np.stack(
        (stretch_major_action - 1.0, stretch_minor_action - 1.0), axis=-1
    )
    nonrigid = np.sqrt(
        np.sum(weights[:, None, None] * np.where(valid[..., None], strain_delta * strain_delta, 0.0), axis=(0, 2))
        / np.maximum(mass, 1.0e-12)
    )

    rest_centroid = rest_tri.mean(axis=1)
    posed_centroid = posed_tri.mean(axis=2)
    response = posed_centroid - rest_centroid[None, :, :]
    response_energy = np.sqrt(np.sum(weights[:, None, None] * response * response, axis=(0, 2)))

    face_valid = mass > 0
    return {
        "surface_face_valid": face_valid.astype(np.bool_),
        "surface_stretch_major_mean": np.where(face_valid, major_mean, 0.0),
        "surface_stretch_minor_mean": np.where(face_valid, minor_mean, 0.0),
        "surface_area_ratio_mean": np.where(face_valid, area_mean, 0.0),
        "surface_anisotropy_mean": np.where(face_valid, anisotropy_mean, 0.0),
        "surface_tangent_direction_2theta": tangent_mean,
        "surface_tangent_valid": tangent_valid.astype(np.bool_),
        "surface_nonrigid_response": np.where(face_valid, nonrigid, 0.0),
        "surface_response_energy": response_energy,
    }, response


def build_surface_response_arrays_v1(rest_vertices, faces, isolated_position, actions,
                                     *, degeneracy_fraction=1e-10, face_chunk_size=128):
    if type(face_chunk_size) is not int or face_chunk_size < 1:
        raise ValueError('SURFACE_RESPONSE_POSITIVE_CHUNK_REQUIRED')
    rest, tri = reference._mesh(rest_vertices, faces)
    posed = np.asarray(isolated_position, dtype=np.float64)
    if posed.ndim != 3 or posed.shape[1:] != rest.shape or not np.isfinite(posed).all():
        raise ValueError('PRESENTATION_RESPONSE_ISOLATED_POSITION_LAYOUT')
    if len(posed) != len(actions):
        raise ValueError('PRESENTATION_RESPONSE_ACTION_RESPONSE_CARDINALITY')
    weights = reference.control_balanced_action_weights_v1(actions)
    scale = reference._scale(rest)
    tolerance = float(degeneracy_fraction) * scale * scale
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('PRESENTATION_RESPONSE_DEGENERACY_POLICY')
    chunks = {}
    response = np.empty((len(posed), len(tri), 3), dtype=np.float64)
    for start in range(0, len(tri), face_chunk_size):
        stop = min(start + face_chunk_size, len(tri))
        block = tri[start:stop]
        measured, block_response = _face_response(rest, block, posed, weights, tolerance)
        for name, values in measured.items():
            if name.startswith('surface_'):
                chunks.setdefault(name, []).append(values)
        response[:, start:stop] = block_response
    result = {name: np.concatenate(values) for name, values in chunks.items()}
    adjacency = reference._face_adjacency(tri)
    coherence = np.zeros(len(adjacency), dtype=np.float64)
    jump = np.zeros(len(adjacency), dtype=np.float64)
    valid = np.zeros(len(adjacency), dtype=np.bool_)
    weighted = response * np.sqrt(weights)[:, None, None]
    flat = np.transpose(weighted, (1, 0, 2)).reshape(len(tri), -1)
    norm = np.linalg.norm(flat, axis=1)
    for index, (a, b) in enumerate(adjacency):
        denominator = float(norm[a] * norm[b])
        if denominator > 1e-12:
            value = float(np.clip(np.dot(flat[a], flat[b]) / denominator, -1., 1.))
            coherence[index], jump[index], valid[index] = value, .5 * (1. - value), True
    result.update(action_weight=weights, response_pair_face=adjacency,
                  response_pair_coherence=coherence, response_pair_jump=jump,
                  response_pair_valid=valid)
    return result
