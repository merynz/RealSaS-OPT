"""Source-chart motion coefficients derived from the sealed canonical skin field.

Harmonic scalar coefficients transport the immutable canonical motion ownership,
not unrelated per-sample displacement vectors. All vertices then evaluate the
same canonical 2D pose palette. This keeps rotation coherent across chart cuts.
The field belongs to presentation; it neither infers nor modifies mechanical W.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .types import QualificationError
from .visual_domain_v2 import validate_domain_binding
from .visual_attachment_motion_v1 import POLICY as ATTACHMENT_POLICY, slot_rigid_transform_2d

OPERATOR_ID = "SOURCE_CHART_CANONICAL_2D_MOTION_BLEND_WITH_SLOT_RIGID_V4"
POLICY = {**ATTACHMENT_POLICY,
    "schema": "RealSaS.VisualDeformationOperatorPolicy.v4",
    "operator_id": OPERATOR_ID,
    "body_xy_contract": "HARMONIC_CANONICAL_SKIN_COEFFICIENTS__SHARED_2D_POSE_PALETTE",
    "body_depth_contract": "UNCHANGED_CANONICAL_SURFACE_HARMONIC_DEPTH",
    "mechanical_weight_inference_or_mutation_authorized": False,
    "maximum_canonical_pose_palette_residual": 1e-8,
}


def build_motion_blend_coefficients(binding, *, visual_faces, mechanical_weights):
    graph = validate_domain_binding(binding, visual_faces)
    w = np.asarray(mechanical_weights, dtype=np.float64)
    if (w.ndim != 2 or w.shape[0] != len(binding["mechanical_rest_xyz"]) or not w.shape[1]
            or not np.isfinite(w).all() or np.any(w < 0)
            or not np.allclose(w.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise QualificationError("VISUAL_MOTION_CANONICAL_COEFFICIENTS_INVALID")
    anchors = np.asarray(binding["anchor_vertex"], dtype=int)
    ancestry = np.asarray(binding["anchor_mechanical_vertices"], dtype=int)
    bary = np.asarray(binding["anchor_barycentric"])
    samples = np.sum(w[ancestry] * bary[:, :, None], axis=1)
    coefficients = np.zeros((graph.shape[0], w.shape[1]))
    coefficients[anchors] = samples
    free = np.setdiff1d(np.arange(graph.shape[0]), anchors)
    lap = diags(np.asarray(graph.sum(axis=1)).ravel()) - graph
    if len(free):
        coefficients[free] = np.asarray(spsolve(lap[free][:, free].tocsc(),
            -(lap[free][:, anchors] @ samples))).reshape(len(free), w.shape[1])
    if (not np.isfinite(coefficients).all() or np.min(coefficients) < -1e-12
            or not np.allclose(coefficients.sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise QualificationError("VISUAL_MOTION_HARMONIC_COEFFICIENTS_INVALID")
    # Only roundoff cleanup; no sparsification, dominance selection or skin fit.
    coefficients = np.maximum(coefficients, 0)
    coefficients /= coefficients.sum(axis=1, keepdims=True)
    return coefficients


def canonical_pose_palette_residual(*, rest_xyz, posed_xyz, mechanical_weights, skin_matrices_source):
    rest, posed = np.asarray(rest_xyz), np.asarray(posed_xyz)
    weights, matrices = np.asarray(mechanical_weights), np.asarray(skin_matrices_source)
    if (rest.shape != posed.shape or rest.ndim != 2 or rest.shape[1] != 3
            or weights.shape != (len(rest), len(matrices)) or matrices.shape[1:] != (4,4)
            or not all(np.isfinite(x).all() for x in (rest,posed,weights,matrices))):
        raise QualificationError("VISUAL_MOTION_CANONICAL_PALETTE_INVALID")
    moved = np.einsum('jab,nb->jna', matrices, np.c_[rest,np.ones(len(rest))], optimize=True)[:,:,:3]
    reconstructed = np.einsum('nj,jna->na',weights,moved,optimize=True)
    return float(np.max(np.abs(reconstructed-posed), initial=0))


def evaluate_motion_blend(field, *, rest_source_xy, coefficients,
                          axis_positions_source, skin_matrices_source, camera):
    result = np.asarray(field, dtype=np.float64).copy()
    rest, blend = np.asarray(rest_source_xy), np.asarray(coefficients)
    axis, matrices = np.asarray(axis_positions_source), np.asarray(skin_matrices_source)
    if (result.shape != (len(rest),3) or rest.shape != (len(rest),2)
            or blend.shape != (len(rest),len(axis)) or matrices.shape != (len(axis),4,4)
            or not all(np.isfinite(x).all() for x in (result,rest,blend,axis,matrices))
            or np.any(blend < 0) or not np.allclose(blend.sum(1),1,atol=1e-8,rtol=0)):
        raise QualificationError("VISUAL_MOTION_BLEND_FIELD_INVALID")
    result[:,:2] = 0
    for joint in np.flatnonzero(np.any(blend > 0,axis=0)):
        rotation, translation = slot_rigid_transform_2d(slot_rest_xyz=axis[joint],
            skin_matrix_source=matrices[joint],camera=camera)
        result[:,:2] += blend[:,joint,None] * (rest @ rotation.T + translation)
    return result
