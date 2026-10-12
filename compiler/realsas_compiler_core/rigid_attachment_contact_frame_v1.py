from __future__ import annotations

"""Generic rigid held-object transport from canonical surface contact.

This module deliberately contains no subject names, no prop class names and no
hand-authored offsets.  A rigid attachment is addressed by its canonical M
vertex ownership.  The holder motion authority is a source-derived carrier
contact patch, not a single skeletal slot, unless the two are independently
proven equivalent.

The setup M remains canonical.  At pose time the closest proper SE(3) motion of
the holder contact patch is applied to the attachment while an explicit
holder/held contact anchor gap is preserved exactly.
"""

from dataclasses import dataclass
import numpy as np

from .types import QualificationError

SCHEMA = "RealSaS.RigidAttachmentContactFrame.v1"
POLICY = {
    "schema": SCHEMA,
    # Contact bandwidth is expressed only in canonical mesh sampling units.
    # It is a research policy parameter and is not claimed generalized until
    # Fit8/LOFO/unseen qualification.
    "contact_band_edge_fraction": 0.25,
    "minimum_holder_patch_vertices": 3,
    "maximum_anchor_gap_drift": 1.0e-10,
    "maximum_rigidity_residual": 1.0e-10,
    "mechanical_topology_mutation_authorized": False,
    "mechanical_skin_mutation_authorized": False,
    "subject_specific_offsets_authorized": False,
}


@dataclass(frozen=True)
class ContactFrameBindingV1:
    attachment_index: int
    attachment_vertex_indices: np.ndarray
    holder_vertex_indices: np.ndarray
    held_anchor_vertex: int
    holder_anchor_vertex: int
    setup_anchor_gap: np.ndarray
    contact_band: float
    minimum_surface_gap: float
    median_edge_length: float
    schema_version: str = SCHEMA


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0):
        raise QualificationError("ATTACHMENT_CONTACT_FACE_DOMAIN_INVALID")
    e = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    return np.unique(np.sort(e, axis=1), axis=0)


def _nearest_indices(query: np.ndarray, reference: np.ndarray, *, chunk: int = 256):
    """Exact chunked nearest-neighbour search; no external NN dependency."""
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1:] != (3,) or r.shape[1:] != (3,):
        raise QualificationError("ATTACHMENT_CONTACT_POINT_DOMAIN_INVALID")
    if not len(q) or not len(r) or not np.isfinite(q).all() or not np.isfinite(r).all():
        raise QualificationError("ATTACHMENT_CONTACT_POINT_DOMAIN_INVALID")
    best_d2 = np.full(len(q), np.inf, dtype=np.float64)
    best_i = np.full(len(q), -1, dtype=np.int64)
    for start in range(0, len(r), int(chunk)):
        stop = min(start + int(chunk), len(r))
        d = q[:, None, :] - r[None, start:stop, :]
        d2 = np.einsum("qrd,qrd->qr", d, d)
        local = np.argmin(d2, axis=1)
        local_d2 = d2[np.arange(len(q)), local]
        take = local_d2 < best_d2
        best_d2[take] = local_d2[take]
        best_i[take] = start + local[take]
    return np.sqrt(best_d2), best_i


def derive_contact_frame_binding(
    *,
    rest_xyz,
    faces,
    attachment_vertex_owner,
    attachment_index: int,
    contact_band_edge_fraction: float | None = None,
) -> ContactFrameBindingV1:
    """Compile a generic holder/held surface binding from setup M only."""
    rest = np.asarray(rest_xyz, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    owners = np.asarray(attachment_vertex_owner)
    index = int(attachment_index)
    if (
        rest.ndim != 2
        or rest.shape[1] != 3
        or not np.isfinite(rest).all()
        or owners.shape != (len(rest),)
        or owners.dtype.kind not in "iu"
        or index <= 0
    ):
        raise QualificationError("ATTACHMENT_CONTACT_BINDING_INPUT_INVALID")
    held = np.flatnonzero(owners == index)
    holder = np.flatnonzero(owners == 0)
    if not len(held) or len(holder) < POLICY["minimum_holder_patch_vertices"]:
        raise QualificationError("ATTACHMENT_CONTACT_BINDING_DOMAIN_EMPTY")

    edges = _unique_edges(f)
    if np.any(edges >= len(rest)):
        raise QualificationError("ATTACHMENT_CONTACT_FACE_DOMAIN_INVALID")
    edge_length = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    median_edge = float(np.median(edge_length))
    if not np.isfinite(median_edge) or median_edge <= 1.0e-12:
        raise QualificationError("ATTACHMENT_CONTACT_MESH_SCALE_INVALID")

    distance, nearest_local = _nearest_indices(rest[held], rest[holder])
    minimum_gap = float(np.min(distance))
    fraction = (
        POLICY["contact_band_edge_fraction"]
        if contact_band_edge_fraction is None
        else float(contact_band_edge_fraction)
    )
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise QualificationError("ATTACHMENT_CONTACT_BAND_POLICY_INVALID")
    band = minimum_gap + fraction * median_edge
    selected = distance <= band
    holder_patch = np.unique(holder[nearest_local[selected]])
    if len(holder_patch) < POLICY["minimum_holder_patch_vertices"]:
        raise QualificationError("ATTACHMENT_CONTACT_PATCH_UNDERCONSTRAINED")

    anchor_row = int(np.argmin(distance))
    held_anchor = int(held[anchor_row])
    holder_anchor = int(holder[int(nearest_local[anchor_row])])
    gap = rest[held_anchor] - rest[holder_anchor]
    return ContactFrameBindingV1(
        attachment_index=index,
        attachment_vertex_indices=held.astype(np.int64),
        holder_vertex_indices=holder_patch.astype(np.int64),
        held_anchor_vertex=held_anchor,
        holder_anchor_vertex=holder_anchor,
        setup_anchor_gap=np.asarray(gap, dtype=np.float64),
        contact_band=float(band),
        minimum_surface_gap=minimum_gap,
        median_edge_length=median_edge,
    )


def closest_proper_rigid_transform(rest_points, posed_points):
    """Least-squares proper rigid transform B ~= R A + t."""
    a = np.asarray(rest_points, dtype=np.float64)
    b = np.asarray(posed_points, dtype=np.float64)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.shape[1] != 3
        or len(a) < 3
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise QualificationError("ATTACHMENT_CONTACT_FRAME_INPUT_INVALID")
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = cb - rotation @ ca
    residual = np.linalg.norm((a @ rotation.T + translation) - b, axis=1)
    return rotation, translation, residual


def apply_contact_frame_binding(
    posed_xyz,
    *,
    rest_xyz,
    binding: ContactFrameBindingV1,
):
    """Transport one rigid attachment by its holder contact frame.

    The Kabsch rotation is taken from the holder patch. Translation is then
    anchored so the canonical holder/held gap vector is preserved exactly under
    that rotation. This prevents an orientation fix from silently opening the
    grip/contact itself.
    """
    rest = np.asarray(rest_xyz, dtype=np.float64)
    posed = np.asarray(posed_xyz, dtype=np.float64)
    if posed.shape != rest.shape or rest.ndim != 2 or rest.shape[1] != 3:
        raise QualificationError("ATTACHMENT_CONTACT_POSE_DOMAIN_INVALID")
    if not np.isfinite(rest).all() or not np.isfinite(posed).all():
        raise QualificationError("ATTACHMENT_CONTACT_POSE_NONFINITE")

    holder = np.asarray(binding.holder_vertex_indices, dtype=np.int64)
    held = np.asarray(binding.attachment_vertex_indices, dtype=np.int64)
    if (
        np.any(holder < 0)
        or np.any(holder >= len(rest))
        or np.any(held < 0)
        or np.any(held >= len(rest))
    ):
        raise QualificationError("ATTACHMENT_CONTACT_BINDING_INDEX_INVALID")
    rotation, _, patch_residual = closest_proper_rigid_transform(rest[holder], posed[holder])
    hv = int(binding.holder_anchor_vertex)
    av = int(binding.held_anchor_vertex)
    target_anchor = posed[hv] + rotation @ np.asarray(binding.setup_anchor_gap)
    translation = target_anchor - rotation @ rest[av]

    out = posed.copy()
    out[held] = rest[held] @ rotation.T + translation
    anchor_gap = out[av] - posed[hv]
    anchor_gap_residual = float(
        np.linalg.norm(anchor_gap - rotation @ np.asarray(binding.setup_anchor_gap))
    )
    rigid_rest = rest[held] - rest[av]
    rigid_pose = out[held] - out[av]
    rigidity_residual = float(np.max(np.abs(rigid_pose - rigid_rest @ rotation.T), initial=0.0))
    return out, {
        "schema": SCHEMA,
        "attachment_index": int(binding.attachment_index),
        "holder_patch_vertex_count": int(len(holder)),
        "minimum_surface_gap": float(binding.minimum_surface_gap),
        "contact_band": float(binding.contact_band),
        "maximum_holder_patch_fit_residual": float(np.max(patch_residual, initial=0.0)),
        "rms_holder_patch_fit_residual": float(np.sqrt(np.mean(patch_residual * patch_residual))),
        "anchor_gap_residual": anchor_gap_residual,
        "rigidity_residual": rigidity_residual,
        "anchor_gap_pass": anchor_gap_residual <= POLICY["maximum_anchor_gap_drift"],
        "rigidity_pass": rigidity_residual <= POLICY["maximum_rigidity_residual"],
    }


def rotation_distance_degrees(a, b) -> float:
    """Proper-rotation distance for diagnostic slot-equivalence proofs."""
    ra = np.asarray(a, dtype=np.float64)
    rb = np.asarray(b, dtype=np.float64)
    if ra.shape != (3, 3) or rb.shape != (3, 3):
        raise QualificationError("ATTACHMENT_CONTACT_ROTATION_DOMAIN_INVALID")
    relative = ra @ rb.T
    value = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))
