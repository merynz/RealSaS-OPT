from __future__ import annotations

"""Generic view-conditioned rigid attachment pose from canonical surface anchors.

The canonical mechanical M is never mutated.  At setup time we compile the
closest held-surface <-> holder-surface pair as two face+barycentric addresses.
At presentation time the attachment uses the already qualified target-slot
proper 2D camera-twist rotation, while translation is solved from the posed
holder surface anchor.  This is the R20 diagnostic principle without a
vertex-nearest contact dependency.

Depth is deliberately untouched: the caller supplies projected M positions and
this operator changes only attachment XY.  Visibility/depth authority remains a
separate contract.
"""

from dataclasses import dataclass
import numpy as np

from .types import QualificationError

SCHEMA = "RealSaS.RigidAttachmentSurfaceAnchor.v1"
POLICY = {
    "schema": SCHEMA,
    "maximum_barycentric_sum_residual": 1.0e-10,
    "minimum_triangle_area_2": 1.0e-14,
    "maximum_anchor_transport_residual_px": 1.0e-9,
    "maximum_rigidity_residual_px": 1.0e-9,
    "mechanical_state_mutation_authorized": False,
    "depth_mutation_authorized": False,
    "subject_specific_offsets_authorized": False,
}


@dataclass(frozen=True)
class SurfaceAddressV1:
    face_index: int
    barycentric: tuple[float, float, float]


@dataclass(frozen=True)
class AttachmentSurfaceBindingV1:
    attachment_index: int
    attachment_vertex_indices: np.ndarray
    held: SurfaceAddressV1
    holder: SurfaceAddressV1
    setup_surface_gap: float
    schema_version: str = SCHEMA


def _validate_domain(rest_xyz, faces, owners):
    rest = np.asarray(rest_xyz, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    own = np.asarray(owners)
    if (
        rest.ndim != 2
        or rest.shape[1] != 3
        or not len(rest)
        or not np.isfinite(rest).all()
        or f.ndim != 2
        or f.shape[1] != 3
        or not len(f)
        or np.any(f < 0)
        or np.any(f >= len(rest))
        or own.shape != (len(rest),)
        or own.dtype.kind not in "iu"
        or np.any(own < 0)
    ):
        raise QualificationError("ATTACHMENT_SURFACE_DOMAIN_INVALID")
    tri = rest[f]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if np.any(area2 <= POLICY["minimum_triangle_area_2"]):
        raise QualificationError("ATTACHMENT_SURFACE_DEGENERATE_FACE")
    return rest, f, own


def _closest_point_barycentric(point, triangle):
    """Ericson closest-point regions; returns (point, barycentric)."""
    p = np.asarray(point, dtype=np.float64)
    a, b, c = np.asarray(triangle, dtype=np.float64)
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = float(ab @ ap), float(ac @ ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0])
    bp = p - b
    d3, d4 = float(ab @ bp), float(ac @ bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0])
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.array([1.0 - v, v, 0.0])
    cp = p - c
    d5, d6 = float(ab @ cp), float(ac @ cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0])
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.array([1.0 - w, 0.0, w])
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.array([0.0, 1.0 - w, w])
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + v * ab + w * ac, np.array([1.0 - v - w, v, w])


def _segment_segment(p1, q1, p2, q2):
    """Closest points on two segments plus their [0,1] parameters."""
    p1, q1, p2, q2 = [np.asarray(x, dtype=np.float64) for x in (p1, q1, p2, q2)]
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1.0e-15
    if a <= eps and e <= eps:
        s = t = 0.0
    elif a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = 0.0 if abs(denom) <= eps else float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return p1 + s * d1, p2 + t * d2, s, t


def _edge_bary(edge_index, t):
    rows = ((0, 1), (1, 2), (2, 0))
    i, j = rows[int(edge_index)]
    b = np.zeros(3, dtype=np.float64)
    b[i], b[j] = 1.0 - float(t), float(t)
    return b


def _triangle_pair_closest(a, b):
    """Exact closest pair on two non-degenerate triangles with barycentrics."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    best = None

    def take(pa, ba, pb, bb):
        nonlocal best
        d2 = float((pa - pb) @ (pa - pb))
        row = (d2, np.asarray(pa), np.asarray(ba), np.asarray(pb), np.asarray(bb))
        if best is None or d2 < best[0] - 1e-18:
            best = row

    for i in range(3):
        pb, bb = _closest_point_barycentric(a[i], b)
        ba = np.zeros(3); ba[i] = 1.0
        take(a[i], ba, pb, bb)
        pa, ba2 = _closest_point_barycentric(b[i], a)
        bb2 = np.zeros(3); bb2[i] = 1.0
        take(pa, ba2, b[i], bb2)
    edges = ((0, 1), (1, 2), (2, 0))
    for ea, (a0, a1) in enumerate(edges):
        for eb, (b0, b1) in enumerate(edges):
            pa, pb, sa, sb = _segment_segment(a[a0], a[a1], b[b0], b[b1])
            take(pa, _edge_bary(ea, sa), pb, _edge_bary(eb, sb))
    return best


def _aabb_distance(lower, upper, qlo, qhi):
    # Exact lower bound between axis-aligned boxes.
    delta = np.maximum(np.maximum(lower - qhi, qlo - upper), 0.0)
    return np.linalg.norm(delta, axis=1)


def derive_surface_binding(
    *, rest_xyz, faces, attachment_vertex_owner, attachment_index: int
) -> AttachmentSurfaceBindingV1:
    """Compile the global closest held/holder triangle-surface pair.

    Candidate pruning is exact: a holder face is skipped only when its AABB
    lower bound is already no better than the current exact surface distance.
    """
    rest, f, owners = _validate_domain(rest_xyz, faces, attachment_vertex_owner)
    idx = int(attachment_index)
    if idx <= 0 or not np.any(owners == idx):
        raise QualificationError("ATTACHMENT_SURFACE_OWNER_INVALID")
    held_vertices = np.flatnonzero(owners == idx)
    held_mask = np.all(owners[f] == idx, axis=1)
    holder_mask = np.all(owners[f] == 0, axis=1)
    held_faces = np.flatnonzero(held_mask)
    holder_faces = np.flatnonzero(holder_mask)
    if not len(held_faces) or not len(holder_faces):
        raise QualificationError("ATTACHMENT_SURFACE_FACE_DOMAIN_EMPTY")

    held_tri = rest[f[held_faces]]
    holder_tri = rest[f[holder_faces]]
    # Seed a finite exact bound with the closest centroid pair.
    hc = held_tri.mean(axis=1)
    bc = holder_tri.mean(axis=1)
    seed_best = (np.inf, None, None, None, None, None, None)
    for hs in range(0, len(hc), 128):
        q = hc[hs : hs + 128, None, :] - bc[None, :, :]
        d2 = np.einsum("hbd,hbd->hb", q, q)
        loc = int(np.argmin(d2))
        i, j = np.unravel_index(loc, d2.shape)
        if float(d2[i, j]) < seed_best[0]:
            seed_best = (float(d2[i, j]), hs + i, j, None, None, None, None)
    _, hi0, bi0, *_ = seed_best
    exact = _triangle_pair_closest(held_tri[hi0], holder_tri[bi0])
    best_d2, pa, ba, pb, bb = exact
    best_hi, best_bi = int(hi0), int(bi0)
    best = float(np.sqrt(best_d2))

    blo = holder_tri.min(axis=1)
    bhi = holder_tri.max(axis=1)
    for hi, tri in enumerate(held_tri):
        qlo, qhi = tri.min(axis=0), tri.max(axis=0)
        lower = _aabb_distance(blo, bhi, qlo, qhi)
        candidates = np.flatnonzero(lower <= best + 1e-14)
        for bi in candidates:
            row = _triangle_pair_closest(tri, holder_tri[bi])
            d = float(np.sqrt(row[0]))
            if d < best - 1e-14:
                best = d
                _, pa, ba, pb, bb = row
                best_hi, best_bi = hi, int(bi)

    for bary in (ba, bb):
        if (
            np.any(bary < -1e-10)
            or np.any(bary > 1.0 + 1e-10)
            or abs(float(np.sum(bary)) - 1.0) > POLICY["maximum_barycentric_sum_residual"]
        ):
            raise QualificationError("ATTACHMENT_SURFACE_BARYCENTRIC_INVALID")
    # Independent replay of both addresses before sealing the binding.
    held_point = ba @ rest[f[held_faces[best_hi]]]
    holder_point = bb @ rest[f[holder_faces[best_bi]]]
    replay = float(np.linalg.norm(held_point - holder_point))
    if abs(replay - best) > 1e-10:
        raise QualificationError("ATTACHMENT_SURFACE_ADDRESS_REPLAY_DRIFT")
    return AttachmentSurfaceBindingV1(
        attachment_index=idx,
        attachment_vertex_indices=held_vertices.astype(np.int64),
        held=SurfaceAddressV1(int(held_faces[best_hi]), tuple(map(float, ba))),
        holder=SurfaceAddressV1(int(holder_faces[best_bi]), tuple(map(float, bb))),
        setup_surface_gap=best,
    )


def _address_point(positions, faces, address: SurfaceAddressV1):
    p = np.asarray(positions, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fi = int(address.face_index)
    bary = np.asarray(address.barycentric, dtype=np.float64)
    if fi < 0 or fi >= len(f) or bary.shape != (3,) or not np.isfinite(bary).all():
        raise QualificationError("ATTACHMENT_SURFACE_ADDRESS_INVALID")
    return bary @ p[f[fi]]


def apply_anchored_camera_twist_2d(
    projected_posed,
    *,
    projected_rest,
    faces,
    binding: AttachmentSurfaceBindingV1,
    rotation_2d,
):
    """Apply R20 as a pure XY presentation transform on one camera/view.

    ``projected_*`` are Nx3 camera projections. Z is copied exactly from the raw
    posed M projection.  The same proper slot rotation is applied to every held
    vertex; translation comes only from the qualified holder surface address.
    """
    rest = np.asarray(projected_rest, dtype=np.float64)
    posed = np.asarray(projected_posed, dtype=np.float64)
    R = np.asarray(rotation_2d, dtype=np.float64)
    if (
        rest.shape != posed.shape
        or rest.ndim != 2
        or rest.shape[1] != 3
        or not np.isfinite(rest).all()
        or not np.isfinite(posed).all()
        or R.shape != (2, 2)
        or not np.isfinite(R).all()
        or np.max(np.abs(R.T @ R - np.eye(2))) > 1e-8
        or abs(float(np.linalg.det(R)) - 1.0) > 1e-8
    ):
        raise QualificationError("ATTACHMENT_ANCHORED_2D_INPUT_INVALID")
    held_rest = _address_point(rest, faces, binding.held)[:2]
    holder_rest = _address_point(rest, faces, binding.holder)[:2]
    holder_pose = _address_point(posed, faces, binding.holder)[:2]
    setup_gap = held_rest - holder_rest
    target_held = holder_pose + R @ setup_gap
    translation = target_held - R @ held_rest

    out = posed.copy()
    ids = np.asarray(binding.attachment_vertex_indices, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= len(rest)):
        raise QualificationError("ATTACHMENT_SURFACE_VERTEX_INDEX_INVALID")
    out[ids, :2] = rest[ids, :2] @ R.T + translation
    # Explicitly prove depth was not touched.
    if not np.array_equal(out[:, 2], posed[:, 2]):
        raise QualificationError("ATTACHMENT_ANCHORED_2D_DEPTH_MUTATION")

    held_out = _address_point(out, faces, binding.held)[:2]
    anchor_residual = float(np.linalg.norm((held_out - holder_pose) - R @ setup_gap))
    local_rest = rest[ids, :2] - held_rest
    local_out = out[ids, :2] - held_out
    rigidity_residual = float(np.max(np.abs(local_out - local_rest @ R.T), initial=0.0))
    proof = {
        "schema": SCHEMA,
        "attachment_index": int(binding.attachment_index),
        "held_face_index": int(binding.held.face_index),
        "holder_face_index": int(binding.holder.face_index),
        "setup_surface_gap_world": float(binding.setup_surface_gap),
        "setup_anchor_gap_px": float(np.linalg.norm(setup_gap)),
        "anchor_transport_residual_px": anchor_residual,
        "rigidity_residual_px": rigidity_residual,
        "anchor_transport_pass": anchor_residual <= POLICY["maximum_anchor_transport_residual_px"],
        "rigidity_pass": rigidity_residual <= POLICY["maximum_rigidity_residual_px"],
        "depth_unchanged": True,
    }
    return out, proof
