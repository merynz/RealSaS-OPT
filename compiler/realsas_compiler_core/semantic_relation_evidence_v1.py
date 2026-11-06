from __future__ import annotations

"""Lift frozen image semantics onto canonical M and derive relation evidence.

No new geometry carrier is introduced.  This module reads source-view semantic
mask proposals and canonical M, then emits abstaining evidence for Compiler
qualification.  It never mutates IRIS geometry, M, rig, skin or motion.
"""

from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np

from models.iris.semantic_evidence_v1 import FrozenSemanticMaskEvidenceV1
from .camera_geometry_v2 import project_points_xyz_v3


SCHEMA = "RealSaS.SemanticRelationEvidence.v1"


def _label(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").split())


@dataclass(frozen=True)
class MeshSemanticEvidenceV1:
    label_vocab: tuple[str, ...]
    face_probabilities: np.ndarray
    face_confidence: np.ndarray
    face_support_views: np.ndarray
    face_label_index: np.ndarray
    face_owner_index: np.ndarray
    owner_ids: tuple[str, ...]
    owner_label_index: np.ndarray
    owner_face_indices: tuple[tuple[int, ...], ...]
    source_evidence_hash: str
    schema_version: str = SCHEMA

    def __post_init__(self) -> None:
        p = np.asarray(self.face_probabilities)
        c = np.asarray(self.face_confidence)
        sv = np.asarray(self.face_support_views)
        li = np.asarray(self.face_label_index)
        oi = np.asarray(self.face_owner_index)
        n = len(p)
        k = len(self.label_vocab)
        if p.shape != (n, k) or c.shape != (n,) or sv.shape != (n, 8):
            raise ValueError("mesh semantic evidence shape mismatch")
        if li.shape != (n,) or oi.shape != (n,):
            raise ValueError("mesh semantic label/owner shape mismatch")
        if not np.isfinite(p).all() or not np.isfinite(c).all():
            raise ValueError("mesh semantic evidence must be finite")
        if np.any(p < 0) or np.any(p > 1) or np.any(c < 0) or np.any(c > 1):
            raise ValueError("mesh semantic probabilities/confidence outside [0,1]")
        if len(self.owner_ids) != len(self.owner_label_index) or len(self.owner_ids) != len(self.owner_face_indices):
            raise ValueError("owner metadata shape mismatch")


@dataclass(frozen=True)
class RelationEvidenceV1:
    relation_kind: str
    a_owner_id: str
    b_owner_id: str
    confidence: float
    support_count: int
    view_index: int | None = None
    a_face_index: int | None = None
    b_face_index: int | None = None
    normalized_distance: float | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class BehaviorEvidenceV1:
    owner_id: str
    semantic_label: str
    behavior_class: str
    confidence: float
    metadata: dict | None = None


def _face_visibility_samples(rest, faces, cameras, *, image_shape):
    """Return visible face-centroid pixels for each source view.

    This is intentionally conservative: a centroid only contributes when its
    projected depth agrees with the canonical opaque first-hit depth at that
    source pixel.  Hidden faces simply abstain in that view.
    """
    rest = np.asarray(rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    centroids = rest[faces].mean(axis=1)
    h, w = map(int, image_shape)
    out = []
    for view, camera in enumerate(cameras):
        projected_vertices = project_points_xyz_v3(rest, camera)
        projected_centroids = project_points_xyz_v3(centroids, camera)
        owner, depth = _rasterize_first_hit(
            projected_vertices,
            faces,
            source_resolution=int(camera.resolution),
        )
        xy = np.floor(projected_centroids[:, :2]).astype(int)
        in_bounds = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < camera.resolution)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < camera.resolution)
        )
        clipped = np.clip(xy, 0, camera.resolution - 1)
        ix, iy = clipped.T
        eps = 2.0 * float(camera.half_extent) / float(camera.resolution)
        visible = in_bounds & (owner[iy, ix] >= 0) & (
            np.abs(projected_centroids[:, 2] - depth[iy, ix]) <= eps
        )
        sx = np.clip(np.floor(projected_centroids[:, 0] * w / camera.resolution), 0, w - 1).astype(int)
        sy = np.clip(np.floor(projected_centroids[:, 1] * h / camera.resolution), 0, h - 1).astype(int)
        out.append((sx, sy, visible))
    return out


def lift_mask_evidence_to_m_v1(
    *,
    rest,
    faces,
    cameras,
    evidence: FrozenSemanticMaskEvidenceV1,
    minimum_face_confidence: float = 0.45,
    minimum_support_views: int = 1,
) -> MeshSemanticEvidenceV1:
    """Fuse modal 2D proposals into abstaining semantic evidence on M faces."""
    if not 0 <= minimum_face_confidence <= 1:
        raise ValueError("minimum_face_confidence outside [0,1]")
    rest = np.asarray(rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if rest.ndim != 2 or rest.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("invalid M domain")
    fcount = len(faces)
    k = len(evidence.labels)
    votes = np.zeros((fcount, k), dtype=np.float64)
    weight = np.zeros(fcount, dtype=np.float64)
    support = np.zeros((fcount, 8), dtype=bool)
    samples = _face_visibility_samples(
        rest, faces, cameras, image_shape=(evidence.height, evidence.width)
    )
    by_view = [np.where(evidence.proposal_view_index == v)[0] for v in range(8)]
    for view in range(8):
        sx, sy, visible = samples[view]
        if not np.any(visible):
            continue
        for pi in by_view[view]:
            hit = visible & evidence.proposal_masks[pi, sy, sx]
            if not np.any(hit):
                continue
            score = float(evidence.proposal_score[pi])
            label_index = int(evidence.proposal_label_index[pi])
            votes[hit, label_index] += score
            weight[hit] += score
            support[hit, view] = True
    probs = np.divide(votes, weight[:, None], out=np.zeros_like(votes), where=weight[:, None] > 0)
    confidence = probs.max(axis=1) if k else np.zeros(fcount)
    label_index = np.where(
        (confidence >= minimum_face_confidence)
        & (support.sum(axis=1) >= int(minimum_support_views)),
        probs.argmax(axis=1),
        -1,
    ).astype(np.int16)
    owner_index, owner_ids, owner_labels, owner_faces = _connected_semantic_owners(
        faces, label_index, evidence.labels
    )
    return MeshSemanticEvidenceV1(
        label_vocab=evidence.labels,
        face_probabilities=probs.astype(np.float32),
        face_confidence=confidence.astype(np.float32),
        face_support_views=support,
        face_label_index=label_index,
        face_owner_index=owner_index,
        owner_ids=owner_ids,
        owner_label_index=owner_labels,
        owner_face_indices=owner_faces,
        source_evidence_hash=evidence.evidence_hash(),
    )


def _connected_semantic_owners(faces, label_index, labels):
    faces = np.asarray(faces, dtype=np.int64)
    n = len(faces)
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    edge_owner = {}
    for fi, tri in enumerate(faces):
        if label_index[fi] < 0:
            continue
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = tuple(sorted((int(a), int(b))))
            other = edge_owner.get(key)
            if other is not None and label_index[other] == label_index[fi]:
                union(fi, other)
            else:
                edge_owner[key] = fi
    groups = {}
    for fi in range(n):
        if label_index[fi] >= 0:
            groups.setdefault((int(label_index[fi]), find(fi)), []).append(fi)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0][0], min(kv[1])))
    owner_index = np.full(n, -1, dtype=np.int32)
    ids, owner_labels, owner_faces = [], [], []
    class_count = {}
    for oi, ((li, _), fis) in enumerate(ordered):
        label = labels[li]
        ordinal = class_count.get(label, 0)
        class_count[label] = ordinal + 1
        oid = f"OWNER:{label}:{ordinal:02d}"
        ids.append(oid)
        owner_labels.append(li)
        owner_faces.append(tuple(map(int, fis)))
        owner_index[np.asarray(fis, dtype=int)] = oi
    return owner_index, tuple(ids), np.asarray(owner_labels, dtype=np.int16), tuple(owner_faces)


def infer_behavior_evidence_v1(mesh: MeshSemanticEvidenceV1) -> tuple[BehaviorEvidenceV1, ...]:
    """Class-to-behavior priors only; these are not motion authority."""
    rigid = {"sword", "weapon", "shield", "armor", "armour", "helmet", "prop", "staff", "gun"}
    articulated = {"body", "torso", "head", "arm", "forearm", "hand", "leg", "foot", "neck"}
    secondary = {"cape", "cloak", "hair", "cloth", "scarf", "skirt", "tail", "tassel", "ribbon"}
    rows = []
    for oi, oid in enumerate(mesh.owner_ids):
        label = mesh.label_vocab[int(mesh.owner_label_index[oi])]
        if label in rigid:
            behavior = "RIGID"
        elif label in secondary:
            behavior = "SECONDARY_DEFORMABLE"
        elif label in articulated:
            behavior = "PRIMARY_ARTICULATED"
        else:
            behavior = "UNKNOWN"
        fis = np.asarray(mesh.owner_face_indices[oi], dtype=int)
        conf = float(np.mean(mesh.face_confidence[fis])) if len(fis) else 0.0
        rows.append(BehaviorEvidenceV1(oid, label, behavior, conf, {"source": "SEMANTIC_CLASS_PRIOR"}))
    return tuple(rows)


def infer_grip_candidates_v1(
    *,
    rest,
    faces,
    mesh: MeshSemanticEvidenceV1,
    maximum_normalized_distance: float = 0.08,
) -> tuple[RelationEvidenceV1, ...]:
    """Propose HAND->held relations from semantic identity plus M proximity."""
    rest = np.asarray(rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    span = float(np.linalg.norm(rest.max(axis=0) - rest.min(axis=0)))
    if span <= 1e-12:
        raise ValueError("degenerate M span")
    holder_labels = {"hand", "grip", "holder"}
    held_labels = {"sword", "weapon", "shield", "prop", "staff", "gun"}
    owner_vertices = []
    for fis in mesh.owner_face_indices:
        owner_vertices.append(np.unique(faces[np.asarray(fis, dtype=int)].ravel()))
    rows = []
    for ai, aid in enumerate(mesh.owner_ids):
        alabel = mesh.label_vocab[int(mesh.owner_label_index[ai])]
        if alabel not in holder_labels:
            continue
        ap = rest[owner_vertices[ai]]
        for bi, bid in enumerate(mesh.owner_ids):
            blabel = mesh.label_vocab[int(mesh.owner_label_index[bi])]
            if blabel not in held_labels:
                continue
            bp = rest[owner_vertices[bi]]
            d = np.linalg.norm(ap[:, None, :] - bp[None, :, :], axis=2)
            arow, brow = np.unravel_index(int(np.argmin(d)), d.shape)
            nd = float(d[arow, brow] / span)
            if nd > maximum_normalized_distance:
                continue
            afaces = np.asarray(mesh.owner_face_indices[ai], dtype=int)
            bfaces = np.asarray(mesh.owner_face_indices[bi], dtype=int)
            sem_conf = float(
                np.sqrt(np.mean(mesh.face_confidence[afaces]) * np.mean(mesh.face_confidence[bfaces]))
            )
            proximity = exp(-((nd / max(maximum_normalized_distance, 1e-9)) ** 2))
            confidence = float(np.clip(sem_conf * proximity, 0, 1))
            av = int(owner_vertices[ai][arow])
            bv = int(owner_vertices[bi][brow])
            a_face = next(int(fi) for fi in afaces if av in faces[fi])
            b_face = next(int(fi) for fi in bfaces if bv in faces[fi])
            rows.append(
                RelationEvidenceV1(
                    "GRIP_CANDIDATE",
                    aid,
                    bid,
                    confidence,
                    1,
                    a_face_index=a_face,
                    b_face_index=b_face,
                    normalized_distance=nd,
                    metadata={"holder_label": alabel, "held_label": blabel},
                )
            )
    return tuple(sorted(rows, key=lambda x: (-x.confidence, x.a_owner_id, x.b_owner_id)))


def infer_source_overlap_relations_v1(
    *,
    rest,
    faces,
    cameras,
    evidence: FrozenSemanticMaskEvidenceV1,
    mesh: MeshSemanticEvidenceV1,
    resolution: int = 256,
    minimum_support_pixels: int = 4,
) -> tuple[RelationEvidenceV1, ...]:
    """Derive source-observed FRONT_OF evidence from M layers + modal semantics.

    A vote is counted only when the externally observed semantic label at a pixel
    agrees with the nearest M owner's semantic label.  Therefore plain M depth is
    not self-certified as semantic truth.
    """
    rest = np.asarray(rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    visible_label = _visible_external_label_maps(evidence)
    accum: dict[tuple[int, int, int], list[float]] = {}
    for view, camera in enumerate(cameras):
        projected = project_points_xyz_v3(rest, camera)
        layers = _rasterize_owner_layers(projected, faces, resolution=resolution, source_resolution=camera.resolution)
        ext = visible_label[view]
        eh, ew = ext.shape
        for pixel, layer_faces in layers.items():
            if len(layer_faces) < 2:
                continue
            y, x = divmod(pixel, resolution)
            ex = min(ew - 1, int((x + 0.5) * ew / resolution))
            ey = min(eh - 1, int((y + 0.5) * eh / resolution))
            ext_label = int(ext[ey, ex])
            if ext_label < 0:
                continue
            front_face = int(layer_faces[0][1])
            front_owner = int(mesh.face_owner_index[front_face])
            if front_owner < 0 or int(mesh.owner_label_index[front_owner]) != ext_label:
                continue
            back_owner = -1
            back_face = -1
            for _, fi in layer_faces[1:]:
                oi = int(mesh.face_owner_index[int(fi)])
                if oi >= 0 and oi != front_owner:
                    back_owner, back_face = oi, int(fi)
                    break
            if back_owner < 0:
                continue
            conf = float(min(mesh.face_confidence[front_face], mesh.face_confidence[back_face]))
            accum.setdefault((view, front_owner, back_owner), []).append(conf)
    rows = []
    for (view, a, b), scores in sorted(accum.items()):
        if len(scores) < int(minimum_support_pixels):
            continue
        rows.append(
            RelationEvidenceV1(
                "FRONT_OF_SOURCE_EVIDENCE",
                mesh.owner_ids[a],
                mesh.owner_ids[b],
                float(np.mean(scores)),
                len(scores),
                view_index=view,
                metadata={"authority": "EXTERNAL_MODAL_SEMANTICS_PLUS_CANONICAL_M_DEPTH"},
            )
        )
    return tuple(rows)


def detect_pose_order_conflicts_v1(
    *,
    posed,
    faces,
    cameras,
    mesh: MeshSemanticEvidenceV1,
    source_relations: Iterable[RelationEvidenceV1],
    resolution: int = 256,
) -> tuple[RelationEvidenceV1, ...]:
    """Report when posed geometric order reverses a supported source relation."""
    posed = np.asarray(posed, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    owner_lookup = {oid: i for i, oid in enumerate(mesh.owner_ids)}
    expected = {}
    for rel in source_relations:
        if rel.relation_kind != "FRONT_OF_SOURCE_EVIDENCE" or rel.view_index is None:
            continue
        expected.setdefault(int(rel.view_index), {})[(owner_lookup[rel.a_owner_id], owner_lookup[rel.b_owner_id])] = rel
    rows = []
    for view, pairs in expected.items():
        if not pairs:
            continue
        projected = project_points_xyz_v3(posed, cameras[view])
        layers = _rasterize_owner_layers(projected, faces, resolution=resolution, source_resolution=cameras[view].resolution)
        reversed_counts = {pair: 0 for pair in pairs}
        total_overlap = {pair: 0 for pair in pairs}
        for layer_faces in layers.values():
            distinct = []
            for _, fi in layer_faces:
                oi = int(mesh.face_owner_index[int(fi)])
                if oi >= 0 and (not distinct or distinct[-1] != oi):
                    distinct.append(oi)
                if len(distinct) >= 4:
                    break
            if len(distinct) < 2:
                continue
            pos = {owner: rank for rank, owner in enumerate(distinct)}
            for pair in pairs:
                a, b = pair
                if a in pos and b in pos:
                    total_overlap[pair] += 1
                    if pos[a] > pos[b]:
                        reversed_counts[pair] += 1
        for pair, base in pairs.items():
            total = total_overlap[pair]
            rev = reversed_counts[pair]
            if total and rev:
                rows.append(
                    RelationEvidenceV1(
                        "POSE_ORDER_CONFLICT",
                        base.a_owner_id,
                        base.b_owner_id,
                        float(base.confidence * rev / total),
                        rev,
                        view_index=view,
                        metadata={"overlap_pixels": total, "reversed_pixels": rev},
                    )
                )
    return tuple(sorted(rows, key=lambda x: (-x.confidence, x.view_index or -1, x.a_owner_id, x.b_owner_id)))


def _visible_external_label_maps(evidence: FrozenSemanticMaskEvidenceV1):
    maps = []
    for view in range(8):
        label = np.full((evidence.height, evidence.width), -1, dtype=np.int16)
        score = np.full((evidence.height, evidence.width), -1.0, dtype=np.float32)
        for pi in np.where(evidence.proposal_view_index == view)[0]:
            mask = evidence.proposal_masks[pi]
            s = float(evidence.proposal_score[pi])
            take = mask & (s > score)
            score[take] = s
            label[take] = int(evidence.proposal_label_index[pi])
        maps.append(label)
    return maps


def _rasterize_first_hit(projected, faces, *, source_resolution):
    layers = _rasterize_owner_layers(projected, faces, resolution=int(source_resolution), source_resolution=int(source_resolution), maximum_layers=1)
    owner = np.full((source_resolution, source_resolution), -1, dtype=np.int32)
    depth = np.full((source_resolution, source_resolution), np.inf, dtype=np.float64)
    for pixel, values in layers.items():
        if values:
            y, x = divmod(pixel, source_resolution)
            depth[y, x], owner[y, x] = values[0]
    return owner, depth


def _rasterize_owner_layers(projected, faces, *, resolution, source_resolution, maximum_layers: int = 8):
    """Small geometry-only front-to-back fragment ledger used for evidence."""
    p = np.asarray(projected, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    p[:, :2] *= float(resolution) / float(source_resolution)
    fragments: dict[int, list[tuple[float, int]]] = {}

    def orient(a, b, x, y):
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])

    for fi, tri_ids in enumerate(f):
        tri = p[tri_ids]
        a, b, c = tri[:, :2]
        area = orient(a, b, c[0], c[1])
        if abs(area) <= 1e-12:
            continue
        lo = np.maximum(0, np.floor(tri[:, :2].min(axis=0) - 0.5).astype(int))
        hi = np.minimum(resolution - 1, np.ceil(tri[:, :2].max(axis=0) - 0.5).astype(int))
        if np.any(lo > hi):
            continue
        yy, xx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        x, y = xx.ravel() + 0.5, yy.ravel() + 0.5
        w0 = orient(b, c, x, y)
        w1 = orient(c, a, x, y)
        w2 = orient(a, b, x, y)
        sign = 1.0 if area > 0 else -1.0
        inside = (sign * w0 >= -1e-12) & (sign * w1 >= -1e-12) & (sign * w2 >= -1e-12)
        if not np.any(inside):
            continue
        w0, w1, w2 = w0[inside] / area, w1[inside] / area, w2[inside] / area
        z = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        pixels = yy.ravel()[inside] * resolution + xx.ravel()[inside]
        for pix, depth in zip(pixels.tolist(), z.tolist()):
            if depth <= 0 or not np.isfinite(depth):
                continue
            fragments.setdefault(int(pix), []).append((float(depth), int(fi)))
    for pixel, values in fragments.items():
        values.sort(key=lambda x: (x[0], x[1]))
        if len(values) > maximum_layers:
            del values[maximum_layers:]
    return fragments
