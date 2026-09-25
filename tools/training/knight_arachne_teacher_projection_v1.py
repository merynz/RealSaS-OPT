from __future__ import annotations

"""Teacher-only Knight Arachne skin target projection.

This module never participates in free-running inference. It projects the exact
source FBX skin field onto the admitted RiggingSurfaceIR for same-witness FIT
supervision:

1. fill source zero-weight geometry by deterministic mechanical attachment:
   - within a connected component, copy the nearest weighted source vertex;
   - for a fully unweighted component, bind the whole component rigidly to the
     dominant joint of the globally nearest weighted source vertex;
2. map source-bone columns onto the mechanically-meaningful Geppetto target
   using provenance-only source indices; synthetic terminal tips receive zero
   direct source mass;
3. closest-point project every RiggingSurfaceIR node onto the exact source
   triangle mesh and barycentrically interpolate the mapped source skin field.

Source mesh, source skin, bone provenance and teacher W are training/evaluation
truth only. Product inference consumes only RiggingSurfaceIR + QualifiedSkeletonIR.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from trimesh.proximity import closest_point_naive
from trimesh.triangles import points_to_barycentric


SCHEMA = "RealSaS.KnightArachneTeacherSurfaceProjection.v1"
RULE = "EXACT_SOURCE_TRIANGLE_CLOSEST_POINT_BARYCENTRIC_WITH_COMPONENT_MECHANICAL_ATTACHMENT"
EPS = 1e-8


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _surface_positions(surface_json: Path) -> tuple[np.ndarray, tuple[str, ...], str]:
    d = json.loads(surface_json.read_text(encoding="utf-8"))
    nodes = tuple(d.get("surface_nodes") or ())
    if not nodes:
        raise ValueError("surface contains no nodes")
    ids = tuple(str(n["surface_id"]) for n in nodes)
    if len(ids) != len(set(ids)):
        raise ValueError("surface contains duplicate ids")
    p = np.asarray([n["P"] for n in nodes], dtype=np.float64)
    if p.shape != (len(nodes), 3) or not np.isfinite(p).all():
        raise ValueError("surface positions invalid")
    lineage = str(d.get("geometry_lineage_hash") or "")
    if not lineage:
        raise ValueError("surface lineage missing")
    return p, ids, lineage


class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int64)
        self.r = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.p
        while int(p[x]) != x:
            p[x] = p[int(p[x])]
            x = int(p[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def _components(vertex_count: int, faces: np.ndarray) -> list[np.ndarray]:
    uf = _UF(vertex_count)
    for a, b, c in np.asarray(faces, dtype=np.int64).tolist():
        uf.union(int(a), int(b))
        uf.union(int(b), int(c))
        uf.union(int(c), int(a))
    groups: dict[int, list[int]] = {}
    for i in range(vertex_count):
        groups.setdefault(uf.find(i), []).append(i)
    return [np.asarray(v, dtype=np.int64) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def _normalized_source_skin(
    vertices: np.ndarray,
    faces: np.ndarray,
    skin: np.ndarray,
) -> tuple[np.ndarray, dict]:
    w = np.asarray(skin, dtype=np.float64).copy()
    if w.ndim != 2 or w.shape[0] != len(vertices):
        raise ValueError("source skin shape mismatch")
    if not np.isfinite(w).all() or np.any(w < -1e-8):
        raise ValueError("source skin invalid")
    w = np.maximum(w, 0.0)
    sums = w.sum(axis=1)
    weighted = sums > EPS
    if not bool(weighted.any()):
        raise ValueError("source skin has no weighted vertices")
    w[weighted] /= sums[weighted, None]

    comps = _components(len(vertices), faces)
    global_weighted = np.flatnonzero(weighted)
    global_tree = cKDTree(vertices[global_weighted])
    partially_filled = 0
    rigid_components = 0
    rigid_vertices = 0
    attachment_rows: list[dict] = []

    for ci, comp in enumerate(comps):
        comp_weighted = comp[weighted[comp]]
        comp_zero = comp[~weighted[comp]]
        if len(comp_zero) == 0:
            continue
        if len(comp_weighted):
            tree = cKDTree(vertices[comp_weighted])
            _, nearest = tree.query(vertices[comp_zero], k=1)
            w[comp_zero] = w[comp_weighted[np.asarray(nearest, dtype=np.int64)]]
            weighted[comp_zero] = True
            partially_filled += int(len(comp_zero))
            continue

        # Entirely unweighted disconnected geometry is a rigid mechanical
        # attachment. Bind every vertex to one source joint selected from the
        # closest already-weighted source locus; do not smear weights over space.
        dist, nearest = global_tree.query(vertices[comp], k=1)
        row = int(np.argmin(np.asarray(dist)))
        anchor_global = int(global_weighted[int(np.asarray(nearest)[row])])
        dominant = int(np.argmax(w[anchor_global]))
        w[comp] = 0.0
        w[comp, dominant] = 1.0
        weighted[comp] = True
        rigid_components += 1
        rigid_vertices += int(len(comp))
        attachment_rows.append(
            {
                "component_index": int(ci),
                "vertex_count": int(len(comp)),
                "anchor_source_vertex": anchor_global,
                "source_joint_index": dominant,
                "anchor_distance": float(np.asarray(dist)[row]),
            }
        )

    residual = np.abs(w.sum(axis=1) - 1.0)
    if float(residual.max(initial=0.0)) > 1e-6:
        raise RuntimeError("mechanical source attachment failed simplex")
    return w.astype(np.float32), {
        "component_count": int(len(comps)),
        "original_zero_weight_vertex_count": int(np.count_nonzero(sums <= EPS)),
        "within_component_propagated_vertex_count": int(partially_filled),
        "fully_unweighted_rigid_component_count": int(rigid_components),
        "fully_unweighted_rigid_vertex_count": int(rigid_vertices),
        "rigid_attachment_rows": attachment_rows,
    }


def _map_to_target_axes(
    source_skin: np.ndarray,
    source_provenance: np.ndarray,
) -> np.ndarray:
    prov = np.asarray(source_provenance, dtype=np.int64)
    out = np.zeros((source_skin.shape[0], len(prov)), dtype=np.float64)
    used = []
    for j, source_index in enumerate(prov.tolist()):
        if source_index < 0:
            continue
        if source_index >= source_skin.shape[1]:
            raise ValueError("target provenance references unknown source bone")
        out[:, j] = source_skin[:, source_index]
        used.append(source_index)
    if len(used) != len(set(used)):
        raise ValueError("target provenance duplicates source skin column")

    selected_mass = out.sum(axis=1)
    source_mass = np.asarray(source_skin, dtype=np.float64).sum(axis=1)
    dropped = np.maximum(0.0, source_mass - selected_mass)
    if float(dropped.max(initial=0.0)) > 1e-5:
        raise RuntimeError(f"target skeleton drops material source skin mass:{float(dropped.max())}")
    nz = selected_mass > EPS
    if not bool(nz.all()):
        raise RuntimeError("mapped source skin contains zero rows")
    out /= selected_mass[:, None]
    return out.astype(np.float32)


def project_teacher_skin(
    *,
    full_source_npz: Path,
    target_npz: Path,
    surface_json: Path,
    chunk_size: int = 32,
) -> tuple[dict[str, np.ndarray], dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with np.load(full_source_npz, allow_pickle=False) as z:
        required = {"vertices_source", "faces", "skin", "parents", "deform_mask"}
        if not required.issubset(z.files):
            raise ValueError(f"full source missing:{sorted(required-set(z.files))}")
        vertices = np.asarray(z["vertices_source"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        source_skin_raw = np.asarray(z["skin"], dtype=np.float64)
    with np.load(target_npz, allow_pickle=False) as z:
        required = {"source_indices_provenance_only", "parent_indices", "positions_world", "role_codes"}
        if not required.issubset(z.files):
            raise ValueError(f"target missing:{sorted(required-set(z.files))}")
        provenance = np.asarray(z["source_indices_provenance_only"], dtype=np.int64)
        target_parents = np.asarray(z["parent_indices"], dtype=np.int64)
        target_positions = np.asarray(z["positions_world"], dtype=np.float32)
        role_codes = np.asarray(z["role_codes"], dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("source vertices invalid")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any((faces < 0) | (faces >= len(vertices))):
        raise ValueError("source faces invalid")

    source_skin, attachment_report = _normalized_source_skin(vertices, faces, source_skin_raw)
    mapped_source = _map_to_target_axes(source_skin, provenance)
    surface_p, surface_ids, surface_lineage = _surface_positions(surface_json)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    projected = np.zeros((len(surface_p), len(provenance)), dtype=np.float64)
    distances = np.zeros(len(surface_p), dtype=np.float64)
    triangle_ids = np.zeros(len(surface_p), dtype=np.int64)
    barycentric = np.zeros((len(surface_p), 3), dtype=np.float64)

    for start in range(0, len(surface_p), int(chunk_size)):
        stop = min(start + int(chunk_size), len(surface_p))
        closest, dist, tri = closest_point_naive(mesh, surface_p[start:stop])
        tri = np.asarray(tri, dtype=np.int64)
        bc = points_to_barycentric(mesh.triangles[tri], np.asarray(closest, dtype=np.float64))
        bc = np.clip(np.asarray(bc, dtype=np.float64), 0.0, 1.0)
        bc /= bc.sum(axis=1, keepdims=True).clip(min=EPS)
        tri_vertices = faces[tri]
        rows = mapped_source[tri_vertices]
        pred = np.einsum("mk,mkj->mj", bc, rows, optimize=True)
        pred = np.maximum(pred, 0.0)
        pred /= pred.sum(axis=1, keepdims=True).clip(min=EPS)
        projected[start:stop] = pred
        distances[start:stop] = np.asarray(dist, dtype=np.float64)
        triangle_ids[start:stop] = tri
        barycentric[start:stop] = bc

    simplex = np.abs(projected.sum(axis=1) - 1.0)
    if not np.isfinite(projected).all() or np.any(projected < -1e-8):
        raise RuntimeError("projected teacher skin numeric failure")
    if float(simplex.max(initial=0.0)) > 1e-6:
        raise RuntimeError("projected teacher skin simplex failure")

    bbox_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if not np.isfinite(bbox_diag) or bbox_diag <= EPS:
        raise RuntimeError("source bbox degenerate")
    p95 = float(np.quantile(distances, 0.95))
    p99 = float(np.quantile(distances, 0.99))
    max_dist = float(distances.max(initial=0.0))
    # Teacher projection validity is a supervision-coverage contract, not a
    # product-geometry gate. This mirrors the prior Arachne FIT2 semantics:
    # rows farther than the frozen same-subject threshold are excluded from
    # teacher objective/evaluation, while free-running Arachne still predicts
    # every admitted surface row and Compiler qualification remains full-surface.
    clean_distance_ratio_max = 0.05
    clean_mask = distances <= (clean_distance_ratio_max * bbox_diag)
    clean_rows = int(np.count_nonzero(clean_mask))
    coverage = float(clean_rows / max(1, len(distances)))
    minimum_clean_rows_required = 192
    if clean_rows < minimum_clean_rows_required:
        raise RuntimeError(
            f"teacher projection insufficient clean supervision:{clean_rows}<{minimum_clean_rows_required}"
        )

    arrays = {
        "weights": projected.astype(np.float32),
        "surface_ids": np.asarray(surface_ids),
        "target_parent_indices": target_parents.astype(np.int64),
        "target_positions_world": target_positions.astype(np.float32),
        "target_role_codes": role_codes.astype(np.int64),
        "target_source_indices_provenance_only": provenance.astype(np.int64),
        "source_triangle_index": triangle_ids.astype(np.int64),
        "source_triangle_barycentric": barycentric.astype(np.float32),
        "source_surface_distance": distances.astype(np.float32),
        "teacher_valid_mask": clean_mask.astype(np.uint8),
    }
    report = {
        "schema": SCHEMA,
        "status": "PASS" if coverage >= 0.90 else "PASS_WITH_COVERAGE_WARNING",
        "rule": RULE,
        "teacher_only": True,
        "free_running_inference_input": False,
        "surface_lineage_hash": surface_lineage,
        "surface_node_count": int(len(surface_ids)),
        "target_joint_count": int(len(provenance)),
        "source_vertex_count": int(len(vertices)),
        "source_face_count": int(len(faces)),
        "source_bbox_diagonal": bbox_diag,
        "closest_surface_distance_p50": float(np.quantile(distances, 0.50)),
        "closest_surface_distance_p95": p95,
        "closest_surface_distance_p99": p99,
        "closest_surface_distance_max": max_dist,
        "closest_surface_distance_p95_ratio": p95 / bbox_diag,
        "closest_surface_distance_max_ratio": max_dist / bbox_diag,
        "clean_distance_ratio_max": clean_distance_ratio_max,
        "clean_row_count": clean_rows,
        "coverage": coverage,
        "coverage_warning": bool(coverage < 0.90),
        "minimum_clean_rows_required": minimum_clean_rows_required,
        "coverage_is_teacher_supervision_mask_not_product_gate": True,
        "max_simplex_residual": float(simplex.max(initial=0.0)),
        "component_attachment": attachment_report,
        "full_source_npz_sha256": _sha(full_source_npz),
        "target_npz_sha256": _sha(target_npz),
        "surface_json_sha256": _sha(surface_json),
    }
    return arrays, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-source-npz", required=True)
    ap.add_argument("--target-npz", required=True)
    ap.add_argument("--surface-json", required=True)
    ap.add_argument("--output-npz", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--chunk-size", type=int, default=32)
    args = ap.parse_args()

    arrays, report = project_teacher_skin(
        full_source_npz=Path(args.full_source_npz),
        target_npz=Path(args.target_npz),
        surface_json=Path(args.surface_json),
        chunk_size=args.chunk_size,
    )
    out_npz = Path(args.output_npz)
    out_json = Path(args.report_json)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    report = dict(report)
    report["output_npz_sha256"] = _sha(out_npz)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("KNIGHT_ARACHNE_TEACHER_PROJECTION_PASS=" + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
