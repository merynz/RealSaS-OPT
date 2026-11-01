#!/usr/bin/env python3
"""Build a declared source-world solid derivative; never relabel raw surfaces.

Domain: Boolean union of independently closed, positive-volume source components
with no detected component self-intersection. Original surface/owner/motion truth
is retained separately. Independent solid, numeric and consumer courts are needed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import manifold3d
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.iris.truth.component_geometry_authority_v1 import build_component_geometry_authority_v1


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def union_boundary(vertices, faces, component_id):
    inputs, volumes = [], []
    for cid in sorted(np.unique(component_id).tolist()):
        ids = np.flatnonzero(component_id == cid)
        used, inverse = np.unique(faces[ids], return_inverse=True)
        mesh = manifold3d.Mesh64(vert_properties=np.ascontiguousarray(vertices[used], np.float64),
            tri_verts=np.ascontiguousarray(inverse.reshape(-1, 3), np.uint64), face_id=ids.astype(np.uint64))
        solid = manifold3d.Manifold(mesh)
        if solid.status() != manifold3d.Error.NoError or solid.volume() <= 0:
            raise ValueError("SOURCE_UNION_COMPONENT_NOT_POSITIVE_MANIFOLD")
        inputs.append(solid)
        volumes.append(solid.volume())
    result = manifold3d.Manifold.batch_boolean(inputs, manifold3d.OpType.Add)
    if result.status() != manifold3d.Error.NoError or result.is_empty():
        raise ValueError("SOURCE_UNION_BOOLEAN_FAILURE")
    if not max(volumes) * (1 - 1e-8) <= result.volume() <= sum(volumes) * (1 + 1e-8):
        raise ValueError("SOURCE_UNION_VOLUME_BOUNDS")
    mesh = result.to_mesh64()
    return (np.asarray(mesh.vert_properties[:, :3], np.float64), np.asarray(mesh.tri_verts, np.int64),
            np.asarray(mesh.face_id, np.int64), float(result.volume()))


def build(route_row, probe_row, output_root, query_count=8192):
    route_ref = route_row["report"]
    if sha(route_ref["path"]) != route_ref["sha256"]:
        raise ValueError("SOURCE_UNION_STRUCTURE_DRIFT")
    if probe_row["subject_id"] != route_row["subject_id"] or probe_row["components_with_self_intersection"]:
        raise ValueError("SOURCE_UNION_SELF_INTERSECTION_NOT_CLEARED")
    if any(c["signed_volume_source_world"] <= 0 for c in probe_row["components"]):
        raise ValueError("SOURCE_UNION_UNDECLARED_SHELL_ORIENTATION")
    route = json.loads(Path(route_ref["path"]).read_text())
    import_ref = route["import_report"]
    if sha(import_ref["path"]) != import_ref["sha256"]:
        raise ValueError("SOURCE_UNION_IMPORT_DRIFT")
    imported = json.loads(Path(import_ref["path"]).read_text())
    geometry_ref = imported["outputs"]["rest_geometry"]
    if sha(geometry_ref["path"]) != geometry_ref["sha256"] or sha(imported["source"]["path"]) != imported["source"]["sha256"]:
        raise ValueError("SOURCE_UNION_SOURCE_DRIFT")
    with np.load(geometry_ref["path"], allow_pickle=False) as z:
        authority = build_component_geometry_authority_v1(z["rest_vertices_source_world"], z["faces"], z["face_owner_privileged"])
    if not (np.all(authority.face_valid) and np.all(authority.sign_component_id >= 0) and np.all(authority.sign_component_valid)):
        raise ValueError("SOURCE_UNION_COMPONENT_CLOSEDNESS")
    v, f, source_face, volume = union_boundary(authority.topology_vertices, authority.topology_faces, authority.sign_component_id)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    sid = route_row["subject_id"]
    seed = int(hashlib.sha256((sid + "|global-solid-union").encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    count = query_count // 2
    surface, face_idx = trimesh.sample.sample_surface(mesh, count, seed=seed)
    scale = float(np.linalg.norm(v.max(0) - v.min(0)))
    offset = rng.uniform(.0002, .03, count) * rng.choice([-1, 1], count) * scale
    near = surface + mesh.face_normals[face_idx] * offset[:, None]
    q = np.vstack([near, rng.uniform(v.min(0) - .1 * scale, v.max(0) + .1 * scale, (query_count - count, 3))])
    # Rtree candidate arrays grow with query count and face count. Bound both
    # nearest-distance and ray workloads; batching changes no query or target.
    distances, closest_faces, occupancies = [], [], []
    for start in range(0, len(q), 64):
        block = q[start:start + 64]
        _, block_distance, block_face = trimesh.proximity.closest_point(mesh, block)
        distances.append(block_distance)
        closest_faces.append(block_face)
        occupancies.append(mesh.contains(block))
    distance, closest, inside = np.concatenate(distances), np.concatenate(closest_faces), np.concatenate(occupancies)
    valid = np.isfinite(distance) & (distance > scale * 1e-9)
    signed_idx = np.flatnonzero(valid)
    signed = distance[signed_idx] * np.where(inside[signed_idx], -1., 1.)
    if not np.any(inside[valid]) or np.all(inside[valid]):
        raise ValueError("SOURCE_UNION_SIGN_CLASSES_EMPTY")
    out = Path(output_root) / sid
    out.mkdir(parents=True, exist_ok=True)
    payload = out / "SOURCE_UNION_SOLID_GEOMETRY_AND_SDF_V1.npz"
    np.savez_compressed(payload, vertices_source_world=v, faces=f.astype(np.int32), source_face_id=source_face.astype(np.int32),
        query_xyz_source_world=q, unsigned_distance_source_world=distance, closest_boundary_face=closest.astype(np.int32),
        signed_query_index=signed_idx.astype(np.int32), signed_distance_source_world=signed, occupancy=inside[signed_idx])
    manifest = {"schema": "RealSaS.SourceUnionSolidCandidate.v1", "status": "CANDIDATE__INDEPENDENT_SOLID_AND_NUMERIC_COURT_REQUIRED",
        "subject_id": sid, "domain": "BOOLEAN_UNION_OF_CLOSED_EMBEDDED_POSITIVE_VOLUME_SOURCE_COMPONENT_SOLIDS",
        "coordinate_frame": "SOURCE_WORLD__CAMERA_NORMALIZATION_BINDING_REQUIRED_FOR_TRAINING",
        "sign_convention": "NEGATIVE_INSIDE__POSITIVE_OUTSIDE", "distance_boundary": "BOOLEAN_UNION_BOUNDARY__INTERNAL_SOURCE_FACES_REMOVED",
        "parents": {"structure": route_ref, "import_report": import_ref, "source_geometry": geometry_ref, "source": imported["source"]},
        "output": ref(payload), "counts": {"source_component_count": len(authority.sign_component_valid), "vertex_count": len(v),
            "face_count": len(f), "query_count": len(q), "valid_signed_query_count": len(signed_idx),
            "inside_count": int(np.count_nonzero(inside[valid])), "outside_count": int(np.count_nonzero(~inside[valid]))},
        "volume_source_world": volume, "runtime": {"manifold3d": version("manifold3d"), "trimesh": trimesh.__version__, "numpy": np.__version__},
        "producer_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__).resolve(), ROOT / "models/iris/truth/component_geometry_authority_v1.py")},
        "raw_source_mutated": False, "raw_surface_replaced": False, "deformation_authority_minted": False,
        "training_authorized": False, "global_sign_authorized": False}
    path = out / "SOURCE_UNION_SOLID_CANDIDATE_V1.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {"subject_id": sid, "status": manifest["status"], "manifest": ref(path), "counts": manifest["counts"]}


def run_one(args):
    row, probe_row, root, count = args
    try:
        return build(row, probe_row, root, count)
    except Exception as exc:
        return {"subject_id": row["subject_id"], "status": "BLOCKED", "error_type": type(exc).__name__, "reason": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--embedding-probe", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--query-count", type=int, default=8192)
    a = p.parse_args()
    census = json.loads(a.census.read_text())
    probe = json.loads(a.embedding_probe.read_text())
    if probe["census"]["sha256"] != sha(a.census) or probe["failures"]:
        raise ValueError("SOURCE_UNION_EMBEDDING_PROBE_BINDING")
    rows = {r["subject_id"]: r for r in census["subjects"]}
    eligible = [r for r in probe["subjects"] if r["components_with_self_intersection"] == 0 and all(c["signed_volume_source_world"] > 0 for c in r["components"])]
    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_one, [(rows[r["subject_id"]], r, a.output_root, a.query_count) for r in eligible]):
            results.append(result)
            print(json.dumps({"done": len(results), "total": len(eligible), "last": result}), flush=True)
    receipt = {"schema": "RealSaS.SourceUnionSolidCandidateBatch.v1", "candidate_count": len(eligible), "subjects": results,
               "parents": {"census": ref(a.census), "embedding_probe": ref(a.embedding_probe)}, "producer_sha256": sha(__file__),
               "training_authorized": False, "global_sign_authorized": False}
    a.output_root.mkdir(parents=True, exist_ok=True)
    (a.output_root / "SOURCE_UNION_SOLID_CANDIDATE_BATCH_V1.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
