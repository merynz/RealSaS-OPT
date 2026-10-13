"""Independent bind geometry/mask court, scoped to image-derived IRIS teachers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from tools.corpus.gltf_bind_scene_v1 import SourceGLTF

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def project_mask(vertices, faces, camera):
    """Conservative triangle/pixel intersection via the separating-axis theorem.

    Pixel centers alone miss partially covered thin tips. Triangle edge normals
    plus screen X/Y are all separating axes for a convex triangle and unit square.
    """
    resolution = camera["resolution"]
    delta = np.asarray(vertices) - camera["origin"]
    xy = np.stack([delta @ camera["right"], -(delta @ camera["screen_up"])], axis=1)
    xy = (xy / camera["half_extent"] + 1.) * (.5 * resolution)
    mask = np.zeros((resolution, resolution), bool)
    for face in faces:
        triangle = xy[face]
        lo = np.maximum(np.floor(triangle.min(0)).astype(int), 0)
        hi = np.minimum(np.floor(triangle.max(0)).astype(int), resolution - 1)
        if np.any(lo > hi):
            continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        centers = np.stack([xx + .5, yy + .5], axis=-1)
        valid = np.ones(xx.shape, bool)
        edges = np.roll(triangle, -1, axis=0) - triangle
        if abs(edges[0,0]*edges[1,1]-edges[0,1]*edges[1,0]) < 1e-12:
            continue
        for edge in edges:
            axis = np.array([-edge[1], edge[0]])
            projections = triangle @ axis
            projected = centers @ axis
            radius = .5 * np.abs(axis).sum()
            valid &= (projected + radius >= projections.min() - 1e-9)
            valid &= (projected - radius <= projections.max() + 1e-9)
        mask[lo[1]:hi[1]+1, lo[0]:hi[0]+1] |= valid
    return mask

def audit_subject(row, source_root, imported_root, observations_root, probe, output_root, policy):
    source = source_root / row["source"]["corpus_relative_path"]
    if digest(source) != row["source"]["sha256"]:
        raise ValueError("ADMISSION_RAW_HASH_DRIFT")
    key = row["subject_key"]
    imported_path = imported_root / key / "imported.json"
    imported = json.loads(imported_path.read_text())
    if source.suffix.lower() in (".gltf", ".glb"):
        raw = SourceGLTF(source).geometry()
    else:
        raw = json.loads(subprocess.check_output([str(probe), str(source)]))
    a, b = np.asarray(raw["vertices"]), np.asarray(imported["vertices"])
    diagonal = float(np.linalg.norm(np.ptp(a, axis=0)))
    relative = float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()) / diagonal)
    if relative > policy["maximum_relative_vertex_error"]:
        raise ValueError(f"ADMISSION_BIND_VERTEX_DRIFT:{key}:{relative}")
    if raw["bone_count"] != imported["bone_count"]:
        raise ValueError("ADMISSION_JOINT_CARDINALITY_DRIFT")
    folder = observations_root / key
    candidate_path = folder / "SOURCE_REST_RENDER_CANDIDATE.json"
    candidate = json.loads(candidate_path.read_text())
    if candidate["source_sha256"] != digest(source) or candidate["resolution"] != policy["resolution"]:
        raise ValueError("ADMISSION_OBSERVATION_SOURCE_OR_POLICY_DRIFT")
    if candidate["import"]["pose_policy"] != policy["pose"] + "__NO_ANIMATION":
        raise ValueError("ADMISSION_POSE_DRIFT")
    cameras_path = folder / "camera_bundle.json"
    if digest(cameras_path) != candidate["camera_bundle"]["sha256"]:
        raise ValueError("ADMISSION_CAMERA_DRIFT")
    bundle = json.loads(cameras_path.read_text())
    if bundle["per_view_recenter"] or bundle["per_view_pose_change"]:
        raise ValueError("ADMISSION_PER_VIEW_SCENE_MUTATION")
    output = output_root / key
    output.mkdir(parents=True, exist_ok=True)
    views = []
    for view, camera in zip(candidate["views"], bundle["cameras"]):
        index = view["view_index"]
        png, binary = folder / f"V{index}.png", folder / f"V{index}.mask.bin"
        if digest(png) != view["raster"]["sha256"] or digest(binary) != view["foreground_mask"]["sha256"]:
            raise ValueError("ADMISSION_RASTER_DRIFT")
        actual = np.asarray(Image.open(png).convert("RGBA"))[:, :, 3] > 0
        declared = np.frombuffer(binary.read_bytes(), np.uint8).reshape(actual.shape) > 0
        if not np.array_equal(actual, declared):
            raise ValueError("ADMISSION_ALPHA_MASK_DRIFT")
        reference = project_mask(a, np.asarray(raw["faces"]), camera)
        reference_path = output / f"V{index}.reference.png"
        Image.fromarray(reference.astype(np.uint8)*255).save(reference_path)
        if not actual.any() or not reference.any():
            raise ValueError("ADMISSION_EMPTY_MASK")
        gap = float(max(distance_transform_edt(~actual)[reference].max(),
                        distance_transform_edt(~reference)[actual].max()))
        metrics = {"view_index": index, "maximum_mask_gap_px": gap,
                   "raster_sha256": digest(png), "mask_sha256": digest(binary),
                   "reference_sha256": digest(reference_path),
                   "border_margin_px": view["border_margin_px"]}
        views.append(metrics)
        if gap > policy["maximum_mask_gap_px"] or view["border_margin_px"] < policy["minimum_border_margin_px"]:
            Image.fromarray(np.stack([reference*255, actual*255, np.zeros_like(actual)],axis=-1).astype("uint8")).save(output/f"V{index}.FAILED.png")
            raise ValueError(f"ADMISSION_MASK_DRIFT:{key}:V{index}:{gap}")
    if len(views) != policy["view_count"]:
        raise ValueError("ADMISSION_VIEW_COUNT")
    receipt = {"schema": "RealSaS.IrisBindObservationAdmission.v1",
               "status": "PASS_IRIS_BIND_OBSERVATIONS", "subject_id": key,
               "source_sha256": digest(source), "candidate_sha256": digest(candidate_path),
               "imported_geometry_sha256": digest(imported_path), "camera_sha256": digest(cameras_path),
               "relative_vertex_error": relative, "raw_vertex_count": len(a),
               "imported_vertex_count": len(b), "bone_count": raw["bone_count"],
               "native_motion_inventory": raw["native_motion_names"],
               "source_material_metadata": raw["materials"], "views": views,
               "scope": policy["scope"], "policy": policy,
               "full_native_motion_truth_admitted": False, "native_lit_parity_claimed": False,
               "go_stage_qualification_claimed": False, "fit8_pass_claimed": False}
    (output / "IRIS_OBSERVATION_ADMISSION.json").write_text(json.dumps(receipt, indent=2))
    print(f"[ADMIT-IRIS] {key} vertices={len(a)} bones={raw['bone_count']} "
          f"relative_error={relative:.3g} max_mask_gap_px={max(v['maximum_mask_gap_px'] for v in views):.2f}", flush=True)
    return receipt

def main():
    p = argparse.ArgumentParser()
    for flag in ("source-root","imported-root","observations-root","probe","output-root"):
        p.add_argument("--"+flag, type=Path, required=True)
    args = p.parse_args()
    roster = json.loads((ROOT/"canonical/FIT8_DIVERSITY_SELECTION_V1.json").read_text())
    policy = json.loads((ROOT/"canonical/FIT8_IRIS_OBSERVATION_POLICY_V1.json").read_text())
    for row in roster["selected"]:
        audit_subject(row, args.source_root, args.imported_root, args.observations_root,
                      args.probe, args.output_root, policy)
if __name__ == "__main__":
    main()
