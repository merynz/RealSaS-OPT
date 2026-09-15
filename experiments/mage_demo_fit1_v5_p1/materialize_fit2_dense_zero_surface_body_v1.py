from __future__ import annotations

"""Materialize Mage BODY on sealed dense zero-surface topology + current FIT2 W.

No training, model query, P1/P1Q adjacency, historical skin transfer, teacher/source
mesh topology, or threshold relaxation is permitted. Only original zero-surface faces
that pass the frozen motion gates and the frozen observation gate are admitted.
This stage does not claim PRODUCT_PASS.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import polygon

from compiler.realsas_compiler_core.appearance_atlas import build_sprite_panel_appearance
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.dense_zero_surface_bridge import (
    DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
    build_dense_zero_surface_candidate,
    project_dense_vertices,
    qualify_dense_zero_surface_mesh,
    replay_dense_zero_surface_compaction,
)
from compiler.realsas_compiler_core.mesh.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_core.motion_rotation_only import build_mage_rotation_only_qualified_motion
from compiler.realsas_compiler_core.v4 import build_mechanical_state, validate_appearance_binding

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1

SCHEMA = "RealSaS.MageFIT2.DenseZeroSurfaceBody.v1"
EXPECTED_ZERO_SURFACE_SHA256 = "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"
EXPECTED_SURFACE_FILE_SHA256 = "170b8e4712fd78ef0721462f19f80ccb94eb008206fc6a7061908bfdc36f202f"
EXPECTED_OWNER_MANIFEST_SHA256 = "4df8d42a273840021624c854bd86414e148f6a5861730e3c3d53bade039d9167"
EXPECTED_FOREGROUND_MANIFEST_SHA256 = "5a331c2a73f2fa8b425dacdde1d4890fa55873d01856fe1d65153b3b03d797bf"
EDGE_STRETCH_LIMIT = 1.55
AREA_CHANGE_LIMIT = 2.10
REQUIRED_RECALL = 0.98
REQUIRED_PRECISION = 0.995
UNIFORM_SAMPLE_COUNT = 9


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = sha256()
    h.update(str(arr.dtype).encode("ascii")); h.update(b"|")
    h.update("x".join(map(str, arr.shape)).encode("ascii")); h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path}")
    return value


def _load_zero_surface(path: Path):
    if _sha(path) != EXPECTED_ZERO_SURFACE_SHA256:
        raise RuntimeError("DENSE_BODY_ZERO_SURFACE_SHA_DRIFT")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces", "normals"}:
            raise RuntimeError(f"DENSE_BODY_ZERO_SURFACE_PAYLOAD_DRIFT:{sorted(z.files)}")
        vertices = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("DENSE_BODY_ZERO_SURFACE_VERTEX_SHAPE_DRIFT")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError("DENSE_BODY_ZERO_SURFACE_FACE_SHAPE_DRIFT")
    return vertices, faces


def _load_cameras(paths):
    if len(paths) != 8:
        raise RuntimeError("DENSE_BODY_REQUIRES_8_CAMERAS")
    out = []
    for view, path in enumerate(paths):
        if _sha(path) != ceiling_v1.CAMERA_SHA256[view]:
            raise RuntimeError(f"DENSE_BODY_CAMERA_SHA_DRIFT_V{view}")
        value = _load_json(path)
        if int(value.get("view_index", -1)) != view or int(value.get("resolution", 0)) != 1024:
            raise RuntimeError(f"DENSE_BODY_CAMERA_CONTRACT_DRIFT_V{view}")
        out.append(value)
    return tuple(out)


def _load_observations(paths):
    if len(paths) != 8:
        raise RuntimeError("DENSE_BODY_REQUIRES_8_OBSERVATIONS")
    out = []
    for view, path in enumerate(paths):
        if _sha(path) != ceiling_v1.OBSERVATION_SHA256[view]:
            raise RuntimeError(f"DENSE_BODY_OBSERVATION_SHA_DRIFT_V{view}")
        with Image.open(path) as im:
            rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        if rgba.shape != (1024, 1024, 4):
            raise RuntimeError(f"DENSE_BODY_OBSERVATION_SHAPE_DRIFT_V{view}:{rgba.shape}")
        out.append(rgba[..., 3] >= 8)
    return tuple(out)


def _load_owner_masks(manifest_path: Path, paths):
    if _sha(manifest_path) != EXPECTED_OWNER_MANIFEST_SHA256:
        raise RuntimeError("DENSE_BODY_OWNER_MANIFEST_SHA_DRIFT")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "PASS__EXACT_SOURCE_FIT_TARGET_COMPONENT_OWNER_RASTER_V0_V7":
        raise RuntimeError("DENSE_BODY_OWNER_MANIFEST_STATUS_INVALID")
    artifacts = {str(r["path"]): str(r["sha256"]) for r in manifest.get("artifacts") or ()}
    if len(paths) != 8:
        raise RuntimeError("DENSE_BODY_REQUIRES_8_OWNER_RASTERS")
    out = []
    for view, path in enumerate(paths):
        name = f"V{view}_SOURCE_COMPONENT_OWNER_RASTER.npz"
        if path.name != name or artifacts.get(name) != _sha(path):
            raise RuntimeError(f"DENSE_BODY_OWNER_RASTER_SHA_DRIFT_V{view}")
        with np.load(path, allow_pickle=False) as z:
            if "owner" not in z.files:
                raise RuntimeError(f"DENSE_BODY_OWNER_RASTER_PAYLOAD_DRIFT_V{view}")
            owner = np.asarray(z["owner"], dtype=np.int16)
        if owner.shape != (1024, 1024):
            raise RuntimeError(f"DENSE_BODY_OWNER_RASTER_SHAPE_DRIFT_V{view}")
        out.append(owner >= 1)  # 0=BODY; 1..4=typed rigid; -1=background
    return manifest, tuple(out)


def _load_foreground_manifest(path: Path):
    if _sha(path) != EXPECTED_FOREGROUND_MANIFEST_SHA256:
        raise RuntimeError("DENSE_BODY_FOREGROUND_MANIFEST_SHA_DRIFT")
    value = _load_json(path)
    if value.get("status") != "PASS__EXACT_OBSERVATION_PLUS_TYPED_RIGID_FOREGROUND_ATLAS_V0_V7":
        raise RuntimeError("DENSE_BODY_FOREGROUND_MANIFEST_STATUS_INVALID")
    rows = {int(r["view"]): r for r in value.get("views") or ()}
    if set(rows) != set(range(8)):
        raise RuntimeError("DENSE_BODY_FOREGROUND_MANIFEST_VIEW_SET_INVALID")
    return value, rows


def _joint_order(skeleton):
    by_id = {str(j.canonical_joint_id): j for j in skeleton.joints}
    order, visiting, done = [], set(), set()
    def visit(jid):
        if jid in done: return
        if jid in visiting: raise RuntimeError("DENSE_BODY_SKELETON_CYCLE")
        visiting.add(jid)
        parent = by_id[jid].parent_canonical_id
        if parent is not None:
            parent = str(parent)
            if parent not in by_id: raise RuntimeError("DENSE_BODY_SKELETON_PARENT_MISSING")
            visit(parent)
        visiting.remove(jid); done.add(jid); order.append(jid)
    for jid in sorted(by_id): visit(jid)
    return tuple(order), by_id


def _surface_skin_matrix(surface, skin, joint_ids):
    rows = {str(r.surface_id): r for r in skin.rows}
    ji = {jid: i for i, jid in enumerate(joint_ids)}
    W = np.zeros((len(surface.surface_nodes), len(joint_ids)), dtype=np.float64)
    for i, node in enumerate(surface.surface_nodes):
        row = rows.get(str(node.surface_id))
        if row is None: raise RuntimeError(f"DENSE_BODY_SKIN_ROW_MISSING:{node.surface_id}")
        for jid, weight in row.influences:
            if str(jid) not in ji: raise RuntimeError(f"DENSE_BODY_SKIN_JOINT_UNKNOWN:{jid}")
            W[i, ji[str(jid)]] = float(weight)
    if not np.isfinite(W).all() or (W < -1e-10).any() or not np.allclose(W.sum(1), 1.0, atol=1e-7, rtol=0.0):
        raise RuntimeError("DENSE_BODY_SURFACE_SKIN_SIMPLEX_DRIFT")
    return W


def _fit_view_affine(surface, skeleton, view):
    rows = []
    for node in surface.surface_nodes:
        xy = [tuple(map(float, q)) for v, q in node.raster_bindings if int(v) == int(view)]
        if len(xy) > 1: raise RuntimeError(f"DENSE_BODY_DUPLICATE_SURFACE_RASTER_V{view}:{node.surface_id}")
        if xy: rows.append((tuple(map(float, node.P)), xy[0]))
    if len(rows) < 8: raise RuntimeError(f"DENSE_BODY_INSUFFICIENT_SURFACE_RASTER_V{view}:{len(rows)}")
    P = np.asarray([p for p, _ in rows], dtype=np.float64)
    X = np.asarray([[*p, 1.0] for p in P], dtype=np.float64)
    Y = np.asarray([q for _, q in rows], dtype=np.float64)
    coeff, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
    if int(rank) < 3: raise RuntimeError(f"DENSE_BODY_AFFINE_RANK_DEFICIENT_V{view}:{rank}")
    residual = np.linalg.norm(X @ coeff - Y, axis=1)
    span = max(float(np.linalg.norm(Y.max(0) - Y.min(0))), 1e-12)
    p95, mx = float(np.percentile(residual, 95))/span, float(residual.max(initial=0.0))/span
    if p95 > 0.015 or mx > 0.05:
        raise RuntimeError(f"DENSE_BODY_AFFINE_QUALIFICATION_FAIL_V{view}:p95={p95}:max={mx}")
    # Same rank-3 guard as the production directional binding: joints must lie near the admitted affine hull.
    center = P.mean(0); centered = P - center
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tol = max(centered.shape) * np.finfo(np.float64).eps * float(singular[0])
    spatial_rank = int(np.sum(singular > tol))
    if spatial_rank < 2: raise RuntimeError(f"DENSE_BODY_AFFINE_HULL_TOO_LOW_V{view}:{spatial_rank}")
    basis = vt[:spatial_rank].T
    scale = max(float(np.linalg.norm(P.max(0)-P.min(0))), 1e-12)
    hull = max(float(np.linalg.norm((np.asarray(j.position)-center) - basis @ (basis.T @ (np.asarray(j.position)-center))) / scale) for j in skeleton.joints)
    if hull > 0.02: raise RuntimeError(f"DENSE_BODY_JOINT_OUTSIDE_AFFINE_HULL_V{view}:{hull}")
    return coeff, {"correspondence_count": len(rows), "rank": int(rank), "p95_residual01": p95, "max_residual01": mx, "max_joint_affine_hull_residual01": hull}


def _motion(mechanical):
    m = compile_motion_quality(build_mage_rotation_only_qualified_motion(mechanical), mechanical).to_dict()
    md = m.get("metadata") or {}
    if md.get("historical_mesh_authority_used") is True or md.get("historical_weight_authority_used") is True:
        raise RuntimeError("DENSE_BODY_HISTORICAL_MOTION_AUTHORITY_FORBIDDEN")
    if m.get("order_tracks") or m.get("visibility_tracks"):
        raise RuntimeError("DENSE_BODY_NON_ROTATION_MOTION_SCOPE")
    for tr in m.get("joint_tracks") or ():
        for key in tr.get("keys") or ():
            if max(abs(float(v)) for v in key.get("translation_xy", (0,0))) > 1e-12: raise RuntimeError("DENSE_BODY_TRANSLATION_SCOPE_NOT_QUALIFIED")
            if max(abs(float(v)-1.0) for v in key.get("scale_xy", (1,1))) > 1e-12: raise RuntimeError("DENSE_BODY_SCALE_SCOPE_NOT_QUALIFIED")
            if abs(float(key.get("depth_offset", 0.0))) > 1e-12: raise RuntimeError("DENSE_BODY_DEPTH_SCOPE_NOT_QUALIFIED")
    return m


def _motion_samples(m):
    clips = {str(c["clip_id"]): c for c in m.get("clips") or ()}
    tracks = {}
    for tr in m.get("joint_tracks") or ():
        tracks.setdefault(str(tr["clip_id"]), {})[str(tr["canonical_joint_id"])] = tr
    samples = []
    for cid in sorted(clips):
        duration = float(clips[cid]["duration_sec"])
        times = {float(x) for x in np.linspace(0.0, duration, UNIFORM_SAMPLE_COUNT)}
        for tr in tracks.get(cid, {}).values():
            times.update(float(k["time_sec"]) for k in tr.get("keys") or ())
        samples.extend((cid, t) for t in sorted(times))
    if not samples: raise RuntimeError("DENSE_BODY_EMPTY_MOTION_SAMPLE_SET")
    return tracks, tuple(samples)


def _rotation_deg(track, t):
    if not track or not track.get("keys"): return 0.0
    keys = tuple(track["keys"])
    if t <= float(keys[0]["time_sec"]): return float(keys[0]["rotation_deg"])
    if t >= float(keys[-1]["time_sec"]): return float(keys[-1]["rotation_deg"])
    for a, b in zip(keys[:-1], keys[1:]):
        ta, tb = float(a["time_sec"]), float(b["time_sec"])
        if ta <= t <= tb:
            if tb <= ta: return float(b["rotation_deg"])
            u = (t-ta)/(tb-ta)
            return (1-u)*float(a["rotation_deg"]) + u*float(b["rotation_deg"])
    return float(keys[-1]["rotation_deg"])


def _T(xy):
    out = np.eye(3, dtype=np.float64); out[0,2], out[1,2] = float(xy[0]), float(xy[1]); return out


def _R(degrees):
    a = math.radians(float(degrees)); c, s = math.cos(a), math.sin(a)
    return np.asarray([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=np.float64)


def _skinning_matrices(order, by_id, joint_ids, pivots, clip_tracks, t):
    posed, bind = {}, {}
    for jid in order:
        p = np.asarray(pivots[jid], dtype=np.float64); parent = by_id[jid].parent_canonical_id
        bind[jid] = _T(p); delta = _R(_rotation_deg(clip_tracks.get(jid), t))
        if parent is None: posed[jid] = _T(p) @ delta
        else:
            parent = str(parent)
            posed[jid] = posed[parent] @ _T(p - np.asarray(pivots[parent])) @ delta
    return np.asarray([posed[j] @ np.linalg.inv(bind[j]) for j in joint_ids], dtype=np.float64)


def _candidate_faces(dense_xy, dense_faces, body_need):
    tri = dense_xy[dense_faces]
    xmin, xmax = np.floor(tri[:,:,0].min(1)).astype(np.int64), np.ceil(tri[:,:,0].max(1)).astype(np.int64)
    ymin, ymax = np.floor(tri[:,:,1].min(1)).astype(np.int64), np.ceil(tri[:,:,1].max(1)).astype(np.int64)
    h, w = body_need.shape
    valid = (xmax >= 0) & (xmin < w) & (ymax >= 0) & (ymin < h) & ((tri >= 0.0) & (tri <= np.asarray([w-1,h-1]))).all(axis=(1,2))
    integral = np.pad(body_need.astype(np.int32), ((1,0),(1,0))).cumsum(0).cumsum(1)
    x0, x1 = np.clip(xmin,0,w-1), np.clip(xmax,0,w-1); y0, y1 = np.clip(ymin,0,h-1), np.clip(ymax,0,h-1)
    region = integral[y1+1,x1+1]-integral[y0,x1+1]-integral[y1+1,x0]+integral[y0,x0]
    coarse = np.flatnonzero(valid & (region > 0)); exact = []
    for fi in coarse:
        q = tri[int(fi)]; rr, cc = polygon(q[:,1], q[:,0], shape=body_need.shape)
        if len(rr) and bool(body_need[rr,cc].any()): exact.append(int(fi))
    return np.asarray(exact, dtype=np.int64)


def _dynamic_safe_faces(dense_xy, dense_faces, candidate_ids, dense_weights, frame_matrices):
    source_faces = dense_faces[candidate_ids]
    used = np.unique(source_faces.reshape(-1)); reverse = np.full(len(dense_xy), -1, dtype=np.int64); reverse[used] = np.arange(len(used))
    faces = reverse[source_faces]; rest = dense_xy[used]; W = dense_weights[used]
    rest_edges = np.stack((
        np.linalg.norm(rest[faces[:,1]]-rest[faces[:,0]],axis=1),
        np.linalg.norm(rest[faces[:,2]]-rest[faces[:,1]],axis=1),
        np.linalg.norm(rest[faces[:,0]]-rest[faces[:,2]],axis=1),
    ), axis=1); rest_edges = np.maximum(rest_edges, 1e-12)
    rest_area = 0.5*((rest[faces[:,1],0]-rest[faces[:,0],0])*(rest[faces[:,2],1]-rest[faces[:,0],1])-(rest[faces[:,1],1]-rest[faces[:,0],1])*(rest[faces[:,2],0]-rest[faces[:,0],0]))
    rest_area_safe = np.maximum(np.abs(rest_area), 1e-12)
    max_edge = np.ones(len(faces)); max_area = np.ones(len(faces)); flipped = np.zeros(len(faces), dtype=bool)
    x, y = rest[:,0], rest[:,1]
    for frame_index, matrices in enumerate(frame_matrices):
        effective = W @ matrices[:,:2,:].reshape(len(matrices),6)
        posed = np.empty_like(rest); posed[:,0] = effective[:,0]*x + effective[:,1]*y + effective[:,2]; posed[:,1] = effective[:,3]*x + effective[:,4]*y + effective[:,5]
        pe = np.stack((np.linalg.norm(posed[faces[:,1]]-posed[faces[:,0]],axis=1),np.linalg.norm(posed[faces[:,2]]-posed[faces[:,1]],axis=1),np.linalg.norm(posed[faces[:,0]]-posed[faces[:,2]],axis=1)),axis=1)
        raw = pe/rest_edges; max_edge = np.maximum(max_edge, np.maximum(raw,1.0/np.maximum(raw,1e-12)).max(1))
        pa = 0.5*((posed[faces[:,1],0]-posed[faces[:,0],0])*(posed[faces[:,2],1]-posed[faces[:,0],1])-(posed[faces[:,1],1]-posed[faces[:,0],1])*(posed[faces[:,2],0]-posed[faces[:,0],0]))
        raw_a = np.abs(pa)/rest_area_safe; max_area = np.maximum(max_area, np.maximum(raw_a,1.0/np.maximum(raw_a,1e-12))); flipped |= (rest_area*pa)<0.0
        if frame_index % 10 == 0:
            print(f"  dynamic frame {frame_index+1}/{len(frame_matrices)} max_edge={max_edge.max():.6f} max_area={max_area.max():.6f} flips={int(flipped.sum())}", flush=True)
    safe = (max_edge <= EDGE_STRETCH_LIMIT+1e-12) & (max_area <= AREA_CHANGE_LIMIT+1e-12) & (~flipped)
    return safe, max_edge, max_area, flipped


def _select_coverage_faces(dense_xy, dense_faces, safe_face_ids, truth, rigid):
    current = rigid.copy().ravel(); flat_truth = truth.ravel(); truth_count = int(truth.sum())
    intersection, predicted = int(np.count_nonzero(current & flat_truth)), int(current.sum())
    patches = []
    for fi in safe_face_ids:
        q = dense_xy[dense_faces[int(fi)]]; rr, cc = polygon(q[:,1],q[:,0],shape=truth.shape)
        pix = np.unique(rr.astype(np.int64)*truth.shape[1]+cc.astype(np.int64)); new = pix[~current[pix]]
        if len(new):
            tp = int(flat_truth[new].sum()); fp = int(len(new)-tp)
            if tp > 0: patches.append((int(fi),pix,tp,fp))
    patches.sort(key=lambda r:(r[3]==0,r[2]/(r[3]+1),r[2],-r[3],-r[0]), reverse=True)
    selected = []
    for fi,pix,_tp,_fp in patches:
        new = pix[~current[pix]]
        if not len(new): continue
        tp = int(flat_truth[new].sum()); fp = int(len(new)-tp)
        if tp <= 0: continue
        ni, npred = intersection+tp, predicted+tp+fp
        if ni/max(npred,1) < REQUIRED_PRECISION-1e-15: continue
        current[new] = True; intersection,predicted = ni,npred; selected.append(fi)
        if intersection/truth_count >= REQUIRED_RECALL-1e-15: break
    recall, precision = intersection/truth_count, intersection/predicted
    if recall < REQUIRED_RECALL-1e-15 or precision < REQUIRED_PRECISION-1e-15:
        raise RuntimeError(f"DENSE_BODY_COVERAGE_GATE_FAIL:recall={recall}:precision={precision}")
    return tuple(selected), {
        "source_alpha_recall": float(recall), "precision_inside_alpha": float(precision),
        "truth_pixel_count": truth_count, "predicted_pixel_count": predicted, "intersection_pixel_count": intersection,
        "rigid_base_recall": float(np.count_nonzero(rigid & truth)/truth_count),
        "rigid_base_precision": float(np.count_nonzero(rigid & truth)/max(int(rigid.sum()),1)),
    }


def run(args) -> dict:
    zero_path, surface_path = Path(args.zero_surface), Path(args.fit2_surface)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if _sha(surface_path) != EXPECTED_SURFACE_FILE_SHA256: raise RuntimeError("DENSE_BODY_SURFACE_FILE_SHA_DRIFT")
    surface = fit2io.load_surface(surface_path); skeleton = fit2io.load_skeleton(Path(args.skeleton)); skin = fit2io.load_skin(Path(args.fit2_skin)); mechanical = build_mechanical_state(surface,skeleton,skin)
    dense_vertices,dense_faces = _load_zero_surface(zero_path)
    replay = replay_dense_zero_surface_compaction(surface,dense_vertices,source_zero_surface_sha256=EXPECTED_ZERO_SURFACE_SHA256)
    if replay.compact_node_count != fit2io.EXPECTED_SURFACE_NODES: raise RuntimeError("DENSE_BODY_REPLAY_GSA_CARDINALITY_DRIFT")
    cameras = _load_cameras(tuple(map(Path,args.cameras))); observations = _load_observations(tuple(map(Path,args.observations)))
    owner_manifest, rigid_masks = _load_owner_masks(Path(args.owner_manifest), tuple(map(Path,args.owner_rasters)))
    foreground_manifest, foreground_rows = _load_foreground_manifest(Path(args.foreground_manifest))

    motion = _motion(mechanical); tracks, sample_rows = _motion_samples(motion)
    order,by_id = _joint_order(skeleton); joint_ids = tuple(sorted(by_id)); surface_W = _surface_skin_matrix(surface,skin,joint_ids); dense_W = surface_W[replay.dense_vertex_to_compact_index]
    if not np.allclose(dense_W.sum(1),1.0,atol=1e-7,rtol=0.0): raise RuntimeError("DENSE_BODY_DENSE_WEIGHT_SIMPLEX_DRIFT")
    print("DENSE_BODY_REPLAY_PASS "+json.dumps(replay.summary(),sort_keys=True),flush=True)
    print("DENSE_BODY_DYNAMIC_POLICY "+json.dumps({"edge":EDGE_STRETCH_LIMIT,"area":AREA_CHANGE_LIMIT,"flip":0,"sample_count":len(sample_rows)},sort_keys=True),flush=True)

    artifacts, view_rows = [], []
    for view in range(8):
        print(f"\n{'='*96}\nDENSE_BODY_V{view}_START\n{'='*96}",flush=True)
        camera = cameras[view]; dense_xy = project_dense_vertices(dense_vertices,camera,view_index=view); body_need = observations[view] & (~rigid_masks[view])
        candidate_ids = _candidate_faces(dense_xy,dense_faces,body_need)
        if not len(candidate_ids): raise RuntimeError(f"DENSE_BODY_NO_CANDIDATE_FACE_V{view}")
        print(f"V{view} candidate_faces={len(candidate_ids)}",flush=True)
        coeff, affine_report = _fit_view_affine(surface,skeleton,view)
        pivots = {jid: np.asarray([*map(float,by_id[jid].position),1.0]) @ coeff for jid in joint_ids}
        frame_matrices, frame_keys = [], []
        for cid,t in sample_rows:
            frame_matrices.append(_skinning_matrices(order,by_id,joint_ids,pivots,tracks.get(cid,{}),float(t))); frame_keys.append((cid,float(t)))
        safe_mask,max_edge,max_area,flipped = _dynamic_safe_faces(dense_xy,dense_faces,candidate_ids,dense_W,tuple(frame_matrices)); safe_ids = candidate_ids[safe_mask]
        print(f"V{view} dynamically_safe_faces={len(safe_ids)}/{len(candidate_ids)}",flush=True)
        selected,coverage = _select_coverage_faces(dense_xy,dense_faces,safe_ids,observations[view],rigid_masks[view]); selected_set = set(selected)
        local = np.asarray([i for i,fi in enumerate(candidate_ids) if int(fi) in selected_set],dtype=np.int64)
        if not len(local): raise RuntimeError(f"DENSE_BODY_EMPTY_SELECTED_SET_V{view}")
        sel_edge,sel_area,sel_flips = float(max_edge[local].max()),float(max_area[local].max()),int(flipped[local].sum())
        if sel_edge > EDGE_STRETCH_LIMIT+1e-12 or sel_area > AREA_CHANGE_LIMIT+1e-12 or sel_flips:
            raise RuntimeError(f"DENSE_BODY_SELECTED_DYNAMIC_GATE_FAIL_V{view}:{sel_edge}:{sel_area}:{sel_flips}")
        selected_sha = _array_sha(np.asarray(sorted(selected_set),dtype=np.int64))
        dynamic = {
            "schema":SCHEMA+".DynamicFaceWitness.v1","view":view,"source_zero_surface_sha256":EXPECTED_ZERO_SURFACE_SHA256,
            "surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,
            "motion_state_hash":str(motion["motion_state_hash"]),"sample_rows":frame_keys,"candidate_face_count":int(len(candidate_ids)),"dynamically_safe_face_count":int(len(safe_ids)),
            "selected_face_count":len(selected_set),"selected_face_set_sha256":selected_sha,"edge_stretch_limit":EDGE_STRETCH_LIMIT,"area_change_limit":AREA_CHANGE_LIMIT,
            "triangle_flip_limit":0,"selected_max_edge_stretch_ratio":sel_edge,"selected_max_area_change_ratio":sel_area,"selected_triangle_flip_count":sel_flips,"affine_projection_report":affine_report,
        }
        dynamic_hash = content_sha256(dynamic); dynamic_path = output_dir/f"V{view}_DENSE_BODY_DYNAMIC_FACE_WITNESS.json"; _write_json(dynamic_path,{**dynamic,"witness_hash":dynamic_hash})
        coverage_witness = {
            "schema":SCHEMA+".CoverageWitness.v1","view":view,"observation_sha256":ceiling_v1.OBSERVATION_SHA256[view],"owner_raster_sha256":_sha(Path(args.owner_rasters[view])),
            "owner_manifest_sha256":EXPECTED_OWNER_MANIFEST_SHA256,"selected_face_set_sha256":selected_sha,"required_recall":REQUIRED_RECALL,"required_precision":REQUIRED_PRECISION,**coverage,
        }
        coverage_hash = content_sha256(coverage_witness); coverage_path = output_dir/f"V{view}_DENSE_BODY_COVERAGE_WITNESS.json"; _write_json(coverage_path,{**coverage_witness,"witness_hash":coverage_hash})
        candidate = build_dense_zero_surface_candidate(surface,dense_vertices,dense_faces,replay,selected_face_indices=selected_set,camera=camera,view_index=view,camera_binding_hash=ceiling_v1.CAMERA_SHA256[view],source_zero_surface_sha256=EXPECTED_ZERO_SURFACE_SHA256,dynamic_witness_hash=dynamic_hash,coverage_witness_hash=coverage_hash)
        mesh,topology_report = qualify_dense_zero_surface_mesh(surface,candidate,dense_vertices,dense_faces,replay,camera=camera,source_zero_surface_sha256=EXPECTED_ZERO_SURFACE_SHA256)
        mesh_skin = bind_mwb2_mesh_skin(surface,skeleton,skin,mesh,require_anchor_unified_topology=False)
        if mesh_skin.skin_binding_hash != fit2io.EXPECTED_SKIN_LINEAGE or mesh_skin.metadata.get("historical_weight_transfer_used") is not False:
            raise RuntimeError(f"DENSE_BODY_MESH_SKIN_AUTHORITY_DRIFT_V{view}")
        fg = foreground_rows[view]
        appearance = build_sprite_panel_appearance(mesh,target_view_index=view,source_observation_hash=ceiling_v1.OBSERVATION_SHA256[view],source_width=int(fg["width"]),source_height=int(fg["height"]),atlas_width=int(fg["atlas_width"]),atlas_height=int(fg["atlas_height"]),panel_x_offset=0,atlas_payload_hash=str(fg["atlas_payload_hash"])); validate_appearance_binding(appearance)
        names = {"mesh":f"V{view}_DENSE_ZERO_SURFACE_BODY_MESH.json","skin":f"V{view}_DENSE_ZERO_SURFACE_BODY_SKIN.json","appearance":f"V{view}_DENSE_ZERO_SURFACE_BODY_APPEARANCE.json","dynamic":dynamic_path.name,"coverage":coverage_path.name}
        for key,obj in (("mesh",mesh),("skin",mesh_skin),("appearance",appearance)): _write_json(output_dir/names[key],obj.to_dict())
        local_artifacts=[]
        for name in names.values():
            p=output_dir/name; row={"path":name,"sha256":_sha(p),"bytes":p.stat().st_size}; local_artifacts.append(row); artifacts.append(row)
        view_rows.append({"view":view,"camera_sha256":ceiling_v1.CAMERA_SHA256[view],"observation_sha256":ceiling_v1.OBSERVATION_SHA256[view],"owner_raster_sha256":_sha(Path(args.owner_rasters[view])),"candidate_face_count":int(len(candidate_ids)),"dynamically_safe_face_count":int(len(safe_ids)),"selected_face_count":len(selected_set),"selected_face_set_sha256":selected_sha,"selected_max_edge_stretch_ratio":sel_edge,"selected_max_area_change_ratio":sel_area,"selected_triangle_flip_count":sel_flips,"coverage":coverage,"topology_qualification":topology_report,"mesh_lineage_hash":mesh.mesh_lineage_hash,"mesh_skin_lineage_hash":mesh_skin.mesh_skin_lineage_hash,"appearance_lineage_hash":appearance.appearance_lineage_hash,"files":names,"artifacts":local_artifacts})
        print(f"DENSE_BODY_V{view}_PASS "+json.dumps({"faces":len(selected_set),"recall":coverage["source_alpha_recall"],"precision":coverage["precision_inside_alpha"],"edge":sel_edge,"area":sel_area,"flips":sel_flips},sort_keys=True),flush=True)

    manifest = {
        "schema":SCHEMA,"status":"PASS__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY_V0_V7","topology_method":DENSE_ZERO_SURFACE_TOPOLOGY_METHOD,
        "zero_surface_sha256":EXPECTED_ZERO_SURFACE_SHA256,"surface_file_sha256":EXPECTED_SURFACE_FILE_SHA256,"surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,
        "motion_state_hash":str(motion["motion_state_hash"]),"owner_manifest_sha256":EXPECTED_OWNER_MANIFEST_SHA256,"foreground_manifest_sha256":EXPECTED_FOREGROUND_MANIFEST_SHA256,"compaction_replay":replay.summary(),
        "policy":{"edge_stretch_limit":EDGE_STRETCH_LIMIT,"area_change_limit":AREA_CHANGE_LIMIT,"triangle_flip_limit":0,"required_recall":REQUIRED_RECALL,"required_precision":REQUIRED_PRECISION,"uniform_sample_count":UNIFORM_SAMPLE_COUNT},
        "views":view_rows,"artifacts":artifacts,"P1_or_P1Q_topology_used":False,"source_or_teacher_mesh_topology_used":False,"historical_weight_transfer_used":False,"model_query_used":False,"training_executed":False,"threshold_relaxation_used":False,"product_pass_claimed":False,
    }
    manifest_path=output_dir/"FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"; _write_json(manifest_path,manifest); manifest_sha=_sha(manifest_path)
    seal={"schema":SCHEMA+".Seal.v1","status":"SEALED__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY","manifest":manifest_path.name,"manifest_sha256":manifest_sha,"artifact_count":len(artifacts),"surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,"product_pass_claimed":False}
    _write_json(output_dir/"FIT2_DENSE_ZERO_SURFACE_BODY_SEAL.json",seal)
    print("DENSE_BODY_MATERIALIZATION="+json.dumps(seal,sort_keys=True),flush=True)
    return {"manifest":manifest,"seal":seal}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--zero-surface",required=True); p.add_argument("--fit2-surface",required=True); p.add_argument("--skeleton",required=True); p.add_argument("--fit2-skin",required=True); p.add_argument("--cameras",nargs=8,required=True); p.add_argument("--observations",nargs=8,required=True); p.add_argument("--owner-manifest",required=True); p.add_argument("--owner-rasters",nargs=8,required=True); p.add_argument("--foreground-manifest",required=True); p.add_argument("--output-dir",required=True); return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
