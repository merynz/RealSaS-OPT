"""Exact CPU replay of the sealed V5.5.1 FIT motion witness, no new mechanics.

The frame/motion/LBS/measurement arithmetic below is preserved from the pinned
notebook, with its sealed inactive coincident-frame receipt applied. This helper
must reproduce every sealed frame metric before downstream use is admissible.
No training, skeleton repair, skin transfer or product qualification occurs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROLE_MAP = {
    "root":"Bone", "hips":"Body", "spine":"Abdomen", "chest":"Torso",
    "upperarm.l":"UpperArm.L", "lowerarm.l":"LowerArm.L", "wrist.l":"Palm.L", "hand.l":"MiddleHand.L", "handslot.l":None,
    "upperarm.r":"UpperArm.R", "lowerarm.r":"LowerArm.R", "wrist.r":"Palm.R", "hand.r":"MiddleHand.R", "handslot.r":None, "head":"Head",
    "upperleg.l":"UpperLeg.L", "lowerleg.l":"LowerLeg.L", "foot.l":None, "toes.l":None,
    "upperleg.r":"UpperLeg.R", "lowerleg.r":"LowerLeg.R", "foot.r":None, "toes.r":None,
}
CORE_NAMES = tuple(ROLE_MAP)


def read_ref(ref, *, raw=False):
    path = Path(ref["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise RuntimeError("SEALED_WITNESS_INPUT_HASH_DRIFT:" + path.name)
    return path if raw else json.loads(path.read_text())


def array_contract_hash(rows):
    h=hashlib.sha256()
    for name,a in rows:
        a=np.ascontiguousarray(a); h.update(str(name).encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()

def carrier_basis_hash(vertices, faces):
    return array_contract_hash([("M.vertices_world",np.asarray(vertices,np.float64)),("M.faces",np.asarray(faces,np.int64))])

def unit(v, code):
    a=np.asarray(v,np.float64); n=float(np.linalg.norm(a))
    if a.shape!=(3,) or not np.isfinite(a).all() or n<=1e-12: raise RuntimeError(code)
    return a/n

def object_frame_from_axis(heads):
    idx={name:i for i,name in enumerate(CORE_NAMES)}
    up=unit(heads[idx["head"]]-heads[idx["hips"]],"V55_UP")
    lateral=sum((heads[idx[b+".r"]]-heads[idx[b+".l"]] for b in ("upperarm","upperleg","foot")),np.zeros(3))
    right=unit(lateral-up*float(np.dot(lateral,up)),"V55_RIGHT")
    forward=unit(np.cross(up,right),"V55_FORWARD")
    right=unit(np.cross(forward,up),"V55_RIGHT_ORTHO")
    C=np.stack((right,forward,up),axis=0)
    assert float(np.linalg.det(C))>0.999999
    return C

def derive_frames(positions,parents,inactive):
    gr=unit((1,0,0),"V55_GR"); gu=unit((0,0,1),"V55_GU")
    gf=unit(np.cross(gr,gu),"V55_GF")
    if float(np.dot(gf,np.asarray((0,1,0),float)))<0: gf=-gf
    gr=unit(np.cross(gu,gf),"V55_GR2")
    out=[]
    for i,pos in enumerate(positions):
        p=int(parents[i])
        if p<0:
            R=np.column_stack((gr,gu,gf))
        elif i in inactive:
            if np.linalg.norm(pos-positions[p]) > 1e-12:
                raise RuntimeError("SEALED_COINCIDENT_FRAME_POSITION_DRIFT")
            R=out[p].copy()
        else:
            y=unit(pos-positions[p],"V55_BONE")
            z=gf-y*float(np.dot(gf,y))
            if np.linalg.norm(z)<=1e-8: z=gu-y*float(np.dot(gu,y))
            if np.linalg.norm(z)<=1e-8: z=gr-y*float(np.dot(gr,y))
            z=unit(z,"V55_SECONDARY"); x=unit(np.cross(y,z),"V55_PRIMARY")
            if float(np.dot(x,gr))<0: x=-x; z=-z
            z=unit(np.cross(x,y),"V55_ORTHO")
            R=np.column_stack((x,y,z))
        assert float(np.linalg.det(R))>0.999999
        out.append(R)
    return np.asarray(out)

def topo_order(parents):
    remaining=set(range(len(parents))); out=[]
    while remaining:
        moved=False
        for i in sorted(tuple(remaining)):
            p=int(parents[i])
            if p<0 or p in out:
                out.append(i); remaining.remove(i); moved=True
        if not moved: raise RuntimeError("V55_AXIS_SKELETON_CYCLE")
    return tuple(out)

def apply_lbs(rest,weights,matrices):
    hom=np.concatenate((rest,np.ones((len(rest),1),np.float64)),axis=1)
    moved=np.einsum("jab,nb->jna",matrices,hom,optimize=True)[:,:,:3]
    return np.einsum("nj,jna->na",weights,moved,optimize=True)

def triangle_metrics_batch(rest,posed,faces):
    r=rest[faces]; p=posed[faces]
    re=np.stack((np.linalg.norm(r[:,1]-r[:,0],axis=1),np.linalg.norm(r[:,2]-r[:,1],axis=1),np.linalg.norm(r[:,0]-r[:,2],axis=1)),1)
    pe=np.stack((np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)),1)
    ratio=pe/np.maximum(re,1e-12)
    r1=r[:,1]-r[:,0]; r2=r[:,2]-r[:,0]
    l1=np.linalg.norm(r1,axis=1); u=r1/np.maximum(l1[:,None],1e-12)
    x2=np.sum(r2*u,axis=1); perp=r2-x2[:,None]*u; y2=np.linalg.norm(perp,axis=1)
    if np.any(l1<=1e-12) or np.any(y2<=1e-12): raise RuntimeError("V55_REST_TRIANGLE_DEGENERATE")
    inv=np.zeros((len(faces),2,2),np.float64)
    inv[:,0,0]=1/l1; inv[:,0,1]=-x2/(l1*y2); inv[:,1,1]=1/y2
    F=np.einsum("nij,njk->nik",np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=2),inv)
    sv=np.linalg.svd(F,compute_uv=False); area=sv[:,0]*sv[:,1]; cond=sv[:,0]/np.maximum(sv[:,1],1e-15)
    return area,cond,ratio.min(1),ratio.max(1)


def replay_witness(config, promotion):
    expected = promotion["sealed_fit1_evidence"]
    for key, evidence_key in (("weights", "mira_v55_w_m_sha256"),
                              ("axis", "axis_v541_qualified_skeleton_sha256"),
                              ("alignment", "axis_v541_alignment_sha256"),
                              ("frame_qualification", "compiler_coincident_frame_qualification_sha256")):
        if config[key]["sha256"] != expected[evidence_key]:
            raise RuntimeError("SEALED_WITNESS_PROMOTION_BINDING_DRIFT:" + key)
    with np.load(read_ref(config["weights"], raw=True), allow_pickle=False) as data:
        rest_source = np.asarray(data["vertices_rest_source_frame"], dtype=np.float64)
        faces = np.asarray(data["faces"], dtype=np.int64)
        weights = np.asarray(data["weights_axis41_canonical"], dtype=np.float64)
        raw_column = np.asarray(data["raw41_index_for_column"], dtype=np.int64)
        joint_ids = np.asarray(data["canonical_joint_ids"]).astype(str)
        basis = str(np.asarray(data["carrier_basis_sha256"]).item())
    if basis != expected["carrier_basis_sha256"] or carrier_basis_hash(rest_source, faces) != basis:
        raise RuntimeError("SEALED_WITNESS_CARRIER_BASIS_DRIFT")
    alignment = read_ref(config["alignment"])["canonical_to_raw_teacher_index_fit_only"]
    axis = read_ref(config["axis"])
    axis_positions = np.zeros((41, 3))
    parents = np.full(41, -1, dtype=int)
    for joint in axis["joints"]:
        raw = alignment[joint["canonical_joint_id"]]
        axis_positions[raw] = joint["position"]
        if joint["parent_canonical_id"] is not None:
            parents[raw] = alignment[joint["parent_canonical_id"]]
    raw_weights = np.zeros_like(weights)
    if sorted(raw_column.tolist()) != list(range(41)):
        raise RuntimeError("SEALED_WITNESS_WEIGHT_COLUMNS_DRIFT")
    for col, raw in enumerate(raw_column):
        if int(alignment[joint_ids[col]]) != int(raw):
            raise RuntimeError("SEALED_WITNESS_ALIGNMENT_DRIFT")
        raw_weights[:, raw] = weights[:, col]
    receipt = read_ref(config["frame_qualification"])
    inactive = set()
    for event in receipt["events"]:
        child, parent = event["child_raw_index"], event["parent_raw_index"]
        if (parents[child] != parent or event["resolution"] != "COINCIDENT_INACTIVE__INHERIT_PARENT_FRAME"
                or np.sum(raw_weights[:, child]) > receipt["inactive_weight_mass_max"]
                or np.max(raw_weights[:, child]) > receipt["inactive_weight_mass_max"]):
            raise RuntimeError("SEALED_WITNESS_COINCIDENT_RECEIPT_DRIFT")
        inactive.add(child)
    C = object_frame_from_axis(axis_positions[:23])
    rest = (C @ rest_source.T).T
    positions = (C @ axis_positions.T).T
    frames = derive_frames(positions, parents, inactive)
    globals_ = np.repeat(np.eye(4)[None], 41, axis=0)
    globals_[:, :3, :3], globals_[:, :3, 3] = frames, positions
    inverse = np.linalg.inv(globals_)
    locals_ = np.asarray([globals_[i] if parents[i] < 0 else inverse[parents[i]] @ globals_[i] for i in range(41)])
    order = topo_order(parents)
    body_scale = max(np.linalg.norm(p - positions[0]) for p in positions[:23])
    sealed_rows = read_ref(config["frame_metrics"])["frames"]
    sealed = {(r["clip_id"], r["frame_index"]): r for r in sealed_rows}
    arrays = {"vertices": rest_source, "faces": faces, "axis_positions_source": axis_positions,
              "axis_parents": parents}
    clips, unsafe = [], set()
    C4 = np.eye(4)
    C4[:3, :3] = C
    count, maximum_error = 0, 0.0
    for ci, ref in enumerate(config["motion"]):
        payload = read_ref(ref)
        if payload["schema"] != "RealSaS.MotionSourceClip.v2" or payload["coordinate_frame"] != "REALSAS_OBJECT_FRAME_V1":
            raise RuntimeError("SEALED_WITNESS_MOTION_SCHEMA_DRIFT")
        tracks = {t["source_joint_id"]: t["keyframes"] for t in payload["tracks"]}
        times = np.asarray([k["time_seconds"] for k in next(iter(tracks.values()))])
        posed_frames, matrices = [], []
        for fi, t in enumerate(times):
            posed = np.zeros_like(globals_)
            posed[:, 3, 3] = 1
            for raw in order:
                delta = np.eye(4)
                if raw < 23 and ROLE_MAP[CORE_NAMES[raw]] is not None:
                    key = tracks[ROLE_MAP[CORE_NAMES[raw]]][fi]
                    delta[:3, :3] = Rotation.from_quat(key["local_rotation_quat_xyzw"]).as_matrix()
                    if CORE_NAMES[raw] == "hips":
                        delta[:3, 3] = np.asarray(key["local_translation_xyz"]) * body_scale
                parent = parents[raw]
                posed[raw] = globals_[raw] @ delta if parent < 0 else posed[parent] @ locals_[raw] @ delta
            skin = posed @ inverse
            moved = apply_lbs(rest, raw_weights, skin)
            area, cond, emin, emax = triangle_metrics_batch(rest, moved, faces)
            invalid = (area < .05) | (area > 20) | (cond > 16) | (emax > 4)
            unsafe.update(np.flatnonzero(invalid).tolist())
            measured = dict(time_seconds=float(t), unsafe_face_count=int(np.count_nonzero(invalid)),
                edge_gt_4=int(np.count_nonzero(emax > 4)), edge_gt_10=int(np.count_nonzero(emax > 10)),
                edge_p99=float(np.quantile(emax, .99)), edge_max=float(emax.max()),
                area_min=float(area.min()), area_max=float(area.max()), condition_max=float(cond.max()))
            recorded = sealed.get((payload["clip_id"], fi))
            if recorded is None:
                raise RuntimeError("SEALED_WITNESS_FRAME_MATRIX_DRIFT")
            for key, number in measured.items():
                error = abs(number - recorded[key])
                maximum_error = max(maximum_error, error)
                if not np.isclose(number, recorded[key], atol=1e-8, rtol=1e-8):
                    raise RuntimeError(f"SEALED_WITNESS_REPLAY_MISMATCH:{payload['clip_id']}:{fi}:{key}:{error}")
            posed_frames.append((C.T @ moved.T).T)
            matrices.append(C4.T @ skin @ C4)
            count += 1
        arrays[f"clip_{ci}_times"] = times
        arrays[f"clip_{ci}_canonical_xyz"] = np.asarray(posed_frames)
        arrays[f"clip_{ci}_skin_matrices_source"] = np.asarray(matrices)
        clips.append({"clip_id": payload["clip_id"], "clip_kind": payload["clip_kind"],
                      "duration_seconds": payload["duration_seconds"], "loop": payload["loop"],
                      "frame_count": len(times), "array_prefix": f"clip_{ci}"})
    if count != len(sealed):
        raise RuntimeError("SEALED_WITNESS_FRAME_MATRIX_INCOMPLETE")
    arrays["unsafe_faces_diagnostic_union"] = np.asarray(sorted(unsafe), dtype=np.int64)
    return arrays, clips, {"sealed_frame_metric_replay_passed": True,
        "frame_count": count, "maximum_metric_replay_error": maximum_error,
        "carrier_basis_sha256": basis, "mechanics_requalified": False,
        "mechanical_inference_executed": False, "product_authority": False}
