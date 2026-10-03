"""Direct LBS deformation-constrained mesh-weight projection audit.

Diagnostic only. Keeps the qualified Arachne surface skin fixed and tests whether
Compiler mesh-weight binding can realize that evidence on the current carrier
with a bounded, subject-free numerical correction.

The optimizer never consumes authored clips. It builds constraints only from the
generic G3 local-frame micro-stress probe bank, then evaluates every candidate
with the frozen G3 thresholds and the separate 51-frame exact motion court.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsmr

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.joint_frames_v1 import (
    derive_joint_frames_from_skeleton,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
    _pose_skin_matrices,
    _triangle_metrics_batch_exact,
)
from compiler.realsas_compiler_core.mesh.dynamic_frame_court_v1 import (
    measure_dynamic_frame_geometry_v1,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    DEFAULT_MAX_EDGE_RATIO,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import (
    CLIPS,
    FULL_MOTION_SAMPLES,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx, _skin, _tracks_for_clip

LAMBDAS = (0.1, 1.0, 10.0, 100.0)
MAX_CONSTRAINTS = 8192
LOWER_EDGE_RATIO = 1.0 / DEFAULT_MAX_EDGE_RATIO
SUPPORT_EPS = 1e-8
NEW_SUPPORT_EPS = 1e-6


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def q(a, p):
    return float(np.quantile(np.asarray(a, dtype=np.float64), p))


def unique_edges(faces):
    values = set()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            u = int(u)
            v = int(v)
            values.add((u, v) if u < v else (v, u))
    edges = np.asarray(sorted(values), dtype=np.int64)
    lookup = {tuple(map(int, e)): i for i, e in enumerate(edges)}
    face_edges = np.asarray(
        [
            [
                lookup[tuple(sorted((int(a), int(b))))],
                lookup[tuple(sorted((int(b), int(c))))],
                lookup[tuple(sorted((int(c), int(a))))],
            ]
            for a, b, c in np.asarray(faces, dtype=np.int64)
        ],
        dtype=np.int64,
    )
    return edges, face_edges


def probe_bank(skeleton, cameras):
    frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)
    joint_ids = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    probes = []
    for jid in sorted(joint_ids):
        for axis_index, axis_name in enumerate(("X", "Y", "Z")):
            for sign in (-1.0, 1.0):
                degrees = sign * G3_LOCAL_MICRO_STRESS_ANGLE_DEG
                by_id = _pose_skin_matrices(
                    skeleton,
                    frames,
                    joint_id=jid,
                    local_axis_index=axis_index,
                    degrees=degrees,
                )
                matrices = np.stack([by_id[x] for x in joint_ids], axis=0)
                probes.append(
                    (
                        f"{jid}:LOCAL_{axis_name}:{degrees:+g}",
                        matrices.astype(np.float64),
                    )
                )
    return joint_ids, tuple(probes)


def transformed_by_joint(rest, matrices):
    hom = np.concatenate(
        [np.asarray(rest, dtype=np.float64), np.ones((len(rest), 1), dtype=np.float64)],
        axis=1,
    )
    return np.stack(
        [(hom @ matrices[j].T)[:, :3] for j in range(len(matrices))],
        axis=1,
    )


def posed_from_weights(rest, weights, matrices):
    per = transformed_by_joint(rest, matrices)
    return np.sum(per * np.asarray(weights, dtype=np.float64)[:, :, None], axis=1)


def simplex_project_rows(values, support_mask):
    """Project each row onto its original Arachne support simplex.

    Mesh-weight binding is not allowed to invent a joint support that is absent
    from the deterministic transfer of the qualified surface skin evidence.
    """
    v = np.asarray(values, dtype=np.float64)
    mask = np.asarray(support_mask, dtype=bool)
    if mask.shape != v.shape:
        raise RuntimeError("DIRECT_LBS_SUPPORT_MASK_SHAPE_INVALID")
    out = np.zeros_like(v)
    for i in range(len(v)):
        active = np.flatnonzero(mask[i])
        if not len(active):
            raise RuntimeError("DIRECT_LBS_SOURCE_SUPPORT_EMPTY")
        row = v[i, active]
        u = np.sort(row)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, len(u) + 1, dtype=np.float64)
        cond = u - cssv / ind > 0.0
        rho = max(int(np.count_nonzero(cond)) - 1, 0)
        theta = float(cssv[rho] / float(rho + 1))
        out[i, active] = np.maximum(row - theta, 0.0)
    mass = out.sum(axis=1, keepdims=True)
    if np.any(mass <= 1e-12) or not np.isfinite(out).all():
        raise RuntimeError("DIRECT_LBS_SIMPLEX_PROJECTION_INVALID")
    out /= mass
    if np.any(out[~mask] != 0.0):
        raise RuntimeError("DIRECT_LBS_NEW_SUPPORT_CREATED")
    return out


def evaluate_g3_and_collect_constraints(
    rest,
    weights,
    faces,
    edges,
    face_edges,
    probes,
    policy,
    *,
    collect_constraints,
):
    rest = np.asarray(rest, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    rest_vec = rest[edges[:, 0]] - rest[edges[:, 1]]
    rest_len = np.linalg.norm(rest_vec, axis=1)
    if np.any(rest_len <= 1e-12):
        raise RuntimeError("DIRECT_LBS_REST_EDGE_DEGENERATE")

    failures = set()
    global_min_area = float("inf")
    global_max_area = 0.0
    global_max_condition = 0.0
    global_min_edge = float("inf")
    global_max_edge = 0.0
    constraints = {}
    per_probe = []

    for probe_index, (probe_id, matrices) in enumerate(probes):
        posed = posed_from_weights(rest, weights, matrices)
        area, condition, edge_min_face, edge_max_face, smin = _triangle_metrics_batch_exact(
            rest, posed, faces
        )
        finite = (
            np.isfinite(area)
            & np.isfinite(condition)
            & np.isfinite(edge_min_face)
            & np.isfinite(edge_max_face)
            & np.isfinite(smin)
        )
        if not np.all(finite):
            failures.add("NONFINITE_DEFORMATION_METRIC")

        pvec = posed[edges[:, 0]] - posed[edges[:, 1]]
        plen = np.linalg.norm(pvec, axis=1)
        ratio = plen / rest_len

        bad_area_low = area < float(policy.g3_min_dynamic_area_ratio)
        bad_area_high = area > float(policy.g3_max_dynamic_area_ratio)
        bad_cond = condition > float(policy.g3_max_dynamic_condition_number)
        if np.any(bad_area_low):
            failures.add("DYNAMIC_AREA_RATIO_BELOW_MIN")
        if np.any(bad_area_high):
            failures.add("DYNAMIC_AREA_RATIO_ABOVE_MAX")
        if np.any(bad_cond):
            failures.add("DYNAMIC_CONDITION_NUMBER_ABOVE_MAX")

        global_min_area = min(global_min_area, float(np.nanmin(area)))
        global_max_area = max(global_max_area, float(np.nanmax(area)))
        global_max_condition = max(global_max_condition, float(np.nanmax(condition)))
        global_min_edge = min(global_min_edge, float(np.nanmin(edge_min_face)))
        global_max_edge = max(global_max_edge, float(np.nanmax(edge_max_face)))

        if collect_constraints:
            edge_severity = np.maximum(
                ratio / float(DEFAULT_MAX_EDGE_RATIO),
                float(LOWER_EDGE_RATIO) / np.maximum(ratio, 1e-12),
            )
            direct_edge_bad = (ratio > float(DEFAULT_MAX_EDGE_RATIO)) | (
                ratio < float(LOWER_EDGE_RATIO)
            )
            for ei in np.flatnonzero(direct_edge_bad):
                key = (int(probe_index), int(ei))
                constraints[key] = max(
                    float(constraints.get(key, 0.0)),
                    float(edge_severity[ei]),
                )

            bad_faces = np.flatnonzero(bad_area_low | bad_area_high | bad_cond)
            for fi in bad_faces:
                sev = max(
                    float(condition[fi])
                    / max(float(policy.g3_max_dynamic_condition_number), 1e-12),
                    float(area[fi])
                    / max(float(policy.g3_max_dynamic_area_ratio), 1e-12),
                    float(policy.g3_min_dynamic_area_ratio)
                    / max(float(area[fi]), 1e-12),
                )
                for ei in face_edges[int(fi)]:
                    key = (int(probe_index), int(ei))
                    constraints[key] = max(
                        float(constraints.get(key, 0.0)),
                        float(sev),
                    )

        per_probe.append(
            {
                "probe_id": probe_id,
                "minimum_area_ratio": float(np.nanmin(area)),
                "maximum_area_ratio": float(np.nanmax(area)),
                "maximum_condition_number": float(np.nanmax(condition)),
                "minimum_edge_ratio": float(np.nanmin(edge_min_face)),
                "maximum_edge_ratio": float(np.nanmax(edge_max_face)),
            }
        )

    ranked_constraints = []
    if collect_constraints:
        ranked_constraints = sorted(
            (
                {
                    "probe_index": pi,
                    "edge_index": ei,
                    "severity": severity,
                }
                for (pi, ei), severity in constraints.items()
            ),
            key=lambda row: (
                -float(row["severity"]),
                int(row["probe_index"]),
                int(row["edge_index"]),
            ),
        )
        ranked_constraints = ranked_constraints[:MAX_CONSTRAINTS]

    return {
        "passed": not failures,
        "failure_invariants": sorted(failures),
        "minimum_area_ratio": float(global_min_area),
        "maximum_area_ratio": float(global_max_area),
        "maximum_condition_number": float(global_max_condition),
        "minimum_edge_ratio": float(global_min_edge),
        "maximum_edge_ratio": float(global_max_edge),
        "per_probe": per_probe,
        "constraint_count": len(ranked_constraints),
        "constraints_truncated": bool(len(constraints) > len(ranked_constraints)),
        "raw_constraint_count": len(constraints),
    }, ranked_constraints


def build_constraint_system(rest, edges, probes, constraints, current_weights):
    """Build direct LBS relative-edge target rows.

    Each selected probe/edge target preserves the current deformed edge direction
    while restoring its length to the rest-edge length. This is a deterministic
    sequential quadratic projection, not a pose or part-specific target.
    """
    rest = np.asarray(rest, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    J = current_weights.shape[1]
    nvar = current_weights.size

    rows = []
    cols = []
    data = []
    rhs = []
    row_cursor = 0

    by_probe = {}
    for item in constraints:
        by_probe.setdefault(int(item["probe_index"]), []).append(item)

    for pi in sorted(by_probe):
        matrices = probes[pi][1]
        items = by_probe[pi]
        edge_ids = np.asarray([int(x["edge_index"]) for x in items], dtype=np.int64)
        endpoints = np.unique(edges[edge_ids].reshape(-1))
        local = {int(v): i for i, v in enumerate(endpoints.tolist())}
        per = transformed_by_joint(rest[endpoints], matrices)
        current_pos = np.sum(
            per * current_weights[endpoints, :, None],
            axis=1,
        )

        for item, ei in zip(items, edge_ids.tolist()):
            u, v = map(int, edges[ei])
            qu = per[local[u]]
            qv = per[local[v]]
            d = current_pos[local[u]] - current_pos[local[v]]
            dlen = float(np.linalg.norm(d))
            rvec = rest[u] - rest[v]
            rlen = float(np.linalg.norm(rvec))
            if rlen <= 1e-12:
                raise RuntimeError("DIRECT_LBS_CONSTRAINT_REST_EDGE_DEGENERATE")
            direction = d / dlen if dlen > 1e-12 else rvec / rlen
            target = direction * rlen
            scale = math.sqrt(max(float(item["severity"]), 1.0)) / rlen

            for dim in range(3):
                rr = row_cursor
                base_u = u * J
                base_v = v * J
                for j in range(J):
                    rows.append(rr)
                    cols.append(base_u + j)
                    data.append(float(scale * qu[j, dim]))
                    rows.append(rr)
                    cols.append(base_v + j)
                    data.append(float(-scale * qv[j, dim]))
                rhs.append(float(scale * target[dim]))
                row_cursor += 1

    A = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(row_cursor, nvar),
        dtype=np.float64,
    )
    b = np.asarray(rhs, dtype=np.float64)
    return A, b


def prepare_simplex_tangent_system(W0, A, b, support_mask):
    """Parameterize the original-support simplex exactly.

    For every vertex choose its highest-mass admitted joint as a pivot. Each
    free variable transfers mass between one admitted joint and that pivot:
        w_j = w0_j + z_j
        w_p = w0_p - sum_j z_j
    Therefore original support and row-sum=1 are hard constraints of the linear
    solve rather than a post-hoc normalization. This also removes the common
    affine/translation mode that made the tiny-edge direct-LBS system poorly
    conditioned.
    """
    W0=np.asarray(W0,dtype=np.float64)
    mask=np.asarray(support_mask,dtype=bool)
    if W0.shape!=mask.shape or W0.ndim!=2:
        raise RuntimeError("DIRECT_LBS_SIMPLEX_TANGENT_SHAPE_INVALID")
    if np.any(W0[~mask]!=0.0):
        raise RuntimeError("DIRECT_LBS_INACTIVE_SOURCE_MASS_NONZERO")
    if not np.allclose(W0.sum(axis=1),1.0,atol=1e-10,rtol=0.0):
        raise RuntimeError("DIRECT_LBS_SOURCE_SIMPLEX_INVALID")

    n,jcount=W0.shape
    rows=[]; cols=[]; data=[]
    pivots=[]
    col=0
    for i in range(n):
        active=np.flatnonzero(mask[i])
        if not len(active):
            raise RuntimeError("DIRECT_LBS_SOURCE_SUPPORT_EMPTY")
        pivot=int(active[np.argmax(W0[i,active])])
        pivots.append(pivot)
        for jid in active:
            jid=int(jid)
            if jid==pivot:
                continue
            rows.extend((i*jcount+jid,i*jcount+pivot))
            cols.extend((col,col))
            data.extend((1.0,-1.0))
            col+=1

    if col<=0:
        raise RuntimeError("DIRECT_LBS_SIMPLEX_TANGENT_EMPTY")
    T=sparse.csr_matrix(
        (np.asarray(data,dtype=np.float64),(rows,cols)),
        shape=(n*jcount,col),dtype=np.float64,
    )
    AT=(A@T).tocsr()
    x0=W0.reshape(-1)
    d=np.asarray(b-A@x0,dtype=np.float64).reshape(-1)
    if not np.isfinite(d).all():
        raise RuntimeError("DIRECT_LBS_SIMPLEX_TANGENT_RHS_NONFINITE")
    column_sum=np.asarray(T.sum(axis=0),dtype=np.float64).reshape(-1)
    if np.max(np.abs(column_sum))>1e-12:
        raise RuntimeError("DIRECT_LBS_SIMPLEX_TANGENT_NOT_CONSERVATIVE")
    return T,AT,d,{
        "tangent_variable_count":int(col),
        "fixed_single_support_vertex_count":int(sum(np.count_nonzero(mask[i])==1 for i in range(n))),
        "pivot_policy":"MAX_SOURCE_WEIGHT_PER_VERTEX",
    }


def solve_variant(W0, lam, support_mask, tangent):
    T,AT,d,tmeta=tangent
    lam=float(lam)
    if lam<=0.0 or not math.isfinite(lam):
        raise RuntimeError("DIRECT_LBS_LAMBDA_INVALID")
    root=math.sqrt(lam)

    t_col=np.asarray(T.power(2).sum(axis=0),dtype=np.float64).reshape(-1)
    a_col=np.asarray(AT.power(2).sum(axis=0),dtype=np.float64).reshape(-1)
    scale=np.sqrt(t_col+lam*a_col)
    if np.any(scale<=1e-14) or not np.isfinite(scale).all():
        raise RuntimeError("DIRECT_LBS_TANGENT_COLUMN_SCALE_INVALID")
    inv_scale=1.0/scale
    n_top=T.shape[0]

    def matvec(y):
        z=np.asarray(y,dtype=np.float64)*inv_scale
        return np.concatenate((
            np.asarray(T@z,dtype=np.float64).reshape(-1),
            root*np.asarray(AT@z,dtype=np.float64).reshape(-1),
        ))

    def rmatvec(v):
        v=np.asarray(v,dtype=np.float64)
        grad=np.asarray(T.T@v[:n_top],dtype=np.float64).reshape(-1)
        grad+=root*np.asarray(AT.T@v[n_top:],dtype=np.float64).reshape(-1)
        return grad*inv_scale

    op=LinearOperator(
        (T.shape[0]+AT.shape[0],T.shape[1]),
        matvec=matvec,rmatvec=rmatvec,dtype=np.float64,
    )
    rhs=np.concatenate((
        np.zeros(T.shape[0],dtype=np.float64),
        root*d,
    ))
    ls=lsmr(op,rhs,atol=1e-9,btol=1e-9,conlim=1e12,maxiter=4000)
    y=np.asarray(ls[0],dtype=np.float64)
    if not np.isfinite(y).all():
        raise RuntimeError("DIRECT_LBS_TANGENT_LSMR_NONFINITE")
    z=y*inv_scale

    delta=np.asarray(T@z,dtype=np.float64).reshape(np.asarray(W0).shape)
    raw=np.asarray(W0,dtype=np.float64)+delta
    row_sum_residual=float(np.max(np.abs(raw.sum(axis=1)-1.0)))
    if row_sum_residual>1e-9:
        raise RuntimeError(
            f"DIRECT_LBS_TANGENT_SIMPLEX_SUM_DRIFT:{row_sum_residual}"
        )

    edge_residual=np.asarray(AT@z-d,dtype=np.float64).reshape(-1)
    displacement_grad=np.asarray(
        T.T@delta.reshape(-1),dtype=np.float64
    ).reshape(-1)
    deformation_grad=lam*np.asarray(
        AT.T@edge_residual,dtype=np.float64
    ).reshape(-1)
    stationarity=displacement_grad+deformation_grad
    backward=float(
        np.linalg.norm(stationarity)
        / max(
            np.linalg.norm(displacement_grad)+np.linalg.norm(deformation_grad),
            1e-12,
        )
    )
    if backward>1e-7:
        raise RuntimeError(
            "DIRECT_LBS_TANGENT_KKT_BACKWARD_ERROR:"
            f"{backward}:istop={int(ls[1])}:itn={int(ls[2])}:"
            f"normar={float(ls[4])}"
        )

    projected=simplex_project_rows(raw,support_mask)
    projection=np.abs(projected-raw).sum(axis=1)
    return projected,{
        "solver":"SIMPLEX_TANGENT_COLUMN_SCALED_LSMR",
        **tmeta,
        "lsmr_istop":int(ls[1]),
        "lsmr_iterations":int(ls[2]),
        "lsmr_normr":float(ls[3]),
        "lsmr_normar":float(ls[4]),
        "tangent_kkt_backward_error":backward,
        "preprojection_row_sum_residual_max":row_sum_residual,
        "raw_negative_mass":float(-np.minimum(raw,0.0).sum()),
        "simplex_projection_l1_mean":float(np.mean(projection)),
        "simplex_projection_l1_max":float(np.max(projection)),
    }


def exact_motion_court(ctx, rest, weights, faces, joint_ids, skeleton, cameras, policy):
    source_report = read("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json")
    rows = []
    mapping_rows = {}
    for clip in CLIPS:
        payload = read(
            ctx["run_root"]
            / "inputs/motion/quaternius_knight_v1"
            / f"{clip}.motion.json"
        )
        tracks, mapping = _tracks_for_clip(
            payload, skeleton, cameras, source_report
        )
        mapping_rows[clip] = mapping
        times = np.linspace(
            0.0,
            float(payload["duration_seconds"]),
            FULL_MOTION_SAMPLES,
            endpoint=not bool(payload.get("loop")),
        )
        for frame_index, t in enumerate(times):
            mats, _, _ = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(t),
                cameras=cameras,
            )
            posed = _skin(rest, weights, joint_ids, mats)
            geometry = measure_dynamic_frame_geometry_v1(
                rest=rest,
                posed=posed,
                faces=faces,
                policy=policy,
            )
            rows.append(
                {
                    "clip_id": clip,
                    "frame_index": int(frame_index),
                    "time_seconds": float(t),
                    "geometry": geometry,
                }
            )
    return {
        "frame_count": len(rows),
        "failed_motion_frame_count": sum(
            not bool(row["geometry"]["passed"]) for row in rows
        ),
        "maximum_motion_edge_ratio": max(
            float(row["geometry"]["maximum_edge_ratio"]) for row in rows
        ),
        "maximum_motion_condition_number": max(
            float(row["geometry"]["maximum_condition_number"]) for row in rows
        ),
        "rows": rows,
        "retarget_mapping": mapping_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--fresh-inference-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    a = ap.parse_args()

    ctx = _ctx(a.authority_root, a.run_id)
    surface = rigging_surface_from_dict(read(a.surface_json))
    candidate = canonical_mesh_candidate_from_dict(read(a.candidate_json))
    skeleton = qualified_skeleton_from_dict(
        read(a.fresh_inference_dir / "fresh_qualified_skeleton.json")
    )
    skin = qualified_skin_from_dict(
        read(a.fresh_inference_dir / "fresh_qualified_skin.json")
    )
    cameraset = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )
    cameras = tuple(sorted(cameraset.cameras, key=lambda c: int(c.view_index)))
    _, envelope = derive_deformation_envelope_v1(
        skeleton=skeleton, camera_set=cameraset
    )

    rest, W0, faces = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=skin
    )
    rest = np.asarray(rest, dtype=np.float64)
    W0 = np.asarray(W0, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    edges, face_edges = unique_edges(faces)
    joint_ids, probes = probe_bank(skeleton, cameras)

    baseline_g3, constraints = evaluate_g3_and_collect_constraints(
        rest,
        W0,
        faces,
        edges,
        face_edges,
        probes,
        policy,
        collect_constraints=True,
    )
    if not constraints:
        raise RuntimeError("DIRECT_LBS_NO_VIOLATING_CONSTRAINTS")
    A, b = build_constraint_system(rest, edges, probes, constraints, W0)

    a.out_root.mkdir(parents=True, exist_ok=True)
    baseline_motion = exact_motion_court(
        ctx, rest, W0, faces, joint_ids, skeleton, cameras, policy
    )
    write(
        a.out_root / "BASELINE.json",
        {
            "g3": baseline_g3,
            "motion_summary": {
                k: baseline_motion[k]
                for k in (
                    "frame_count",
                    "failed_motion_frame_count",
                    "maximum_motion_edge_ratio",
                    "maximum_motion_condition_number",
                )
            },
        },
    )

    manifest = {
        "schema": "RealSaS.DirectLBSMeshWeightProjectionAudit.v1",
        "status": "DIAGNOSTIC_ONLY",
        "product_authority_minted": False,
        "training_used": False,
        "teacher_data_used": False,
        "motion_clip_data_used_for_optimization": False,
        "optimization_domain": "MESH_WEIGHT_REALIZATION_ONLY",
        "original_joint_support_hard_preserved": True,
        "simplex_hard_parameterized_in_linear_solve": True,
        "source_surface_skin_mutated": False,
        "objective": "L2_TO_DETERMINISTIC_BINDER_WEIGHTS_PLUS_DIRECT_G3_LBS_RELATIVE_EDGE_TARGETS",
        "g3_probe_count": len(probes),
        "constraint_count": len(constraints),
        "raw_constraint_count": int(baseline_g3["raw_constraint_count"]),
        "constraints_truncated": bool(baseline_g3["constraints_truncated"]),
        "max_constraints": MAX_CONSTRAINTS,
        "edge_ratio_upper_reference": float(DEFAULT_MAX_EDGE_RATIO),
        "edge_ratio_lower_reference": float(LOWER_EDGE_RATIO),
        "baseline": {
            "g3_passed": bool(baseline_g3["passed"]),
            "g3_max_condition": float(baseline_g3["maximum_condition_number"]),
            "g3_max_edge_ratio": float(baseline_g3["maximum_edge_ratio"]),
            "failed_motion_frames": int(baseline_motion["failed_motion_frame_count"]),
            "max_motion_edge_ratio": float(baseline_motion["maximum_motion_edge_ratio"]),
        },
        "variants": [],
    }

    source_support_mask = W0 > 0.0
    tangent = prepare_simplex_tangent_system(
        W0, A, b, source_support_mask
    )
    print(
        "DIRECT_LBS_TANGENT_PREP="
        + json.dumps(
            {
                "active_source_weight_count": int(np.count_nonzero(source_support_mask)),
                "tangent_variable_count": int(tangent[0].shape[1]),
                "constraint_dimension": int(tangent[1].shape[0]),
                "constraint_tangent_nnz": int(tangent[1].nnz),
                **tangent[3],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for lam in LAMBDAS:
        W, solver = solve_variant(
            W0, lam, source_support_mask, tangent
        )
        correction = np.abs(W - W0).sum(axis=1)
        new_support_mass = np.where(source_support_mask, 0.0, W).sum(axis=1)
        new_support_count = np.count_nonzero(
            (~source_support_mask) & (W > NEW_SUPPORT_EPS), axis=1
        )
        g3, _ = evaluate_g3_and_collect_constraints(
            rest,
            W,
            faces,
            edges,
            face_edges,
            probes,
            policy,
            collect_constraints=False,
        )
        motion = exact_motion_court(
            ctx, rest, W, faces, joint_ids, skeleton, cameras, policy
        )
        tag = f"lambda_{str(lam).replace('.', 'p')}"
        out = a.out_root / tag
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "mesh_weights.npz",
            rest=rest,
            weights=W,
            baseline_weights=W0,
            faces=faces,
            edges=edges,
            joint_ids=np.asarray(joint_ids),
        )
        report = {
            "tag": tag,
            "lambda": float(lam),
            "solver": solver,
            "p50_correction_l1": q(correction, 0.50),
            "p95_correction_l1": q(correction, 0.95),
            "p99_correction_l1": q(correction, 0.99),
            "max_correction_l1": float(np.max(correction)),
            "mean_new_support_mass": float(np.mean(new_support_mass)),
            "p95_new_support_mass": q(new_support_mass, 0.95),
            "vertices_with_new_support": int(np.count_nonzero(new_support_count)),
            "g3_passed": bool(g3["passed"]),
            "g3_failure_invariants": g3["failure_invariants"],
            "g3_min_area_ratio": float(g3["minimum_area_ratio"]),
            "g3_max_area_ratio": float(g3["maximum_area_ratio"]),
            "g3_max_condition": float(g3["maximum_condition_number"]),
            "g3_min_edge_ratio": float(g3["minimum_edge_ratio"]),
            "g3_max_edge_ratio": float(g3["maximum_edge_ratio"]),
            "failed_motion_frames": int(motion["failed_motion_frame_count"]),
            "max_motion_edge_ratio": float(motion["maximum_motion_edge_ratio"]),
            "max_motion_condition_number": float(
                motion["maximum_motion_condition_number"]
            ),
            "product_authority_minted": False,
        }
        write(out / "REPORT.json", report)
        manifest["variants"].append(report)
        print("DIRECT_LBS_RESULT=" + json.dumps(report, sort_keys=True), flush=True)

    write(a.out_root / "RESULT.json", manifest)


if __name__ == "__main__":
    main()
