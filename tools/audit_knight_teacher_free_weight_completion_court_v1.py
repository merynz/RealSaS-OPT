from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    normalization_domain_from_dict,
    signed_zero_surface_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    _skin_l1_per_face,
    _stress_angle,
    _triangle_metrics_batch,
    run_skin_topology_compatibility_v1,
)
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    replay_compacted_face_provenance_v1,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)

EXACT = {
    "candidate": (
        "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json",
        "0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3",
    ),
    "skeleton": (
        "artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json",
        "e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987",
    ),
    "skin": (
        "artifacts/32_SKIN_QUALIFIED/qualified_skin.json",
        "f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09",
    ),
    "cameras": (
        "artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json",
        "312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a",
    ),
}
CLIPS = ("demo_idle_v1", "demo_run_v1", "demo_slash_v1")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def exact(rr: Path, key: str, codec):
    rel, expected = EXACT[key]
    path = rr / rel
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"SHA_DRIFT:{key}:{actual}:{expected}")
    return codec(json.loads(path.read_text()))


def load(path: Path, codec):
    return codec(json.loads(path.read_text()))


def face_indices(candidate):
    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    return np.asarray(
        [[vi[str(x)] for x in face] for face in candidate.faces],
        dtype=np.int64,
    )


def candidate_surface_ids(candidate):
    out = []
    for v in candidate.vertices:
        coeff = tuple(v.support_binding.coefficients)
        if len(coeff) != 1 or abs(float(coeff[0][1]) - 1.0) > 1e-12:
            raise RuntimeError("NON_IDENTITY_SURFACE_BINDING")
        out.append(str(coeff[0][0]))
    return tuple(out)


def dense_supported_face_mask(rr, candidate, surface):
    zero = load(
        rr / "artifacts/12_ZERO_SURFACE_DECODED/signed_zero_surface_seal.json",
        signed_zero_surface_from_dict,
    )
    norm = load(
        rr / "artifacts/08_NORMALIZATION_DOMAIN_QUALIFIED/normalization_domain.json",
        normalization_domain_from_dict,
    )
    npz = Path(zero.npz_path)
    if not npz.is_file() or sha(npz) != zero.npz_sha256:
        raise RuntimeError("ZERO_NPZ_DRIFT")
    with np.load(npz, allow_pickle=False) as z:
        vn = np.asarray(z["vertices_normalized"], dtype=np.float64)
        df = np.asarray(z["faces"], dtype=np.int64)
    world = np.asarray(norm.center_xyz, dtype=np.float64)[None, :] + vn * float(norm.half_extent)
    explicit = set(
        replay_compacted_face_provenance_v1(
            world,
            df,
            surface,
            position_tolerance=1e-9,
        )
    )
    sids = candidate_surface_ids(candidate)
    faces = face_indices(candidate)
    mask = np.zeros(len(faces), dtype=bool)
    for fi, face in enumerate(faces):
        tri = tuple(sorted(sids[int(v)] for v in face))
        mask[fi] = tri in explicit
    return mask


def build_safe_graph(rest, faces, safe_face_mask):
    edges = {}
    for face in faces[np.asarray(safe_face_mask, dtype=bool)]:
        for a, b in ((int(face[0]), int(face[1])),
                     (int(face[1]), int(face[2])),
                     (int(face[2]), int(face[0]))):
            if a == b:
                continue
            x, y = (a, b) if a < b else (b, a)
            length = float(np.linalg.norm(rest[x] - rest[y]))
            if not np.isfinite(length) or length <= 1e-12:
                continue
            # Deterministic inverse-length conductance.
            w = 1.0 / length
            edges[(x, y)] = max(edges.get((x, y), 0.0), w)
    return edges


def harmonic_complete(weights, unknown_mask, edges):
    W0 = np.asarray(weights, dtype=np.float64)
    n, k = W0.shape
    unknown = np.asarray(unknown_mask, dtype=bool)
    anchors = ~unknown

    nbr = [[] for _ in range(n)]
    for (a, b), w in edges.items():
        nbr[a].append((b, w))
        nbr[b].append((a, w))

    # Any unknown component with no anchor boundary cannot be solved without invention.
    seen = np.zeros(n, dtype=bool)
    solvable = np.zeros(n, dtype=bool)
    unresolved_components = []
    for start in np.where(unknown)[0]:
        if seen[start]:
            continue
        stack = [int(start)]
        seen[start] = True
        comp = []
        boundary = set()
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in nbr[u]:
                if unknown[v]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
                else:
                    boundary.add(int(v))
        if boundary:
            solvable[np.asarray(comp, dtype=np.int64)] = True
        else:
            unresolved_components.append(comp)

    U = np.where(solvable)[0]
    uindex = {int(v): i for i, v in enumerate(U)}
    rows = []
    cols = []
    vals = []
    B = np.zeros((len(U), k), dtype=np.float64)

    for i, u in enumerate(U):
        total = 0.0
        for v, w in nbr[int(u)]:
            total += w
            if solvable[v]:
                rows.append(i); cols.append(uindex[int(v)]); vals.append(-w)
            elif anchors[v]:
                B[i] += w * W0[v]
            # unresolved unknown neighbors are deliberately excluded from the
            # Dirichlet solve; they cannot act as invented anchors.
        if total <= 0.0:
            raise RuntimeError(f"HARMONIC_ZERO_DEGREE:{u}")
        rows.append(i); cols.append(i); vals.append(total)

    out = W0.copy()
    if len(U):
        A = csr_matrix((vals, (rows, cols)), shape=(len(U), len(U)))
        for j in range(k):
            out[U, j] = spsolve(A, B[:, j])

    # Fail-closed simplex projection for numerical residue only.
    solved = solvable
    if np.any(solved):
        out[solved] = np.maximum(out[solved], 0.0)
        sums = out[solved].sum(axis=1, keepdims=True)
        if np.any(sums <= 1e-12):
            raise RuntimeError("HARMONIC_SIMPLEX_COLLAPSE")
        out[solved] /= sums

    return out, {
        "unknown_vertex_count": int(np.count_nonzero(unknown)),
        "anchor_vertex_count": int(np.count_nonzero(anchors)),
        "solved_vertex_count": int(np.count_nonzero(solvable)),
        "unresolved_vertex_count": int(np.count_nonzero(unknown & ~solvable)),
        "unresolved_component_count": int(len(unresolved_components)),
        "unresolved_component_sizes": sorted(
            [len(x) for x in unresolved_components],
            reverse=True,
        )[:64],
    }


def stress_arbitrary_weights(rest, weights, faces, skeleton, cameras, envelope, policy):
    stress_angle = _stress_angle(envelope)
    frames = derive_joint_frames_from_skeleton(skeleton, cameras=cameras)
    joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
    max_area = np.ones(len(faces), dtype=np.float64)
    min_area = np.ones(len(faces), dtype=np.float64)
    max_condition = np.ones(len(faces), dtype=np.float64)
    max_edge = np.ones(len(faces), dtype=np.float64)

    hom = np.concatenate(
        [rest, np.ones((len(rest), 1), dtype=np.float64)],
        axis=1,
    )
    for jid in sorted(joint_ids):
        for axis_index in range(3):
            for sign in (-1.0, 1.0):
                skin_by_id = _pose_skin_matrices(
                    skeleton,
                    frames,
                    joint_id=jid,
                    local_axis_index=axis_index,
                    degrees=sign * stress_angle,
                )
                matrices = np.stack([skin_by_id[x] for x in joint_ids], axis=0)
                per = np.stack(
                    [(hom @ matrices[j].T)[:, :3] for j in range(len(joint_ids))],
                    axis=1,
                )
                posed = np.sum(per * weights[:, :, None], axis=1)
                area, cond, _, emax = _triangle_metrics_batch(rest, posed, faces)
                max_area = np.maximum(max_area, area)
                min_area = np.minimum(min_area, area)
                max_condition = np.maximum(max_condition, cond)
                max_edge = np.maximum(max_edge, emax)

    unsafe = (
        (max_area > float(policy.g3_max_dynamic_area_ratio))
        | (min_area < float(policy.g3_min_dynamic_area_ratio))
        | (max_condition > float(policy.g3_max_dynamic_condition_number))
        | (max_edge > 4.0)
    )
    return {
        "unsafe_face_count": int(np.count_nonzero(unsafe)),
        "unsafe_face_indices": np.nonzero(unsafe)[0].astype(int).tolist(),
        "max_edge_max": float(np.max(max_edge)),
        "max_edge_p95": float(np.quantile(max_edge, .95)),
        "max_condition_max": float(np.max(max_condition)),
        "max_area_max": float(np.max(max_area)),
        "min_area_min": float(np.min(min_area)),
    }


def motion_metrics(rest, weights, faces, joint_ids, skeleton, cameras, rr, source_report):
    rows = []
    for clip in CLIPS:
        payload = json.loads(
            (rr / "inputs/motion/quaternius_knight_v1" / f"{clip}.motion.json").read_text()
        )
        tracks, _ = _tracks_for_clip(payload, skeleton, cameras, source_report)
        times = np.linspace(
            0.0,
            float(payload["duration_seconds"]),
            4,
            endpoint=not bool(payload.get("loop")),
        )
        for frame, t in enumerate(times):
            mats, _, _ = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(t),
                cameras=cameras,
            )
            posed = _skin(rest, weights, joint_ids, mats)
            r = rest[faces]
            p = posed[faces]
            rl = np.stack(
                [
                    np.linalg.norm(r[:, 1] - r[:, 0], axis=1),
                    np.linalg.norm(r[:, 2] - r[:, 1], axis=1),
                    np.linalg.norm(r[:, 0] - r[:, 2], axis=1),
                ],
                axis=1,
            )
            pl = np.stack(
                [
                    np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                    np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                    np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
                ],
                axis=1,
            )
            e = (pl / np.maximum(rl, 1e-15)).max(axis=1)
            rows.append(
                {
                    "clip_id": clip,
                    "frame": int(frame),
                    "time_seconds": float(t),
                    "edge_p95": float(np.quantile(e, .95)),
                    "edge_p99": float(np.quantile(e, .99)),
                    "edge_max": float(np.max(e)),
                    "edge_gt_4": int(np.count_nonzero(e > 4.0)),
                    "edge_gt_10": int(np.count_nonzero(e > 10.0)),
                }
            )
    return {
        "frames": rows,
        "worst_edge_max": max(x["edge_max"] for x in rows),
        "max_edge_gt_4": max(x["edge_gt_4"] for x in rows),
        "max_edge_gt_10": max(x["edge_gt_10"] for x in rows),
        "max_edge_p99": max(x["edge_p99"] for x in rows),
    }


def teacher_matrix(bank_path, skeleton, candidate, joint_ids):
    with np.load(bank_path, allow_pickle=False) as z:
        sids = tuple(map(str, z["surface_ids"].tolist()))
        BW = np.asarray(z["weights"], dtype=np.float64)
        valid = np.asarray(z["teacher_valid_mask"], dtype=np.uint8).astype(bool)
        bpos = np.asarray(z["target_positions_world"], dtype=np.float64)
        bpar = np.asarray(z["target_parent_indices"], dtype=np.int64)

    joints = tuple(skeleton.joints)
    ids = [str(j.canonical_joint_id) for j in joints]
    ididx = {x: i for i, x in enumerate(ids)}
    spos = np.asarray([j.position for j in joints], dtype=np.float64)
    spar = np.asarray(
        [
            -1 if j.parent_canonical_id is None else ididx[str(j.parent_canonical_id)]
            for j in joints
        ],
        dtype=np.int64,
    )
    C = np.linalg.norm(bpos[:, None, :] - spos[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(C)
    assign = np.empty(len(bpos), dtype=np.int64)
    assign[ri] = ci
    graph_matches = sum(
        int(
            (-1 if bpar[b] < 0 else assign[int(bpar[b])])
            == spar[int(assign[b])]
        )
        for b in range(len(bpar))
    )
    if graph_matches != len(bpar):
        raise RuntimeError("TEACHER_TARGET_GRAPH_ALIGNMENT_NOT_EXACT")

    jix = {jid: i for i, jid in enumerate(joint_ids)}
    WT = np.zeros((len(sids), len(joint_ids)), dtype=np.float64)
    for bc in range(len(assign)):
        jid = ids[int(assign[bc])]
        WT[:, jix[jid]] = BW[:, bc]

    row = {sid: i for i, sid in enumerate(sids)}
    csids = candidate_surface_ids(candidate)
    idx = np.asarray([row[sid] for sid in csids], dtype=np.int64)
    return WT[idx], valid[idx]


def l1_summary(A, B, mask):
    err = np.sum(np.abs(A - B), axis=1)
    x = err[np.asarray(mask, dtype=bool)]
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else None,
        "p50": float(np.quantile(x, .50)) if len(x) else None,
        "p95": float(np.quantile(x, .95)) if len(x) else None,
        "p99": float(np.quantile(x, .99)) if len(x) else None,
        "max": float(np.max(x)) if len(x) else None,
        "gt_0_1": int(np.count_nonzero(x > .1)),
        "gt_1_0": int(np.count_nonzero(x > 1.0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--teacher-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    rr = _ctx(a.authority_root, a.run_id)["run_root"]
    candidate = exact(rr, "candidate", canonical_mesh_candidate_from_dict)
    skeleton = exact(rr, "skeleton", qualified_skeleton_from_dict)
    skin = exact(rr, "skin", qualified_skin_from_dict)
    cameras = tuple(
        sorted(
            exact(rr, "cameras", qualified_camera_set_from_dict).cameras,
            key=lambda x: int(x.view_index),
        )
    )
    surface = load(
        rr / "artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",
        rigging_surface_from_dict,
    )
    envelope = load(
        rr / "artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",
        deformation_envelope_from_dict,
    )
    policy = load(
        rr / "artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",
        mesh_policy_from_dict,
    )

    faces = face_indices(candidate)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    joint_ids, Wpred = _candidate_skin_weights(candidate, skin, skeleton)

    # Teacher-free contamination detector: all-face synthetic mechanical stress.
    full = run_skin_topology_compatibility_v1(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
        risk_l1_min=0.0,
        stress_all_faces=True,
    )
    unsafe_face_mask = np.zeros(len(faces), dtype=bool)
    unsafe_face_mask[np.asarray(full["unsafe_face_indices"], dtype=np.int64)] = True

    dense_supported = dense_supported_face_mask(rr, candidate, surface)
    # Never propagate through invented/clique-only faces, and never use a face already
    # convicted by mechanical stress as harmonic support.
    graph_face_mask = dense_supported & (~unsafe_face_mask)
    edges = build_safe_graph(rest, faces, graph_face_mask)

    unknown = np.zeros(len(rest), dtype=bool)
    if np.any(unsafe_face_mask):
        unknown[np.unique(faces[unsafe_face_mask].reshape(-1))] = True

    Wcomp, completion = harmonic_complete(Wpred, unknown, edges)

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    pred_motion = motion_metrics(
        rest, Wpred, faces, joint_ids, skeleton, cameras, rr, source_report
    )
    comp_motion = motion_metrics(
        rest, Wcomp, faces, joint_ids, skeleton, cameras, rr, source_report
    )

    pred_stress = stress_arbitrary_weights(
        rest, Wpred, faces, skeleton, cameras, envelope, policy
    )
    comp_stress = stress_arbitrary_weights(
        rest, Wcomp, faces, skeleton, cameras, envelope, policy
    )

    # Teacher is evaluation-only. It does not enter contamination detection, graph
    # construction, anchor selection, or the harmonic solve.
    Wteach, teacher_valid = teacher_matrix(
        a.teacher_bank, skeleton, candidate, joint_ids
    )
    teacher_eval = {
        "predicted_vs_teacher_valid": l1_summary(
            Wpred, Wteach, teacher_valid
        ),
        "completed_vs_teacher_valid": l1_summary(
            Wcomp, Wteach, teacher_valid
        ),
        "predicted_vs_teacher_invalid": l1_summary(
            Wpred, Wteach, ~teacher_valid
        ),
        "completed_vs_teacher_invalid": l1_summary(
            Wcomp, Wteach, ~teacher_valid
        ),
        "completed_changed_vertex_count": int(
            np.count_nonzero(np.sum(np.abs(Wcomp - Wpred), axis=1) > 1e-10)
        ),
    }

    report = {
        "schema": "RealSaS.KnightTeacherFreeMechanicalWeightCompletionCourt.v1",
        "status": "DIAGNOSTIC__NO_PRODUCT_AUTHORITY_MUTATION",
        "operator": {
            "contamination_detector": "STAGE35_ALL_FACE_STRESS_V1",
            "unknown_rule": "VERTEX_INCIDENT_TO_ANY_UNSAFE_FACE",
            "anchor_rule": "ALL_OTHER_PREDICTED_ARACHNE_ROWS",
            "graph_authority": (
                "CURRENT_CANDIDATE_DENSE_FACE_SUPPORTED_AND_MECHANICALLY_SAFE_FACES_ONLY"
            ),
            "conductance": "INVERSE_REST_EDGE_LENGTH",
            "solve": "DIRICHLET_GRAPH_HARMONIC_PER_JOINT",
            "teacher_used_by_solver": False,
            "simplex_postprocess": "CLAMP_NUMERICAL_NEGATIVE_AND_RENORMALIZE_SOLVED_ROWS",
        },
        "static": {
            "face_count": int(len(faces)),
            "dense_supported_face_count": int(np.count_nonzero(dense_supported)),
            "clique_only_face_count": int(np.count_nonzero(~dense_supported)),
            "initial_unsafe_face_count": int(np.count_nonzero(unsafe_face_mask)),
            "graph_face_count": int(np.count_nonzero(graph_face_mask)),
            "graph_edge_count": int(len(edges)),
            **completion,
        },
        "teacher_evaluation_only": teacher_eval,
        "synthetic_stress": {
            "predicted": pred_stress,
            "completed": comp_stress,
        },
        "actual_motion": {
            "predicted": pred_motion,
            "completed": comp_motion,
        },
        "finding": {
            "completion_is_fully_resolved": bool(
                completion["unresolved_vertex_count"] == 0
            ),
            "completion_reduces_actual_edge_gt_10": bool(
                comp_motion["max_edge_gt_10"] < pred_motion["max_edge_gt_10"]
            ),
            "completion_reduces_actual_edge_gt_4": bool(
                comp_motion["max_edge_gt_4"] < pred_motion["max_edge_gt_4"]
            ),
            "completion_reduces_all_face_unsafe": bool(
                comp_stress["unsafe_face_count"] < pred_stress["unsafe_face_count"]
            ),
            "completion_matches_teacher_better_on_invalid_rows": bool(
                teacher_eval["completed_vs_teacher_invalid"]["mean"]
                < teacher_eval["predicted_vs_teacher_invalid"]["mean"]
            ),
        },
        "claim_boundary": (
            "Teacher weights are evaluation-only. A PASS would prove that mechanical "
            "stress can identify a useful repair support set and that a topology-local "
            "harmonic compiler completion can improve it without teacher input. It "
            "would not yet authorize production adoption; remaining unsafe topology "
            "must be separately attributed and remeshed rather than face-deleted."
        ),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_TEACHER_FREE_WEIGHT_COMPLETION_COURT_PASS",
        json.dumps(
            {
                "static": report["static"],
                "teacher_eval": teacher_eval,
                "pred_motion": {
                    k: v for k, v in pred_motion.items() if k != "frames"
                },
                "comp_motion": {
                    k: v for k, v in comp_motion.items() if k != "frames"
                },
                "pred_stress": pred_stress,
                "comp_stress": comp_stress,
                "finding": report["finding"],
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
