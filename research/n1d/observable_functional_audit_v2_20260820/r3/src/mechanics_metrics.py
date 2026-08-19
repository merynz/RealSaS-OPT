from __future__ import annotations
import numpy as np

EFFECT_THRESHOLDS = {
    "F_response_NRMS": .15,
    "F_kernel_RMS": .10,
    "F_rigid_residual_NRMS": .20,
    "D_surface_action_NRMS": .15,
    "D_residual_NRMS": .20,
    "D_gradient_NRMS": .20,
    "R_motion_NRMS": .20,
    "R_transfer_NRMS": .20,
    "R_surface_affinity_RMS": .15,
    "G_direction_disagree": .15,
    "G_axis_line_norm_RMS": .20,
    "G_support_RMS": .15,
}


def rms(x):
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def nrms_truth(pred, truth):
    """Prediction-vs-truth NRMS, normalized by the truth block scale."""
    return rms(np.asarray(pred, np.float64) - np.asarray(truth, np.float64)) / (rms(truth) + 1e-8)


def nrms_effect(a, b):
    """Symmetric state-vs-state NRMS for functional equivalence tests."""
    aa = rms(a); bb = rms(b)
    return rms(np.asarray(a, np.float64) - np.asarray(b, np.float64)) / (max(aa, bb) + 1e-8)


def motif_from_neighbors(idx, i):
    idx = np.asarray(idx, np.int64)
    S = {int(i)}
    S.update(map(int, idx[i]))
    rev = np.where(np.any(idx == int(i), axis=1))[0]
    S.update(map(int, rev))
    return np.asarray(sorted(S), np.int64)


def local_edges(idx, S):
    idx = np.asarray(idx, np.int64); S = np.asarray(S, np.int64)
    keep = np.zeros(idx.shape, bool); sm = np.zeros(idx.shape[0], bool); sm[S] = True
    keep[S] = sm[idx[S]]
    return keep


def _axis_distance(a, b, support, radius):
    ids = np.where(np.asarray(support) >= .25)[0]
    if not len(ids):
        return 0.0, 0.0, True
    da = np.asarray(a["G_axis_direction"])[ids].astype(np.float64)
    db = np.asarray(b["G_axis_direction"])[ids].astype(np.float64)
    da /= np.maximum(np.linalg.norm(da, axis=1, keepdims=True), 1e-8)
    db /= np.maximum(np.linalg.norm(db, axis=1, keepdims=True), 1e-8)
    disagree = float(np.mean(1.0 - np.abs(np.sum(da * db, axis=1))))
    pa = np.asarray(a["G_axis_point"])[ids].astype(np.float64)
    pb = np.asarray(b["G_axis_point"])[ids].astype(np.float64)
    d = pa - pb
    # Axis point is gauge-free along the axis: compare only perpendicular displacement.
    perp = np.linalg.norm(np.cross(d, db), axis=1) / max(float(radius), 1e-8)
    return disagree, rms(perp), False


def effect_blocks(base, swap, P_A, i):
    S = motif_from_neighbors(base["R_neighbor_idx"], i)
    E = local_edges(base["R_neighbor_idx"], S)
    P = np.asarray(P_A, np.float64)[S]
    c = P.mean(0); radius = rms(P - c)
    support = np.maximum(np.asarray(base["G_support"])[S], np.asarray(swap["G_support"])[S])
    gb = {k: np.asarray(base[k])[S] for k in ("G_axis_direction", "G_axis_point")}
    gs = {k: np.asarray(swap[k])[S] for k in ("G_axis_direction", "G_axis_point")}
    gd, gl, abstain = _axis_distance(gb, gs, support, radius)
    vals = {
        "F_response_NRMS": nrms_effect(np.asarray(base["F_response_features"])[S], np.asarray(swap["F_response_features"])[S]),
        "F_kernel_RMS": rms(np.asarray(base["F_coresponse_kernel"])[np.ix_(S, S)] - np.asarray(swap["F_coresponse_kernel"])[np.ix_(S, S)]),
        "F_rigid_residual_NRMS": nrms_effect(np.asarray(base["F_rigid_residual_normalized"])[S], np.asarray(swap["F_rigid_residual_normalized"])[S]),
        "D_surface_action_NRMS": nrms_effect(np.asarray(base["D_surface_action"])[S], np.asarray(swap["D_surface_action"])[S]),
        "D_residual_NRMS": nrms_effect(np.asarray(base["D_local_residual_rms"])[S], np.asarray(swap["D_local_residual_rms"])[S]),
        "D_gradient_NRMS": nrms_effect(np.asarray(base["D_J_surface_gradient"])[S], np.asarray(swap["D_J_surface_gradient"])[S]),
        "R_motion_NRMS": nrms_effect(np.asarray(base["R_motion_distance"])[E], np.asarray(swap["R_motion_distance"])[E]),
        "R_transfer_NRMS": nrms_effect(np.asarray(base["R_transfer_error_normalized"])[E], np.asarray(swap["R_transfer_error_normalized"])[E]),
        "R_surface_affinity_RMS": rms(np.asarray(base["R_surface_affinity"])[E] - np.asarray(swap["R_surface_affinity"])[E]),
        "G_direction_disagree": gd,
        "G_axis_line_norm_RMS": gl,
        "G_support_RMS": rms(np.asarray(base["G_support"])[S] - np.asarray(swap["G_support"])[S]),
    }
    passes = {k: (bool(v <= EFFECT_THRESHOLDS[k]) if np.isfinite(v) else False) for k, v in vals.items()}
    if abstain:
        passes["G_direction_disagree"] = True
        passes["G_axis_line_norm_RMS"] = True
    return {"motif": S.tolist(), "values": vals, "passes": passes, "equivalent": bool(all(passes.values())), "G_abstain": bool(abstain)}


def truth_block_errors(pred, truth, P_A, i):
    S = motif_from_neighbors(truth["R_neighbor_idx"], i)
    E = local_edges(truth["R_neighbor_idx"], S)
    P = np.asarray(P_A, np.float64)[S]; c = P.mean(0); radius = rms(P - c)
    support = np.asarray(truth["G_support"])[S]
    gp = {k: np.asarray(pred[k])[S] for k in ("G_axis_direction", "G_axis_point")}
    gt = {k: np.asarray(truth[k])[S] for k in ("G_axis_direction", "G_axis_point")}
    gd, gl, abstain = _axis_distance(gp, gt, support, radius)
    vals = {
        "F_response": nrms_truth(np.asarray(pred["F_response_features"])[S], np.asarray(truth["F_response_features"])[S]),
        "F_kernel": rms(np.asarray(pred["F_coresponse_kernel"])[np.ix_(S, S)] - np.asarray(truth["F_coresponse_kernel"])[np.ix_(S, S)]),
        "F_rigid_residual": nrms_truth(np.asarray(pred["F_rigid_residual_normalized"])[S], np.asarray(truth["F_rigid_residual_normalized"])[S]),
        "D_surface_action": nrms_truth(np.asarray(pred["D_surface_action"])[S], np.asarray(truth["D_surface_action"])[S]),
        "D_residual": nrms_truth(np.asarray(pred["D_local_residual_rms"])[S], np.asarray(truth["D_local_residual_rms"])[S]),
        "D_gradient": nrms_truth(np.asarray(pred["D_J_surface_gradient"])[S], np.asarray(truth["D_J_surface_gradient"])[S]),
        "R_motion": nrms_truth(np.asarray(pred["R_motion_distance"])[E], np.asarray(truth["R_motion_distance"])[E]),
        "R_transfer": nrms_truth(np.asarray(pred["R_transfer_error_normalized"])[E], np.asarray(truth["R_transfer_error_normalized"])[E]),
        "R_surface_affinity": rms(np.asarray(pred["R_surface_affinity"])[E] - np.asarray(truth["R_surface_affinity"])[E]),
        "G_direction": gd,
        "G_axis_line": gl,
        "G_support": rms(np.asarray(pred["G_support"])[S] - np.asarray(truth["G_support"])[S]),
    }
    finite = [min(float(v), 10.0) for v in vals.values() if np.isfinite(v)]
    composite = float(np.median(finite)) if finite else float("nan")
    return {"values": vals, "composite_median_capped10": composite, "G_abstain": bool(abstain)}
