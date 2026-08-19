from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DEPS = HERE / "frozen_deps"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


runner = _load("realsas_v11_frozen", DEPS / "realsas_n1d_hybrid_v11_frozen_runner.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _paths(root: Path, family: int, episode: str):
    A = sorted((root / str(family) / "A").glob("*.png"))
    B = sorted((root / str(family) / episode / "B").glob("*.png"))
    if len(A) != 8 or len(B) != 8:
        raise RuntimeError((family, episode, len(A), len(B)))
    return A, B


def _route_module(route: str):
    if route == "V8_BASE":
        return runner.v8
    if route == "SEED_BASIN_WSEED":
        return runner.v5
    raise RuntimeError(f"unknown frozen V11 route: {route}")


def _model_context(root: Path, family: int, episode: str, route: str):
    m = _route_module(route)
    A, B = _paths(root, family, episode)
    PA, VA, seedB, output_idx, hb, prb, diag = m.exact_problem_a(A, B)
    cfg = m.SEESConfig(
        image_size=128, base_dim=64, token_dim=192, descriptor_dim=64,
        transformer_depth=4, transformer_heads=6, camera_residual_enabled=False,
        input_channels=4, max_delta=.35, correspondence_radius_f4=5,
        correspondence_temperature=.03, correspondence_position_lambda=.02,
        correspondence_gain_threshold=.10, correspondence_gain_width=.10,
    )
    model = m.IRISSEESN1(cfg)
    ck = torch.load(m.CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck, strict=True)
    model.eval()
    with torch.no_grad():
        field = model(m.load_model_tensor(A, B))
    xyA = np.stack([m.project_points(PA, v) for v in range(8)]).astype(np.float32)
    return m, A, B, PA, VA, seedB, np.asarray(output_idx, np.int32), hb, prb, float(diag), field, xyA


def _candidate_pools(m, B, PA, VA, seedB, field, xyA):
    zA = m.descriptor_consensus(m.sample_field_np(field["descriptor"][0, 0], xyA), VA)
    masks = [m.foreground_mask(p) for p in B]
    cache = [m.prepare_b_search(field["descriptor"][0, 1, v], masks[v]) for v in range(8)]
    pools = []
    for i in range(len(PA)):
        cands, scores = [], []
        for v in range(8):
            if bool(VA[v, i]):
                q, s = m.top4_with_scores(field["descriptor"][0, 1, v], zA[i], masks[v], cache[v])
            else:
                q, s = np.empty((0, 2), np.float32), np.empty(0, np.float32)
            cands.append(q); scores.append(s)
        pools.append(m.candidate_pool(cands, scores, PA[i], seedB[i]))
    return pools


def _flatten_pools(pools):
    offsets = [0]
    xyz, reproj, desc = [], [], []
    for H in pools:
        for p, e, d in H:
            xyz.append(np.asarray(p, np.float32)); reproj.append(float(e)); desc.append(float(d))
        offsets.append(len(xyz))
    return (
        np.asarray(xyz, np.float32).reshape(-1, 3),
        np.asarray(reproj, np.float32),
        np.asarray(desc, np.float32),
        np.asarray(offsets, np.int64),
    )


def decorate_pb_observable(root: Path, family: int, episode: str, route: str, expected_pa: np.ndarray, PB: np.ndarray):
    m, A, B, PA, VA, seedB, output_idx, hb, prb, diag, field, xyA = _model_context(root, family, episode, route)
    P = np.asarray(PA[output_idx], np.float32)
    V = np.asarray(VA[:, output_idx], np.uint8)
    if not np.array_equal(P, np.asarray(expected_pa, np.float32)):
        raise RuntimeError("observable P_A replay mismatch")
    PB = np.asarray(PB, np.float32)
    VB = m.exact_b_visibility(hb, prb, PB, diag)
    xyB = np.stack([m.project_points(PB, v) for v in range(8)]).astype(np.float32)
    NB = m.consensus_normals(m.sample_field_np(field["normal_B"][0, 1], xyB), VB)
    return NB.astype(np.float32), VB.astype(np.uint8), xyB.astype(np.float32)


def run_family(root: Path, family: int, episode: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{family}_{episode}_baseline.npz"
    meta = runner.run(root, family, episode, pred_path)
    if meta.get("truth_access") != "NONE":
        raise RuntimeError("frozen runner violated observable-only contract")
    z = np.load(pred_path, allow_pickle=False)
    base = {k: np.asarray(z[k]) for k in z.files}
    route = str(meta["route"])
    m, A, B, PA, VA, seedB, output_idx, hb, prb, diag, field, xyA = _model_context(root, family, episode, route)
    P = np.asarray(PA[output_idx], np.float32)
    V = np.asarray(VA[:, output_idx], np.uint8)
    if not np.array_equal(P, np.asarray(base["P_A"], np.float32)):
        raise RuntimeError("candidate-frontdoor P_A mismatch against frozen V11 prediction")
    pools_all = _candidate_pools(m, B, PA, VA, seedB, field, xyA)
    pools = [pools_all[int(i)] for i in output_idx]
    pool_xyz, pool_reproj, pool_desc, pool_offsets = _flatten_pools(pools)
    PB_base = np.asarray(base["P_B"], np.float32)
    VB2 = m.exact_b_visibility(hb, prb, PB_base, diag)
    XYB2 = np.stack([m.project_points(PB_base, v) for v in range(8)]).astype(np.float32)
    NB2 = m.consensus_normals(m.sample_field_np(field["normal_B"][0, 1], XYB2), VB2)
    if not np.array_equal(VB2, np.asarray(base["V_B"], np.uint8)):
        raise RuntimeError("observable visibility replay mismatch")
    nb_max = float(np.max(np.abs(NB2 - np.asarray(base["N_B"], np.float32))))
    if nb_max > 1e-7:
        raise RuntimeError(f"observable normal replay mismatch: {nb_max}")
    state_path = out_dir / f"{family}_{episode}_OBSERVABLE_STATE.npz"
    np.savez_compressed(
        state_path,
        P_A=np.asarray(base["P_A"], np.float32), P_B=np.asarray(base["P_B"], np.float32),
        N_A=np.asarray(base["N_A"], np.float32), N_B=np.asarray(base["N_B"], np.float32),
        V_A=np.asarray(base["V_A"], np.uint8), V_B=np.asarray(base["V_B"], np.uint8),
        XY_A=np.asarray(base["XY_A"], np.float32), XY_B=np.asarray(base["XY_B"], np.float32),
        H_xyz=pool_xyz, H_reproj_px=pool_reproj, H_desc_score=pool_desc, H_offsets=pool_offsets,
    )
    smeta = {
        "schema": "RealSaS.N1D.ObservableFunctionalAuditV2.ObservableFamilyState.v1",
        "family": int(family), "episode": episode, "route": route,
        "truth_access": "NONE",
        "baseline_prediction_sha256": sha256_file(pred_path),
        "observable_state_sha256": sha256_file(state_path),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "raster_sha256": meta["raster_sha256"],
        "candidate_pool": {
            "carriers": 64,
            "total": int(len(pool_xyz)),
            "min": int(np.min(np.diff(pool_offsets))),
            "median": float(np.median(np.diff(pool_offsets))),
            "max": int(np.max(np.diff(pool_offsets))),
        },
        "decorator_replay": {"V_B_exact": True, "N_B_max_abs": nb_max},
    }
    meta_path = state_path.with_suffix(".json")
    meta_path.write_text(json.dumps(smeta, indent=2, sort_keys=True))
    return smeta


def run_panel(root: Path, families, episode: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fam = []
    for f in families:
        fam.append(run_family(root, int(f), episode, out_dir))
    manifest = {
        "schema": "RealSaS.N1D.ObservableFunctionalAuditV2.ObservablePanel.v1",
        "truth_access": "NONE",
        "episode": episode,
        "families": fam,
    }
    path = out_dir / "OBSERVABLE_PANEL_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--families", nargs="+", type=int, required=True)
    ap.add_argument("--episode", default="e00")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(run_panel(a.root, a.families, a.episode, a.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
