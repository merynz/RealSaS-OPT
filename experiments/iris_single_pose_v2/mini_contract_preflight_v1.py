from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import build_mini_seed_v1 as seed_builder
import dataset
import finalize_mini_v2 as finalize
import losses
import matcher
import run_mini_extractability_v1 as runner
import train_mini_v2 as train

EXPECTED_MEMBERSHIP_CANONICAL_SHA256 = "4e223c799cf479a210716a86701ab96459673fe21852142e7cc7bd3a9d30e055"
EXPECTED_PANEL_DIGEST = "366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961"
EXPECTED_REP_SEED_SHA = "f3d43da7766f104cab08f19fd24b515d54fde47545cc3288da023779c6d4c9af"
ROLE_SHA = {
    "FIT_TRAIN": "3c2df914b33708024eb81cf555d00087c66f45983a55ab96310356df130ba757",
    "FIT_SELECT": "fe62a1dc625d086e107673a6e3261c76b33064e6f5826e782c0c40d52bd50b14",
    "TUNE_FINAL": "b820cd30ae6954bd33d64c2950066c780210f11bb42d0becfadcf1c5aaca10ba",
}


def canonical_json_sha(path):
    obj = json.load(open(path, encoding="utf-8"))
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), obj


def ids_sha(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def gpu_preflight_regression(here: Path):
    script = here / "gpu_training_preflight_v1.py"
    with tempfile.TemporaryDirectory(prefix="iris_gpu_preflight_contract_") as td:
        td = Path(td)
        cpu_json = td / "cpu_semantic.json"
        subprocess.check_call(
            [
                sys.executable,
                str(script),
                "--out",
                str(cpu_json),
                "--resolution",
                "256",
                "--cpu-semantic-smoke",
            ],
            cwd=here,
        )
        cpu = json.load(open(cpu_json, encoding="utf-8"))
        assert cpu["schema"] == "RealSaS.IRISSinglePoseV2.MiniGPUCapacityPreflight.v1"
        assert cpu["status"] == "CPU_SEMANTIC_PASS"
        assert cpu["mode"] == "CPU_SEMANTIC_SMOKE"
        assert cpu["full_loss_enabled"] is True
        assert cpu["forward_backward_executed"] is True
        assert cpu["finite_gradients"] is True
        assert cpu["scientific_optimizer_steps"] == 0
        assert cpu["geom_samples"] == 1024 and cpu["track_samples"] == 128

        no_cuda_json = td / "no_cuda.json"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [sys.executable, str(script), "--out", str(no_cuda_json), "--resolution", "256"],
            cwd=here,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
        assert no_cuda_json.is_file()
        no_cuda = json.load(open(no_cuda_json, encoding="utf-8"))
        assert no_cuda["schema"] == "RealSaS.IRISSinglePoseV2.MiniGPUCapacityPreflight.v1"
        assert no_cuda["status"] == "NO_CUDA"
        assert no_cuda["diagnostic_stage"] == "cuda_availability"
        assert no_cuda["environment"]["cuda_available"] is False
        assert no_cuda["scientific_optimizer_steps"] == 0
        assert no_cuda["forward_backward_executed"] is False
        assert no_cuda["adamw_moment_memory_accounted_without_step"] is False
        assert no_cuda["error_type"] == "RuntimeError"
        assert "CUDA is not available" in no_cuda["error"]
        assert "Traceback" in no_cuda["traceback"]
    return {
        "cpu_semantic_full_loss_backward": True,
        "no_cuda_exit_code": 2,
        "no_cuda_structured_report": True,
        "blind_calledprocesserror_regression_closed": True,
    }


def main():
    ap = argparse.ArgumentParser(description="Frozen mini contract semantic preflight")
    ap.add_argument(
        "--membership",
        default=str(Path(__file__).with_name("MINI_EXTRACTABILITY_MEMBERSHIP_V1.json")),
    )
    ap.add_argument(
        "--provenance",
        default=str(Path(__file__).with_name("MINI_MEMBERSHIP_PROVENANCE_V1.json")),
    )
    a = ap.parse_args()
    here = Path(__file__).resolve().parent

    msh, mem = canonical_json_sha(a.membership)
    prov = json.load(open(a.provenance, encoding="utf-8"))
    if msh != EXPECTED_MEMBERSHIP_CANONICAL_SHA256:
        raise RuntimeError(f"membership canonical SHA drift {msh}")
    assert mem["source_panel_asset_id_digest"] == EXPECTED_PANEL_DIGEST
    assert mem["source_representation_seed_sha256"] == EXPECTED_REP_SEED_SHA
    assert mem["sealed_splits_opened"] is False
    assert {k: len(mem["roles"][k]) for k in ("FIT_TRAIN", "FIT_SELECT", "TUNE_FINAL")} == {
        "FIT_TRAIN": 128,
        "FIT_SELECT": 32,
        "TUNE_FINAL": 26,
    }
    sets = [set(mem["roles"][k]) for k in ("FIT_TRAIN", "FIT_SELECT", "TUNE_FINAL")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert prov["schema"] == "RealSaS.IRISSinglePoseV2.MiniMembershipProvenance.v1"
    assert prov["verified_against_source_representation_seed"] is True
    assert prov["source_representation_seed_sha256"] == EXPECTED_REP_SEED_SHA
    assert prov["source_panel_asset_id_digest"] == EXPECTED_PANEL_DIGEST
    assert prov["membership_canonical_sha256"] == msh
    assert prov["role_counts"] == {"FIT_TRAIN": 128, "FIT_SELECT": 32, "TUNE_FINAL": 26}
    assert prov["roles_disjoint"] is True
    assert prov["fit_train_all_source_split_fit"] is True
    assert prov["fit_select_all_source_split_fit"] is True
    assert prov["tune_final_all_source_split_tune"] is True
    assert prov["runtime_carries_source_registry_or_capability_metadata"] is False
    assert prov["runtime_representation_seed_consumed"] is False
    assert prov["sealed_splits_opened"] is False
    for role, d in ROLE_SHA.items():
        assert ids_sha(mem["roles"][role]) == d == prov["role_asset_id_list_sha256"][role]
    used = sorted(mem["roles"]["FIT_TRAIN"] + mem["roles"]["FIT_SELECT"] + mem["roles"]["TUNE_FINAL"])
    assert ids_sha(used) == prov["sorted_used_asset_id_set_sha256"] == "3f9d3dbe3aa4722a907882eb64ecc2c1e5d1cf5b23e1ddcb9a374c96d8001c5f"

    ms = inspect.getsource(matcher.match_query)
    assert 'outputs["N"]' not in ms and "outputs['N']" not in ms
    assert 'outputs["Z_coarse"]' in ms and 'outputs["P"]' in ms and 'outputs["Z_fine"]' in ms
    cs = inspect.getsource(losses.coarse_correspondence_loss)
    fs = inspect.getsource(losses.fine_local_loss)
    gs = inspect.getsource(losses.geometry_loss)
    ds = inspect.getsource(dataset.IRISV2Dataset.__getitem__)
    assert "track_n_view" not in cs + fs + ds
    assert 'batch["geom_n"]' in gs
    assert "_uniform_track_subsample" in fs and "[:max_tracks_per_pair]" not in fs

    base = {
        "P_p95": 0.004,
        "Zc_top8": 0.93,
        "oracle_Zf_top1_p95_native_px": 12.0,
        "family_Zc_top8_p10": 0.80,
        "family_P_p95_p90": 0.005,
        "min_style_Zc_top8": 0.90,
        "max_style_P_p95": 0.005,
        "N_p95_deg": 10.0,
    }
    score = train.selection_score(base)
    assert train.selection_score(dict(base, N_p95_deg=170.0)) == score
    assert train.selection_score(dict(base, P_p95=0.006)) > score
    assert train.EPOCHS == 16
    assert train.CANDIDATE_EPOCHS == (4, 8, 12, 16)
    assert train.WARMUP_EPOCHS == 3
    assert train.GRAD_ACCUM == 4
    assert train.TRACK_SAMPLES_TRAIN == 128
    assert train.TRACK_SAMPLES_EVAL == 512

    bs = inspect.getsource(seed_builder)
    assert "--representation-seed" not in bs
    assert "json.load(open(a.representation_seed" not in bs
    assert "'representation_seed_consumed_at_runtime':False" in bs
    assert "'runtime_tune_fields':['asset_id','split','mini_role']" in bs

    ts = inspect.getsource(train)
    tm = inspect.getsource(train.main)
    assert "--tune-cache" not in ts
    assert "split='TUNE'" not in tm
    assert "build_report(model,ck" not in tm
    assert "CHECKPOINT_SELECTION_FROZEN.json" in ts
    assert "'tune_seen_during_selection':False" in ts
    assert "'tune_stage_cache_exists_during_selection':False" in ts

    rs = inspect.getsource(runner.main)
    assert "--representation-seed" not in rs
    i_fit = rs.index("'--splits','FIT'")
    i_train = rs.index("'train_mini_v2.py'")
    i_freeze = rs.index("'CHECKPOINT_SELECTION_FROZEN.json'")
    i_tune = rs.index("'--splits','TUNE'")
    i_final = rs.index("'finalize_mini_v2.py'")
    assert i_fit < i_train < i_freeze < i_tune < i_final
    assert "if tune_stage.exists() or tune_cache.exists():raise RuntimeError('TUNE stage/cache exists before checkpoint freeze')" in rs
    assert "ms.get('representation_seed_consumed_at_runtime') is not False" in rs

    fz = inspect.getsource(finalize.main)
    assert "CHECKPOINT_SELECTION_FROZEN.json" in fz and "--tune-cache" in fz and "build_report" in fz

    gpu_reg = gpu_preflight_regression(here)
    print(
        json.dumps(
            {
                "status": "PASS",
                "membership_frozen": True,
                "membership_canonical_sha256": msh,
                "membership_provenance_verified": True,
                "roles": {"FIT_TRAIN": 128, "FIT_SELECT": 32, "TUNE_FINAL": 26},
                "matcher_N_forbidden": True,
                "N_local_geometry_only": True,
                "N_checkpoint_selection_forbidden": True,
                "fine_track_prefix_bias_removed": True,
                "checkpoint_selection_frozen": True,
                "runtime_representation_seed_consumed": False,
                "runtime_tune_fields": ["asset_id", "split", "mini_role"],
                "tune_cli_absent_from_training_phase": True,
                "tune_stage_after_checkpoint_freeze_enforced": True,
                "gpu_preflight_regression": gpu_reg,
                "sealed_splits_opened": False,
                "optimizer_steps": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
