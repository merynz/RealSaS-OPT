from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call([str(x) for x in cmd])


def make_synthetic_master(root: Path):
    aid = "asset_repr_stage_synth"
    ar = root / "master" / "assets" / aid
    (ar / "renders").mkdir(parents=True, exist_ok=True)
    vertices = np.asarray([[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.4, 0.0]], np.float32)
    faces = np.asarray([[0, 1, 2]], np.int32)
    np.savez(
        ar / "primary_geometry.npz",
        vertices=vertices,
        faces=faces,
        parents=np.asarray([-1, 0], np.int32),
        skin=np.ones((3, 2), np.float32),
        bone_heads=np.zeros((2, 3), np.float32),
    )

    rng = np.random.default_rng(123)
    uv = []
    while len(uv) < 700:
        a, b = rng.random(2)
        if a + b <= 1:
            uv.append((a, b))
    uv = np.asarray(uv, np.float32)
    w = np.stack([uv[:, 0], uv[:, 1], 1 - uv[:, 0] - uv[:, 1]], -1)
    p = (vertices[faces[0]][None] * w[:, :, None]).sum(1)
    res = 1024
    gx = p[:, 0] / 0.5
    gy = -p[:, 1] / 0.5
    x = np.floor((gx + 1) * 0.5 * res).astype(np.int64)
    y = np.floor((gy + 1) * 0.5 * res).astype(np.int64)
    pix = y * res + x
    order = np.argsort(pix)
    pix = pix[order]
    uv = uv[order]
    _, first = np.unique(pix, return_index=True)
    pix = pix[first]
    uv = uv[first]

    for v in range(8):
        vd = ar / "renders" / f"V{v}"
        vd.mkdir(parents=True, exist_ok=True)
        np.savez(
            vd / "raster_authority.npz",
            resolution=np.asarray([res], np.int32),
            pixel_linear_index=pix,
            triangle_id=np.zeros(len(pix), np.int32),
            barycentric_uv=uv,
        )
        (vd / "camera.json").write_text(
            json.dumps({"yaw_deg": float(v * 45), "right": [1, 0, 0], "up": [0, 1, 0], "half_extent": 0.5}),
            encoding="utf-8",
        )
        # Deliberately invalid decoy image bytes: representation staging must never read them.
        (vd / "cel_clean.png").write_bytes(b"DECOY_RGB_MUST_NOT_BE_READ")
        (vd / "ink_cel.png").write_bytes(b"DECOY_RGB_MUST_NOT_BE_READ")
    return aid, vertices, faces


def write_seed(path: Path, aid: str):
    path.write_text(json.dumps({
        "schema": "synthetic",
        "records": [{"asset_id": aid, "split": "FIT", "source_registry_id": "SYNTH"}],
    }), encoding="utf-8")


def stage_fingerprint(stage: Path, aid: str):
    marker = json.load((stage / "assets" / aid / "STAGE.json").open(encoding="utf-8"))
    assert marker["stage_profile"] == "representation_authority_geometry_only"
    assert marker["images_staged"] is False
    assert marker["geometry_fields"] == ["vertices", "faces"]
    for v in range(8):
        vd = stage / "assets" / aid / "renders" / f"V{v}"
        assert {p.name for p in vd.iterdir() if p.is_file()} == {"camera.json", "raster_authority.npz"}
    return marker["source_fingerprint"]


def restage(here: Path, master: Path, seed: Path, stage: Path):
    shutil.rmtree(stage, ignore_errors=True)
    run([
        sys.executable, here / "stage_representation_authority_v1.py",
        "--root", master,
        "--seed-manifest", seed,
        "--out", stage,
        "--splits", "FIT,TUNE",
    ])


def main():
    here = Path(__file__).resolve().parent
    root = Path(tempfile.mkdtemp(prefix="irisv2_repr_data_preflight_"))
    try:
        master = root / "masterroot"
        stage = root / "stage"
        cache = root / "cache"
        seed = root / "seed.json"
        audit = root / "audit.json"
        audit_fail = root / "audit_fail.json"
        aid, vertices, faces = make_synthetic_master(master)
        write_seed(seed, aid)

        # Initial stage: no image decoding/copying is allowed.
        restage(here, master, seed, stage)
        fp0 = stage_fingerprint(stage, aid)
        with np.load(stage / "assets" / aid / "primary_geometry.npz", allow_pickle=False) as z:
            assert set(z.files) == {"vertices", "faces"}

        # RGB changes are intentionally outside this exact P/N Representation Authority apparatus.
        decoy = master / "master" / "assets" / aid / "renders" / "V0" / "cel_clean.png"
        decoy.write_bytes(b"MUTATED_RGB_STILL_MUST_NOT_BE_READ")
        restage(here, master, seed, stage)
        fp_rgb = stage_fingerprint(stage, aid)
        assert fp_rgb == fp0, "RGB mutation incorrectly changed representation-stage fingerprint"

        # Hidden rig/mechanics changes are also outside the legal representation source boundary.
        np.savez(
            master / "master" / "assets" / aid / "primary_geometry.npz",
            vertices=vertices,
            faces=faces,
            parents=np.asarray([99, 77], np.int32),
            skin=np.zeros((3, 2), np.float32),
            bone_heads=np.ones((2, 3), np.float32),
        )
        restage(here, master, seed, stage)
        fp_hidden = stage_fingerprint(stage, aid)
        assert fp_hidden == fp0, "hidden rig mutation incorrectly changed legal representation fingerprint"

        # A legal raster-authority change must invalidate the representation source fingerprint.
        rp = master / "master" / "assets" / aid / "renders" / "V0" / "raster_authority.npz"
        with np.load(rp, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        uv = arrays["barycentric_uv"].copy()
        uv[0, 0] = np.float32(max(0.0, min(0.49, float(uv[0, 0]) * 0.9)))
        arrays["barycentric_uv"] = uv
        np.savez(rp, **arrays)
        restage(here, master, seed, stage)
        fp_raster = stage_fingerprint(stage, aid)
        assert fp_raster != fp_hidden, "legal raster mutation failed to invalidate representation source fingerprint"

        run([
            sys.executable, here / "prepare_representation_cache_v1.py",
            "--stage-manifest", stage / "STAGE_MANIFEST.json",
            "--out", cache,
            "--geom-samples", "64",
            "--anchors-per-view", "64",
            "--max-tracks", "128",
            "--radius-px", "3",
            "--max-surface-error", "0.004",
        ])
        cm = json.load((cache / "CACHE_MANIFEST.json").open(encoding="utf-8"))
        assert cm["stage_profile"] == "representation_authority_geometry_only"
        assert cm["rgb_consumed"] is False
        assert cm["record_count"] == 1
        assert cm["records"][0]["track_count"] > 0

        run([
            sys.executable, here / "audit_representation_stage_cache_v1.py",
            "--stage-manifest", stage / "STAGE_MANIFEST.json",
            "--cache-manifest", cache / "CACHE_MANIFEST.json",
            "--seed-manifest", seed,
            "--out", audit,
            "--progress-every", "1",
        ])
        report = json.load(audit.open(encoding="utf-8"))
        assert report["status"] == "PASS", report
        assert report["fatal_asset_count"] == 0, report
        assert report["images_required"] is False, report

        # Auditor must fail if any image/unexpected file leaks into representation staging.
        leaked = stage / "assets" / aid / "renders" / "V0" / "cel_clean_input.png"
        leaked.write_bytes(b"LEAK")
        run([
            sys.executable, here / "audit_representation_stage_cache_v1.py",
            "--stage-manifest", stage / "STAGE_MANIFEST.json",
            "--cache-manifest", cache / "CACHE_MANIFEST.json",
            "--seed-manifest", seed,
            "--out", audit_fail,
            "--progress-every", "1",
        ])
        failed = json.load(audit_fail.open(encoding="utf-8"))
        assert failed["status"] == "FAIL"
        assert failed["fatal_asset_count"] == 1
        assert any("unexpected_files" in x for x in failed["fatal_assets"][0]["fatal"])

        print(json.dumps({
            "status": "PASS",
            "representation_stage_profile": "geometry+raster+camera only",
            "rgb_decoy_never_read": True,
            "rgb_mutation_excluded_from_fingerprint": True,
            "hidden_rig_mutation_excluded_from_fingerprint": True,
            "raster_mutation_invalidates_fingerprint": True,
            "physical_firewall": True,
            "cache_rgb_consumed": False,
            "auditor_rejects_image_leak": True,
            "optimizer_steps": 0,
        }, indent=2))
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
