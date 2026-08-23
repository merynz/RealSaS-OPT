from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from representation_authority_study_v1 import (
    arm_definitions,
    observed_fields,
    perturb_normals,
    philox_rng,
)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_asset(root: Path, asset_id: str, split: str):
    ar = root / "stage" / "assets" / asset_id
    ar.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]], np.float32)
    faces = np.asarray([[0, 1, 2]], np.int32)
    np.savez(ar / "primary_geometry.npz", vertices=vertices, faces=faces)

    # Track 0 and 1 deliberately form one legal set-valued locus under tol=0.003.
    p = np.asarray([
        [-0.30, -0.20, 0.0],
        [-0.299, -0.20, 0.0],
        [-0.10, -0.10, 0.0],
        [0.00, 0.00, 0.0],
        [0.10, 0.05, 0.0],
        [0.20, 0.10, 0.0],
        [0.30, 0.15, 0.0],
        [-0.20, 0.20, 0.0],
        [-0.05, 0.25, 0.0],
        [0.05, 0.28, 0.0],
        [0.18, 0.30, 0.0],
        [0.32, 0.34, 0.0],
    ], np.float32)
    t = len(p)
    vis = np.ones((t, 8), np.uint8)
    support = np.full((t,), 8, np.uint8)
    n = np.zeros((t, 8, 3), np.float32)
    n[..., 2] = 1.0
    xy = np.zeros((t, 8, 2), np.float32)
    truth = root / "cache" / "truth" / f"{asset_id}.npz"
    truth.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        truth,
        track_p=p,
        track_n_view=n,
        track_visible=vis,
        track_support=support,
        track_xy=xy,
    )
    import hashlib
    h = hashlib.sha256(truth.read_bytes()).hexdigest()
    return {
        "schema": "RealSaS.IRISSinglePoseV2.CacheAsset.v2",
        "asset_id": asset_id,
        "split": split,
        "asset_dir": str(ar),
        "truth_path": str(truth),
        "truth_sha256": h,
    }


def main():
    # Deterministic observation namespace: repeated calls must be byte-identical.
    p = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]], np.float32)
    n = np.zeros((2, 8, 3), np.float32)
    n[..., 2] = 1.0
    arm = {"id": "TEST", "p_sigma": 0.005, "n_deg": 20.0, "use_n": True}
    p1, n1 = observed_fields("asset_repeat", arm, p, n)
    p2, n2 = observed_fields("asset_repeat", arm, p, n)
    assert np.array_equal(p1, p2)
    assert np.array_equal(n1, n2)

    # Exact angular perturbation, not xyz-Gaussian approximation.
    base = np.zeros((32, 3), np.float32)
    base[:, 2] = 1.0
    out = perturb_normals(base, 20.0, philox_rng("asset_angle", "ANGLE20", 0))
    angle = np.degrees(np.arccos(np.clip((out * base).sum(-1), -1.0, 1.0)))
    assert float(np.max(np.abs(angle - 20.0))) < 1e-3, angle

    arms = arm_definitions()
    assert len(arms) == 38
    assert len([a for a in arms if a["family"] == "R3"]) == 30

    with tempfile.TemporaryDirectory(prefix="repr_authority_preflight_") as td:
        root = Path(td)
        records = [
            make_asset(root, "asset_a", "FIT"),
            make_asset(root, "asset_b", "FIT"),
            make_asset(root, "asset_c", "TUNE"),
        ]
        cache_manifest = root / "cache" / "CACHE_MANIFEST.json"
        write_json(cache_manifest, {
            "schema": "RealSaS.IRISSinglePoseV2.CacheManifest.v2",
            "record_count": len(records),
            "records": records,
        })
        selection = {
            "asset_a": {"split": "FIT", "source_registry_id": "provider_a", "capabilities": {"iris": True, "geppetto": False, "arachne": False}},
            "asset_b": {"split": "FIT", "source_registry_id": "provider_b", "capabilities": {"iris": True, "geppetto": True, "arachne": True}},
            "asset_c": {"split": "TUNE", "source_registry_id": "provider_a", "capabilities": {"iris": True, "geppetto": True, "arachne": False}},
        }
        selection_path = root / "CANONICAL_VARIANT_SELECTION.json"
        write_json(selection_path, selection)
        audit_path = root / "STAGE_CACHE_AUDIT.json"
        write_json(audit_path, {"status": "PASS", "fatal_asset_count": 0, "asset_count": 3})
        out_path = root / "RESULT.json"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("representation_authority_study_v1.py")),
            "--cache-manifest", str(cache_manifest),
            "--selection-json", str(selection_path),
            "--stage-cache-audit", str(audit_path),
            "--out", str(out_path),
            "--dev-max-assets", "3",
            "--dev-per-category", "2",
            "--progress-every", "99",
        ]
        subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        result = json.load(open(out_path, encoding="utf-8"))
        assert result["optimizer_steps"] == 0
        assert result["status"] == "DEVELOPMENT_ONLY"
        assert result["arm_count"] == 38
        assert result["asset_count"] == 3
        assert result["structural_confusability_diagnostic"]["semantic_symmetry_claim"] is False

        r0_full = result["arms"]["R0_P_EXACT"]
        r0 = r0_full["pooled"]
        r1 = result["arms"]["R1_PN_EXACT"]["pooled"]
        assert r0["top1"] == 1.0 and r0["top8"] == 1.0, r0
        assert r1["top1"] == 1.0 and r1["top8"] == 1.0, r1
        assert r0["reciprocal_success"] == 1.0 and r0["cycle_success"] == 1.0, r0
        assert r0["ambiguous_fraction"] > 0.0, r0

        gap = r0["nearest_non_equivalent_physical_gap"]
        assert gap["n"] > 0 and gap["median"] > 0.003, gap
        family = r0_full["family_tail"]
        assert family["assets"] == 3
        for key in ("asset_error_median_distribution", "asset_error_p90_distribution", "asset_error_p95_distribution"):
            assert "median" in family[key] and "p90" in family[key] and "p95" in family[key], family[key]

        panel = json.load(open(root / "REPRESENTATION_AUTHORITY_PANEL_V1.json", encoding="utf-8"))
        assert len(panel["selected_asset_ids"]) == 3
        assert panel["mode"] == "DEVELOPMENT_ONLY"

    print(json.dumps({
        "status": "PASS",
        "optimizer_steps": 0,
        "arm_count": 38,
        "set_valued_exact_R0": "PASS",
        "reciprocal_cycle_exact_R0": "PASS",
        "philox_repeatability": "PASS",
        "exact_angular_noise": "PASS",
        "family_p90_p95_tail": "PASS",
        "structural_confusability_not_semantic_symmetry": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
