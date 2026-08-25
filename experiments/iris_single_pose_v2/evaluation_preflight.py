from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_preflight import make_synthetic_master
from stage_source import stage_asset
from prepare_cache import build_asset
from evaluate_v2 import evaluate_style
from matcher import MatcherConfig
from model import IRISSinglePoseV2, IRISV2Config


def make_authority_aligned_rgba(master: Path, aid: str):
    ar = master / "master" / "assets" / aid
    for v in range(8):
        vd = ar / "renders" / f"V{v}"
        with np.load(vd / "raster_authority.npz", allow_pickle=False) as ra:
            pix = ra["pixel_linear_index"].astype(np.int64)
        rgba = np.zeros((1024, 1024, 4), np.uint8)
        yy = pix // 1024
        xx = pix % 1024
        rgba[yy, xx, :3] = 255
        rgba[yy, xx, 3] = 255
        native = Image.fromarray(rgba, "RGBA")
        for style in ("cel_clean", "ink_cel"):
            native.save(vd / f"{style}.png")
            native.resize((512, 512), resample=Image.Resampling.BILINEAR).save(vd / f"{style}_512.png")


def tiny_cfg():
    return IRISV2Config(
        widths=(8, 16, 24, 32),
        coarse_dim=16,
        fine_dim=8,
        context_hw=8,
        within_heads=4,
        cross_heads=4,
        cross_layers=1,
    )


def main():
    root = Path(tempfile.mkdtemp(prefix="irisv2_eval_preflight_"))
    try:
        master = root / "masterroot"
        stage = root / "stage"
        cache = root / "cache"
        aid = make_synthetic_master(master)
        make_authority_aligned_rgba(master, aid)
        smeta = stage_asset(master, stage, aid, 256)
        record = {
            "asset_id": aid,
            "split": "FIT",
            "asset_dir": str(stage / "assets" / aid),
            "source_fingerprint": smeta["source_fingerprint"],
        }
        cm = build_asset(
            stage,
            record,
            cache,
            geom_samples=32,
            anchors_per_view=32,
            max_tracks=32,
            radius_px=3,
            max_surface_error=0.004,
        )
        manifest = {
            "schema": "RealSaS.IRISSinglePoseV2.CacheManifest.v2",
            "authority_resolution": 1024,
            "input_resolution": 256,
            "record_count": 1,
            "records": [cm],
        }
        cache_manifest = cache / "CACHE_MANIFEST.json"
        cache_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        seed_rows = {
            aid: {
                "asset_id": aid,
                "split": "FIT",
                "source_registry_id": "synthetic_provider",
                "candidate_id": "synthetic_candidate",
            }
        }
        torch.manual_seed(7)
        model = IRISSinglePoseV2(tiny_cfg()).eval()
        result = evaluate_style(
            model,
            str(cache_manifest),
            seed_rows,
            "FIT",
            "cel_clean",
            torch.device("cpu"),
            track_samples=16,
            query_limit=1,
            qualification_limit=0,
            cfg=MatcherConfig(coarse_keep=8, p_rescue_keep=4, final_topk=8, fine_radius_cells=2, fine_per_basin=2),
        )
        if result["family"]["asset_count"] != 1:
            raise AssertionError(result["family"])
        if result["assets"][0]["source_registry_id"] != "synthetic_provider":
            raise AssertionError(result["assets"][0])
        qn = result["assets"][0]["correspondence"].get("query_count", 0)
        if qn < 1:
            raise AssertionError("evaluator produced no correspondence query")
        p95 = result["assets"][0]["geometry"]["P"]["p95"]
        if p95 is None or not np.isfinite(p95):
            raise AssertionError(result["assets"][0]["geometry"])
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "asset_count": result["family"]["asset_count"],
                    "query_count": qn,
                    "source_registry_id": result["assets"][0]["source_registry_id"],
                    "P_p95": p95,
                },
                indent=2,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
