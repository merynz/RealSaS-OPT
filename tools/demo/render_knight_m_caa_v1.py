#!/usr/bin/env python3
"""Reproduce the canonical Knight development baseline without inference/LBS.

Only immutable recorded M poses are replayed. This is regression evidence,
never a new scientific qualification or ProductRevision.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_core.single_m_caa_v1 import (
    array_hash,
    load_material_bundle,
    render_single_m_caa,
    write_material_bundle,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.types import QualificationError

CONTEXT = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    for name, expected in manifest["inputs"].items():
        if digest(root / name) != expected:
            raise QualificationError("CANONICAL_BASELINE_INPUT_BYTES_DRIFT:" + name)
    with np.load(root / "run_poses.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if array_hash(arrays["poses"]) != manifest["pose_sha256"]:
        raise QualificationError("CANONICAL_BASELINE_M_POSES_DRIFT")
    source = []
    with zipfile.ZipFile(root / "source_rgba.zip") as archive:
        for view in range(8):
            raw = archive.read(f"observations/V{view}.png")
            if (
                hashlib.sha256(raw).hexdigest()
                != manifest["source_png_sha256"][str(view)]
            ):
                raise QualificationError("CANONICAL_BASELINE_SOURCE_PNG_DRIFT")
            source.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA")))
    cameras = []
    for row in manifest["cameras"]:
        validated = qualify_camera_v3(
            row, view_id=row["view_id"], view_index=row["view_index"]
        )
        cameras.append(type(validated)(**row))
    with np.load(root / "compiled_caa.npz", allow_pickle=False) as data:
        caa_arrays = {k: data[k] for k in data.files}
    return manifest, arrays, source, cameras, caa_arrays


def render_view(view):
    material = CONTEXT["material"]
    rows = []
    for index, posed in enumerate(CONTEXT["poses"]):
        result = render_single_m_caa(
            **{
                k: material[k]
                for k in ("rest", "faces", "source", "maps", "cameras", "caa")
            },
            posed=posed,
            view=view,
            resolution=256,
        )
        rgba = result.straight_rgba_u8
        Image.fromarray(rgba).save(CONTEXT["out"] / f"run_v{view}_{index:03d}.png")
        rows.append(
            {
                "frame_index": index,
                "rgba_sha256": array_hash(rgba),
                "depth_tie_count": result.unresolved_depth_tie_count,
                "fragment_overflow_count": result.fragment_overflow_count,
                "material_fragments": result.material_fragments,
            }
        )
    return view, rows


def tile(rgba, label):
    resolution = rgba.shape[0]
    yy, xx = np.indices((resolution, resolution))
    background = np.full((resolution, resolution, 4), (38, 44, 55, 255), np.uint8)
    background[(xx // 16 + yy // 16) % 2 == 1] = (44, 50, 61, 255)
    image = Image.alpha_composite(Image.fromarray(background), Image.fromarray(rgba))
    result = Image.new("RGB", (resolution, resolution + 32), (23, 28, 37))
    result.paste(image.convert("RGB"), (0, 32))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    ImageDraw.Draw(result).text((10, 7), label, font=font, fill=(235, 240, 247))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest, arrays, source, cameras, caa_arrays = load_inputs(args.baseline)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref = write_material_bundle(
        out / "single_m_caa_material.npz",
        rest=arrays["rest"],
        faces=arrays["faces"],
        source=source,
        cameras=cameras,
        caa_arrays=caa_arrays,
        input_identities=manifest["inputs"],
    )
    material = load_material_bundle(
        ref["path"],
        expected_sha256=ref["sha256"],
        rest=arrays["rest"],
        faces=arrays["faces"],
    )
    CONTEXT.update(material=material, poses=arrays["poses"], out=out)
    views = {}
    with ProcessPoolExecutor(
        max_workers=max(1, min(args.workers, 8)),
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        jobs = [pool.submit(render_view, v) for v in range(8)]
        for job in as_completed(jobs):
            view, rows = job.result()
            views[f"V{view}"] = rows
            print(f"V{view}: {len(rows)} fresh frames", flush=True)
    raw_mismatches = [
        key
        for key, expected in manifest["golden_rgba"].items()
        if views[key.split("/")[0]][int(key.split("/")[1])]["rgba_sha256"] != expected
    ]
    sheets = []
    for index in range(len(arrays["poses"])):
        sheet = Image.new("RGB", (1024, 576), (23, 28, 37))
        for view in range(8):
            rgba = np.asarray(
                Image.open(out / f"run_v{view}_{index:03d}.png").convert("RGBA")
            )
            sheet.paste(tile(rgba, f"V{view}"), ((view % 4) * 256, (view // 4) * 288))
        sheet.save(out / f"run_grid_{index:03d}.png")
        sheets.append(sheet)
    times = arrays["times"]
    durations = []
    for i, timestamp in enumerate(times):
        dt = (
            float(times[i + 1] - timestamp)
            if i + 1 < len(times)
            else float(times[1] - times[0])
        )
        start = float(timestamp - times[0])
        durations.append((round((start + dt) * 100) - round(start * 100)) * 10)
    gif = out / "Knight_RUN_M_CAA_8_views.gif"
    sheets[0].save(
        gif, save_all=True, append_images=sheets[1:], duration=durations, loop=0
    )
    golden = args.baseline.parent / "approved_run_8views.gif"
    if digest(golden) != manifest["approved_gif_sha256"]:
        raise QualificationError("CANONICAL_BASELINE_GOLDEN_BYTES_DRIFT")
    with Image.open(golden) as before, Image.open(gif) as after:
        previous = [
            np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(before)
        ]
        current = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(after)]
        gif_mismatches = sum(
            int(np.count_nonzero(a != b)) for a, b in zip(previous, current)
        )
        frame_count_equal = len(previous) == len(current) == len(times)
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = {
        "schema": "RealSaS.CanonicalSingleMCAAReplay.v1",
        "source_sha": source_sha,
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "baseline_manifest_sha256": digest(args.baseline),
        "material": ref,
        "pose_sha256_before": manifest["pose_sha256"],
        "pose_sha256_after": array_hash(arrays["poses"]),
        "skinning_executed": False,
        "retarget_executed": False,
        "inference_executed": False,
        "fresh_frames": len(times) * 8,
        "reused_render_frames": 0,
        "raw_rgba_golden_frames": len(manifest["golden_rgba"]),
        "raw_rgba_mismatches": raw_mismatches,
        "eight_view_gif_channel_mismatches": gif_mismatches,
        "frame_count_equal": frame_count_equal,
        "gif_sha256": digest(gif),
        "views": views,
        "open_gates": manifest["open_gates"],
        "product_qualification_claimed": False,
    }
    report["regression_passed"] = (
        not raw_mismatches
        and gif_mismatches == 0
        and frame_count_equal
        and report["pose_sha256_before"] == report["pose_sha256_after"]
    )
    (out / "render_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "source_sha",
                    "fresh_frames",
                    "raw_rgba_mismatches",
                    "eight_view_gif_channel_mismatches",
                    "regression_passed",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["regression_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
