from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.mage_full_subject_reclosure_v1.run_geppetto_fit2_corrected_substrate_refit_v1 import (
    ACTIVE_GSA_LINEAGE,
    EXPECTED_TENSORIZATION_HASH,
    _load_inputs,
)

SCHEMA = "RealSaS.MageFIT2.GSA8ViewArtifact.v1"


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _project(points, camera):
    p = np.asarray(points, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64)
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["screen_up"], dtype=np.float64)
    half = float(camera["half_extent"])
    res = int(camera["resolution"])
    d = p - origin[None, :]
    gx = (d @ right) / half
    gy = -(d @ up) / half
    return np.stack(((gx + 1.0) * 0.5 * res - 0.5, (gy + 1.0) * 0.5 * res - 0.5), axis=1)


def run(args):
    _, _, obs_paths, cameras, surface, tensor, _, _ = _load_inputs(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    panels = []
    points = np.asarray(tensor.positions_world, dtype=np.float64)
    edges = np.asarray(tensor.edge_index, dtype=np.int64)

    for view, (camera, observation_path) in enumerate(zip(cameras, obs_paths)):
        with Image.open(observation_path) as im:
            base = im.convert("RGBA")
        xy = _project(points, camera)
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for a, b in edges:
            xa, ya = xy[a]
            xb, yb = xy[b]
            if ((-16 <= xa <= 1039 and -16 <= ya <= 1039) or (-16 <= xb <= 1039 and -16 <= yb <= 1039)):
                draw.line((float(xa), float(ya), float(xb), float(yb)), fill=(255, 255, 255, 42), width=1)
        for i, (x, y) in enumerate(xy):
            if not (-6 <= x <= 1029 and -6 <= y <= 1029):
                continue
            if bool(tensor.support[i, view]):
                radius, fill = 2.2, (30, 255, 110, 235)
            elif bool(tensor.completed[i]):
                radius, fill = 1.5, (255, 80, 220, 150)
            else:
                radius, fill = 1.2, (80, 180, 255, 115)
            draw.ellipse((float(x-radius), float(y-radius), float(x+radius), float(y+radius)), fill=fill)
        canvas = Image.alpha_composite(base, overlay)
        label = ImageDraw.Draw(canvas)
        label.rectangle((0, 0, 500, 42), fill=(0, 0, 0, 190))
        label.text((10, 10), f"FIT2 REAL GSA8192 V{view} / yaw {camera['yaw_deg']} support={int(tensor.support[:, view].sum())}", fill=(255, 255, 255, 255))
        path = out / f"FIT2_GSA8192_V{view}_REAL_OVERLAY.png"
        canvas.save(path)
        files.append(path.name)
        panels.append(canvas.convert("RGB"))

    sheet = Image.new("RGB", (2048, 4096))
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * 1024, (i // 2) * 1024))
    sheet_path = out / "FIT2_GSA8192_8VIEW_REAL_CONTACT_SHEET.png"
    sheet.save(sheet_path)
    files.append(sheet_path.name)

    manifest = {
        "schema": SCHEMA,
        "status": "PASS_REAL_8VIEW_GSA_ARTIFACT_EMITTED",
        "gsa_lineage_hash": surface.geometry_lineage_hash,
        "tensorization_hash": tensor.tensorization_hash,
        "nodes": tensor.node_count,
        "edges": tensor.edge_count,
        "observed_nodes": int(tensor.observed.sum()),
        "completed_nodes": int(tensor.completed.sum()),
        "support_counts": [int(x) for x in tensor.support.sum(axis=0).tolist()],
        "files": [{"name": name, "sha256": _sha(out / name)} for name in files],
        "synthetic_image_generation_used": False,
        "render_basis": "REAL_OBSERVATIONS_PLUS_REAL_GSA_GRAPH_PROJECTION",
    }
    if manifest["gsa_lineage_hash"] != ACTIVE_GSA_LINEAGE or manifest["tensorization_hash"] != EXPECTED_TENSORIZATION_HASH:
        raise RuntimeError("FIT2_GSA_ARTIFACT_AUTHORITY_DRIFT")
    (out / "FIT2_GSA8192_8VIEW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--normalized-corpus", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
