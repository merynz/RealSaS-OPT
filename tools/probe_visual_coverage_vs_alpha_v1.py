"""Unqualified geometric-support versus artwork-alpha raster diagnostic.

An opaque-texture control preserves positions, UVs, triangles and depth. It
measures covered pixels without interpreting alpha as intended empty support.
It does not supply an intended silhouette or qualify semantic visibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_core.visual_depth_v2 import render_visual_depth


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(field, faces, uv, texture, resolution):
    arguments = dict(positions=field[:, :2], depths=field[:, 2],
                     faces=faces, uv=uv, resolution=resolution)
    # This is a numerical support ablation, not replacement appearance.
    opaque = np.full_like(texture, 255)
    geometry = render_visual_depth(texture=opaque, **arguments).straight_rgba_u8[:, :, 3] > 0
    artwork = render_visual_depth(texture=texture, **arguments).straight_rgba_u8[:, :, 3] > 0
    enclosed = binary_fill_holes(geometry) & ~geometry
    components, _ = label(enclosed)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    regions = []
    for component in np.argsort(sizes)[::-1][:5]:
        if not sizes[component]:
            continue
        y, x = np.where(components == component)
        regions.append(dict(pixel_count=int(sizes[component]),
                            bbox=[int(x.min()), int(y.min()), int(x.max()), int(y.max())]))
    return dict(geometry_covered_pixel_count=int(geometry.sum()),
                geometry_covered_but_art_alpha_zero_pixel_count=int((geometry & ~artwork).sum()),
                enclosed_geometric_void_pixel_count=int(enclosed.sum()),
                largest_enclosed_voids=regions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-arrays", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-template", required=True,
                        help="Absolute NPZ path template containing {view}; each has clip-prefix XYZ arrays")
    parser.add_argument("--views", type=int, nargs="+", default=[0, 4, 6])
    parser.add_argument("--clip-prefix", default="clip_1")
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.projection_arrays, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    bindings, rows = [], []
    for view in args.views:
        source_path = args.source_root / f"V{view}.png"
        candidate_path = Path(args.candidate_template.format(view=view))
        with np.load(candidate_path, allow_pickle=False) as source:
            candidate = source[args.clip_prefix][args.frame_index]
        texture = np.asarray(Image.open(source_path).convert("RGBA"))
        prefix = f"{args.clip_prefix}_view_{view}"
        baseline = np.c_[arrays[f"{prefix}_positions"][args.frame_index],
                         arrays[f"{prefix}_depths"][args.frame_index]]
        bindings.append(dict(view_id=f"V{view}", source_png_sha256=digest(source_path),
                             candidate_fields_npz_sha256=digest(candidate_path)))
        for variant, field in (("baseline", baseline), ("connected_diagnostic", candidate)):
            rows.append(dict(view_id=f"V{view}", clip_prefix=args.clip_prefix,
                             frame_index=args.frame_index, variant=variant,
                             **measure(field, arrays[f"view_{view}_faces"], arrays[f"view_{view}_uv"],
                                       texture, args.resolution)))
    report = dict(schema="RealSaS.VisualSupportAlphaDiagnostic.v1",
                  scope="UNQUALIFIED_LOCAL_ENGINEERING_DIAGNOSTIC_ONLY",
                  intended_silhouette_or_occlusion_qualified=False,
                  enclosed_void_semantics="ENCLOSED_EMPTY_SUPPORT__NOT_AUTOMATICALLY_AN_ERROR",
                  product_authority=False, resolution=args.resolution,
                  projection_arrays_sha256=digest(args.projection_arrays),
                  probe_source_sha256=digest(Path(__file__)), bindings=bindings, rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(report=str(args.out), rows=len(rows), qualification=report["scope"])))


if __name__ == "__main__":
    main()
