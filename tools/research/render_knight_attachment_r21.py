#!/usr/bin/env python3
from __future__ import annotations

"""Render the Knight R21 attachment court without mutating canonical M.

Research-only diagnostic.  The generic operator is imported from compiler core;
the Knight component/slot rows below are sealed Fit1 presentation evidence, not
generic policy and not a product authority claim.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import (
    apply_anchored_camera_twist_2d,
    derive_surface_binding,
)
from compiler.realsas_compiler_core.single_m_caa_v1 import (
    material_sampler,
    prepare_caa_field,
    render_m_depth,
)
from compiler.realsas_compiler_core.visual_attachment_motion_v1 import (
    slot_rigid_transform_2d,
)

# Exact qualified Knight Fit1 evidence recovered from the Stage37/42 presentation
# package.  These rows only select already-authored pieces and their qualified
# target slots.  R21 itself contains no Knight/component/slot constants.
KNIGHT_FIT1_ATTACHMENTS = (
    dict(name="SWORD", owner=1, components=(4, 5, 6, 34, 35, 36), slot=13),
    dict(
        name="SHIELD",
        owner=2,
        components=(7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 30, 31, 32, 33, 37, 38),
        slot=8,
    ),
)
CLIPS = {"RUN": 1, "SLASH": 2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connected_components(vertex_count: int, faces: np.ndarray):
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(x):
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    for a, b, c in np.asarray(faces, dtype=np.int64):
        union(a, b)
        union(b, c)
    groups = {}
    for i in range(vertex_count):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(v, dtype=np.int64) for v in sorted(groups.values(), key=lambda x: min(x))]


def owner_field(rest, faces):
    comps = connected_components(len(rest), faces)
    if len(comps) != 39:
        raise RuntimeError(f"KNIGHT_COMPONENT_COUNT_DRIFT:{len(comps)}")
    owners = np.zeros(len(rest), dtype=np.int32)
    for row in KNIGHT_FIT1_ATTACHMENTS:
        for ci in row["components"]:
            owners[comps[ci]] = row["owner"]
    counts = {row["name"]: int(np.count_nonzero(owners == row["owner"])) for row in KNIGHT_FIT1_ATTACHMENTS}
    if counts != {"SWORD": 371, "SHIELD": 831}:
        raise RuntimeError(f"KNIGHT_ATTACHMENT_OWNER_DRIFT:{counts}")
    return owners, comps


def load_material(fixture: Path):
    manifest = json.loads((fixture / "baseline.json").read_text())
    with np.load(fixture / "run_poses.npz", allow_pickle=False) as z:
        rest = z["rest"].astype(np.float64)
        faces = z["faces"].astype(np.int64)
    cameras = []
    for row in manifest["cameras"]:
        c = qualify_camera_v3(row, view_id=row["view_id"], view_index=row["view_index"])
        cameras.append(type(c)(**row))
    source = []
    with zipfile.ZipFile(fixture / "source_rgba.zip") as archive:
        for v in range(8):
            source.append(np.asarray(Image.open(io.BytesIO(archive.read(f"observations/V{v}.png"))).convert("RGBA")))
    with np.load(fixture / "compiled_caa.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    caa = prepare_caa_field(arrays, rest, faces)
    # Compile rest visibility exactly as the canonical material path does.
    from compiler.realsas_compiler_core.single_m_caa_v1 import compile_rest_visibility
    maps = compile_rest_visibility(rest, faces, cameras)
    return manifest, rest, faces, cameras, source, caa, maps


def locate_witness(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = []
    for p in path.rglob("*.npz"):
        try:
            with np.load(p, allow_pickle=False) as z:
                keys = set(z.files)
                needed = {
                    "vertices",
                    "faces",
                    "axis_positions_source",
                    "clip_1_canonical_xyz",
                    "clip_1_skin_matrices_source",
                    "clip_2_canonical_xyz",
                    "clip_2_skin_matrices_source",
                }
                if needed <= keys:
                    candidates.append(p)
        except Exception:
            pass
    if len(candidates) != 1:
        raise RuntimeError(f"R21_WITNESS_DISCOVERY_AMBIGUOUS:{[str(x) for x in candidates]}")
    return candidates[0]


def render_projected(projected, *, rest, faces, source, maps, cameras, caa, view, resolution):
    sampler, counts = material_sampler(rest, faces, source, maps, cameras, caa, view, True)
    result = render_m_depth(
        positions=projected[:, :2] - 0.5,
        depths=projected[:, 2],
        faces=faces,
        uv=np.zeros((len(rest), 2)),
        texture=source[view],
        resolution=resolution,
        sample_material=sampler,
    )
    result.material_fragments = counts
    return result.straight_rgba_u8


def composite(rgba):
    h, w = rgba.shape[:2]
    yy, xx = np.indices((h, w))
    bg = np.full((h, w, 4), (38, 44, 55, 255), np.uint8)
    bg[(xx // 16 + yy // 16) % 2 == 1] = (44, 50, 61, 255)
    return Image.alpha_composite(Image.fromarray(bg), Image.fromarray(rgba)).convert("RGB")


def compare_frame(base, fixed, title):
    base, fixed = composite(base), composite(fixed)
    w, h = base.size
    out = Image.new("RGB", (2 * w, h + 32), (23, 28, 37))
    out.paste(base, (0, 32))
    out.paste(fixed, (w, 32))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 7), f"{title}  BASE", font=font, fill=(235, 240, 247))
    draw.text((w + 8, 7), "R21 SURFACE-ANCHORED", font=font, fill=(235, 240, 247))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--witness", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--views", default="6,5")
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()

    fixture = ROOT / "tests/fixtures/single_m_caa_knight_v1"
    manifest, rest, faces, cameras, source, caa, maps = load_material(fixture)
    witness_path = locate_witness(args.witness)
    with np.load(witness_path, allow_pickle=False) as z:
        witness = {k: z[k] for k in z.files}
    if not np.array_equal(witness["vertices"], rest) or not np.array_equal(witness["faces"], faces):
        raise RuntimeError("R21_PARENT_WITNESS_DOMAIN_DRIFT")
    owners, _ = owner_field(rest, faces)
    bindings = {
        row["name"]: derive_surface_binding(
            rest_xyz=rest,
            faces=faces,
            attachment_vertex_owner=owners,
            attachment_index=row["owner"],
        )
        for row in KNIGHT_FIT1_ATTACHMENTS
    }
    views = [int(x) for x in args.views.split(",")]
    if len(views) != 2 or any(v < 0 or v >= 8 for v in views):
        raise RuntimeError("R21_REQUIRES_EXACT_TWO_VALID_VIEWS")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "RealSaS.KnightAttachmentR21Diagnostic.v1",
        "product_qualification_claimed": False,
        "witness": str(witness_path),
        "witness_sha256": sha256(witness_path),
        "views": views,
        "attachments": {},
        "clips": {},
    }
    for row in KNIGHT_FIT1_ATTACHMENTS:
        b = bindings[row["name"]]
        report["attachments"][row["name"]] = {
            "owner": row["owner"],
            "slot": row["slot"],
            "component_indices": list(row["components"]),
            "vertex_count": int(len(b.attachment_vertex_indices)),
            "held_face_index": int(b.held.face_index),
            "held_barycentric": list(b.held.barycentric),
            "holder_face_index": int(b.holder.face_index),
            "holder_barycentric": list(b.holder.barycentric),
            "setup_surface_gap_world": float(b.setup_surface_gap),
        }

    for clip_name, clip_index in CLIPS.items():
        poses = witness[f"clip_{clip_index}_canonical_xyz"].astype(np.float64)
        mats = witness[f"clip_{clip_index}_skin_matrices_source"].astype(np.float64)
        times = witness[f"clip_{clip_index}_times"].astype(np.float64)
        axis = witness["axis_positions_source"].astype(np.float64)
        clip_report = {"frame_count": int(len(poses)), "views": {}}
        for view in views:
            camera = cameras[view]
            rest_proj = project_points_xyz_v3(rest, camera)
            frames = []
            proofs = []
            for fi, pose in enumerate(poses):
                raw_proj = project_points_xyz_v3(pose, camera)
                fixed_proj = raw_proj.copy()
                frame_proof = {"frame_index": fi, "attachments": {}}
                for row in KNIGHT_FIT1_ATTACHMENTS:
                    R2, _ = slot_rigid_transform_2d(
                        slot_rest_xyz=axis[row["slot"]],
                        skin_matrix_source=mats[fi, row["slot"]],
                        camera=camera,
                    )
                    fixed_proj, proof = apply_anchored_camera_twist_2d(
                        fixed_proj,
                        projected_rest=rest_proj,
                        faces=faces,
                        binding=bindings[row["name"]],
                        rotation_2d=R2,
                    )
                    frame_proof["attachments"][row["name"]] = proof
                base = render_projected(
                    raw_proj, rest=rest, faces=faces, source=source, maps=maps,
                    cameras=cameras, caa=caa, view=view, resolution=args.resolution,
                )
                fixed = render_projected(
                    fixed_proj, rest=rest, faces=faces, source=source, maps=maps,
                    cameras=cameras, caa=caa, view=view, resolution=args.resolution,
                )
                frames.append(compare_frame(base, fixed, f"{clip_name} V{view} F{fi:02d}"))
                proofs.append(frame_proof)
            # Preserve recorded timing; last frame uses the previous interval.
            if len(times) > 1:
                ms = [max(20, int(round(float(times[i + 1] - times[i]) * 1000))) for i in range(len(times) - 1)]
                ms.append(ms[-1])
            else:
                ms = [80]
            gif = out / f"KNIGHT_{clip_name}_V{view}_BASE_vs_R21.gif"
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=ms, loop=0)
            clip_report["views"][f"V{view}"] = {
                "gif": gif.name,
                "sha256": sha256(gif),
                "frames": proofs,
            }
        report["clips"][clip_name] = clip_report
    (out / "R21_ATTACHMENT_PROOF.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "witness": str(witness_path),
        "outputs": [x.name for x in sorted(out.glob('*.gif'))],
        "attachments": report["attachments"],
    }, indent=2))


if __name__ == "__main__":
    main()
