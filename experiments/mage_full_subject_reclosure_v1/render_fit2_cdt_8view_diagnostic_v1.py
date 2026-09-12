from __future__ import annotations

"""Render the exact current Mage FIT2 observation-domain CDT result in all 8 views.

Diagnostic only: this reproduces the already-closed S->directional CDT behavioral result.
It does not require or claim fresh Geppetto/Arachne mechanics and therefore is not a
FIT2 product-mesh PASS.  The output is intended to answer one narrow question:
"does the actual CDT deformation domain still contain visually obvious holes?"
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import polygon

from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
    build_mwb2_observation_cdt_candidate,
    qualify_mwb2_observation_cdt_mesh,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from experiments.mage_full_subject_reclosure_v1.run_frozen_geppetto_replay_v2 import (
    ACTIVE_GSA_LINEAGE,
    ACTIVE_ZERO_SURFACE_SHA256,
    CAMERA_SHA256,
    OBSERVATION_SHA256,
    _build_active_surface,
    _load_cameras,
    _load_observation_masks,
)

SCHEMA = "RealSaS.MageFIT2.CDT8ViewDiagnostic.v1"
H1_RUN_ID = "20260912T074348Z"
H1_CHECKPOINT_SHA256 = "76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5"
EXPECTED_CANDIDATE_LINEAGES = (
    "b063921e23a6ce6f27f15925810c62811453a4a70eef22664ba385b0d0c71b89",
    "87fe281cba69fa52484eafbe327cc429ca8c8b0182e34f60f2f083c5b3add8ef",
    "3e9142fe2f57da50f9208bf428a98820545871f94a8d531d1226d0a9efb5a714",
    "3f3d42464f139603f2fcf40c83258530710316ec11a2cd2a69df503ff9fd1720",
    "147934d64ba5728da36bd065536ca6db261372b5a9acf08ec699ba079f81b768",
    "bc3cdb7973b4b03353e7f49c8be917b15162ea6630b39bd3fbf8065c2c176592",
    "3c929b7a8918466be2e691f4b1b0ad1435d1bd0a339320f9755ae3143967db28",
    "34216cd50d0637f9f13c0e7012ae805b56d3671686e8ac53f00d4b824e1d4ee1",
)
EXPECTED_MESH_LINEAGES = (
    "b2e18ce5864e6e722b53bd21c66ebd7ff912e339dd69dc9fa12bd7b5fc2fad01",
    "7ea8b6eddde49a2745b8af8b28a9b3c3e458e3ab13e1d8572b1d40176569aea5",
    "5de17063149bd190d40ed188d7df299816e6ebb7fe045c1510fbf33e28180ae6",
    "8e23b9d50a38c0d3936b26c99b69bfbf47fd6b7950cc3fc86069db919cf7ae57",
    "f99f76efc0a38dab800e294963b9a533ba2aab7cf7fcfe86b5bf5cb84cc16a2b",
    "73e1c31ead96e202a38af2ba0df7c7ad7ed42fa00551ba1ab5dabbdc475ac006",
    "487827c265fd9dfa41a4e345437db6516f14f040ab31f7fd456e6430ace33185",
    "4ffbc77f2f24ffd590e064dbeb1a5e23505117c4def5bf98b4ca3d1cc9d00b4d",
)
EXPECTED_RECALL = (
    0.9258877389925143,
    0.9406053363705958,
    0.9055737721419602,
    0.9412975358672925,
    0.9284963409210367,
    0.9379882275563154,
    0.9075763715671493,
    0.943664854740135,
)
EXPECTED_FACE_COUNT = (4801, 5219, 4022, 5730, 5245, 5411, 3995, 5305)


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _candidate_xy(candidate):
    return {
        v.candidate_vertex_id: tuple(map(float, v.metadata["raster_xy"]))
        for v in candidate.vertices
    }


def _predicted_mask(candidate, width: int, height: int) -> np.ndarray:
    xy = _candidate_xy(candidate)
    out = np.zeros((height, width), dtype=np.bool_)
    for face in candidate.faces:
        pts = [xy[v] for v in face]
        rr, cc = polygon(
            [p[1] for p in pts], [p[0] for p in pts], shape=(height, width)
        )
        out[rr, cc] = True
    return out


def _draw_wire(base: Image.Image, candidate) -> Image.Image:
    canvas = base.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    xy = _candidate_xy(candidate)
    # translucent face fill makes true uncovered artist pixels obvious without hiding art
    for face in candidate.faces:
        pts = [xy[v] for v in face]
        draw.polygon(pts, fill=(40, 170, 255, 34))
    for a, b in candidate.edges:
        pa, pb = xy[a], xy[b]
        draw.line((pa[0], pa[1], pb[0], pb[1]), fill=(0, 235, 255, 155), width=1)
    return Image.alpha_composite(canvas, overlay)


def _draw_uncovered(base: Image.Image, authority: np.ndarray, predicted: np.ndarray) -> Image.Image:
    canvas = base.convert("RGBA")
    arr = np.zeros((authority.shape[0], authority.shape[1], 4), dtype=np.uint8)
    missing = authority & ~predicted
    covered = authority & predicted
    arr[covered] = (30, 220, 90, 36)
    arr[missing] = (255, 35, 35, 210)
    return Image.alpha_composite(canvas, Image.fromarray(arr, mode="RGBA"))


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, min(out.width, 1000), 42), fill=(0, 0, 0, 205))
    d.text((10, 10), text, fill=(255, 255, 255, 255))
    return out


def run(args) -> dict:
    zero = Path(args.zero_surface)
    camera_paths = tuple(Path(x) for x in args.cameras)
    observation_paths = tuple(Path(x) for x in args.observations)
    if _sha(zero) != ACTIVE_ZERO_SURFACE_SHA256:
        raise RuntimeError("ZERO_SURFACE_SHA_DRIFT")
    for i, path in enumerate(camera_paths):
        if _sha(path) != CAMERA_SHA256[i]:
            raise RuntimeError(f"CAMERA_SHA_DRIFT_V{i}")
    for i, path in enumerate(observation_paths):
        if _sha(path) != OBSERVATION_SHA256[i]:
            raise RuntimeError(f"OBSERVATION_SHA_DRIFT_V{i}")

    cameras = _load_cameras(list(camera_paths))
    masks = _load_observation_masks(observation_paths)
    surface_args = SimpleNamespace(
        zero_surface=str(zero),
        iris_run_id=H1_RUN_ID,
        iris_checkpoint_sha256=H1_CHECKPOINT_SHA256,
    )
    surface, tensor = _build_active_surface(surface_args, cameras, masks)
    if surface.geometry_lineage_hash != ACTIVE_GSA_LINEAGE:
        raise RuntimeError("GSA_LINEAGE_DRIFT")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    contact_rows = []
    wire_panels = []

    for view, (obs_path, mask) in enumerate(zip(observation_paths, masks)):
        with Image.open(obs_path) as im:
            source = im.convert("RGBA")
        domain = ObservationRasterDomain.from_rows(
            mask.tolist(),
            view_index=view,
            source_alpha_sha256=OBSERVATION_SHA256[view],
        )
        candidate = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=view,
            camera_binding_hash=CAMERA_SHA256[view],
            observation_domain=domain,
        )
        qualified = qualify_mwb2_observation_cdt_mesh(surface, candidate)

        predicted = _predicted_mask(candidate, domain.width, domain.height)
        authority = np.frombuffer(domain.mask_bytes, dtype=np.uint8).reshape(domain.height, domain.width).astype(bool)
        coverage = domain.coverage(
            tuple(
                tuple(_candidate_xy(candidate)[v] for v in face)
                for face in candidate.faces
            )
        )

        # Geometry must reproduce the sealed exact-CDT report, not merely look plausible.
        recall = float(coverage["source_alpha_recall"])
        if abs(recall - EXPECTED_RECALL[view]) > 1.0e-12:
            raise RuntimeError(f"CDT_RECALL_DRIFT_V{view}:{recall}:{EXPECTED_RECALL[view]}")
        if len(candidate.faces) != EXPECTED_FACE_COUNT[view]:
            raise RuntimeError(f"CDT_FACE_COUNT_DRIFT_V{view}:{len(candidate.faces)}:{EXPECTED_FACE_COUNT[view]}")
        if candidate.candidate_lineage_hash != EXPECTED_CANDIDATE_LINEAGES[view]:
            raise RuntimeError(
                f"CDT_CANDIDATE_LINEAGE_DRIFT_V{view}:{candidate.candidate_lineage_hash}:{EXPECTED_CANDIDATE_LINEAGES[view]}"
            )
        if qualified.mesh_lineage_hash != EXPECTED_MESH_LINEAGES[view]:
            raise RuntimeError(
                f"CDT_MESH_LINEAGE_DRIFT_V{view}:{qualified.mesh_lineage_hash}:{EXPECTED_MESH_LINEAGES[view]}"
            )

        wire = _label(
            _draw_wire(source, candidate),
            f"V{view} EXACT CDT | faces={len(candidate.faces)} | recall={recall*100:.2f}% | precision={float(coverage['precision_inside_alpha'])*100:.2f}%",
        )
        holes = _label(
            _draw_uncovered(source, authority, predicted),
            f"V{view} UNCOVERED RED | missing={(authority & ~predicted).sum()} px | largest-hole={float(coverage['largest_uncovered_component_fraction'])*100:.2f}% fg",
        )
        src = _label(source, f"V{view} REAL SOURCE RGBA")

        wire_path = outdir / f"FIT2_EXACT_CDT_V{view}_WIRE.png"
        holes_path = outdir / f"FIT2_EXACT_CDT_V{view}_UNCOVERED.png"
        triptych_path = outdir / f"FIT2_EXACT_CDT_V{view}_TRIPTYCH.png"
        wire.save(wire_path)
        holes.save(holes_path)
        strip = Image.new("RGB", (3072, 1024))
        strip.paste(src.convert("RGB"), (0, 0))
        strip.paste(wire.convert("RGB"), (1024, 0))
        strip.paste(holes.convert("RGB"), (2048, 0))
        strip.save(triptych_path)

        wire_panels.append(wire.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS))
        contact_rows.append(strip.resize((1536, 512), Image.Resampling.LANCZOS))
        rows.append(
            {
                "view": view,
                "candidate_lineage_hash": candidate.candidate_lineage_hash,
                "mesh_lineage_hash": qualified.mesh_lineage_hash,
                "vertex_count": len(candidate.vertices),
                "edge_count": len(candidate.edges),
                "face_count": len(candidate.faces),
                "source_alpha_recall": recall,
                "precision_inside_alpha": float(coverage["precision_inside_alpha"]),
                "alpha_iou": float(coverage["alpha_iou"]),
                "largest_uncovered_component_fraction": float(coverage["largest_uncovered_component_fraction"]),
                "wire_png": wire_path.name,
                "uncovered_png": holes_path.name,
                "triptych_png": triptych_path.name,
            }
        )

    wires = Image.new("RGB", (2048, 1024))
    for i, panel in enumerate(wire_panels):
        wires.paste(panel, ((i % 4) * 512, (i // 4) * 512))
    wire_sheet = outdir / "FIT2_EXACT_CDT_8VIEW_WIREFRAME_CONTACT_SHEET.png"
    wires.save(wire_sheet)

    diagnostic = Image.new("RGB", (1536, 4096))
    for i, row in enumerate(contact_rows):
        diagnostic.paste(row, (0, i * 512))
    diag_sheet = outdir / "FIT2_EXACT_CDT_8VIEW_SOURCE_WIRE_UNCOVERED.png"
    diagnostic.save(diag_sheet)

    manifest = {
        "schema": SCHEMA,
        "status": "PASS__EXACT_SEALED_CDT_GEOMETRY_REPRODUCED_AND_RENDERED",
        "scope": "DIAGNOSTIC_PRE_GW__NOT_PRODUCT_PASS",
        "gsa_lineage_hash": surface.geometry_lineage_hash,
        "zero_surface_sha256": ACTIVE_ZERO_SURFACE_SHA256,
        "views": rows,
        "minimum_recall": min(r["source_alpha_recall"] for r in rows),
        "minimum_precision": min(r["precision_inside_alpha"] for r in rows),
        "wire_contact_sheet": wire_sheet.name,
        "diagnostic_contact_sheet": diag_sheet.name,
        "synthetic_image_generation_used": False,
        "fresh_geppetto_required_for_this_diagnostic": False,
        "fresh_arachne_required_for_this_diagnostic": False,
        "product_mesh_pass_claimed": False,
    }
    manifest_path = outdir / "FIT2_EXACT_CDT_8VIEW_DIAGNOSTIC_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
