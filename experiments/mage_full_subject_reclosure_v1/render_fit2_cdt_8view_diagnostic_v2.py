from __future__ import annotations

"""Render the exact Mage FIT2 observation-domain CDT result in all 8 views.

V2 explicitly separates *historical sealed lineage* from *current enriched diagnostics*.

The 2026-09-12 sealed CDT report was produced before ObservationRasterDomain.coverage()
gained IoU/F1/connected-component diagnostics and before mesh_binding qualification gained
additional report fields / 5-digit MV ids. Those later diagnostic-contract enrichments are
included in content hashes, so directly comparing today's candidate/mesh lineage hashes with
the historical sealed hashes reports a false geometry drift.

This renderer therefore:
1. verifies exact H1/GSA/camera/RGBA authority;
2. builds today's CDT candidate;
3. reconstructs the historical v3 candidate hash by projecting residual_report to the exact
   field set that existed at the sealing commit;
4. reconstructs the historical qualified-mesh hash with the exact old identity qualifier;
5. requires both reconstructed hashes + all sealed structural/coverage metrics to match;
6. separately records today's enriched candidate/mesh lineage;
7. renders source | actual CDT | uncovered pixels.

Diagnostic only. This is not a FIT2 product-mesh PASS and does not require Geppetto/Arachne.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from skimage.draw import polygon

from compiler.realsas_compiler_core.mesh._historical_v05 import HISTORICAL_CDT_SOURCE_SHA256
from compiler.realsas_compiler_core.mesh.mesh_binding import (
    mesh_candidate_lineage_hash,
    mesh_lineage_hash,
)
from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
    build_mwb2_observation_cdt_candidate,
    qualify_mwb2_observation_cdt_mesh,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedMeshVertex,
)
from experiments.mage_full_subject_reclosure_v1.run_frozen_geppetto_replay_v2 import (
    ACTIVE_GSA_LINEAGE,
    ACTIVE_ZERO_SURFACE_SHA256,
    CAMERA_SHA256,
    OBSERVATION_SHA256,
    _build_active_surface,
    _load_cameras,
    _load_observation_masks,
)

SCHEMA = "RealSaS.MageFIT2.CDT8ViewDiagnostic.v2"
H1_RUN_ID = "20260912T074348Z"
H1_CHECKPOINT_SHA256 = "76fc68a8c6f2bed80ae8a678649006c875bc96b586065da8914e7f61528c8ce5"
SEALED_REPORT_PATH = (
    Path(__file__).resolve().parent / "MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json"
)

# Exact residual field set present when historical candidate lineages were sealed.
# New coverage diagnostics are useful, but are not geometry authority and must not rewrite
# the meaning of an already-sealed historical lineage comparison.
HISTORICAL_RESIDUAL_KEYS = (
    "predicted_pixel_count",
    "inside_alpha_pixel_count",
    "foreground_pixel_count",
    "precision_inside_alpha",
    "source_alpha_recall",
    "face_count",
    "edge_count",
    "vertex_count",
    "visible_surface_node_count",
    "safe_relation_edge_count",
    "full_safe_relation_edge_count",
    "safe_component_count",
    "full_safe_component_count",
    "cdt_component_count",
    "tiny_component_count",
    "kernel_triangle_count",
    "alpha_rejected_face_count",
    "duplicate_projected_node_count",
    "kernel_constraint_split_count",
    "contracted_boundary_recovery_vertex_count",
    "post_contraction_triangle_count",
)


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
    for face in candidate.faces:
        pts = [xy[v] for v in face]
        draw.polygon(pts, fill=(40, 170, 255, 34))
    for a, b in candidate.edges:
        pa, pb = xy[a], xy[b]
        draw.line((pa[0], pa[1], pb[0], pb[1]), fill=(0, 235, 255, 155), width=1)
    return Image.alpha_composite(canvas, overlay)


def _draw_uncovered(
    base: Image.Image, authority: np.ndarray, predicted: np.ndarray
) -> Image.Image:
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


def _historical_candidate_projection(candidate):
    missing = [k for k in HISTORICAL_RESIDUAL_KEYS if k not in candidate.residual_report]
    if missing:
        raise RuntimeError("HISTORICAL_RESIDUAL_FIELD_MISSING:" + ",".join(missing))
    residual = {k: candidate.residual_report[k] for k in HISTORICAL_RESIDUAL_KEYS}
    projected = replace(candidate, residual_report=residual, candidate_lineage_hash="")
    projected = replace(
        projected, candidate_lineage_hash=mesh_candidate_lineage_hash(projected)
    )
    return projected


def _historical_qualified_mesh(surface, candidate):
    """Reproduce the exact f02df44 identity/CDT qualification hash semantics."""
    ordered = tuple(sorted(candidate.vertices, key=lambda v: v.candidate_vertex_id))
    id_map = {
        v.candidate_vertex_id: f"MV:{i:04d}" for i, v in enumerate(ordered)
    }
    vertices = tuple(
        QualifiedMeshVertex(
            id_map[v.candidate_vertex_id],
            v.P,
            v.support_binding,
            v.candidate_vertex_id,
            dict(v.metadata),
        )
        for v in ordered
    )
    faces = tuple(tuple(id_map[x] for x in face) for face in candidate.faces)
    edges = tuple((id_map[a], id_map[b]) for a, b in candidate.edges)
    report = {
        "status": "PASS_IDENTITY_SUBSET_QUALIFICATION",
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "edge_count": len(edges),
        "no_new_geometry_authority": True,
        "camera_geometry_consumed": bool(
            candidate.metadata.get("camera_geometry_consumed", False)
        ),
        "numerical_solver_promoted": False,
    }
    mesh = QualifiedEditableMeshIR(
        vertices,
        faces,
        edges,
        surface.geometry_lineage_hash,
        candidate.view_index,
        candidate.camera_binding_hash,
        report,
        "",
        candidate.boundary_constraints,
        candidate.coverage_classification,
        metadata={"source_candidate_lineage_hash": candidate.candidate_lineage_hash},
    )
    mesh = replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))

    report = dict(mesh.qualification_report)
    report.update(
        {
            "status": "PASS_OBSERVATION_DOMAIN_CDT_QUALIFICATION",
            "source_alpha_recall": float(
                candidate.residual_report["source_alpha_recall"]
            ),
            "precision_inside_alpha": float(
                candidate.residual_report["precision_inside_alpha"]
            ),
            "historical_cdt_source_sha256": HISTORICAL_CDT_SOURCE_SHA256,
            "historical_kernel_role": "NUMERICAL_ONLY",
            "cdt_behavioral_gate_pass": True,
            "numerical_solver_promoted": True,
        }
    )
    updated = replace(
        mesh,
        qualification_report=report,
        metadata={
            **mesh.metadata,
            "cdt_adapter": "RealSaS.MWB2.ObservationDomainCDT.v3",
        },
        mesh_lineage_hash="",
    )
    return replace(updated, mesh_lineage_hash=mesh_lineage_hash(updated))


def _require_exact_metric(label: str, got, expected) -> None:
    if isinstance(expected, float):
        if abs(float(got) - float(expected)) > 1.0e-12:
            raise RuntimeError(f"{label}_DRIFT:{got}:{expected}")
    else:
        if int(got) != int(expected):
            raise RuntimeError(f"{label}_DRIFT:{got}:{expected}")


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

    if not SEALED_REPORT_PATH.is_file():
        raise RuntimeError(f"SEALED_CDT_REPORT_MISSING:{SEALED_REPORT_PATH}")
    sealed = json.loads(SEALED_REPORT_PATH.read_text(encoding="utf-8"))
    if sealed.get("status") != "PASS_MWB2_CDT_EXACT_BEHAVIORAL_RECLOSURE":
        raise RuntimeError("SEALED_CDT_REPORT_STATUS_DRIFT")
    sealed_views = {int(row["view"]): row for row in sealed["views"]}

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

    print(
        "LINEAGE_POLICY: historical sealed hashes are reconstructed with the exact "
        "pre-enrichment diagnostic schema; current enriched hashes are recorded separately.",
        flush=True,
    )

    for view, (obs_path, mask) in enumerate(zip(observation_paths, masks)):
        expected = sealed_views[view]
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

        # Current enriched qualifier is still required to pass today's contract.
        current_qualified = qualify_mwb2_observation_cdt_mesh(surface, candidate)

        # Reconstruct the historical sealed hash semantics without discarding today's
        # additional diagnostics.
        historical_candidate = _historical_candidate_projection(candidate)
        historical_mesh = _historical_qualified_mesh(surface, historical_candidate)

        if historical_candidate.candidate_lineage_hash != expected["candidate_lineage_hash"]:
            raise RuntimeError(
                f"HISTORICAL_CDT_CANDIDATE_LINEAGE_DRIFT_V{view}:"
                f"{historical_candidate.candidate_lineage_hash}:"
                f"{expected['candidate_lineage_hash']}"
            )
        if historical_mesh.mesh_lineage_hash != expected["mesh_lineage_hash"]:
            raise RuntimeError(
                f"HISTORICAL_CDT_MESH_LINEAGE_DRIFT_V{view}:"
                f"{historical_mesh.mesh_lineage_hash}:{expected['mesh_lineage_hash']}"
            )

        # Structural + behavioral sealed metrics must also reproduce exactly.
        sealed_metric_map = {
            "source_alpha_recall": candidate.residual_report["source_alpha_recall"],
            "precision_inside_alpha": candidate.residual_report["precision_inside_alpha"],
            "predicted_pixel_count": candidate.residual_report["predicted_pixel_count"],
            "foreground_pixel_count": candidate.residual_report["foreground_pixel_count"],
            "visible_surface_node_count": candidate.residual_report[
                "visible_surface_node_count"
            ],
            "face_count": len(candidate.faces),
            "edge_count": len(candidate.edges),
            "vertex_count": len(candidate.vertices),
            "kernel_triangle_count": candidate.residual_report["kernel_triangle_count"],
            "alpha_rejected_face_count": candidate.residual_report[
                "alpha_rejected_face_count"
            ],
            "kernel_constraint_split_count": candidate.residual_report[
                "kernel_constraint_split_count"
            ],
            "contracted_boundary_recovery_vertex_count": candidate.residual_report[
                "contracted_boundary_recovery_vertex_count"
            ],
            "post_contraction_triangle_count": candidate.residual_report[
                "post_contraction_triangle_count"
            ],
        }
        for key, got in sealed_metric_map.items():
            _require_exact_metric(f"SEALED_{key.upper()}_V{view}", got, expected[key])

        predicted = _predicted_mask(candidate, domain.width, domain.height)
        authority = (
            np.frombuffer(domain.mask_bytes, dtype=np.uint8)
            .reshape(domain.height, domain.width)
            .astype(bool)
        )
        coverage = domain.coverage(
            tuple(
                tuple(_candidate_xy(candidate)[v] for v in face)
                for face in candidate.faces
            )
        )

        recall = float(coverage["source_alpha_recall"])
        precision = float(coverage["precision_inside_alpha"])
        wire = _label(
            _draw_wire(source, candidate),
            f"V{view} EXACT CDT | faces={len(candidate.faces)} | "
            f"recall={recall*100:.2f}% | precision={precision*100:.2f}%",
        )
        holes = _label(
            _draw_uncovered(source, authority, predicted),
            f"V{view} UNCOVERED RED | missing={(authority & ~predicted).sum()} px | "
            f"largest-hole={float(coverage['largest_uncovered_component_fraction'])*100:.2f}% fg",
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

        wire_panels.append(
            wire.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
        )
        contact_rows.append(
            strip.resize((1536, 512), Image.Resampling.LANCZOS)
        )

        row = {
            "view": view,
            "historical_candidate_lineage_hash": (
                historical_candidate.candidate_lineage_hash
            ),
            "historical_candidate_lineage_match": True,
            "historical_mesh_lineage_hash": historical_mesh.mesh_lineage_hash,
            "historical_mesh_lineage_match": True,
            "current_enriched_candidate_lineage_hash": candidate.candidate_lineage_hash,
            "current_enriched_mesh_lineage_hash": current_qualified.mesh_lineage_hash,
            "vertex_count": len(candidate.vertices),
            "edge_count": len(candidate.edges),
            "face_count": len(candidate.faces),
            "source_alpha_recall": recall,
            "precision_inside_alpha": precision,
            "alpha_iou": float(coverage["alpha_iou"]),
            "alpha_f1": float(coverage["alpha_f1"]),
            "largest_uncovered_component_fraction": float(
                coverage["largest_uncovered_component_fraction"]
            ),
            "largest_uncovered_component_pixels": int(
                coverage["largest_uncovered_component_pixels"]
            ),
            "uncovered_component_count": int(
                coverage["uncovered_component_count"]
            ),
            "wire_png": wire_path.name,
            "uncovered_png": holes_path.name,
            "triptych_png": triptych_path.name,
        }
        rows.append(row)
        print(
            f"V{view} PASS | faces={row['face_count']} verts={row['vertex_count']} "
            f"recall={recall*100:.3f}% precision={precision*100:.3f}% "
            f"IoU={row['alpha_iou']*100:.3f}% "
            f"largest_hole={row['largest_uncovered_component_fraction']*100:.3f}% "
            f"| historical lineage EXACT",
            flush=True,
        )

    wires = Image.new("RGB", (2048, 1024))
    for i, panel in enumerate(wire_panels):
        wires.paste(panel, ((i % 4) * 512, (i // 4) * 512))
    wire_sheet = outdir / "FIT2_EXACT_CDT_8VIEW_WIREFRAME_CONTACT_SHEET.png"
    wires.save(wire_sheet)

    diagnostic = Image.new("RGB", (1536, 4096))
    for i, row_img in enumerate(contact_rows):
        diagnostic.paste(row_img, (0, i * 512))
    diag_sheet = outdir / "FIT2_EXACT_CDT_8VIEW_SOURCE_WIRE_UNCOVERED.png"
    diagnostic.save(diag_sheet)

    manifest = {
        "schema": SCHEMA,
        "status": "PASS__HISTORICAL_SEALED_CDT_GEOMETRY_REPRODUCED__CURRENT_DIAGNOSTICS_RENDERED",
        "scope": "DIAGNOSTIC_PRE_GW__NOT_PRODUCT_PASS",
        "lineage_policy": {
            "historical_candidate_and_mesh_hashes_reconstructed": True,
            "reason": (
                "post-seal diagnostic/report-schema enrichment changed content hashes "
                "without changing CDT geometry authority"
            ),
            "current_enriched_hashes_recorded_separately": True,
        },
        "gsa_lineage_hash": surface.geometry_lineage_hash,
        "zero_surface_sha256": ACTIVE_ZERO_SURFACE_SHA256,
        "views": rows,
        "minimum_recall": min(r["source_alpha_recall"] for r in rows),
        "minimum_precision": min(r["precision_inside_alpha"] for r in rows),
        "maximum_largest_uncovered_component_fraction": max(
            r["largest_uncovered_component_fraction"] for r in rows
        ),
        "wire_contact_sheet": wire_sheet.name,
        "diagnostic_contact_sheet": diag_sheet.name,
        "synthetic_image_generation_used": False,
        "fresh_geppetto_required_for_this_diagnostic": False,
        "fresh_arachne_required_for_this_diagnostic": False,
        "product_mesh_pass_claimed": False,
    }
    manifest_path = outdir / "FIT2_EXACT_CDT_8VIEW_DIAGNOSTIC_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
