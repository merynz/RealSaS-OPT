from __future__ import annotations

"""Materialize exact per-view runtime atlas + rigid foreground sprite geometry.

Panel 0 is the exact source observation used by the P1Q underlay. Panels 1..4 are
exact pixel copies masked by the qualified source-component owner raster for the four
rigid foreground roles. No new artist pixels are generated. This stage creates only
atlas/geometry/appearance authority; rigid carry skin is compiled later from the
qualified component assembly to avoid circular attachment authority.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_atlas import build_sprite_panel_appearance
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedMeshVertex,
    SurfaceSupportBinding,
)

SCHEMA = "RealSaS.MageRuntimeForegroundAtlas.v1"
EXPECTED_SOURCE_TRUTH_SHA256 = "a23565b0904e9683d11083984a87405f4e0a1069984ca11b690ad431f35f5e86"

RIGID_COMPONENTS = (
    ("BOOK_VARIANTS", "BOOK_FOREGROUND"),
    ("WAND_STAFF_VARIANTS", "STAFF_FOREGROUND"),
    ("HAT", "HAT_FOREGROUND"),
    ("CAPE", "CAPE_FOREGROUND"),
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sprite_mesh(*, component_id: str, view: int, width: int, height: int, camera_hash: str, owner_manifest_sha: str, component_family: str):
    coords = ((0.0, 0.0), (float(width - 1), 0.0), (float(width - 1), float(height - 1)), (0.0, float(height - 1)))
    carrier_p = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    surface_hash = content_sha256({
        "schema": "RealSaS.RigidForegroundSpriteSupport.v1",
        "owner_manifest_sha256": owner_manifest_sha,
        "component_family": component_family,
        "component_id": component_id,
        "view": view,
        "width": width,
        "height": height,
    })
    vertices = []
    for index, (xy, P) in enumerate(zip(coords, carrier_p)):
        sid = f"FG:{component_id}:V{view}:S{index}"
        vertices.append(QualifiedMeshVertex(
            f"FG:{component_id}:V{view}:Q{index}",
            P,
            SurfaceSupportBinding("RIGID_SPRITE_CARRIER", ((sid, 1.0),)),
            source_candidate_vertex_id=f"OWNER:{component_family}:V{view}:Q{index}",
            metadata={
                "raster_xy": xy,
                "source_raster_xy": xy,
                "runtime_sprite_carrier": True,
                "owner_manifest_sha256": owner_manifest_sha,
                "component_family": component_family,
            },
        ))
    ids = tuple(v.canonical_mesh_vertex_id for v in vertices)
    mesh = QualifiedEditableMeshIR(
        tuple(vertices),
        ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])),
        ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[3]), (ids[3], ids[0]), (ids[0], ids[2])),
        surface_hash,
        int(view),
        camera_hash,
        {
            "status": "PASS_EXACT_SOURCE_OWNER_MASK_SPRITE_GEOMETRY",
            "runtime_representation_only": True,
            "source_owner_manifest_sha256": owner_manifest_sha,
            "component_family": component_family,
            "new_pixels_generated": False,
        },
        "",
        support_coverage_classification="RIGID_FOREGROUND_SPRITE",
        metadata={
            "runtime_sprite_carrier": True,
            "source_owner_manifest_sha256": owner_manifest_sha,
            "component_family": component_family,
            "scientific_surface_authority": False,
        },
    )
    return replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))


def run(args) -> dict:
    source_truth_path = Path(args.source_truth)
    owner_dir = Path(args.owner_dir)
    owner_manifest_path = owner_dir / "SOURCE_COMPONENT_OWNER_RASTER_AUTHORITY.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _sha(source_truth_path) != EXPECTED_SOURCE_TRUTH_SHA256:
        raise RuntimeError("FOREGROUND_ATLAS_SOURCE_TRUTH_SHA_DRIFT")
    owner_manifest_sha = _sha(owner_manifest_path)
    if args.expected_owner_manifest and owner_manifest_sha != args.expected_owner_manifest:
        raise RuntimeError("FOREGROUND_ATLAS_OWNER_MANIFEST_SHA_DRIFT")
    owner_manifest = json.loads(owner_manifest_path.read_text(encoding="utf-8"))
    if owner_manifest.get("status") != "PASS__EXACT_SOURCE_FIT_TARGET_COMPONENT_OWNER_RASTER_V0_V7":
        raise RuntimeError("FOREGROUND_ATLAS_OWNER_MANIFEST_STATUS_INVALID")
    if owner_manifest.get("source_truth_sha256") != EXPECTED_SOURCE_TRUTH_SHA256:
        raise RuntimeError("FOREGROUND_ATLAS_OWNER_TRUTH_BINDING_DRIFT")

    cameras = tuple(Path(x) for x in args.cameras)
    observations = tuple(Path(x) for x in args.observations)
    if len(cameras) != 8 or len(observations) != 8:
        raise RuntimeError("FOREGROUND_ATLAS_REQUIRES_8_VIEWS")
    owner_rows = {int(row["view"]): row for row in owner_manifest["views"]}
    if set(owner_rows) != set(range(8)):
        raise RuntimeError("FOREGROUND_ATLAS_OWNER_VIEWS_INCOMPLETE")

    component_names = tuple(map(str, owner_manifest["component_names"]))
    component_index = {name: i for i, name in enumerate(component_names)}
    if any(family not in component_index for family, _ in RIGID_COMPONENTS):
        raise RuntimeError("FOREGROUND_ATLAS_REQUIRED_SOURCE_COMPONENT_MISSING")

    rows, artifacts = [], []
    for view in range(8):
        camera_path, observation_path = cameras[view], observations[view]
        camera_hash, observation_hash = _sha(camera_path), _sha(observation_path)
        if owner_rows[view]["camera_sha256"] != camera_hash or owner_rows[view]["observation_sha256"] != observation_hash:
            raise RuntimeError(f"FOREGROUND_ATLAS_VIEW_AUTHORITY_DRIFT_V{view}")

        owner_path = owner_dir / f"V{view}_SOURCE_COMPONENT_OWNER_RASTER.npz"
        if _sha(owner_path) != owner_rows[view]["owner_raster_sha256"]:
            raise RuntimeError(f"FOREGROUND_ATLAS_OWNER_RASTER_SHA_DRIFT_V{view}")
        with np.load(owner_path, allow_pickle=False) as payload:
            owner = np.asarray(payload["owner"], dtype=np.int16)
            payload_names = tuple(map(str, payload["component_names"].tolist()))
        if payload_names != component_names:
            raise RuntimeError(f"FOREGROUND_ATLAS_COMPONENT_ORDER_DRIFT_V{view}")

        with Image.open(observation_path) as image:
            observation = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        height, width = observation.shape[:2]
        if owner.shape != (height, width):
            raise RuntimeError(f"FOREGROUND_ATLAS_OWNER_DIMENSION_DRIFT_V{view}")
        atlas_width = width * (1 + len(RIGID_COMPONENTS))
        atlas = np.zeros((height, atlas_width, 4), dtype=np.uint8)
        atlas[:, :width] = observation

        component_rows = []
        for panel_index, (family, runtime_id) in enumerate(RIGID_COMPONENTS, start=1):
            mask = owner == int(component_index[family])
            panel = np.zeros_like(observation)
            panel[mask] = observation[mask]
            x0 = panel_index * width
            atlas[:, x0:x0 + width] = panel
            pixel_count = int(np.count_nonzero(panel[..., 3]))
            if pixel_count <= 0:
                raise RuntimeError(f"FOREGROUND_ATLAS_EMPTY_REQUIRED_COMPONENT_V{view}:{family}")

            mesh = _sprite_mesh(
                component_id=runtime_id,
                view=view,
                width=width,
                height=height,
                camera_hash=camera_hash,
                owner_manifest_sha=owner_manifest_sha,
                component_family=family,
            )
            component_rows.append({
                "component_family": family,
                "runtime_component_id": runtime_id,
                "panel_index": panel_index,
                "panel_x_offset": x0,
                "visible_pixel_count": pixel_count,
                "mesh": mesh,
            })

        atlas_name = f"V{view}_MAGE_PRODUCT_ATLAS.png"
        atlas_path = output_dir / atlas_name
        Image.fromarray(atlas, mode="RGBA").save(atlas_path, format="PNG", optimize=False, compress_level=9)
        atlas_sha = _sha(atlas_path)
        atlas_relpath = f"textures/{atlas_name}"
        atlas_payload_hash = content_sha256({"image_sha256": atlas_sha, "image_relpath": atlas_relpath})
        artifacts.append({"path": atlas_name, "sha256": atlas_sha, "bytes": atlas_path.stat().st_size})

        persisted_components = []
        for component in component_rows:
            mesh = component.pop("mesh")
            appearance = build_sprite_panel_appearance(
                mesh,
                target_view_index=view,
                source_observation_hash=observation_hash,
                source_width=width,
                source_height=height,
                atlas_width=atlas_width,
                atlas_height=height,
                panel_x_offset=int(component["panel_x_offset"]),
                atlas_payload_hash=atlas_payload_hash,
            )
            mesh_name = f"V{view}_{component['runtime_component_id']}_SPRITE_MESH.json"
            app_name = f"V{view}_{component['runtime_component_id']}_APPEARANCE.json"
            mesh_path, app_path = output_dir / mesh_name, output_dir / app_name
            _write_json(mesh_path, mesh.to_dict())
            _write_json(app_path, appearance.to_dict())
            artifacts.extend([
                {"path": mesh_name, "sha256": _sha(mesh_path), "bytes": mesh_path.stat().st_size},
                {"path": app_name, "sha256": _sha(app_path), "bytes": app_path.stat().st_size},
            ])
            persisted_components.append({
                **component,
                "mesh_file": mesh_name,
                "mesh_file_sha256": _sha(mesh_path),
                "mesh_lineage_hash": mesh.mesh_lineage_hash,
                "appearance_file": app_name,
                "appearance_file_sha256": _sha(app_path),
                "appearance_lineage_hash": appearance.appearance_lineage_hash,
            })

        rows.append({
            "view": view,
            "camera_sha256": camera_hash,
            "observation_sha256": observation_hash,
            "owner_raster_sha256": owner_rows[view]["owner_raster_sha256"],
            "width": width,
            "height": height,
            "atlas_width": atlas_width,
            "atlas_height": height,
            "atlas_file": atlas_name,
            "atlas_relpath": atlas_relpath,
            "atlas_sha256": atlas_sha,
            "atlas_payload_hash": atlas_payload_hash,
            "components": persisted_components,
        })
        print("FOREGROUND_ATLAS_V" + str(view) + "=" + json.dumps({
            "atlas_sha256": atlas_sha,
            "component_pixels": {c["runtime_component_id"]: c["visible_pixel_count"] for c in persisted_components},
        }, sort_keys=True), flush=True)

    manifest = {
        "schema": SCHEMA,
        "status": "PASS__EXACT_OBSERVATION_PLUS_TYPED_RIGID_FOREGROUND_ATLAS_V0_V7",
        "source_truth_sha256": EXPECTED_SOURCE_TRUTH_SHA256,
        "owner_manifest_sha256": owner_manifest_sha,
        "panel_policy": {
            "panel_0": "EXACT_FULL_OBSERVATION_UNDERLAY",
            "panels_1_4": "EXACT_SOURCE_OWNER_MASKED_OBSERVATION_RGBA",
            "new_pixels_generated": False,
            "cross_view_donor_used": False,
        },
        "rigid_components": [runtime_id for _family, runtime_id in RIGID_COMPONENTS],
        "views": rows,
        "artifacts": artifacts,
        "product_pass_claimed": False,
    }
    manifest_path = output_dir / "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json"
    _write_json(manifest_path, manifest)
    print("FOREGROUND_ATLAS_MANIFEST_SHA256=" + _sha(manifest_path), flush=True)
    return manifest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-truth", required=True)
    p.add_argument("--owner-dir", required=True)
    p.add_argument("--expected-owner-manifest", default="")
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
