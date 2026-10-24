#!/usr/bin/env python3
"""Build the offline Knight investor-demo core from canonical RealSaS authorities.

The exporter is intentionally read-only with respect to Compiler/runtime authority.
It binds the investor viewer to the same canonical M + source-wins/CAA render line
used by ``tools/demo/render_knight_m_caa_v1.py`` and discovers G/W by schema from
an exact sealed Knight run. Motion clips are a separate plug-in boundary and are
left empty until the user selects the exact animations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.product_artifact_codec_v1 import (  # noqa: E402
    qualified_mesh_from_dict,
    qualified_mesh_skin_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.single_m_caa_v1 import (  # noqa: E402
    array_hash,
    load_material_bundle,
    render_single_m_caa,
    write_material_bundle,
)
from compiler.realsas_compiler_core.types import QualificationError  # noqa: E402
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (  # noqa: E402
    sha256_file,
)
from compiler.realsas_compiler_services.orchestrator.status_semantics import (  # noqa: E402
    dependency_status_admissible,
)
from tools.demo.render_knight_m_caa_v1 import load_inputs  # noqa: E402

SCHEMA_G = "RealSaS.QualifiedSkeletonIR.v1"
SCHEMA_M = "RealSaS.QualifiedMeshIR.v1"
SCHEMA_W = "RealSaS.QualifiedMeshSkinIR.v1"
BASELINE = ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json"
WEB_ROOT = ROOT / "tools/investor_demo/web"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _resolve_output_path(run_root: Path, raw: str) -> Path:
    value = Path(str(raw)).expanduser()
    candidates = [value] if value.is_absolute() else [run_root / value, ROOT / value, value]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise QualificationError(f"INVESTOR_DEMO_AUTHORITY_FILE_MISSING:{raw}")


def _schema_output(run_root: Path, schema: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    ledger_path = run_root / "ACTIVE_RUN_V2.json"
    if not ledger_path.is_file():
        raise QualificationError("INVESTOR_DEMO_ACTIVE_RUN_MISSING")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    matches: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for stage in ledger.get("stages") or ():
        status = str(stage.get("status") or "")
        if not dependency_status_admissible(ledger, status):
            continue
        for output in stage.get("outputs") or ():
            if str(output.get("schema") or "") != schema:
                continue
            path = _resolve_output_path(run_root, str(output.get("path") or ""))
            expected = str(output.get("sha256") or "")
            if len(expected) != 64 or sha256_file(path) != expected:
                raise QualificationError(f"INVESTOR_DEMO_AUTHORITY_BYTES_DRIFT:{schema}")
            matches.append((stage, output, path))
    if not matches:
        raise QualificationError(f"INVESTOR_DEMO_AUTHORITY_SCHEMA_MISSING:{schema}")
    unique = {str(output["sha256"]) for _, output, _ in matches}
    if len(unique) != 1:
        rows = [(str(stage.get("id")), str(output.get("sha256"))) for stage, output, _ in matches]
        raise QualificationError(
            f"INVESTOR_DEMO_AUTHORITY_SCHEMA_AMBIGUOUS:{schema}:{rows}"
        )
    return matches[-1]


def _load_authorities(run_root: Path):
    g_stage, g_ref, g_path = _schema_output(run_root, SCHEMA_G)
    m_stage, m_ref, m_path = _schema_output(run_root, SCHEMA_M)
    w_stage, w_ref, w_path = _schema_output(run_root, SCHEMA_W)
    skeleton = qualified_skeleton_from_dict(json.loads(g_path.read_text(encoding="utf-8")))
    mesh = qualified_mesh_from_dict(json.loads(m_path.read_text(encoding="utf-8")))
    mesh_skin = qualified_mesh_skin_from_dict(json.loads(w_path.read_text(encoding="utf-8")))
    refs = {
        "G": {"stage_id": g_stage["id"], "schema": SCHEMA_G, "sha256": g_ref["sha256"]},
        "M": {"stage_id": m_stage["id"], "schema": SCHEMA_M, "sha256": m_ref["sha256"]},
        "W": {"stage_id": w_stage["id"], "schema": SCHEMA_W, "sha256": w_ref["sha256"]},
    }
    return skeleton, mesh, mesh_skin, refs


def _mesh_arrays(mesh):
    vertex_ids = [str(vertex.canonical_mesh_vertex_id) for vertex in mesh.vertices]
    if len(vertex_ids) != len(set(vertex_ids)):
        raise QualificationError("INVESTOR_DEMO_DUPLICATE_M_VERTEX")
    index = {vertex_id: i for i, vertex_id in enumerate(vertex_ids)}
    rest = np.asarray([vertex.P for vertex in mesh.vertices], dtype=np.float64)
    faces = np.asarray([[index[str(v)] for v in face] for face in mesh.faces], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise QualificationError("INVESTOR_DEMO_NON_TRIANGULAR_M")
    edges = set()
    for a, b, c in faces.tolist():
        for x, y in ((a, b), (b, c), (c, a)):
            edges.add((min(x, y), max(x, y)))
    return vertex_ids, rest, faces, sorted(edges)


def _assert_same_render_m(rest: np.ndarray, faces: np.ndarray, baseline_arrays: dict[str, np.ndarray]) -> None:
    baseline_rest = np.asarray(baseline_arrays["rest"], dtype=np.float64)
    baseline_faces = np.asarray(baseline_arrays["faces"], dtype=np.int32)
    if rest.shape != baseline_rest.shape or not np.array_equal(rest, baseline_rest):
        raise QualificationError("INVESTOR_DEMO_M_REST_NOT_RENDER_BASELINE")
    if faces.shape != baseline_faces.shape or not np.array_equal(faces, baseline_faces):
        raise QualificationError("INVESTOR_DEMO_M_TOPOLOGY_NOT_RENDER_BASELINE")


def _weights_payload(mesh_skin, vertex_ids: list[str], skeleton):
    joint_ids = [str(j.canonical_joint_id) for j in skeleton.joints]
    joint_index = {joint_id: i for i, joint_id in enumerate(joint_ids)}
    rows = {str(row.canonical_mesh_vertex_id): row for row in mesh_skin.rows}
    if set(rows) != set(vertex_ids):
        raise QualificationError("INVESTOR_DEMO_W_VERTEX_ACCOUNTING_DRIFT")
    sparse: list[list[list[float | int]]] = []
    for vertex_id in vertex_ids:
        values: list[list[float | int]] = []
        total = 0.0
        for joint_id, weight in rows[vertex_id].influences:
            joint_id = str(joint_id)
            if joint_id not in joint_index:
                raise QualificationError("INVESTOR_DEMO_W_UNKNOWN_JOINT")
            w = float(weight)
            if not np.isfinite(w) or w < -1.0e-10:
                raise QualificationError("INVESTOR_DEMO_W_INVALID")
            if w > 0.0:
                values.append([joint_index[joint_id], w])
            total += w
        if abs(total - 1.0) > 1.0e-8:
            raise QualificationError("INVESTOR_DEMO_W_SIMPLEX_DRIFT")
        sparse.append(values)
    return joint_ids, sparse


def _skeleton_payload(skeleton):
    ids = [str(j.canonical_joint_id) for j in skeleton.joints]
    index = {joint_id: i for i, joint_id in enumerate(ids)}
    joints = []
    for joint in skeleton.joints:
        parent = None if joint.parent_canonical_id is None else index[str(joint.parent_canonical_id)]
        joints.append(
            {
                "id": str(joint.canonical_joint_id),
                "parent_index": parent,
                "rest": list(map(float, joint.position)),
            }
        )
    return joints


def _camera_payload(cameras):
    return [
        {
            "view_id": str(camera.view_id),
            "view_index": int(camera.view_index),
            "origin": list(map(float, camera.origin)),
            "right": list(map(float, camera.right)),
            "screen_up": list(map(float, camera.screen_up)),
            "forward": list(map(float, camera.forward)),
            "half_extent": float(camera.half_extent),
        }
        for camera in cameras
    ]


def _copy_web_shell(out: Path) -> None:
    for name in ("START_DEMO.html", "styles.css", "app.js"):
        source = WEB_ROOT / name
        if not source.is_file():
            raise QualificationError(f"INVESTOR_DEMO_WEB_TEMPLATE_MISSING:{name}")
        shutil.copy2(source, out / name)


def _write_rest_appearance(out: Path, material: dict[str, Any], rest: np.ndarray, resolution: int):
    target = out / "appearance"
    target.mkdir(parents=True, exist_ok=True)
    paths = {}
    for view in range(8):
        result = render_single_m_caa(
            **{
                key: material[key]
                for key in ("rest", "faces", "source", "maps", "cameras", "caa")
            },
            posed=rest,
            view=view,
            resolution=resolution,
        )
        path = target / f"rest_v{view}.png"
        Image.fromarray(result.straight_rgba_u8).save(path)
        paths[str(view)] = str(path.relative_to(out)).replace("\\", "/")
    return paths


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build(*, run_root: Path, baseline_path: Path, out: Path, preview_resolution: int) -> dict[str, Any]:
    if preview_resolution < 256 or preview_resolution > 1024:
        raise QualificationError("INVESTOR_DEMO_PREVIEW_RESOLUTION_OUT_OF_RANGE")
    run_root = run_root.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    (out / "authority").mkdir(exist_ok=True)

    baseline_manifest, baseline_arrays, source, cameras, caa_arrays = load_inputs(baseline_path)
    skeleton, mesh, mesh_skin, authority_refs = _load_authorities(run_root)
    vertex_ids, rest, faces, edges = _mesh_arrays(mesh)
    _assert_same_render_m(rest, faces, baseline_arrays)
    joint_ids, sparse_weights = _weights_payload(mesh_skin, vertex_ids, skeleton)

    material_ref = write_material_bundle(
        out / "authority/single_m_source_wins_caa_material.npz",
        rest=rest,
        faces=faces,
        source=source,
        cameras=cameras,
        caa_arrays=caa_arrays,
        input_identities=baseline_manifest["inputs"],
    )
    material = load_material_bundle(
        material_ref["path"],
        expected_sha256=material_ref["sha256"],
        rest=rest,
        faces=faces,
    )
    rest_frames = _write_rest_appearance(out, material, rest, preview_resolution)
    shutil.copy2(baseline_path, out / "authority/render_baseline.json")

    data = {
        "schema": "RealSaS.KnightInvestorInteractiveDemo.v1",
        "title": "RealSaS — KNIGHT",
        "subtitle": "Compiled Character Structure",
        "source_sha": _git_head(),
        "authority": {
            "geometry": "CANONICAL_M",
            "rig": "QUALIFIED_G",
            "weights": "QUALIFIED_W_ON_M",
            "appearance": "CANONICAL_M_SOURCE_WINS_PLUS_SEALED_CAA_V1",
            "refs": authority_refs,
            "render_line": {
                "renderer": "tools/demo/render_knight_m_caa_v1.py",
                "core": "compiler/realsas_compiler_core/single_m_caa_v1.py",
                "baseline": "tests/fixtures/single_m_caa_knight_v1/baseline.json",
                "baseline_manifest_sha256": sha256_file(baseline_path),
                "baseline_pose_sha256": baseline_manifest["pose_sha256"],
                "compiled_caa_sha256": baseline_manifest["inputs"]["compiled_caa.npz"],
                "source_rgba_sha256": baseline_manifest["inputs"]["source_rgba.zip"],
                "material_sha256": material_ref["sha256"],
            },
        },
        "cameras": _camera_payload(cameras),
        "mesh": {
            "vertex_ids": vertex_ids,
            "rest_xyz": rest.tolist(),
            "faces": faces.tolist(),
            "edges": [list(edge) for edge in edges],
            "rest_array_sha256": array_hash(rest),
            "face_array_sha256": array_hash(faces),
        },
        "skeleton": {
            "joint_ids": joint_ids,
            "root_id": str(skeleton.root_id),
            "joints": _skeleton_payload(skeleton),
        },
        "weights": {
            "representation": "SPARSE_PER_VERTEX_CANONICAL_M_WEIGHTS",
            "influences": sparse_weights,
        },
        "appearance": {
            "rest_frames": rest_frames,
            "preview_resolution": preview_resolution,
            "material_path": "authority/single_m_source_wins_caa_material.npz",
        },
        "motion": {
            "status": "PENDING_EXACT_USER_SELECTION",
            "contract": "CLIP_FRAMES_SHARE_THE_SAME_M_G_W_CAA_AND_CANONICAL_TIME_INDEX",
            "clips": [],
        },
        "presentation": {
            "bone_glyph": "TAPERED_EDITOR_BONE_V1",
            "topology": "CANONICAL_TRIANGLE_WIREFRAME_V1",
            "weight_field": "LINEAR_PER_VERTEX_INTERPOLATION_ON_POSED_M_V1",
            "debug_claimed": False,
            "product_qualification_claimed": False,
        },
    }
    data_json = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    data_script = "window.REALSAS_DEMO_DATA=" + data_json + ";\n"
    (out / "data/demo-data.js").write_text(data_script, encoding="utf-8")
    _copy_web_shell(out)

    manifest = {
        "schema": "RealSaS.KnightInvestorDemoPackageManifest.v1",
        "source_sha": data["source_sha"],
        "demo_data_sha256": _sha256_text(data_script),
        "authority": data["authority"],
        "motion_status": data["motion"]["status"],
        "entrypoint": "START_DEMO.html",
        "offline": True,
    }
    (out / "DEMO_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "README.txt").write_text(
        "RealSaS KNIGHT Interactive Investor Demo\n\n"
        "Open START_DEMO.html in a modern desktop browser.\n"
        "This package is an offline, read-only presentation consumer.\n"
        "M/G/W/CAA are sealed here; motion clips are intentionally pending exact selection.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Exact sealed Knight run root containing ACTIVE_RUN_V2.json and canonical G/M/W outputs.",
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-resolution", type=int, default=512)
    args = parser.parse_args()
    manifest = build(
        run_root=args.run_root,
        baseline_path=args.baseline.resolve(),
        out=args.output,
        preview_resolution=args.preview_resolution,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
