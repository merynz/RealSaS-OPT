from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

Json = dict[str, Any]
VIEW_LABELS = ("S", "SE", "E", "NE", "N", "NW", "W", "SW")
PRODUCT_SCHEMA = "RealSaS.CanonicalPuppetGraph.v3"
PROOF_SCHEMA = "RealSaS.ProductProofBundleIR.v1"
DIRECTIONAL_BINDING_SCHEMA = "RealSaS.DirectionalJointViewBindingSetIR.v1"
MOTION_BAKE_SCHEMA = "RealSaS.QualificationOwnedMotionBakeIR.v1"
USER_LAYER_SCHEMA = "RealSaS.UserPuppetEditLayer.v3"


class LivingCompileError(RuntimeError):
    pass


def json_load(path: Path) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LivingCompileError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LivingCompileError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LivingCompileError(f"expected JSON object: {path}")
    return value


def json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def finite(value: Any, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise LivingCompileError(f"{field} must be numeric") from exc
    if not math.isfinite(out):
        raise LivingCompileError(f"{field} must be finite")
    return out


def finite_pair(value: Any, *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise LivingCompileError(f"{field} must be a two-value sequence")
    return finite(value[0], field=f"{field}[0]"), finite(value[1], field=f"{field}[1]")


def safe_id(value: Any, *, field: str, max_len: int = 128) -> str:
    text = str(value or "").strip()
    if not text or len(text) > max_len or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", text):
        raise LivingCompileError(f"invalid {field}")
    return text


def view_id(view_index: int) -> str:
    return f"V{int(view_index)}"


def parse_view(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text.startswith("V") and text[1:].isdigit():
        idx = int(text[1:])
    elif text.isdigit():
        idx = int(text)
    elif text in VIEW_LABELS:
        idx = VIEW_LABELS.index(text)
    else:
        raise LivingCompileError(f"unknown view: {value}")
    if idx < 0 or idx > 7:
        raise LivingCompileError(f"view outside 0..7: {value}")
    return idx


def bundle_path(value: Any) -> Path:
    if value is None:
        raise LivingCompileError("bundle path is required")
    root = Path(str(value)).expanduser().resolve()
    if not root.is_dir():
        raise LivingCompileError(f"bundle directory does not exist: {root}")
    return root


def within(root: Path, relative: str) -> Path:
    rel = Path(str(relative).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise LivingCompileError("unsafe relative path")
    path = (root / rel).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise LivingCompileError("path escapes bundle root") from exc
    return path


def _product_path(root: Path) -> Path:
    for path in (root / "puppet" / "canonical_puppet_graph_v3.json", root / "canonical_puppet_graph_v3.json"):
        if path.is_file():
            return path
    raise LivingCompileError("V4 canonical product not found (puppet/canonical_puppet_graph_v3.json)")


def _proof_path(root: Path) -> Path:
    for path in (root / "proof" / "product_proof_bundle_ir.json", root / "product_proof_bundle_ir.json"):
        if path.is_file():
            return path
    raise LivingCompileError("V4 product proof not found (proof/product_proof_bundle_ir.json)")


def _directional_binding_path(root: Path) -> Path | None:
    for path in (
        root / "renderables" / "directional_joint_view_binding_set_ir.json",
        root / "directional_joint_view_binding_set_ir.json",
    ):
        if path.is_file():
            return path
    return None


def load_product(root: Path) -> Json:
    product = json_load(_product_path(root))
    if product.get("schema_version") != PRODUCT_SCHEMA:
        raise LivingCompileError(f"unsupported product schema: {product.get('schema_version')}")
    if product.get("representation_class") != "DIRECTIONAL_2D_2P5D_PUPPET":
        raise LivingCompileError("Living Compile requires the directional 2D/2.5D product representation")
    if bool(product.get("full_3d_reconstruction_authority", False)):
        raise LivingCompileError("full 3D reconstruction authority is forbidden")
    return product


def load_proof(root: Path, product: Mapping[str, Any]) -> Json:
    proof = json_load(_proof_path(root))
    if proof.get("schema_version") != PROOF_SCHEMA:
        raise LivingCompileError(f"unsupported proof schema: {proof.get('schema_version')}")
    if proof.get("source_product_state_hash") != product.get("product_state_hash"):
        raise LivingCompileError("proof/product lineage mismatch")
    return proof


def load_directional_binding(root: Path, product: Mapping[str, Any], *, required: bool = False) -> Json | None:
    path = _directional_binding_path(root)
    if path is None:
        if required:
            raise LivingCompileError("qualified directional joint/view binding not found")
        return None
    binding = json_load(path)
    if binding.get("schema_version") != DIRECTIONAL_BINDING_SCHEMA:
        raise LivingCompileError(f"unsupported directional binding schema: {binding.get('schema_version')}")
    mechanical = product.get("mechanical_state") or {}
    surface = mechanical.get("surface") or {}
    skeleton = mechanical.get("skeleton") or {}
    visuals = product.get("directional_renderables") or {}
    expected = {
        "source_product_state_hash": product.get("product_state_hash"),
        "surface_lineage_hash": surface.get("geometry_lineage_hash"),
        "skeleton_lineage_hash": skeleton.get("skeleton_lineage_hash"),
        "directional_visual_state_hash": visuals.get("directional_visual_state_hash"),
    }
    for field, value in expected.items():
        if binding.get(field) != value:
            raise LivingCompileError(f"directional binding lineage mismatch: {field}")
    directions = {int(row.get("view_index", -1)): row for row in list(visuals.get("directions") or [])}
    projections = list(binding.get("projections") or [])
    if len(projections) != 8 or set(directions) != set(range(8)):
        raise LivingCompileError("directional binding requires exact views 0..7")
    projection_hashes: dict[int, str] = {}
    for row in projections:
        view = int(row.get("view_index", -1))
        if view not in directions or view in projection_hashes:
            raise LivingCompileError("directional binding projection set is invalid")
        if row.get("camera_binding_hash") != directions[view].get("camera_binding_hash"):
            raise LivingCompileError(f"directional binding camera mismatch: V{view}")
        ph = str(row.get("projection_binding_hash") or "")
        if not ph:
            raise LivingCompileError(f"directional binding projection hash missing: V{view}")
        projection_hashes[view] = ph
    joint_ids = {str(j.get("canonical_joint_id") or "") for j in list(skeleton.get("joints") or [])}
    pivots: set[tuple[int, str]] = set()
    for row in list(binding.get("joint_pivots") or []):
        view = int(row.get("view_index", -1))
        jid = str(row.get("canonical_joint_id") or "")
        key = (view, jid)
        if view not in projection_hashes or jid not in joint_ids or key in pivots:
            raise LivingCompileError("directional binding pivot set is invalid")
        finite_pair(row.get("raster_xy"), field=f"directional pivot V{view}:{jid}")
        if row.get("projection_binding_hash") != projection_hashes[view]:
            raise LivingCompileError(f"directional pivot projection mismatch: V{view}:{jid}")
        pivots.add(key)
    expected_pivots = {(view, jid) for view in range(8) for jid in joint_ids}
    if pivots != expected_pivots:
        raise LivingCompileError("directional binding pivot set is incomplete")
    if not str(binding.get("binding_set_hash") or ""):
        raise LivingCompileError("directional binding set hash missing")
    return binding


def surface_raster(product: Mapping[str, Any], view_index: int) -> dict[str, tuple[float, float]]:
    surface = ((product.get("mechanical_state") or {}).get("surface") or {})
    out: dict[str, tuple[float, float]] = {}
    for node in list(surface.get("surface_nodes") or []):
        sid = str(node.get("surface_id") or "")
        rows = []
        for raw_view, raw_xy in list(node.get("raster_bindings") or []):
            if int(raw_view) == int(view_index):
                rows.append(finite_pair(raw_xy, field=f"surface raster {sid}"))
        if len(rows) > 1:
            raise LivingCompileError(f"duplicate raster binding for {sid} view {view_index}")
        if rows:
            out[sid] = rows[0]
    return out


def weighted_xy(coeffs: Iterable[Iterable[Any]], raster: Mapping[str, tuple[float, float]], *, field: str) -> tuple[float, float]:
    x = y = total = 0.0
    for pair in coeffs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise LivingCompileError(f"invalid {field} support coefficient")
        sid = str(pair[0])
        if sid not in raster:
            raise LivingCompileError(f"{field} references surface without raster binding: {sid}")
        c = finite(pair[1], field=f"{field}:{sid}")
        if c < -1e-9:
            raise LivingCompileError(f"{field} has negative support coefficient")
        x += c * raster[sid][0]
        y += c * raster[sid][1]
        total += c
    if abs(total - 1.0) > 1e-6:
        raise LivingCompileError(f"{field} support coefficients must sum to one")
    return x, y
