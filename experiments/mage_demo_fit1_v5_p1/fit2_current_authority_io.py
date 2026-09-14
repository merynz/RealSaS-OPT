from __future__ import annotations

"""Exact loaders for the corrected Mage FIT2 mechanical authority.

This module intentionally does not reuse the FIT1 static loader because that loader
hard-pins the historical 950-node surface/skeleton/skin lineage.  The FIT2 product
closure must bind the current 8171-node GSA, fresh Geppetto skeleton, and fresh
Arachne QualifiedSkinIR as one mechanically consistent state.
"""

import hashlib
import json
from pathlib import Path

from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)
from compiler.realsas_compiler_core.v4 import build_mechanical_state
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2


EXPECTED_SURFACE_LINEAGE = "65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da"
EXPECTED_TENSORIZATION_HASH = "fe351362e195164805cebb0f63b74d1861ef123458b20ac56607016e00a6c67e"
EXPECTED_SURFACE_NODES = 8171
EXPECTED_SURFACE_RELATIONS = 23656
EXPECTED_SKELETON_LINEAGE = "69f05e4fdef65f2cd86fed66503911210e1dbc7212ad017d69bf3dcb7b0896a1"
EXPECTED_SKELETON_SHA256 = "319ae46d94450d97dd6f40f21d571ac9432968104e7a517af00ace442d05b9fe"
EXPECTED_SKIN_LINEAGE = "eb96b398282e4e25cd6df662e7a02e8ff1afe881818127190979bddd2f6c4006"
EXPECTED_SKIN_SHA256 = "722fdeb60fc32aa09e2296adbdc0e589de0379bfd49f6239f214330517f01ddd"
EXPECTED_MODEL_SHA256 = "8749843ee673f5c4b84b5d58a6285bd108e9867af5d87bd4afcf2a5e64d98353"
EXPECTED_CANONICAL_WEIGHTS_SHA256 = "c5bdb807f28b87cd472e808b6dc4594ef881db4418fdd873887a8bbcc0fa14ab"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object:{path}")
    return value


def load_surface(path: Path) -> RiggingSurfaceIR:
    raw = _load_json(path)
    nodes = tuple(
        SurfaceNode(
            surface_id=str(row["surface_id"]),
            P=tuple(map(float, row["P"])),
            support_views=tuple(map(int, row.get("support_views") or ())),
            provenance_refs=tuple(map(str, row.get("provenance_refs") or ())),
            source_observation_ids=tuple(map(str, row.get("source_observation_ids") or ())),
            raster_bindings=tuple((int(view), tuple(map(float, xy))) for view, xy in row.get("raster_bindings") or ()),
            persistence_group_id=row.get("persistence_group_id"),
            derived_normal=None if row.get("derived_normal") is None else tuple(map(float, row["derived_normal"])),
            validity_flags=tuple(map(str, row.get("validity_flags") or ())),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in raw.get("surface_nodes") or ()
    )
    relations = tuple(
        SurfaceRelation(
            relation_id=str(row["relation_id"]),
            a_surface_id=str(row["a_surface_id"]),
            b_surface_id=str(row["b_surface_id"]),
            relation_kind=str(row["relation_kind"]),
            score=float(row.get("score", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in raw.get("local_relations") or ()
    )
    value = RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash=str(raw["geometry_lineage_hash"]),
        builder_id=str(raw.get("builder_id") or "RealSaS.GeometricSubstrateAssembler.current"),
        schema_version=str(raw.get("schema_version") or "RealSaS.RiggingSurfaceIR.v1"),
        metadata=dict(raw.get("metadata") or {}),
    )
    if value.geometry_lineage_hash != EXPECTED_SURFACE_LINEAGE:
        raise ValueError("MAGE_FIT2_SURFACE_LINEAGE_DRIFT")
    if len(value.surface_nodes) != EXPECTED_SURFACE_NODES or len(value.local_relations) != EXPECTED_SURFACE_RELATIONS:
        raise ValueError("MAGE_FIT2_SURFACE_CARDINALITY_DRIFT")
    return value


def load_skeleton(path: Path) -> QualifiedSkeletonIRV2:
    if sha256_file(path) != EXPECTED_SKELETON_SHA256:
        raise ValueError("MAGE_FIT2_SKELETON_FILE_SHA_DRIFT")
    raw = _load_json(path)
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=str(row["canonical_joint_id"]),
            position=tuple(map(float, row["position"])),
            parent_canonical_id=None if row.get("parent_canonical_id") is None else str(row["parent_canonical_id"]),
            support_surface_ids=tuple(map(str, row.get("support_surface_ids") or ())),
            source_proposal_id=str(row.get("source_proposal_id") or ""),
        )
        for row in raw.get("joints") or ()
    )
    value = QualifiedSkeletonIRV2(
        joints=joints,
        deform_root_ids=tuple(map(str, raw.get("deform_root_ids") or ())),
        assembly_root_binding=dict(raw.get("assembly_root_binding") or {}),
        qualification_report=dict(raw.get("qualification_report") or {}),
        skeleton_lineage_hash=str(raw["skeleton_lineage_hash"]),
        schema_version=str(raw.get("schema_version") or "RealSaS.QualifiedSkeletonIR.v2"),
    )
    if value.skeleton_lineage_hash != EXPECTED_SKELETON_LINEAGE or len(value.joints) != 22:
        raise ValueError("MAGE_FIT2_SKELETON_AUTHORITY_DRIFT")
    return value


def load_skin(path: Path) -> QualifiedSkinIR:
    if sha256_file(path) != EXPECTED_SKIN_SHA256:
        raise ValueError("MAGE_FIT2_SKIN_FILE_SHA_DRIFT")
    raw = _load_json(path)
    rows = tuple(
        QualifiedSkinRow(
            surface_id=str(row["surface_id"]),
            influences=tuple((str(jid), float(weight)) for jid, weight in row.get("influences") or ()),
            simplex_residual_before=float(row.get("simplex_residual_before", 0.0)),
            correction_l1=float(row.get("correction_l1", 0.0)),
        )
        for row in raw.get("rows") or ()
    )
    value = QualifiedSkinIR(
        rows=rows,
        surface_binding_hash=str(raw["surface_binding_hash"]),
        skeleton_binding_hash=str(raw["skeleton_binding_hash"]),
        qualification_report=dict(raw.get("qualification_report") or {}),
        skin_lineage_hash=str(raw["skin_lineage_hash"]),
        schema_version=str(raw.get("schema_version") or "RealSaS.QualifiedSkinIR.v1"),
    )
    if value.skin_lineage_hash != EXPECTED_SKIN_LINEAGE or len(value.rows) != EXPECTED_SURFACE_NODES:
        raise ValueError("MAGE_FIT2_SKIN_AUTHORITY_DRIFT")
    if value.surface_binding_hash != EXPECTED_SURFACE_LINEAGE:
        raise ValueError("MAGE_FIT2_SKIN_SURFACE_BINDING_DRIFT")
    if value.skeleton_binding_hash != EXPECTED_SKELETON_LINEAGE:
        raise ValueError("MAGE_FIT2_SKIN_SKELETON_BINDING_DRIFT")
    return value


def build_exact_mechanical(surface_path: Path, skeleton_path: Path, skin_path: Path):
    surface = load_surface(surface_path)
    skeleton = load_skeleton(skeleton_path)
    skin = load_skin(skin_path)
    mechanical = build_mechanical_state(surface, skeleton, skin)
    return surface, skeleton, skin, mechanical
