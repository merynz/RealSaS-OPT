from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import rigging_surface_from_dict
from compiler.realsas_compiler_core.rig import qualify_skeleton
from compiler.realsas_compiler_core.types import (
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
)
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.arachne_candidate_v4 import ArachneA1ConfigV4, ArachneA1V4


SCHEMA = "RealSaS.MIRAV6Teacher23QueryPreflight.v1"
EXPECTED_SURFACE_LINEAGE = "452cf6564a2cde35a4d6d420021491d231f90cec267df9f41d4424dc8d74895a"
EXPECTED_MIRA_MODEL_SHA256 = "bfd2bd24f55bdce2dc36b0ae6a7afe01b8381c680de545c66a11a3d707fe73ed"
EXPECTED_RUNNER_SHA256 = "39ff4f7a2051f0d9c1251a5999f472b9926e83deacbedcedc8e92eea424dce01"
EXPECTED_MIRA_SCHEMA = "RealSaS.KnightArachneV6DemoClosure.v1.Model.v1"
EXPECTED_MIRA_ARCH = "RealSaS.Arachne.A1.RawSurfaceK4AttentionDirectSimplex.v6"
EXPECTED_BACKBONE_ARCH = "RealSaS.Arachne.A1.RichQualifiedBidirectional.v4"
EXPECTED_BACKBONE_PARAMS = 138_053_153
EXPECTED_DECODER_PARAMS = 1_619_491
EXPECTED_RAW_BONES = 41
EXPECTED_TEACHER_VERTS = 3665
EXPECTED_TEACHER_FACES = 6952
EXPECTED_TEST_JOINTS = 23
SUPPORT_K = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def import_exact_runner(path: Path):
    spec = importlib.util.spec_from_file_location("realsas_exact_mira_v6_closure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("MIRA_V6_EXACT_RUNNER_IMPORT_SPEC_FAIL")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher23_indices(skin: np.ndarray, parents: np.ndarray) -> tuple[int, ...]:
    if skin.shape != (EXPECTED_TEACHER_VERTS, EXPECTED_RAW_BONES):
        raise RuntimeError(f"TEACHER_SKIN_SHAPE_DRIFT::{skin.shape}")
    mass = np.asarray(skin, dtype=np.float64).sum(axis=0)
    positive = tuple(int(i) for i in np.flatnonzero(mass > 0.0))
    expected_positive = tuple(range(1, 23))
    if positive != expected_positive:
        raise RuntimeError(f"TEACHER_POSITIVE_SKIN_COLUMNS_DRIFT::{positive}")
    selected = (0,) + positive
    if len(selected) != EXPECTED_TEST_JOINTS:
        raise RuntimeError("TEACHER23_COUNT_DRIFT")
    selected_set = set(selected)
    for idx in selected:
        p = int(parents[idx])
        if p >= 0 and p not in selected_set:
            raise RuntimeError(f"TEACHER23_PARENT_ESCAPES_SELECTION::{idx}->{p}")
    if float(mass[0]) != 0.0:
        raise RuntimeError("TEACHER_ROOT0_EXPECTED_ZERO_SKIN_MASS")
    return selected


def nearest_support_ids(surface, joint_position: np.ndarray, k: int = SUPPORT_K) -> tuple[str, ...]:
    nodes = tuple(surface.surface_nodes)
    if len(nodes) < k:
        raise RuntimeError("GSA_TOO_SMALL_FOR_SUPPORT_BINDING")
    rows = []
    jp = np.asarray(joint_position, dtype=np.float64)
    for node in nodes:
        p = np.asarray(node.P, dtype=np.float64)
        d2 = float(np.dot(p - jp, p - jp))
        rows.append((d2, str(node.surface_id)))
    rows.sort(key=lambda x: (x[0], x[1]))
    return tuple(sid for _, sid in rows[:k])


def build_teacher23_proposal(surface, teacher_npz: Path) -> tuple[SkeletonProposalIR, dict]:
    with np.load(teacher_npz, allow_pickle=False) as z:
        required = {
            "vertices_source", "faces", "bone_heads_source", "bone_tails_source",
            "parents", "rest_local_source", "rest_world_source", "skin",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise RuntimeError(f"TEACHER_NPZ_ARRAYS_MISSING::{missing}")
        vertices = np.asarray(z["vertices_source"], dtype=np.float32)
        faces = np.asarray(z["faces"], dtype=np.int32)
        heads = np.asarray(z["bone_heads_source"], dtype=np.float32)
        parents = np.asarray(z["parents"], dtype=np.int64)
        skin = np.asarray(z["skin"], dtype=np.float32)
    if vertices.shape != (EXPECTED_TEACHER_VERTS, 3) or faces.shape != (EXPECTED_TEACHER_FACES, 3):
        raise RuntimeError(f"TEACHER_MESH_SHAPE_DRIFT::{vertices.shape}::{faces.shape}")
    if heads.shape != (EXPECTED_RAW_BONES, 3) or parents.shape != (EXPECTED_RAW_BONES,):
        raise RuntimeError(f"TEACHER_RIG_SHAPE_DRIFT::{heads.shape}::{parents.shape}")

    selected = teacher23_indices(skin, parents)
    source_to_local = {src: local for local, src in enumerate(selected)}
    proposal_ids = {src: f"P:RAW41:{src:02d}" for src in selected}
    joints = []
    support_rows = []
    for src in selected:
        support_ids = nearest_support_ids(surface, heads[src])
        support_rows.append({
            "raw_bone_index": int(src),
            "support_surface_ids": list(support_ids),
        })
        joints.append(SkeletonProposalJoint(
            proposal_id=proposal_ids[src],
            position=tuple(map(float, heads[src].tolist())),
            root_score=1.0 if src == 0 else 0.0,
            confidence=1.0,
            support_surface_ids=support_ids,
            metadata={
                "raw_teacher_bone_index": int(src),
                "raw41_source_truth_retained": True,
                "teacher_skin_is_not_support_binding_input": True,
                "support_binding_policy": "GSA_NEAREST8_TO_RAW_TEACHER_BONE_HEAD_V1",
                "fbx_object_packaging_authority": False,
            },
        ))

    edges = []
    for src in selected:
        parent = int(parents[src])
        if parent < 0:
            continue
        if parent not in source_to_local:
            raise RuntimeError(f"TEACHER23_PARENT_OUTSIDE::{src}->{parent}")
        edges.append(SkeletonProposalEdge(
            edge_id=f"E:{proposal_ids[parent]}->{proposal_ids[src]}",
            parent_proposal_id=proposal_ids[parent],
            child_proposal_id=proposal_ids[src],
            score=1.0,
            confidence=1.0,
            hard_required=True,
            hard_forbidden=False,
            reason="RAW41_TEACHER23_EXACT_PARENT_FOR_SUBSTITUTION_COURT",
            metadata={
                "test_authority_only": True,
                "product_authority": False,
            },
        ))

    proposal = SkeletonProposalIR(
        joints=tuple(joints),
        edges=tuple(edges),
        surface_binding_hash=str(surface.geometry_lineage_hash),
        model_provenance="RAW41_TEACHER23_TEST_AUTHORITY_V1",
        metadata={
            "scope": "RESEARCH_SUBSTITUTION_COURT_FIXED_TEACHER_RIG",
            "raw_teacher_bone_count": EXPECTED_RAW_BONES,
            "selected_raw_bone_indices": list(selected),
            "selection_rule": "ROOT0_PLUS_SKIN_POSITIVE_RAW_BONES_1_TO_22",
            "teacher_skin_role": "TEST_RIG_MEMBERSHIP_ONLY__NEVER_MIRA_PREDICTOR_FEATURE",
            "support_binding_policy": "GSA_NEAREST8_TO_RAW_TEACHER_BONE_HEAD_V1",
            "support_binding_uses_teacher_skin": False,
            "fbx_object_packaging_authority": False,
            "equipment_semantics": "GEOMETRY_COMPONENT_PLUS_QUALIFIED_JOINT_SUPPORT",
            "product_authority_claimed": False,
            "generalization_claimed": False,
        },
    )
    binding = {
        "selected_raw_bone_indices": list(selected),
        "raw_parent_indices": [int(parents[i]) for i in selected],
        "support_rows": support_rows,
        "support_binding_hash": canonical_json_hash(support_rows),
    }
    return proposal, binding


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-json", type=Path, required=True)
    ap.add_argument("--teacher-npz", type=Path, required=True)
    ap.add_argument("--mira-checkpoint", type=Path, required=True)
    ap.add_argument("--exact-v6-runner", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    surface_path = args.surface_json.resolve()
    teacher_path = args.teacher_npz.resolve()
    checkpoint_path = args.mira_checkpoint.resolve()
    runner_path = args.exact_v6_runner.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("MIRA23_PREFLIGHT_PHASE=VERIFY_INPUT_BYTES", flush=True)
    model_sha = sha256(checkpoint_path)
    runner_sha = sha256(runner_path)
    if model_sha != EXPECTED_MIRA_MODEL_SHA256:
        raise RuntimeError(f"MIRA_MODEL_SHA_DRIFT::{model_sha}")
    if runner_sha != EXPECTED_RUNNER_SHA256:
        raise RuntimeError(f"MIRA_RUNNER_SHA_DRIFT::{runner_sha}")

    surface = rigging_surface_from_dict(load_json(surface_path))
    if str(surface.geometry_lineage_hash) != EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError(f"SURFACE_LINEAGE_DRIFT::{surface.geometry_lineage_hash}")
    print(f"MIRA23_PREFLIGHT_SURFACE nodes={len(surface.surface_nodes)} edges={len(surface.local_relations)}", flush=True)

    print("MIRA23_PREFLIGHT_PHASE=MATERIALIZE_AND_QUALIFY_TEACHER23", flush=True)
    proposal, binding = build_teacher23_proposal(surface, teacher_path)
    write_json(out_dir / "TEACHER23_SKELETON_PROPOSAL.json", proposal.to_dict())
    qualified = qualify_skeleton(
        surface,
        proposal,
        run_ilp_shadow=False,
        authority_bindings={
            "court_scope": "MIRA_ONLY_FIXED_TEACHER23_RIG",
            "support_policy": "GSA_NEAREST8_TO_RAW_TEACHER_BONE_HEAD_V1",
        },
    )
    if len(qualified.joints) != EXPECTED_TEST_JOINTS:
        raise RuntimeError(f"QUALIFIED_TEACHER23_COUNT_DRIFT::{len(qualified.joints)}")
    roots = [j for j in qualified.joints if j.parent_canonical_id is None]
    if len(roots) != 1:
        raise RuntimeError(f"QUALIFIED_TEACHER23_ROOT_COUNT_DRIFT::{len(roots)}")
    if any(len(j.support_surface_ids) != SUPPORT_K for j in qualified.joints):
        raise RuntimeError("QUALIFIED_TEACHER23_SUPPORT_CARDINALITY_DRIFT")
    write_json(out_dir / "TEACHER23_QUALIFIED_SKELETON_IR.json", qualified.to_dict())
    print(f"MIRA23_PREFLIGHT_TEACHER23_QUALIFIED lineage={qualified.skeleton_lineage_hash}", flush=True)

    print("MIRA23_PREFLIGHT_PHASE=BUILD_V3_CONDITIONING", flush=True)
    conditioning = ArachneRichConditioningAdapterV3(require_scene_first=True)([surface], [qualified])
    n = int(conditioning.surface_mask[0].sum())
    e = int(conditioning.edge_mask[0].sum())
    j = int(conditioning.joint_mask[0].sum())
    if j != EXPECTED_TEST_JOINTS:
        raise RuntimeError(f"MIRA23_CONDITIONING_JOINT_COUNT_DRIFT::{j}")
    if tuple(conditioning.pair_geometry.shape[-2:]) != (EXPECTED_TEST_JOINTS, 10):
        raise RuntimeError(f"MIRA23_PAIR_GEOMETRY_SHAPE_DRIFT::{conditioning.pair_geometry.shape}")
    print(f"MIRA23_PREFLIGHT_CONDITIONING_PASS N={n} E={e} J={j}", flush=True)

    print("MIRA23_PREFLIGHT_PHASE=STRICT_LOAD_EXACT_V6", flush=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError("MIRA_V6_CHECKPOINT_NOT_DICT")
    if ckpt.get("schema") != EXPECTED_MIRA_SCHEMA or ckpt.get("architecture_id") != EXPECTED_MIRA_ARCH:
        raise RuntimeError(f"MIRA_V6_CHECKPOINT_AUTHORITY_DRIFT::{ckpt.get('schema')}::{ckpt.get('architecture_id')}")
    if ckpt.get("backbone_architecture_id") != EXPECTED_BACKBONE_ARCH:
        raise RuntimeError(f"MIRA_V6_BACKBONE_ARCH_DRIFT::{ckpt.get('backbone_architecture_id')}")
    if ckpt.get("surface_lineage_hash") != EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("MIRA_V6_SURFACE_LINEAGE_DRIFT")
    if ckpt.get("teacher_predictor_input_used") is not False:
        raise RuntimeError("MIRA_V6_TEACHER_PREDICTOR_INPUT_BOUNDARY_DRIFT")

    backbone = ArachneA1V4(ArachneA1ConfigV4()).cpu().float()
    backbone.load_state_dict(ckpt["backbone"], strict=True)
    if backbone.config.architecture_id != EXPECTED_BACKBONE_ARCH:
        raise RuntimeError("MIRA_V6_INSTANTIATED_BACKBONE_ARCH_DRIFT")
    backbone_params = int(sum(p.numel() for p in backbone.parameters()))
    if backbone_params != EXPECTED_BACKBONE_PARAMS:
        raise RuntimeError(f"MIRA_V6_BACKBONE_PARAMETER_COUNT_DRIFT::{backbone_params}")

    runner = import_exact_runner(runner_path)
    decoder = runner.ArachneV6RawReadout(
        surface_dim=int(backbone.config.model_dim),
        token_dim=int(backbone.config.latent_channels),
        pair_dim=int(backbone.config.pair_geometry_dim),
        geom_dim=7,
        relation_dim=int(runner.RELATION_DIM),
    ).cpu().float()
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder_params = int(sum(p.numel() for p in decoder.parameters()))
    if decoder_params != EXPECTED_DECODER_PARAMS:
        raise RuntimeError(f"MIRA_V6_DECODER_PARAMETER_COUNT_DRIFT::{decoder_params}")

    joint_vocab_params = [
        name for name, tensor in backbone.named_parameters()
        if "embedding" in name.lower() and tensor.ndim >= 2 and int(tensor.shape[0]) in {28, EXPECTED_TEST_JOINTS}
    ]
    if joint_vocab_params:
        raise RuntimeError(f"MIRA_V6_FIXED_JOINT_VOCAB_DETECTED::{joint_vocab_params}")

    print("MIRA23_PREFLIGHT_PHASE=DECODER_J23_SMOKE", flush=True)
    r = 3
    with torch.no_grad():
        z, w = decoder.forward_rows(
            torch.arange(r, dtype=torch.long),
            torch.zeros((r, backbone.config.model_dim), dtype=torch.float32),
            torch.zeros((r, 7), dtype=torch.float32),
            torch.zeros((r, EXPECTED_TEST_JOINTS, backbone.config.pair_geometry_dim), dtype=torch.float32),
            torch.zeros((EXPECTED_TEST_JOINTS, backbone.config.field_tokens, backbone.config.latent_channels), dtype=torch.float32),
            torch.ones((r, EXPECTED_TEST_JOINTS), dtype=torch.bool),
        )
    if tuple(z.shape) != (r, EXPECTED_TEST_JOINTS) or tuple(w.shape) != (r, EXPECTED_TEST_JOINTS):
        raise RuntimeError(f"MIRA_V6_DECODER_J23_SHAPE_FAIL::{z.shape}::{w.shape}")
    if not torch.isfinite(z).all() or not torch.isfinite(w).all():
        raise RuntimeError("MIRA_V6_DECODER_J23_NONFINITE")
    simplex_resid = float((w.sum(-1) - 1.0).abs().max().item())
    if simplex_resid > 1e-6:
        raise RuntimeError(f"MIRA_V6_DECODER_J23_SIMPLEX_FAIL::{simplex_resid}")

    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "scope": "STRUCTURAL_PREFLIGHT_ONLY__NO_FULL_BACKBONE_FORWARD__NO_PRODUCT_AUTHORITY",
        "surface": {
            "sha256": sha256(surface_path),
            "lineage_hash": surface.geometry_lineage_hash,
            "node_count": len(surface.surface_nodes),
            "edge_count": len(surface.local_relations),
        },
        "teacher_source": {
            "sha256": sha256(teacher_path),
            "raw_bone_count": EXPECTED_RAW_BONES,
            "test_joint_count": EXPECTED_TEST_JOINTS,
            "selected_raw_bone_indices": binding["selected_raw_bone_indices"],
            "raw_parent_indices": binding["raw_parent_indices"],
            "selection_rule": "ROOT0_PLUS_SKIN_POSITIVE_RAW_BONES_1_TO_22",
            "teacher_skin_role": "FIXED_TEST_RIG_MEMBERSHIP_ONLY__NOT_MIRA_PREDICTOR_FEATURE",
        },
        "support_binding": {
            "policy": "GSA_NEAREST8_TO_RAW_TEACHER_BONE_HEAD_V1",
            "k": SUPPORT_K,
            "uses_teacher_skin": False,
            "binding_hash": binding["support_binding_hash"],
            "fbx_object_packaging_authority": False,
        },
        "qualified_teacher23": {
            "joint_count": len(qualified.joints),
            "root_count": len(roots),
            "skeleton_lineage_hash": qualified.skeleton_lineage_hash,
        },
        "conditioning": {
            "schema": conditioning.schema_version,
            "surface_count": n,
            "edge_count": e,
            "joint_count": j,
            "pair_geometry_shape": list(conditioning.pair_geometry.shape),
            "conditioning_hash": conditioning.conditioning_hashes[0],
        },
        "mira_v6": {
            "checkpoint_sha256": model_sha,
            "exact_runner_sha256": runner_sha,
            "checkpoint_schema": ckpt["schema"],
            "architecture_id": ckpt["architecture_id"],
            "backbone_architecture_id": ckpt["backbone_architecture_id"],
            "backbone_parameter_count": backbone_params,
            "decoder_parameter_count": decoder_params,
            "strict_backbone_load": True,
            "strict_decoder_load": True,
            "closure_guard_expected_joints_historical": int(getattr(runner, "EXPECTED_JOINTS")),
            "architecture_cardinality_dynamic_evidence": {
                "v4_fixed_joint_vocab_detected": False,
                "v6_decoder_j23_smoke": True,
                "decoder_output_shape": list(w.shape),
                "decoder_simplex_max_abs_residual": simplex_resid,
            },
            "original_checkpoint_skeleton_lineage_hash": ckpt.get("skeleton_lineage_hash"),
            "requery_skeleton_lineage_hash": qualified.skeleton_lineage_hash,
        },
        "authority": {
            "fbx_object_packaging_authority": False,
            "carrier_identity": "GEOMETRY_COMPONENT_SUPPORT_SEMANTICS",
            "teacher23_is_fixed_test_authority_not_product_authority": True,
            "compiler_qualified_teacher23": True,
            "mira_checkpoint_product_authority_claimed": bool(ckpt.get("product_authority_claimed", False)),
            "generalization_claimed": False,
        },
        "next": "A100_EXACT_V6_FULL_REQUERY_ON_TEACHER23",
    }
    write_json(out_dir / "MIRA_V6_TEACHER23_QUERY_PREFLIGHT_V1.json", result)
    print("MIRA23_PREFLIGHT_PASS=" + json.dumps({
        "joint_count": j,
        "teacher23_skeleton_lineage": qualified.skeleton_lineage_hash,
        "conditioning_hash": conditioning.conditioning_hashes[0],
        "decoder_j23_simplex_residual": simplex_resid,
        "next": result["next"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
