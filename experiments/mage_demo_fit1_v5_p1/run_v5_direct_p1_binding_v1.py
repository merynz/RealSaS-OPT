from __future__ import annotations

"""Optimizer-free Mage demo skin query on the exact P1_B2_G10 render meshes.

The sealed FIT1 V5 backbone is NOT rerun on the 8171-node FIT2 substrate. Instead this
runner consumes the already sealed K4x512 joint-field tokens emitted by the FIT1 V5
closure and applies only the sealed 325,313-parameter DirectSimplexDecoderV5 to exact
P1 mesh vertices. Thus there is no historical barycentric skin transfer, no optimizer,
no backward pass and no A100 requirement.

Inputs are hash-pinned and every output QualifiedMeshSkinIR is bound to the exact
P1 mesh lineage and exact FIT1 G22 skeleton lineage by the Compiler.
"""

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.mesh.direct_model_skin import (
    qualify_direct_model_mesh_skin,
)
from compiler.realsas_compiler_core.mesh.mesh_binding import validate_qualified_mesh
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedJoint,
    QualifiedMeshVertex,
    QualifiedSkinIR,
    QualifiedSkinRow,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4_types import QualifiedSkeletonIRV2
from models.arachne.v2.arachne_geometry_v2 import arachne_pair_geometry_v2
from models.arachne.v5.arachne_candidate_v5 import DirectSimplexDecoderV5
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)

import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2


SCHEMA = "RealSaS.MageDemo.FIT1V5DirectP1Binding.v1"
EXPECTED_FIELD_TOKENS_FILE_SHA256 = "67c33f61fd3a417fb1c8b36d753b62d23d6167b1720c352458dde54784573ec6"
EXPECTED_CONDITIONING_WITNESS_SHA256 = "bdb2e5ed620fd8cb2891df0adfd622e83f22a931bc5116377f526fd91ba1ea2c"
EXPECTED_DECODER_DELTA_SHA256 = "13344178bf1b3ce96c9356456db0ad2c8a3945182a5ec63617c50137b8c52137"
EXPECTED_SKELETON_FILE_SHA256 = "48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992"
EXPECTED_SOURCE_SKIN_FILE_SHA256 = "401e0a563a974681c3f32a8a31c01eaa728aaa385750d69b2b811723d41c6773"
EXPECTED_SOURCE_SURFACE_LINEAGE = "67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb"
EXPECTED_SKELETON_LINEAGE = "738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306"
EXPECTED_SOURCE_SKIN_LINEAGE = "ef28f75e0306dbbabc32e75b837412ede39b180248502f6e504994310d91edaf"
EXPECTED_BACKBONE_SHA256 = "95c441f97b02123de1a5bc83bdf5ad223363c4b97927e8d420a0d246efbc1763"
EXPECTED_PARENT_DECODER_SHA256 = "078d5155f798d7b926c19463a31bf01c19973be30787f5fbff7916be87894832"
EXPECTED_V5_SOURCE_SHA256 = "a66adaebe92e9181873888ef90941ad87e4b3d835d21647e65176df4441a0f9e"
EXPECTED_DECODER_PARAMETERS = 325_313
EXPECTED_P1_MESH_LINEAGE = {
    0: "0c59f380b4817a2a05ff0489a37cf0c745b03bdf0ec219a4f41ad67abc0dd5c4",
    1: "6748278190007d6f887f5f6b2cace482d938f077bc4c4bfe25d5390ccca78c07",
    2: "eb8368c63cf73c05fb1ec017d724e3730617e595d9cce28d57273dcfa0e2be4c",
    3: "795e56aad0aa73774f5e9f7c96f655e49877b0c99f459defe8046a232cae5c49",
    4: "4f5b7a74670a1380a91048eccbcfaee1cd9beb8d4943a57d5cd1f560338bf466",
    5: "6b006a76a529ff718017ecb13bdd137aa5aba506387894e007933b98ccc0aee1",
    6: "1eacb8a0a73275f34e5c707a2c5ca7ad7add96b285b6446680c6c634d5c2ccfa",
    7: "fd4afa11aa7eda12bdf3e842d3ed90fb143d5d18f76c1087adecd3ad0c9e2c33",
}


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label}_MISSING::{path}")
    got = _sha(path)
    if got != expected:
        raise RuntimeError(f"{label}_SHA_DRIFT::{got}::{expected}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_skeleton(path: Path) -> QualifiedSkeletonIRV2:
    _require_sha(path, EXPECTED_SKELETON_FILE_SHA256, "FIT1_SKELETON")
    d = json.loads(path.read_text(encoding="utf-8"))
    joints = tuple(
        QualifiedJoint(
            canonical_joint_id=str(row["canonical_joint_id"]),
            position=tuple(map(float, row["position"])),
            parent_canonical_id=None if row.get("parent_canonical_id") is None else str(row["parent_canonical_id"]),
            support_surface_ids=tuple(map(str, row.get("support_surface_ids", ()))),
            source_proposal_id=str(row.get("source_proposal_id", "")),
        )
        for row in d["joints"]
    )
    skeleton = QualifiedSkeletonIRV2(
        joints=joints,
        deform_root_ids=tuple(map(str, d.get("deform_root_ids", ()))),
        assembly_root_binding=dict(d.get("assembly_root_binding", {})),
        qualification_report=dict(d.get("qualification_report", {})),
        skeleton_lineage_hash=str(d["skeleton_lineage_hash"]),
        schema_version=str(d.get("schema_version", "RealSaS.QualifiedSkeletonIR.v2")),
    )
    if skeleton.skeleton_lineage_hash != EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("FIT1_SKELETON_LINEAGE_DRIFT")
    return skeleton


def _load_source_skin(path: Path) -> QualifiedSkinIR:
    _require_sha(path, EXPECTED_SOURCE_SKIN_FILE_SHA256, "FIT1_V5_SOURCE_SKIN")
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(
        QualifiedSkinRow(
            surface_id=str(row["surface_id"]),
            influences=tuple((str(jid), float(weight)) for jid, weight in row["influences"]),
            simplex_residual_before=float(row["simplex_residual_before"]),
            correction_l1=float(row["correction_l1"]),
        )
        for row in d["rows"]
    )
    skin = QualifiedSkinIR(
        rows=rows,
        surface_binding_hash=str(d["surface_binding_hash"]),
        skeleton_binding_hash=str(d["skeleton_binding_hash"]),
        qualification_report=dict(d.get("qualification_report", {})),
        skin_lineage_hash=str(d["skin_lineage_hash"]),
        schema_version=str(d.get("schema_version", "RealSaS.QualifiedSkinIR.v1")),
    )
    if skin.surface_binding_hash != EXPECTED_SOURCE_SURFACE_LINEAGE:
        raise RuntimeError("FIT1_V5_SOURCE_SKIN_SURFACE_LINEAGE_DRIFT")
    if skin.skeleton_binding_hash != EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("FIT1_V5_SOURCE_SKIN_SKELETON_LINEAGE_DRIFT")
    if skin.skin_lineage_hash != EXPECTED_SOURCE_SKIN_LINEAGE:
        raise RuntimeError("FIT1_V5_SOURCE_SKIN_LINEAGE_DRIFT")
    return skin


def _load_mesh(path: Path, surface, expected_lineage: str) -> QualifiedEditableMeshIR:
    if not path.is_file():
        raise RuntimeError(f"P1_MESH_ARTIFACT_MISSING::{path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    vertices = tuple(
        QualifiedMeshVertex(
            canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
            P=tuple(map(float, row["P"])),
            support_binding=SurfaceSupportBinding(
                mode=str(row["support_binding"]["mode"]),
                coefficients=tuple((str(sid), float(coeff)) for sid, coeff in row["support_binding"]["coefficients"]),
                metadata=dict(row["support_binding"].get("metadata", {})),
                schema_version=str(row["support_binding"].get("schema_version", "RealSaS.SurfaceSupportBinding.v1")),
            ),
            source_candidate_vertex_id=str(row.get("source_candidate_vertex_id", "")),
            metadata=dict(row.get("metadata", {})),
        )
        for row in d["vertices"]
    )
    mesh = QualifiedEditableMeshIR(
        vertices=vertices,
        faces=tuple(tuple(map(str, face)) for face in d["faces"]),
        edges=tuple((str(a), str(b)) for a, b in d["edges"]),
        surface_binding_hash=str(d["surface_binding_hash"]),
        view_index=int(d["view_index"]),
        camera_binding_hash=str(d["camera_binding_hash"]),
        qualification_report=dict(d.get("qualification_report", {})),
        mesh_lineage_hash=str(d["mesh_lineage_hash"]),
        boundary_constraints=tuple(d.get("boundary_constraints", ())),
        support_coverage_classification=str(d.get("support_coverage_classification", "")),
        schema_version=str(d.get("schema_version", "RealSaS.QualifiedEditableMeshIR.v1")),
        metadata=dict(d.get("metadata", {})),
    )
    validate_qualified_mesh(mesh, surface)
    if mesh.mesh_lineage_hash != expected_lineage:
        raise RuntimeError(f"P1_MESH_LINEAGE_DRIFT_V{mesh.view_index}:{mesh.mesh_lineage_hash}:{expected_lineage}")
    return mesh


def _load_v5_authority(field_tokens_path: Path, witness_path: Path, decoder_path: Path, device: torch.device):
    _require_sha(field_tokens_path, EXPECTED_FIELD_TOKENS_FILE_SHA256, "V5_FIELD_TOKENS")
    _require_sha(witness_path, EXPECTED_CONDITIONING_WITNESS_SHA256, "V5_CONDITIONING_WITNESS")
    _require_sha(decoder_path, EXPECTED_DECODER_DELTA_SHA256, "V5_DECODER_DELTA")

    with np.load(field_tokens_path, allow_pickle=False) as z:
        field_tokens = np.asarray(z["field_tokens"], dtype=np.float32)
        joint_ids = tuple(map(str, z["joint_ids"].tolist()))
        declared_token_sha = str(z["field_tokens_sha256"].reshape(-1)[0])
        backbone_sha = str(z["v4_backbone_sha256"].reshape(-1)[0])
        parent_decoder_sha = str(z["parent_z_head_sha256"].reshape(-1)[0])
    token_sha = sha256(np.ascontiguousarray(field_tokens).tobytes()).hexdigest()
    if token_sha != declared_token_sha:
        raise RuntimeError("V5_FIELD_TOKEN_PAYLOAD_SHA_DRIFT")
    if field_tokens.shape != (1, 22, 4, 512):
        raise RuntimeError(f"V5_FIELD_TOKEN_SHAPE_DRIFT::{field_tokens.shape}")
    if backbone_sha != EXPECTED_BACKBONE_SHA256 or parent_decoder_sha != EXPECTED_PARENT_DECODER_SHA256:
        raise RuntimeError("V5_FIELD_TOKEN_PARENT_AUTHORITY_DRIFT")

    with np.load(witness_path, allow_pickle=False) as z:
        witness_joint_ids = tuple(map(str, z["joint_ids"].tolist()))
        joint_positions = np.asarray(z["joint_positions_normalized"], dtype=np.float32)
        joint_mask = np.asarray(z["joint_mask"], dtype=bool)
        parent_indices = np.asarray(z["parent_indices"], dtype=np.int64)
        source_surface_hash = str(z["source_surface_hash"].reshape(-1)[0])
        source_skeleton_hash = str(z["source_skeleton_hash"].reshape(-1)[0])
    if witness_joint_ids != joint_ids:
        raise RuntimeError("V5_FIELD_TOKEN_WITNESS_JOINT_ORDER_DRIFT")
    if joint_positions.shape != (1, 22, 3) or joint_mask.shape != (1, 22) or parent_indices.shape != (1, 22):
        raise RuntimeError("V5_WITNESS_JOINT_GEOMETRY_SHAPE_DRIFT")
    if not bool(joint_mask.all()):
        raise RuntimeError("V5_WITNESS_JOINT_MASK_DRIFT")
    if source_surface_hash != EXPECTED_SOURCE_SURFACE_LINEAGE or source_skeleton_hash != EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("V5_WITNESS_LINEAGE_DRIFT")

    payload = torch.load(decoder_path, map_location="cpu", weights_only=False)
    if str(payload.get("architecture_id")) != "RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5":
        raise RuntimeError("V5_DECODER_ARCHITECTURE_DRIFT")
    if str(payload.get("backbone_model_sha256")) != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError("V5_DECODER_BACKBONE_AUTHORITY_DRIFT")
    if str(payload.get("decoder_parent_sha256")) != EXPECTED_PARENT_DECODER_SHA256:
        raise RuntimeError("V5_DECODER_PARENT_AUTHORITY_DRIFT")
    if str(payload.get("v5_source_sha256")) != EXPECTED_V5_SOURCE_SHA256:
        raise RuntimeError("V5_DECODER_SOURCE_AUTHORITY_DRIFT")
    if int(payload.get("decoder_parameter_count", -1)) != EXPECTED_DECODER_PARAMETERS:
        raise RuntimeError("V5_DECODER_PARAMETER_COUNT_DRIFT")

    source_path = Path(inspect.getsourcefile(DirectSimplexDecoderV5) or "")
    if not source_path.is_file() or _sha(source_path) != EXPECTED_V5_SOURCE_SHA256:
        raise RuntimeError("V5_RUNTIME_SOURCE_SHA_DRIFT")

    decoder = DirectSimplexDecoderV5().to(device=device, dtype=torch.float32)
    decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    decoder.eval()
    if decoder.parameter_count != EXPECTED_DECODER_PARAMETERS:
        raise RuntimeError("V5_RUNTIME_DECODER_PARAMETER_COUNT_DRIFT")

    return {
        "field_tokens": torch.from_numpy(field_tokens).to(device=device, dtype=torch.float32),
        "joint_ids": joint_ids,
        "joint_positions": joint_positions,
        "joint_mask": joint_mask,
        "parent_indices": parent_indices,
        "decoder": decoder,
        "field_token_payload_sha256": token_sha,
    }


def _mesh_query_geometry(mesh, surface, tensor):
    normal_by_id = {
        str(sid): (np.asarray(tensor.normals[i], dtype=np.float64), bool(tensor.normal_valid[i]))
        for i, sid in enumerate(tensor.surface_ids)
    }
    positions, normals, valid = [], [], []
    for vertex in mesh.vertices:
        positions.append(tuple(map(float, vertex.P)))
        rows = []
        all_valid = True
        for sid, coeff in vertex.support_binding.coefficients:
            if str(sid) not in normal_by_id:
                raise RuntimeError(f"P1_NORMAL_SUPPORT_ID_MISSING:{sid}")
            n, ok = normal_by_id[str(sid)]
            all_valid = all_valid and bool(ok)
            rows.append(float(coeff) * n)
        n = np.sum(rows, axis=0) if rows else np.zeros(3, dtype=np.float64)
        norm = float(np.linalg.norm(n))
        ok = bool(all_valid and norm > 1.0e-8 and np.isfinite(n).all())
        normals.append(tuple((n / norm).tolist()) if ok else (0.0, 0.0, 0.0))
        valid.append(ok)
    positions = np.asarray(positions, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    geometry7 = np.concatenate([2.0 * positions, normals, valid[:, None].astype(np.float32)], axis=-1).astype(np.float32)
    if not np.isfinite(geometry7).all():
        raise RuntimeError("P1_V5_QUERY_GEOMETRY_NONFINITE")
    return positions, normals, valid, geometry7


def _predict_mesh_weights(mesh, surface, tensor, authority, device: torch.device, chunk: int):
    positions, normals, normal_valid, geometry7 = _mesh_query_geometry(mesh, surface, tensor)
    joint_positions = authority["joint_positions"]
    joint_mask = authority["joint_mask"]
    parent_indices = authority["parent_indices"]
    decoder = authority["decoder"]
    field_tokens = authority["field_tokens"]

    blocks = []
    for start in range(0, len(positions), int(chunk)):
        stop = min(len(positions), start + int(chunk))
        sp = positions[None, start:stop]
        sn = normals[None, start:stop]
        sv = normal_valid[None, start:stop]
        sm = np.ones((1, stop - start), dtype=bool)
        pair, pair_mask = arachne_pair_geometry_v2(
            sp,
            sn,
            sv,
            joint_positions,
            parent_indices,
            sm,
            joint_mask,
        )
        with torch.inference_mode():
            _logits, weights = decoder(
                torch.from_numpy(geometry7[None, start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(pair).to(device=device, dtype=torch.float32),
                torch.from_numpy(pair_mask).to(device=device, dtype=torch.bool),
                field_tokens,
            )
        blocks.append(weights[0].detach().cpu().float().numpy())
    predicted = np.concatenate(blocks, axis=0).astype(np.float32)
    if predicted.shape != (len(mesh.vertices), len(authority["joint_ids"])):
        raise RuntimeError(f"P1_V5_WEIGHT_SHAPE_DRIFT::{predicted.shape}")
    if not np.isfinite(predicted).all() or float(predicted.min()) < -1.0e-8:
        raise RuntimeError("P1_V5_WEIGHT_NUMERIC_INVALID")
    return geometry7, predicted


def run(args) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    if int(args.chunk) <= 0:
        raise ValueError("chunk must be positive")

    # Rebuild only the exact FIT2 surface authority needed to validate/materialize M.
    # This is not Arachne FIT2 and performs no optimization.
    surface, tensor, replay = ceiling_v2._preflight_surface(args)
    if not bool(replay.get("gsa_lineage_exact_match")):
        raise RuntimeError("P1_DIRECT_BINDING_REQUIRES_EXACT_FIT2_GSA_LINEAGE")

    skeleton = _load_skeleton(Path(args.skeleton))
    source_skin = _load_source_skin(Path(args.source_skin))
    authority = _load_v5_authority(
        Path(args.field_tokens),
        Path(args.conditioning_witness),
        Path(args.decoder_delta),
        device,
    )
    if set(authority["joint_ids"]) != {str(j.canonical_joint_id) for j in skeleton.joints}:
        raise RuntimeError("V5_FIELD_TOKEN_SKELETON_JOINT_SET_DRIFT")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh_dir = Path(args.materialized_mesh_dir)
    rows = []
    artifacts = []

    for view in range(8):
        mesh_path = mesh_dir / f"P1_B2_G10_V{view}_QUALIFIED_MESH_IR.json"
        mesh = _load_mesh(mesh_path, surface, EXPECTED_P1_MESH_LINEAGE[view])
        if int(mesh.view_index) != view:
            raise RuntimeError(f"P1_VIEW_INDEX_DRIFT::{mesh.view_index}::{view}")

        geometry7, weights = _predict_mesh_weights(mesh, surface, tensor, authority, device, int(args.chunk))
        weight_map = {
            str(vertex.canonical_mesh_vertex_id): tuple(float(x) for x in weights[i])
            for i, vertex in enumerate(mesh.vertices)
        }
        bound = qualify_direct_model_mesh_skin(
            surface,
            skeleton,
            source_skin,
            mesh,
            joint_ids=authority["joint_ids"],
            predicted_weights_by_vertex=weight_map,
            model_provenance={
                "architecture_id": "RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5",
                "field_tokens_file_sha256": EXPECTED_FIELD_TOKENS_FILE_SHA256,
                "field_token_payload_sha256": authority["field_token_payload_sha256"],
                "conditioning_witness_sha256": EXPECTED_CONDITIONING_WITNESS_SHA256,
                "decoder_delta_sha256": EXPECTED_DECODER_DELTA_SHA256,
                "backbone_checkpoint_sha256": EXPECTED_BACKBONE_SHA256,
                "optimizer_constructed": False,
                "backward_executed": False,
                "parameter_update_performed": False,
            },
        )

        weight_name = f"P1_B2_G10_V{view}_V5_DIRECT_WEIGHTS.npz"
        binding_name = f"P1_B2_G10_V{view}_QUALIFIED_MESH_SKIN_IR.json"
        weight_path, binding_path = out / weight_name, out / binding_name
        np.savez_compressed(
            weight_path,
            canonical_mesh_vertex_id=np.asarray([str(v.canonical_mesh_vertex_id) for v in mesh.vertices]),
            joint_ids=np.asarray(authority["joint_ids"]),
            weights=weights,
            geometry7=geometry7,
            mesh_lineage_hash=np.asarray([mesh.mesh_lineage_hash]),
            mesh_skin_lineage_hash=np.asarray([bound.mesh_skin_lineage_hash]),
        )
        _write_json(binding_path, bound.to_dict())
        artifacts.extend(
            [
                {"path": weight_name, "sha256": _sha(weight_path), "bytes": weight_path.stat().st_size},
                {"path": binding_name, "sha256": _sha(binding_path), "bytes": binding_path.stat().st_size},
            ]
        )
        report = bound.qualification_report
        rows.append(
            {
                "view": view,
                "mesh_lineage_hash": mesh.mesh_lineage_hash,
                "mesh_skin_lineage_hash": bound.mesh_skin_lineage_hash,
                "vertex_count": len(mesh.vertices),
                "joint_count": len(authority["joint_ids"]),
                "direct_model_query": True,
                "source_skin_rows_consumed": False,
                "historical_barycentric_weight_transfer_used": False,
                "max_simplex_residual_before": float(report["max_simplex_residual_before"]),
                "total_correction_l1": float(report["total_correction_l1"]),
            }
        )

    manifest_payload = {
        "schema": SCHEMA,
        "status": "PASS__V5_DIRECT_QUERY_AND_COMPILER_EXACT_P1_BINDING_V0_V7",
        "authority": "canonical/MAGE_DEMO_FIT1_V5_P1_EXECUTION_AUTHORITY_V1.json",
        "fit2_arachne_training_executed": False,
        "a100_required": False,
        "optimizer_constructed": False,
        "backward_executed": False,
        "parameter_update_performed": False,
        "historical_barycentric_weight_transfer_used": False,
        "source_skin_rows_consumed": False,
        "field_tokens_file_sha256": EXPECTED_FIELD_TOKENS_FILE_SHA256,
        "conditioning_witness_sha256": EXPECTED_CONDITIONING_WITNESS_SHA256,
        "decoder_delta_sha256": EXPECTED_DECODER_DELTA_SHA256,
        "fit1_skeleton_file_sha256": EXPECTED_SKELETON_FILE_SHA256,
        "fit1_source_skin_file_sha256": EXPECTED_SOURCE_SKIN_FILE_SHA256,
        "fit1_skeleton_lineage_hash": EXPECTED_SKELETON_LINEAGE,
        "fit1_source_skin_lineage_hash": EXPECTED_SOURCE_SKIN_LINEAGE,
        "views": rows,
        "artifacts": artifacts,
        "product_pass_claimed": False,
        "unseen_generalization_claimed": False,
    }
    manifest_path = out / "V5_DIRECT_P1_BINDING_MANIFEST.json"
    _write_json(manifest_path, manifest_payload)
    seal = {
        "schema": SCHEMA + ".Seal.v1",
        "status": "SEALED__V5_DIRECT_P1_BINDING",
        "manifest": manifest_path.name,
        "manifest_sha256": _sha(manifest_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "product_pass_claimed": False,
    }
    seal_path = out / "V5_DIRECT_P1_BINDING_SEAL.json"
    _write_json(seal_path, seal)
    print("V5_DIRECT_P1_BINDING=" + json.dumps(seal, sort_keys=True), flush=True)
    return {"manifest": manifest_payload, "seal": seal}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--materialized-mesh-dir", required=True)
    p.add_argument("--field-tokens", required=True)
    p.add_argument("--conditioning-witness", required=True)
    p.add_argument("--decoder-delta", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--source-skin", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=2048)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
