from __future__ import annotations

"""Optimizer-free Geppetto replay on the active corrected Mage FIT1 substrate.

V1 is preserved as historical repair apparatus. V2 is the current replay runner bound to
`canonical/MAGE_FIT1_REPLAY_SUBSTRATE_V1.json`: H1 run 20260912T074348Z, GSA target
8192 / exact lineage, and exact observation support. The promoted Geppetto checkpoint
is loaded strictly and evaluated without optimizer, backward, parameter updates, old
QualifiedSkeletonIR reuse, or teacher geometry at inference.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from experiments.geppetto_reference_strength_fullstack_v1.run_geppetto_reference_strength_fit1_v1 import (
    _build_target,
    _check,
    _load_cameras,
)
from models.geppetto.reference_strength_v1.checkpoint_authority_v1 import (
    FINAL_CHECKPOINT_SHA256,
)
from models.geppetto.reference_strength_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    GeppettoReferenceStrengthNoLearnedSlotV1,
    assert_no_learned_view_slot_identity_v1,
)
from models.geppetto.reference_strength_v1.rigging_surface_tensorization_v1 import (
    tensorize_rigging_surface_v1,
)


SCHEMA = "RealSaS.MageFullSubject.FrozenGeppettoReplay.v2"
ACTIVE_SUBSTRATE_ID = "MAGE_FIT1_REPLAY_SUBSTRATE_V1"
ACTIVE_GSA_TARGET_NODES = 8192
ACTIVE_GSA_NODE_COUNT = 8171
ACTIVE_GSA_EDGE_COUNT = 23656
ACTIVE_GSA_LINEAGE = "65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da"
ACTIVE_ZERO_SURFACE_SHA256 = "56073e8b348b828350c812ac44982b823237196d5ec2f361241877e9ae301925"
NORMALIZED_CORPUS_SHA256 = "528bef491eceb358ebc8ecb2a46af1d37b4322a7ef500281403a8207fe7c648f"
CAMERA_SHA256 = (
    "73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541",
    "bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4",
    "3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c",
    "24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d",
    "5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39",
    "7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe",
    "daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec",
    "68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f",
)
OBSERVATION_SHA256 = (
    "8e9875c16bba3047c8f2fc211b984f2399d1145f5720c7427c6fc03a3fec0616",
    "fa94283780d5e5f3ad3943bbffc1f0592a70fc362fdd03b94a1d1cb46889dc30",
    "777bf4f7c505b405d3a5d2a111b2f297949454f77cae7c33064bf02479d384a5",
    "2a771c91c1d1c0b75dab14f6e98b7905a8429bbf0c0a8dd690261f93f6476d86",
    "354bb239feb6c1a515917fee4efb42fb1e0cb9f173d0eb3191fb0901a02a6228",
    "158fc14a75aa69f6133f2ba3ec5df2ad146afc62f477d7d3b4a41ca2cb0e42f2",
    "e3b08836c187863819d8b8aaa53aef79eddfee167e1fabf3fb1dd8ec0484563a",
    "87d4ad46edffec3c6bff034d305194820288d3cc6ef9a9480ac2234fb56451bc",
)
ALPHA_THRESHOLD = 8


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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(name)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    return d


def _load_observation_masks(paths: tuple[Path, ...]) -> tuple[np.ndarray, ...]:
    if len(paths) != 8:
        raise RuntimeError("EXACTLY_8_OBSERVATIONS_REQUIRED")
    masks = []
    for view, path in enumerate(paths):
        _require_sha(path, OBSERVATION_SHA256[view], f"OBSERVATION_V{view}")
        with Image.open(path) as im:
            if im.mode != "RGBA" or im.size != (1024, 1024):
                raise RuntimeError(f"OBSERVATION_CONTRACT_DRIFT_V{view}")
            rgba = np.asarray(im, dtype=np.uint8)
        masks.append(np.ascontiguousarray(rgba[..., 3] >= ALPHA_THRESHOLD))
    return tuple(masks)


def _build_active_surface(args, cameras, observation_masks):
    zero_path = Path(args.zero_surface)
    _require_sha(zero_path, ACTIVE_ZERO_SURFACE_SHA256, "ACTIVE_ZERO_SURFACE")
    with np.load(zero_path, allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces", "normals"}:
            raise RuntimeError(f"ZERO_SURFACE_PAYLOAD_DRIFT::{sorted(z.files)}")
        world = np.asarray(z["vertices"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        hints = np.asarray(z["normals"], dtype=np.float64)
    center = np.asarray(cameras[0]["center"], dtype=np.float64)
    half = float(cameras[0]["half_extent"])
    vertices_normalized = (world - center[None, :]) / half
    surface = rigging_surface_from_scene_first_zero_mesh_v1(
        vertices_normalized,
        faces,
        hints,
        cameras,
        normalization_center=center,
        normalization_half_extent=half,
        authority_label="H1_OBSERVABLE_PRODUCT_SURFACE_MAGE_FIT1_REPLAY_V1",
        source_run_id=str(args.iris_run_id),
        source_checkpoint_sha256=str(args.iris_checkpoint_sha256),
        source_zero_surface_sha256=ACTIVE_ZERO_SURFACE_SHA256,
        target_nodes=ACTIVE_GSA_TARGET_NODES,
        normal_k=64,
        visibility_depth_tolerance_norm=0.02,
        observation_alpha_masks=observation_masks,
        observation_ids=tuple(f"MAGE_FIT1_V{i}" for i in range(8)),
        observation_hashes=OBSERVATION_SHA256,
        observation_support_radius_px=1,
        require_observation_support=True,
        metadata={
            "camera_contract": "CANONICAL_8_ORTHOGRAPHIC_YAW_45_DEG",
            "full_subject_reclosure": True,
            "active_substrate_id": ACTIVE_SUBSTRATE_ID,
            "teacher_truth_used": False,
        },
    )
    tensor = tensorize_rigging_surface_v1(surface, require_scene_first=True)
    if surface.geometry_lineage_hash != ACTIVE_GSA_LINEAGE:
        raise RuntimeError(
            f"ACTIVE_GSA_LINEAGE_DRIFT::{surface.geometry_lineage_hash}::{ACTIVE_GSA_LINEAGE}"
        )
    if int(tensor.node_count) != ACTIVE_GSA_NODE_COUNT:
        raise RuntimeError(f"ACTIVE_GSA_NODE_COUNT_DRIFT::{tensor.node_count}::{ACTIVE_GSA_NODE_COUNT}")
    if int(tensor.edge_count) != ACTIVE_GSA_EDGE_COUNT:
        raise RuntimeError(f"ACTIVE_GSA_EDGE_COUNT_DRIFT::{tensor.edge_count}::{ACTIVE_GSA_EDGE_COUNT}")
    if not tensor.raster_valid.any() or not tensor.support.any():
        raise RuntimeError("ACTIVE_GSA_OBSERVATION_SUPPORT_EMPTY")
    return surface, tensor


def run(args) -> dict:
    normalized = Path(args.normalized_corpus)
    checkpoint = Path(args.checkpoint)
    _require_sha(normalized, NORMALIZED_CORPUS_SHA256, "NORMALIZED_CORPUS")
    _require_sha(checkpoint, FINAL_CHECKPOINT_SHA256, "FROZEN_GEPPETTO_CHECKPOINT")

    camera_paths = tuple(Path(x) for x in args.cameras)
    observation_paths = tuple(Path(x) for x in args.observations)
    for view, path in enumerate(camera_paths):
        _require_sha(path, CAMERA_SHA256[view], f"CAMERA_V{view}")
    cameras = _load_cameras(list(camera_paths))
    observation_masks = _load_observation_masks(observation_paths)
    surface, tensor = _build_active_surface(args, cameras, observation_masks)
    target, _prepared = _build_target(normalized, tensor)

    device = _device(args.device)
    model = GeppettoReferenceStrengthNoLearnedSlotV1().to(device)
    assert_no_learned_view_slot_identity_v1(model)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("FROZEN_GEPPETTO_CHECKPOINT_PAYLOAD_INVALID")
    if payload.get("config_hash") != model.config.config_hash:
        raise RuntimeError(
            f"FROZEN_GEPPETTO_CONFIG_HASH_DRIFT::{payload.get('config_hash')}::{model.config.config_hash}"
        )
    incompat = model.load_state_dict(payload["model"], strict=True)
    if getattr(incompat, "missing_keys", None) or getattr(incompat, "unexpected_keys", None):
        raise RuntimeError("FROZEN_GEPPETTO_STRICT_LOAD_DRIFT")
    model.eval()

    check = _check(model, surface, tensor, target, int(payload.get("step", -1)))
    passed = bool(check.get("pass", False))
    status = (
        "PASS__FROZEN_GEPPETTO_TRANSFERS_TO_ACTIVE_MAGE_FIT1_REPLAY_SUBSTRATE"
        if passed
        else "FAIL__FROZEN_GEPPETTO_DOES_NOT_TRANSFER_TO_ACTIVE_SUBSTRATE__REFIT_DECISION_REQUIRED"
    )
    qualified_rows = [
        row.get("qualified")
        for row in check.get("diffusion_seed_reports", [])
        if isinstance(row, dict) and row.get("qualified") is not None
    ]
    report = {
        "schema": SCHEMA,
        "status": status,
        "active_substrate_id": ACTIVE_SUBSTRATE_ID,
        "device": str(device),
        "optimizer_constructed": False,
        "backward_executed": False,
        "parameter_update_performed": False,
        "old_qualified_skeleton_reused": False,
        "teacher_geometry_used_at_inference": False,
        "frozen_checkpoint_sha256": FINAL_CHECKPOINT_SHA256,
        "corrected_zero_surface_sha256": ACTIVE_ZERO_SURFACE_SHA256,
        "corrected_surface_lineage_hash": surface.geometry_lineage_hash,
        "surface_node_count": int(tensor.node_count),
        "surface_edge_count": int(tensor.edge_count),
        "surface_tensorization_hash": tensor.tensorization_hash,
        "observation_support_authority": surface.metadata.get("observation_support_authority"),
        "observation_hashes": list(OBSERVATION_SHA256),
        "camera_hashes": list(CAMERA_SHA256),
        "check": check,
        "qualified_skeleton_candidates": qualified_rows,
        "geppetto_refit_authorized_by_this_result": not passed,
        "product_pass_claimed": False,
        "unseen_generalization_claimed": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(status)
    print("active_substrate=", ACTIVE_SUBSTRATE_ID)
    print("surface_lineage=", surface.geometry_lineage_hash)
    print("nodes=", tensor.node_count, "edges=", tensor.edge_count)
    print(out)
    if not passed:
        raise SystemExit(3)
    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--normalized-corpus", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--iris-run-id", required=True)
    p.add_argument("--iris-checkpoint-sha256", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
