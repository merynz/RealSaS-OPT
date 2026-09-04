from __future__ import annotations

import base64
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.substrate.scene_first_signed import zero_surface_normal_operator_hash_v1
from compiler.realsas_compiler_core.types import RiggingSurfaceIR, SurfaceNode, SurfaceRelation
from models.geppetto.v2.geppetto_candidate_v2 import GeppettoCandidateV2
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from models.geppetto.v2.geppetto_eval_v2 import geppetto_metrics_v2
from models.geppetto.v2.geppetto_train_v2 import geppetto_train_step_v2
from models.geppetto.v2.training_targets_v1 import GeppettoTeacherTargetV1


FIXTURE_PATH = Path(__file__).with_name("MAGE_E2E_COMPACT_FIXTURE_V1.npz.b64")
FIXTURE_SHA256 = "65bec754f4806a45e01af78a3b72e1db0d0632094454ff4b31e9f61315d8c266"
SOURCE_ZERO_SURFACE_SHA256 = "987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b"
SOURCE_RUN_ID = "20260904T220929Z"
SEED = 20260905
MAX_STEPS = 2048
CHECK_EVERY = 32
REQUIRED_STABLE = 3
INFERENCE_RESOURCE_STEP_LIMIT = 128  # execution budget only; not a learned/frozen product slot count
LR = 3e-4
WEIGHT_DECAY = 1e-4


def _load_fixture() -> dict[str, np.ndarray]:
    raw = base64.b64decode("".join(FIXTURE_PATH.read_text().split()))
    got = sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise RuntimeError(f"fixture hash drift:{got}")
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _surface(fx: dict[str, np.ndarray]) -> RiggingSurfaceIR:
    points = np.asarray(fx["points"], np.float32)
    normals = np.asarray(fx["normals"], np.float32)
    edges = np.asarray(fx["edges"], np.int64)
    view_mask = np.asarray(fx["view_mask"], np.uint8)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("bad compact fixture surface")
    if len(points) != 950:
        raise ValueError(f"unexpected compact node count:{len(points)}")
    op_hash = zero_surface_normal_operator_hash_v1(k=64)
    nodes = []
    for i, (p, n, mask) in enumerate(zip(points, normals, view_mask)):
        views = tuple(v for v in range(8) if int(mask) & (1 << v))
        sid = f"MAGE:SFS:{i:04d}"
        nodes.append(SurfaceNode(
            surface_id=sid,
            P=tuple(map(float, p)),
            support_views=views,
            provenance_refs=(f"IRIS_V3_SIGNED:{SOURCE_RUN_ID}",),
            source_observation_ids=(),
            raster_bindings=(),
            persistence_group_id=f"MAGE:SFS:PG:{i:04d}",
            derived_normal=tuple(map(float, n)),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",) if views else ("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",),
            metadata={
                "normal_operator": "RealSaS.GSA.ZeroSurfaceRobustLocalPCA.v1",
                "normal_operator_hash": op_hash,
                "teacher_truth_used": False,
            },
        ))
    rels = []
    for ri, (a, b) in enumerate(edges.tolist()):
        rels.append(SurfaceRelation(
            relation_id=f"MAGE:SFS:R:{ri:05d}",
            a_surface_id=nodes[int(a)].surface_id,
            b_surface_id=nodes[int(b)].surface_id,
            relation_kind="SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
            score=1.0,
            metadata={"teacher_truth_used": False},
        ))
    lineage = content_sha256({
        "schema": "RealSaS.MageSceneFirstCompactSurfaceWitness.v1",
        "source_zero_surface_sha256": SOURCE_ZERO_SURFACE_SHA256,
        "fixture_sha256": FIXTURE_SHA256,
        "points": points.tolist(),
        "edges": edges.tolist(),
        "view_mask": view_mask.tolist(),
    })
    return RiggingSurfaceIR(
        tuple(nodes), tuple(rels), lineage,
        builder_id="RealSaS.GeometricSubstrateAssembler.current",
        metadata={
            "raster_coordinate_system": "NORMALIZED_GRID_XY",
            "resolution": 1024,
            "Nd_operator_sha256": op_hash,
            "scene_first_signed_geometry": True,
            "source_run_id": SOURCE_RUN_ID,
            "source_zero_surface_sha256": SOURCE_ZERO_SURFACE_SHA256,
            "teacher_truth_used": False,
        },
    )


def _teacher(fx: dict[str, np.ndarray], conditioning) -> GeppettoTeacherTargetV1:
    heads = np.asarray(fx["bone_heads_world"], np.float32)
    parents = np.asarray(fx["parents"], np.int64)
    if heads.shape != (41, 3) or parents.shape != (41,) or int(np.sum(parents < 0)) != 1:
        raise ValueError("Mage skeleton teacher drift")
    norm = conditioning.normalizations[0].normalize(heads)
    return GeppettoTeacherTargetV1(
        positions_normalized=norm,
        parent_indices=parents,
        root_mask=(parents < 0),
        valid=True,
    )


def _distinct_unique_radius(target: GeppettoTeacherTargetV1) -> float:
    p = torch.as_tensor(target.positions_normalized, dtype=torch.float32)
    d = torch.cdist(p, p).numpy()
    d[np.eye(len(d), dtype=bool)] = np.inf
    positive = d[np.isfinite(d) & (d > 1e-5)]
    if len(positive) == 0:
        raise ValueError("teacher has no distinct loci")
    return 0.5 * float(positive.min())


def _shipping(model, conditioning, target) -> dict[str, object]:
    device = next(model.parameters()).device
    f = torch.as_tensor(conditioning.features, device=device, dtype=torch.float32)
    p = torch.as_tensor(conditioning.positions_normalized, device=device, dtype=torch.float32)
    m = torch.as_tensor(conditioning.valid_mask, device=device, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        out, counts = model.generate(
            f, p, m,
            resource_step_limit=min(INFERENCE_RESOURCE_STEP_LIMIT, int(m[0].sum().item())),
        )
    k = int(counts[0])
    metrics = dict(geppetto_metrics_v2(out.positions_normalized[0, :k], target.positions_normalized))
    metrics["generated_count"] = k
    metrics["resource_step_limit"] = INFERENCE_RESOURCE_STEP_LIMIT
    return metrics


def _qualified_mechanical(model, surface, conditioning, target) -> tuple[bool, dict[str, object], object | None]:
    try:
        proposal = model.propose(
            conditioning,
            device=next(model.parameters()).device,
            resource_step_limit=INFERENCE_RESOURCE_STEP_LIMIT,
        )[0]
        if len(proposal.joints) != len(target.positions_normalized):
            return False, {"status": "COUNT_FAIL", "proposal_count": len(proposal.joints)}, None
        qualified = qualify_skeleton_v2(surface, proposal)
        ids = {j.canonical_joint_id for j in qualified.joints}
        illegal = sum(j.parent_canonical_id is not None and j.parent_canonical_id not in ids for j in qualified.joints)
        unsupported = sum(not j.support_surface_ids for j in qualified.joints)
        report = {
            "status": "PASS" if len(qualified.joints) == len(target.positions_normalized) and len(qualified.deform_root_ids) >= 1 and illegal == 0 and unsupported == 0 else "FAIL",
            "qualified_count": len(qualified.joints),
            "deform_root_count": len(qualified.deform_root_ids),
            "illegal_parent_count": int(illegal),
            "unsupported_joint_count": int(unsupported),
            "skeleton_lineage_hash": qualified.skeleton_lineage_hash,
        }
        return report["status"] == "PASS", report, qualified
    except Exception as exc:
        return False, {"status": "EXCEPTION", "error": f"{type(exc).__name__}:{exc}"}, None


def run(output_path: Path) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_REAL_MAGE_FIT")
    device = torch.device("cuda")

    fx = _load_fixture()
    surface = _surface(fx)
    conditioning = GeppettoConditioningAdapterV2()([surface])
    target = _teacher(fx, conditioning)
    unique_radius = _distinct_unique_radius(target)

    model = GeppettoCandidateV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    initial = _shipping(model, conditioning, target)
    trace = []
    stable = 0
    pass_step = None
    qualified = None
    mech_report = {"status": "NOT_EVALUATED"}

    for step_idx in range(1, MAX_STEPS + 1):
        losses = geppetto_train_step_v2(model, optimizer, conditioning, [target])
        if not all(math.isfinite(float(v)) for v in losses.values()):
            raise RuntimeError(f"nonfinite Geppetto loss at {step_idx}")
        if step_idx != 1 and step_idx % CHECK_EVERY:
            continue
        shipping = _shipping(model, conditioning, target)
        geometry_pass = (
            int(shipping["generated_count"]) == len(target.positions_normalized)
            and float(shipping["matched_p95"]) < unique_radius
        )
        mechanical_pass = False
        if geometry_pass:
            mechanical_pass, mech_report, qualified = _qualified_mechanical(model, surface, conditioning, target)
        product_pass = bool(geometry_pass and mechanical_pass)
        stable = stable + 1 if product_pass else 0
        row = {
            "step": step_idx,
            "loss_total": float(losses["total"]),
            "shipping": shipping,
            "unique_radius": unique_radius,
            "geometry_pass": geometry_pass,
            "mechanical_pass": mechanical_pass,
            "mechanical_report": mech_report,
            "stable_passes": stable,
        }
        trace.append(row)
        print("MAGE_GEPPETTO_CHECK=" + json.dumps(row, sort_keys=True), flush=True)
        if stable >= REQUIRED_STABLE:
            pass_step = step_idx
            break

    final = _shipping(model, conditioning, target)
    result = {
        "schema": "RealSaS.MageSceneFirstGeppettoFit.v1",
        "status": "PASS" if pass_step is not None else "FAIL",
        "seed": SEED,
        "source_run_id": SOURCE_RUN_ID,
        "fixture_sha256": FIXTURE_SHA256,
        "surface_nodes": len(surface.surface_nodes),
        "surface_relations": len(surface.local_relations),
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "geppetto_architecture_id": model.config.architecture_id,
        "geppetto_config_hash": model.config.config_hash,
        "target_joint_count": len(target.positions_normalized),
        "distinct_unique_radius": unique_radius,
        "initial": initial,
        "final": final,
        "pass_step": pass_step,
        "required_stable_passes": REQUIRED_STABLE,
        "stable_passes": stable,
        "mechanical_report": mech_report,
        "trace": trace,
        "teacher_role": "FIT_EVALUATOR_ONLY",
        "teacher_used_to_construct_surface": False,
        "generalization_claim": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MAGE_GEPPETTO_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise AssertionError(result)
    # Save only ephemeral job-local state for the next E2E stage; workflow artifact, not repo authority.
    torch.save({
        "schema": "RealSaS.MageSceneFirstGeppettoFitCheckpoint.v1",
        "model": model.state_dict(),
        "config_hash": model.config.config_hash,
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "step": pass_step,
    }, output_path.with_suffix(".pt"))
    return result


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/MAGE_GEPPETTO_FIT_RESULT_V1.json")
    run(out)
