from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    repair_candidate_endpoint_collapses_v1,
    repair_candidate_fixed_vertex_flips_v1,
    repair_candidate_projected_relaxation_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,
    build_mesh_qualification_policy,
    canonical_mesh_candidate_lineage_hash,
)
from compiler.realsas_compiler_core.types import SurfaceSupportBinding

EXPECTED_T1B_SHA = "b7ccb058ef23f8c54d053e64580150dffffeb6a76a126ecaf48b265dd093f390"
EXPECTED_NORM_SHA = "01b63a57390de9a5d2c731d9644e65ff779980caa066c19204db3b4a3e0c634b"
EXPECTED_V2_SHA = "ab6278d55577e0e46f0faf9535dfe16238d815d81a3854b972effc7d440967a9"
EXPECTED_V3_SHA = "1fa2c1d67314124d4c22af55e57cc8eb9b5e65bf1ff65870acefc7ea1d104bd1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quality_policy():
    return build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH", 0.999, 0.999, 0.00025, 0.0005),
            CarrierCoverageThresholdIR("PLANAR", 0.99, 0.995, 0.0025, 0.0025),
        ),
        g3_min_dynamic_area_ratio=0.05,
        g3_max_dynamic_area_ratio=20.0,
        g3_max_dynamic_condition_number=16.0,
    )


def build_original(t1b: Path, normalization: Path) -> CanonicalMeshCandidateIR:
    if sha256(t1b) != EXPECTED_T1B_SHA:
        raise RuntimeError("TESSA_REPLAY_T1B_SHA_DRIFT")
    if sha256(normalization) != EXPECTED_NORM_SHA:
        raise RuntimeError("TESSA_REPLAY_NORMALIZATION_SHA_DRIFT")
    with np.load(t1b, allow_pickle=False) as z:
        vn = np.asarray(z["vertices_normalized"], np.float64)
        fi = np.asarray(z["faces"], np.int64)
        vci = np.asarray(z["vertex_component_indices"], np.int64)
        fci = np.asarray(z["face_component_indices"], np.int64)
    if vn.shape != (3665, 3) or fi.shape != (6952, 3) or vci.shape != (3665,) or fci.shape != (6952,):
        raise RuntimeError("TESSA_REPLAY_T1B_SHAPE_DRIFT")
    if len(np.unique(vci)) != 39:
        raise RuntimeError("TESSA_REPLAY_COMPONENT_COUNT_DRIFT")
    norm = json.loads(normalization.read_text(encoding="utf-8"))
    if norm.get("normalization_hash") != "4d1fd4943d2815156974b8967d5ba180e424dc7343f9da68549cff8146efdb41":
        raise RuntimeError("TESSA_REPLAY_NORMALIZATION_LINEAGE_DRIFT")
    center = np.asarray(norm["center_xyz"], np.float64)
    world = vn * (2.0 * float(norm["half_extent"])) + center[None, :]
    ids = tuple(f"TESSA:V:{i:04d}" for i in range(len(world)))
    vertices = tuple(
        CanonicalMeshVertexCandidateIR(
            candidate_vertex_id=ids[i],
            support_binding=SurfaceSupportBinding(
                "IDENTITY_SURFACE_NODE",
                ((ids[i], 1.0),),
                metadata={"tessa_t1b_reference_support": True},
            ),
            component_id=f"TESSA:C:{int(vci[i]):02d}",
            P=tuple(map(float, world[i])),
            metadata={"decoded_tessa_vertex_index": int(i)},
        )
        for i in range(len(world))
    )
    faces = tuple(tuple(ids[int(x)] for x in row) for row in fi)
    edges = tuple(sorted({tuple(sorted((face[i], face[j]))) for face in faces for i, j in ((0, 1), (1, 2), (2, 0))}))
    candidate = CanonicalMeshCandidateIR(
        vertices=vertices,
        faces=faces,
        edges=edges,
        surface_binding_hash="TESSA_T1B_REFERENCE_SURFACE_V1",
        partition_binding_hash="TESSA_T1B_COMPONENTS_39_V1",
        carrier_policy_binding_hash="TESSA_T1B_MESH_CARRIER_V1",
        producer_id="RealSaS.TESSAT1BDecodedReferenceCandidate.v1",
        producer_policy_hash=hashlib.sha256(b"TESSA_T1B_STATIC_G3_REPAIR_REFERENCE_V2").hexdigest(),
        candidate_lineage_hash="",
        metadata={
            "fit1_knight": True,
            "teacher_geometry_used_to_repair": False,
            "reference_is_decoded_tessa_itself": True,
            "threshold_relaxation_performed": False,
        },
    )
    return replace(candidate, candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate))


def boundary_support_ids(candidate: CanonicalMeshCandidateIR) -> frozenset[str]:
    incidence = defaultdict(int)
    for a, b, c in candidate.faces:
        for edge in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
            incidence[edge] += 1
    if sum(1 for n in incidence.values() if n > 2) != 0:
        raise RuntimeError("TESSA_REPLAY_INITIAL_NONMANIFOLD")
    edges = [edge for edge, n in incidence.items() if n == 1]
    if len(edges) != 278:
        raise RuntimeError(f"TESSA_REPLAY_BOUNDARY_EDGE_DRIFT:{len(edges)}")
    return frozenset(vertex for edge in edges for vertex in edge)


def serialize_candidate(path: Path, candidate: CanonicalMeshCandidateIR) -> str:
    path.write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1b", type=Path, required=True)
    ap.add_argument("--normalization", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policy = quality_policy()
    original = build_original(args.t1b, args.normalization)
    protected = boundary_support_ids(original)

    # Historical V2 replay: exact notebook wiring, including the now-known overprotection bug.
    flipped, flip_report = repair_candidate_fixed_vertex_flips_v1(original, policy, max_passes=12)
    current = flipped
    collapse_total = 0
    collapse_batches = []
    while collapse_total < 256:
        current, report = repair_candidate_endpoint_collapses_v1(
            current, policy, max_collapses=min(32, 256 - collapse_total), seed_only_violating_faces=True
        )
        accepted = int(report["accepted_collapse_count"])
        collapse_total += accepted
        collapse_batches.append({"accepted": accepted, "residual": int(report["after"]["policy_violating_face_count"])})
        if int(report["after"]["policy_violating_face_count"]) == 0 or accepted == 0:
            break
    collapsed = current
    relax_total = 0
    relax_batches = []
    while relax_total < 512:
        current, report = repair_candidate_projected_relaxation_v1(
            current,
            original,
            policy,
            protected_surface_ids=set(protected),
            max_moves=min(32, 512 - relax_total),
        )
        accepted = int(report["accepted_move_count"])
        relax_total += accepted
        relax_batches.append({"accepted": accepted, "residual": int(report["after"]["policy_violating_face_count"])})
        if int(report["after"]["policy_violating_face_count"]) == 0 or accepted == 0:
            break
    v2 = current
    v2_report = report["after"]
    v2_path = args.out / "TESSA_STATIC_G3_REPAIRED_CANDIDATE_V2_REPLAY.json"
    v2_sha = serialize_candidate(v2_path, v2)

    # Exact historical endpoint court.
    if (len(v2.vertices), len(v2.faces)) != (3653, 6928):
        raise RuntimeError("TESSA_REPLAY_V2_COUNT_DRIFT")
    if int(flip_report["accepted_flip_count"]) != 72 or collapse_total != 12 or relax_total != 3:
        raise RuntimeError("TESSA_REPLAY_V2_OPERATOR_COUNT_DRIFT")
    if int(v2_report["policy_violating_face_count"]) != 2:
        raise RuntimeError("TESSA_REPLAY_V2_RESIDUAL_DRIFT")
    if abs(float(v2_report["min_angle_deg"]) - 7.129214652904797) > 1e-12:
        raise RuntimeError("TESSA_REPLAY_V2_MIN_ANGLE_DRIFT")
    if abs(float(v2_report["max_aspect_longest_over_min_altitude"]) - 9.089329469223644) > 1e-12:
        raise RuntimeError("TESSA_REPLAY_V2_ASPECT_DRIFT")
    if v2_sha != EXPECTED_V2_SHA:
        raise RuntimeError(f"TESSA_REPLAY_V2_SHA_DRIFT:{v2_sha}")

    # Historical V3 correction: same exact V2 candidate, correct operator wiring only.
    v3, v3_report = repair_candidate_projected_relaxation_v1(
        v2,
        original,
        policy,
        protected_surface_ids=frozenset(),
        max_moves=16,
    )
    q = v3_report["after"]
    v3_path = args.out / "TESSA_STATIC_G3_REPAIRED_CANDIDATE_V3_REPLAY.json"
    v3_sha = serialize_candidate(v3_path, v3)
    if int(v3_report["accepted_move_count"]) != 2:
        raise RuntimeError("TESSA_REPLAY_V3_MOVE_COUNT_DRIFT")
    if (len(v3.vertices), len(v3.faces)) != (3653, 6928):
        raise RuntimeError("TESSA_REPLAY_V3_COUNT_DRIFT")
    if int(q["policy_violating_face_count"]) != 0:
        raise RuntimeError("TESSA_REPLAY_V3_RESIDUAL")
    if abs(float(q["min_angle_deg"]) - 7.524628286281861) > 1e-12:
        raise RuntimeError("TESSA_REPLAY_V3_MIN_ANGLE_DRIFT")
    if abs(float(q["max_aspect_longest_over_min_altitude"]) - 9.089329469223644) > 1e-12:
        raise RuntimeError("TESSA_REPLAY_V3_ASPECT_DRIFT")
    if v3_sha != EXPECTED_V3_SHA:
        raise RuntimeError(f"TESSA_REPLAY_V3_SHA_DRIFT:{v3_sha}")

    result = {
        "schema": "RealSaS.TESSAKnightStaticSurvivorReplay.v1",
        "status": "PASS_EXACT_HISTORICAL_V2_TO_V3_REPLAY",
        "inputs": {
            "t1b_sha256": EXPECTED_T1B_SHA,
            "normalization_sha256": EXPECTED_NORM_SHA,
        },
        "v2": {
            "sha256": v2_sha,
            "vertices": len(v2.vertices),
            "faces": len(v2.faces),
            "accepted_flips": int(flip_report["accepted_flip_count"]),
            "accepted_collapses": collapse_total,
            "accepted_relaxations": relax_total,
            "residual": int(v2_report["policy_violating_face_count"]),
            "min_angle_deg": float(v2_report["min_angle_deg"]),
            "max_aspect": float(v2_report["max_aspect_longest_over_min_altitude"]),
        },
        "v3": {
            "sha256": v3_sha,
            "vertices": len(v3.vertices),
            "faces": len(v3.faces),
            "accepted_corrective_moves": int(v3_report["accepted_move_count"]),
            "residual": int(q["policy_violating_face_count"]),
            "min_angle_deg": float(q["min_angle_deg"]),
            "max_aspect": float(q["max_aspect_longest_over_min_altitude"]),
        },
        "claims": {
            "static_geometry_qualification_only": True,
            "threshold_relaxation": False,
            "knight_specific_exemption": False,
            "product_authority": False,
            "dynamic_motion_pass": False,
            "generalization": False,
        },
    }
    (args.out / "TESSA_KNIGHT_STATIC_SURVIVOR_REPLAY_V1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("TESSA_KNIGHT_STATIC_SURVIVOR_REPLAY=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
