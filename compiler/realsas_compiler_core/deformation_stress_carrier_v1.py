from __future__ import annotations

"""G3 local-frame numerical conditioning directly on frozen (M,G,W_M)."""

from dataclasses import replace
import numpy as np

from .carrier_mechanics_v1 import carrier_skin_matrix_v1
from .hashing import content_sha256
from .joint_frames_v2 import derive_joint_frames_post_bind_v2, frame_set_hash_v2
from .mesh.deformation_stress_v1 import G3DeformationStressReportIR
from .mesh.deformation_stress_v2 import (
    G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
    _pose_skin_matrices,
    _triangle_metrics_batch_exact,
)
from .motion_3d_v1 import apply_lbs_matrix_v1
from .product_authority_v1 import (
    validate_deformation_capability_envelope,
    validate_mesh_qualification_policy,
)
from .types import QualificationError


def run_g3_carrier_native_v1(
    carrier,
    *,
    skeleton,
    skin,
    envelope,
    cameras,
    policy,
):
    validate_mesh_qualification_policy(policy)
    validate_deformation_capability_envelope(
        envelope, known_joint_ids={j.canonical_joint_id for j in skeleton.joints}
    )
    if envelope.skeleton_lineage_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("G3_CARRIER_ENVELOPE_SKELETON_DRIFT")

    frames, frame_report = derive_joint_frames_post_bind_v2(
        skeleton, carrier_skin=skin, cameras=cameras
    )
    frames_hash = frame_set_hash_v2(
        frames, carrier_skin_lineage_hash=skin.skin_lineage_hash
    )
    rest, weights, faces, joint_ids = carrier_skin_matrix_v1(carrier, skeleton, skin)

    probes = [("REST", None, None, 0.0)]
    for jid in sorted(joint_ids):
        for axis_index, axis_name in enumerate(("X", "Y", "Z")):
            for sign in (-1.0, 1.0):
                degrees = sign * G3_LOCAL_MICRO_STRESS_ANGLE_DEG
                probes.append((f"{jid}:LOCAL_{axis_name}:{degrees:+g}", jid, axis_index, degrees))

    per_probe = []
    failures = set()
    global_min_area = float("inf")
    global_max_area = 0.0
    global_max_condition = 0.0
    global_min_edge = float("inf")
    global_max_edge = 0.0
    for probe_id, jid, axis_index, degrees in probes:
        skin_by_id = _pose_skin_matrices(
            skeleton, frames, joint_id=jid, local_axis_index=axis_index, degrees=degrees
        )
        matrices = np.stack([skin_by_id[x] for x in joint_ids], axis=0)
        posed = apply_lbs_matrix_v1(rest, weights, matrices)
        area, condition, edge_min, edge_max, smin = _triangle_metrics_batch_exact(
            rest, posed, faces
        )
        finite = (
            np.isfinite(area) & np.isfinite(condition) & np.isfinite(edge_min)
            & np.isfinite(edge_max) & np.isfinite(smin)
        )
        if not np.all(finite):
            failures.add("NONFINITE_DEFORMATION_METRIC")
        if np.any(finite):
            af, cf, emin, emax = area[finite], condition[finite], edge_min[finite], edge_max[finite]
            min_area = float(np.min(af)); max_area = float(np.max(af))
            max_condition = float(np.max(cf)); min_edge = float(np.min(emin)); max_edge = float(np.max(emax))
            if np.any(af < policy.g3_min_dynamic_area_ratio): failures.add("DYNAMIC_AREA_RATIO_BELOW_MIN")
            if np.any(af > policy.g3_max_dynamic_area_ratio): failures.add("DYNAMIC_AREA_RATIO_ABOVE_MAX")
            if np.any(cf > policy.g3_max_dynamic_condition_number): failures.add("DYNAMIC_CONDITION_NUMBER_ABOVE_MAX")
        else:
            min_area=float("inf"); max_area=0.0; max_condition=0.0; min_edge=float("inf"); max_edge=0.0
        global_min_area=min(global_min_area,min_area); global_max_area=max(global_max_area,max_area)
        global_max_condition=max(global_max_condition,max_condition)
        global_min_edge=min(global_min_edge,min_edge); global_max_edge=max(global_max_edge,max_edge)
        per_probe.append({
            "probe_id":probe_id,"joint_id":jid,"local_axis_index":axis_index,
            "rotation_degrees":degrees,"derived_joint_frame_set_hash":frames_hash,
            "minimum_area_ratio":min_area,"maximum_area_ratio":max_area,
            "maximum_condition_number":max_condition,"minimum_edge_ratio":min_edge,
            "maximum_edge_ratio":max_edge,
        })

    probe_plan_hash = content_sha256({
        "schema":"RealSaS.G3CarrierNativeLocalFrameMicroStress.v1",
        "carrier_evidence_hash":carrier.carrier_evidence_hash,
        "carrier_topology_hash":carrier.topology_hash,
        "carrier_geometry_hash":carrier.geometry_hash,
        "skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "skin_lineage_hash":skin.skin_lineage_hash,
        "derived_joint_frame_set_hash":frames_hash,
        "angle_deg":G3_LOCAL_MICRO_STRESS_ANGLE_DEG,
        "probe_ids":tuple(row["probe_id"] for row in per_probe),
    })
    value = G3DeformationStressReportIR(
        candidate_lineage_hash=carrier.candidate_mesh_binding_hash,
        surface_lineage_hash=carrier.source_surface_binding_hash,
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        envelope_lineage_hash=envelope.envelope_lineage_hash,
        axis_contract_hash=frames_hash,
        qualification_policy_hash=policy.qualification_policy_lineage_hash,
        probe_plan_hash=probe_plan_hash,
        probe_count=len(per_probe), face_count=len(faces),
        minimum_area_ratio=float(global_min_area), maximum_area_ratio=float(global_max_area),
        maximum_condition_number=float(global_max_condition), minimum_edge_ratio=float(global_min_edge),
        maximum_edge_ratio=float(global_max_edge), failure_invariants=tuple(sorted(failures)),
        passed=not failures, per_probe=tuple(per_probe), report_hash="",
        metadata={
            "measurement_role":"STAGE35_CARRIER_NATIVE_LOCAL_3D_NUMERICAL_CONDITIONING_ONLY",
            "mechanical_topology_host":"STAGE19_EXACT_CARRIER_M",
            "skin_output_domain":"MECHANICAL_CARRIER_M",
            "semantic_skin_transfer_performed":False,
            "derived_joint_frame_set_hash":frames_hash,
            "joint_frame_qualification":frame_report,
            "actual_motion_capability_claimed":False,
            "mesh_skin_product_authority_minted":False,
        },
    )
    payload=value.to_dict(); payload.pop("report_hash",None)
    return replace(value, report_hash=content_sha256(payload))


__all__=["run_g3_carrier_native_v1"]
