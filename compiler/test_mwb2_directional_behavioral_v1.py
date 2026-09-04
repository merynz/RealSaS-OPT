from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math

import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mwb2 import build_mwb2_candidate, qualify_mwb2_mesh
from compiler.realsas_compiler_core.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.rig import qualify_skeleton_v2
from compiler.realsas_compiler_core.skin import qualify_skin
from compiler.realsas_compiler_core.substrate.iris_v2 import compile_surface_v2
from compiler.realsas_compiler_core.types import (
    ObservationEvidenceIR,
    ObservationSample,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
    SkinInfluenceProposal,
    SkinProposalIR,
)
from models.skin_field_codec.v1.codec_deformation_loss_v1 import torch_verified_lbs_v1


RESOLUTION = 1024
PLANE_DISTANCE = 2.0
SIGMA = 0.11
AREA_RATIO_MIN = 0.999999
AREA_RATIO_MAX = 1.000001
SKIN_TOL = 1e-9
DEFORMATION_RMS_MAX = 1e-7


@dataclass(frozen=True)
class Witness:
    name: str
    rows: int
    cols: int
    stride_px: int
    world_step: float
    slope: float
    curvature: float
    offset_px: tuple[float, float]


WITNESSES = (
    Witness("tilted_grid_4x4", 4, 4, 16, 0.05, 0.41421356237309503, 0.0, (0.0, 0.0)),
    Witness("curved_grid_5x4", 5, 4, 20, 0.045, 0.31, 0.18, (0.0, 0.0)),
    Witness("offset_grid_4x5", 4, 5, 24, 0.04, -0.27, 0.08, (37.0, -29.0)),
)


def _unit(v):
    a = np.asarray(v, np.float64)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n <= 1e-12:
        raise ValueError("bad vector")
    return a / n


def _camera(view: int):
    theta = math.radians(45.0 * int(view))
    f = _unit((math.cos(theta), 0.0, math.sin(theta)))
    up = np.asarray((0.0, 1.0, 0.0), np.float64)
    right = _unit(np.cross(f, up))
    return f, right, up


def _points(w: Witness):
    us = np.asarray([(c - (w.cols - 1) / 2.0) * w.world_step for c in range(w.cols)], np.float64)
    mean_u2 = float(np.mean(us * us))
    out = []
    for r in range(w.rows):
        v = (r - (w.rows - 1) / 2.0) * w.world_step
        for c, u in enumerate(us):
            x = w.slope * float(u) + w.curvature * (float(u * u) - mean_u2)
            out.append((r, c, np.asarray((x, v, float(u)), np.float64)))
    return tuple(out)


def _evidence(w: Witness) -> ObservationEvidenceIR:
    scale = float(w.stride_px) / float(w.world_step)
    cx = 511.5 + float(w.offset_px[0])
    cy = 511.5 + float(w.offset_px[1])
    samples = []
    groups = {}
    anchors = {}
    sigma = {}
    for idx, (row, col, P) in enumerate(_points(w)):
        gid = f"Q{idx:06d}:M00"
        ids = []
        for view in range(8):
            f, right, up = _camera(view)
            sx = float(np.dot(P, right))
            sy = float(np.dot(P, up))
            px = cx + scale * sx
            py = cy + scale * sy
            if not (0.0 <= px < RESOLUTION and 0.0 <= py < RESOLUTION):
                raise AssertionError(f"preregistered point left raster:{w.name}:{view}:{px}:{py}")
            origin = right * sx + up * sy - PLANE_DISTANCE * f
            depth = PLANE_DISTANCE + float(np.dot(P, f))
            oid = f"MWB2SYN:{w.name}:{gid}:V{view}"
            samples.append(ObservationSample(
                observation_id=oid,
                view_index=view,
                raster_xy=(px, py),
                ray_origin=tuple(map(float, origin)),
                ray_forward=tuple(map(float, f)),
                depth=float(depth),
                support=True,
                provenance_ref=f"MWB2_SYNTHETIC_Q_LATTICE|{w.name}|{gid}",
                validity_flags=("Q_HYPOTHESIS", "ANALYTIC_REPROJECTION"),
            ))
            ids.append(oid)
            if view == 0:
                expected_x = cx + (col - (w.cols - 1) / 2.0) * float(w.stride_px)
                expected_y = cy + (row - (w.rows - 1) / 2.0) * float(w.stride_px)
                if abs(px - expected_x) > 1e-10 or abs(py - expected_y) > 1e-10:
                    raise AssertionError("anchor raster lattice drift")
                anchors[gid] = {
                    "view_index": 0,
                    "raster_xy": (px, py),
                    "anchor_grid_xy": (
                        2.0 * (px + 0.5) / RESOLUTION - 1.0,
                        2.0 * (py + 0.5) / RESOLUTION - 1.0,
                    ),
                }
        groups[gid] = ids
        sigma[gid] = 0.0
    return ObservationEvidenceIR(
        tuple(samples),
        camera_model="KNOWN_ORTHOGRAPHIC_8VIEW",
        observation_frame="REALSAS_OBJECT_FRAME",
        evidence_version="RealSaS.ObservationEvidenceIR.v2.MWB2_SYNTHETIC_Q_LATTICE",
        metadata={
            "hypothesis_groups": groups,
            "hypothesis_anchor_raster": anchors,
            "mode_sigma": sigma,
            "raster_coordinate_system": "PIXEL_CENTER_XY",
            "resolution": RESOLUTION,
            "mechanical_authority": "FORWARD_DEPTH_SUPPORT_UNCERTAINTY_ONLY",
            "local_relation_authority": "OBSERVED_ANCHOR_RASTER_LOCALITY_ONLY",
            "production_q_lattice_semantics": True,
            "source_mesh_used": False,
            "teacher_topology_used": False,
        },
    )


def _nearest_surface_id(surface, position):
    nodes = tuple(surface.surface_nodes)
    P = np.asarray([n.P for n in nodes], np.float64)
    q = np.asarray(position, np.float64)
    return nodes[int(np.argmin(np.linalg.norm(P - q[None], axis=1)))].surface_id


def _qualified_g(surface, w: Witness):
    pts = np.asarray([n.P for n in surface.surface_nodes], np.float64)
    lo, hi = pts.min(0), pts.max(0)
    center = pts.mean(0)
    loci = (
        tuple(map(float, 0.8 * lo + 0.2 * center)),
        tuple(map(float, center)),
        tuple(map(float, 0.8 * hi + 0.2 * center)),
    )
    joints = tuple(
        SkeletonProposalJoint(
            f"MWB2:{w.name}:J:{i}",
            p,
            root_score=1.0 if i == 0 else 0.0,
            confidence=1.0,
            support_surface_ids=(_nearest_surface_id(surface, p),),
            metadata={"oracle_mechanical_g": True, "source_rig_identity_used": False},
        )
        for i, p in enumerate(loci)
    )
    edges = tuple(
        SkeletonProposalEdge(
            f"MWB2:{w.name}:E:{i-1}:{i}",
            joints[i - 1].proposal_id,
            joints[i].proposal_id,
            1.0,
            confidence=1.0,
            hard_required=True,
            reason="MWB2_DIRECTIONAL_BEHAVIORAL_ORACLE_G",
        )
        for i in range(1, len(joints))
    )
    proposal = SkeletonProposalIR(
        joints,
        edges,
        surface.geometry_lineage_hash,
        model_provenance="MWB2_DIRECTIONAL_BEHAVIORAL_ORACLE_G_V1",
        metadata={"source_mesh_used": False, "compiler_owns_canonical_ids": True},
    )
    return qualify_skeleton_v2(surface, proposal)


def _qualified_w(surface, skeleton):
    joints = tuple(skeleton.joints)
    J = np.asarray([j.position for j in joints], np.float64)
    influences = []
    for node in surface.surface_nodes:
        p = np.asarray(node.P, np.float64)
        d2 = np.sum((J - p[None, :]) ** 2, axis=1)
        a = np.exp(-d2 / (2.0 * SIGMA * SIGMA))
        a = a / a.sum()
        for joint, weight in zip(joints, a):
            influences.append(SkinInfluenceProposal(node.surface_id, joint.canonical_joint_id, float(weight)))
    proposal = SkinProposalIR(
        tuple(influences),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        model_provenance="MWB2_DIRECTIONAL_BEHAVIORAL_ORACLE_W_V1",
        metadata={"source_mesh_used": False, "teacher_topology_used": False},
    )
    return qualify_skin(surface, skeleton, proposal)


def _signed_area(a, b, c):
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _convex_hull(points):
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_area(poly):
    return 0.5 * abs(sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    ))


def _mesh_geometry_metrics(mesh):
    xy = {v.canonical_mesh_vertex_id: tuple(map(float, v.metadata["raster_xy"])) for v in mesh.vertices}
    signed = []
    incidence = {}
    for face in mesh.faces:
        if len(face) != 3:
            raise AssertionError("MWB2 prereg requires triangular directional faces")
        signed.append(_signed_area(xy[face[0]], xy[face[1]], xy[face[2]]))
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((a, b)))
            incidence[key] = incidence.get(key, 0) + 1
    hull = _convex_hull(xy.values())
    hull_area = _polygon_area(hull)
    face_area_sum = float(sum(abs(x) for x in signed))
    pos = sum(x > 1e-12 for x in signed)
    neg = sum(x < -1e-12 for x in signed)
    zero = len(signed) - pos - neg
    return {
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "face_count": len(mesh.faces),
        "euler_characteristic": len(mesh.vertices) - len(mesh.edges) + len(mesh.faces),
        "face_area_sum": face_area_sum,
        "hull_area": hull_area,
        "area_multiplicity": face_area_sum / hull_area if hull_area > 0 else float("inf"),
        "positive_winding_faces": pos,
        "negative_winding_faces": neg,
        "zero_area_faces": zero,
        "max_face_incidence_per_edge": max(incidence.values(), default=0),
    }


def _skin_and_deformation_metrics(surface, skeleton, skin, mesh, mesh_skin):
    source_rows = {r.surface_id: dict(r.influences) for r in skin.rows}
    mesh_rows = {r.canonical_mesh_vertex_id: dict(r.influences) for r in mesh_skin.rows}
    joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
    max_weight_error = 0.0
    rest_source = []
    rest_mesh = []
    source_w = []
    mesh_w = []
    for vertex in sorted(mesh.vertices, key=lambda v: v.canonical_mesh_vertex_id):
        assert vertex.support_binding.mode == "IDENTITY_SURFACE_NODE"
        assert len(vertex.support_binding.coefficients) == 1
        sid, coeff = vertex.support_binding.coefficients[0]
        assert abs(float(coeff) - 1.0) <= SKIN_TOL
        src = source_rows[sid]
        dst = mesh_rows[vertex.canonical_mesh_vertex_id]
        for jid in joint_ids:
            max_weight_error = max(max_weight_error, abs(float(src.get(jid, 0.0)) - float(dst.get(jid, 0.0))))
        node = next(n for n in surface.surface_nodes if n.surface_id == sid)
        rest_source.append(node.P)
        rest_mesh.append(vertex.P)
        source_w.append([float(src.get(jid, 0.0)) for jid in joint_ids])
        mesh_w.append([float(dst.get(jid, 0.0)) for jid in joint_ids])
    rest_source = torch.tensor([rest_source], dtype=torch.float32)
    rest_mesh = torch.tensor([rest_mesh], dtype=torch.float32)
    source_w = torch.tensor([source_w], dtype=torch.float32)
    mesh_w = torch.tensor([mesh_w], dtype=torch.float32)
    J = len(joint_ids)
    transforms = torch.eye(4, dtype=torch.float32)[None, None].repeat(1, 4, J, 1, 1)
    for ji in range(J):
        u = float(ji + 1) / float(J)
        transforms[0, 1, ji, 0, 3] = 0.08 * u
        transforms[0, 2, ji, 1, 3] = 0.07 * u
        transforms[0, 3, ji, 0, 3] = -0.04 * (ji - (J - 1) / 2.0)
        transforms[0, 3, ji, 2, 3] = 0.06 * u
    source_def = torch_verified_lbs_v1(rest_source, source_w, transforms)
    mesh_def = torch_verified_lbs_v1(rest_mesh, mesh_w, transforms)
    deformation_rms = float(torch.sqrt((source_def - mesh_def).square().mean()).item())
    simplex = max(abs(sum(row.influences[i][1] for i in range(len(row.influences))) - 1.0) for row in mesh_skin.rows)
    return {
        "mesh_skin_row_count": len(mesh_skin.rows),
        "max_weight_error": max_weight_error,
        "max_simplex_residual": float(simplex),
        "deformation_rms": deformation_rms,
    }


def _run_clean(w: Witness):
    evidence = _evidence(w)
    surface = compile_surface_v2(evidence, max_common_frame_error=1e-8)
    skeleton = _qualified_g(surface, w)
    skin = _qualified_w(surface, skeleton)
    relation_count = len(surface.local_relations)
    assert relation_count > 0, {"witness": w.name, "reason": "ZERO_NATIVE_LATTICE_RELATIONS"}
    per_view = []
    for view in range(8):
        candidate = build_mwb2_candidate(surface, view_index=view, camera_binding_hash=f"MWB2_SYN_CAMERA:{view}")
        mesh = qualify_mwb2_mesh(surface, candidate)
        mesh_skin = bind_mwb2_mesh_skin(surface, skeleton, skin, mesh)
        geom = _mesh_geometry_metrics(mesh)
        skin_metrics = _skin_and_deformation_metrics(surface, skeleton, skin, mesh, mesh_skin)
        total_visible = sum(1 for n in surface.surface_nodes if view in n.support_views and sum(1 for v, _ in n.raster_bindings if v == view) == 1)
        result = {
            "view": view,
            "visible_surface_nodes": total_visible,
            "vertex_coverage": len(mesh.vertices) / float(total_visible),
            "candidate_source_mesh_used": candidate.metadata.get("source_mesh_used"),
            "mesh_source_candidate": mesh.metadata.get("source_candidate_lineage_hash"),
            **geom,
            **skin_metrics,
        }
        per_view.append(result)
    return surface, skeleton, skin, per_view


def _assert_view_pass(result):
    assert result["candidate_source_mesh_used"] is False, result
    assert abs(result["vertex_coverage"] - 1.0) <= 1e-12, result
    assert result["zero_area_faces"] == 0, result
    assert (result["positive_winding_faces"] == 0) != (result["negative_winding_faces"] == 0), result
    assert result["max_face_incidence_per_edge"] <= 2, result
    assert result["euler_characteristic"] == 1, result
    assert AREA_RATIO_MIN <= result["area_multiplicity"] <= AREA_RATIO_MAX, result
    assert result["mesh_skin_row_count"] == result["vertex_count"], result
    assert result["max_simplex_residual"] <= SKIN_TOL, result
    assert result["max_weight_error"] <= SKIN_TOL, result
    assert result["deformation_rms"] <= DEFORMATION_RMS_MAX, result


@pytest.mark.parametrize("witness", WITNESSES, ids=lambda w: w.name)
def test_mwb2_directional_mesh_skin_behavior(witness: Witness):
    _, _, _, per_view = _run_clean(witness)
    payload = {"witness": witness.name, "views": per_view}
    print("MWB2_DIRECTIONAL_BEHAVIORAL_V1=" + json.dumps(payload, sort_keys=True))
    for result in per_view:
        _assert_view_pass(result)


def test_mwb2_unknown_cut_cannot_bridge_or_increase_coverage():
    witness = WITNESSES[0]
    surface, _, _, baseline_views = _run_clean(witness)
    baseline = baseline_views[0]
    x0 = {n.surface_id: next(xy[0] for v, xy in n.raster_bindings if v == 0) for n in surface.surface_nodes}
    xs = sorted(set(round(float(x), 9) for x in x0.values()))
    cut = 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
    changed = []
    crossing = 0
    for rel in surface.local_relations:
        a_left = x0[rel.a_surface_id] < cut
        b_left = x0[rel.b_surface_id] < cut
        if a_left != b_left:
            md = dict(rel.metadata)
            md.update({"crosses_unknown": True, "unknown_bridge": True, "diagnostic_mutation": "MWB2_CENTRAL_UNKNOWN_CUT_V1"})
            changed.append(replace(rel, relation_kind="UNKNOWN_BRIDGE", metadata=md))
            crossing += 1
        else:
            changed.append(rel)
    assert crossing > 0
    lineage = content_sha256({
        "base": surface.geometry_lineage_hash,
        "mutation": "MWB2_CENTRAL_UNKNOWN_CUT_V1",
        "relations": [r.to_dict() for r in changed],
    })
    mutated = replace(surface, local_relations=tuple(changed), geometry_lineage_hash=lineage)
    candidate = build_mwb2_candidate(mutated, view_index=0, camera_binding_hash="MWB2_SYN_CAMERA:0")
    mesh = qualify_mwb2_mesh(mutated, candidate)
    geom = _mesh_geometry_metrics(mesh)
    source_id = {v.canonical_mesh_vertex_id: v.support_binding.coefficients[0][0] for v in mesh.vertices}
    crossing_faces = 0
    for face in mesh.faces:
        sides = {x0[source_id[v]] < cut for v in face}
        if len(sides) > 1:
            crossing_faces += 1
    result = {
        "crossing_relations_mutated": crossing,
        "crossing_output_faces": crossing_faces,
        "baseline_face_count": baseline["face_count"],
        "mutated_face_count": geom["face_count"],
        "baseline_face_area_sum": baseline["face_area_sum"],
        "mutated_face_area_sum": geom["face_area_sum"],
        "source_mesh_used": candidate.metadata.get("source_mesh_used"),
    }
    print("MWB2_UNKNOWN_CUT_V1=" + json.dumps(result, sort_keys=True))
    assert crossing_faces == 0, result
    assert geom["face_count"] <= baseline["face_count"], result
    assert geom["face_area_sum"] <= baseline["face_area_sum"] + 1e-9, result
    assert candidate.metadata.get("source_mesh_used") is False, result
