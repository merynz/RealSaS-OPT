from __future__ import annotations

"""Current V4 authored-motion dynamic measurement + proof-owned runtime bake.

The same sampled deformed frames used to measure authored motion consequences are
optionally sealed into RuntimeDeployBake.v2. Exporters may project/decode those
frames but may not rerun deformation. Dynamic order/visibility tracks are kept
fail-closed until their JSON key semantics are separately typed/qualified.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json, math, struct
import numpy as np

from compiler.realsas_compiler_services.export.runtime_deploy_bake import encode_runtime_deploy_bake
from .motion_probe_geometry import (
    bbox_diag, clip_tracks, component_measurements, deform_component,
    joint_topological_order, mesh_arrays, sample_times, skinning_transforms,
)

SERVICE_ID = "RealSaS.CompilerServices.AuthoredMotionProbe.v2"
HISTORICAL_SEMANTIC_ORIGIN = "compiler/realsas_deformation/motion_proof.py"
RUNTIME_BAKE_POLICY = "PROOF_OWNED_SAMPLED_FRAMES_NO_EXPORT_REPLAY_V1"


@dataclass(frozen=True)
class AuthoredMotionProbePolicyV1:
    uniform_sample_count: int = 9
    max_edge_relative_change: float = 0.50
    min_triangle_area_ratio: float = 0.20
    max_triangle_area_ratio: float = 5.00
    max_loop_seam_normalized: float = 1.0e-5
    min_effective_motion_normalized: float = 1.0e-6
    geometry_epsilon: float = 1.0e-12
    schema_version: str = "RealSaS.AuthoredMotionProbePolicy.v1"

    def validate(self):
        if self.uniform_sample_count < 3: raise ValueError("motion probe requires >=3 uniform samples")
        if not 0.0 < self.max_edge_relative_change < 10.0: raise ValueError("invalid edge bound")
        if not 0.0 < self.min_triangle_area_ratio <= 1.0: raise ValueError("invalid min area ratio")
        if self.max_triangle_area_ratio < 1.0: raise ValueError("invalid max area ratio")
        if min(self.max_loop_seam_normalized, self.min_effective_motion_normalized) < 0.0: raise ValueError("negative motion tolerance")
        if self.geometry_epsilon <= 0.0: raise ValueError("geometry epsilon must be positive")

    @property
    def policy_hash(self):
        self.validate()
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _threshold_aliases(policy):
    return {
        "max_edge_stretch_ratio": 1.0 + float(policy.max_edge_relative_change),
        "max_area_change_ratio": max(float(policy.max_triangle_area_ratio), 1.0 / float(policy.min_triangle_area_ratio)),
        "max_return_to_rest_error01": float(policy.max_loop_seam_normalized),
    }


def _empty(policy):
    return {
        "schema": "RealSaS.AuthoredMotionMeasurement.v2", "service_id": SERVICE_ID,
        "policy_hash": policy.policy_hash, "clip_count": 0, "effective_clip_count": 0,
        "component_probe_count": 0, "sample_frame_count": 0, "nonfinite_value_count": 0,
        "degenerate_triangle_instances": 0, "max_edge_relative_change": 0.0,
        "p95_edge_relative_change": 0.0, "min_triangle_area_ratio": 1.0,
        "max_triangle_area_ratio": 1.0, "max_motion_normalized": 0.0,
        "max_loop_seam_normalized": 0.0, "max_edge_stretch_ratio": 1.0,
        "max_area_change_ratio": 1.0, "return_to_rest_error01": 0.0,
        "frozen_policy_thresholds": _threshold_aliases(policy), "clip_reports": [],
        "policy": asdict(policy), "causal_owner_attribution": "NOT_PERFORMED",
        "runtime_bake_status": "NO_CLIPS", "runtime_deploy_bakes": [],
    }


def _frame_hash(time_seconds: float, mesh_vertices_by_id: dict, render_order_by_view: dict) -> str:
    h = sha256()
    h.update(struct.pack("<d", float(time_seconds)))
    for mesh_id in sorted(mesh_vertices_by_id):
        h.update(mesh_id.encode("utf-8") + b"\0")
        for point in mesh_vertices_by_id[mesh_id]:
            h.update(struct.pack("<dd", float(point[0]), float(point[1])))
    for view_id in sorted(render_order_by_view):
        h.update(view_id.encode("utf-8") + b"\0")
        for mesh_id in render_order_by_view[view_id]:
            h.update(mesh_id.encode("utf-8") + b"\0")
    return h.hexdigest()


def _static_render_order(product) -> dict[str, list[str]]:
    result = {}
    for direction in sorted(product.directional_renderables.directions, key=lambda d: int(d.view_index)):
        visible = [c for c in direction.components if bool(getattr(c, "default_visible", True))]
        ordered = sorted(visible, key=lambda c: (int(getattr(c, "setup_order", 0)), str(c.component_id)))
        result[f"V{int(direction.view_index)}"] = [f"V{int(direction.view_index)}:{c.component_id}" for c in ordered]
    return result


def _runtime_bake_supported(product) -> tuple[bool, str]:
    if tuple(getattr(product.motion_state, "order_tracks", ()) or ()):
        return False, "ABSTAIN_DYNAMIC_ORDER_TRACK_SCHEMA_NOT_TYPED_FOR_RUNTIME_BAKE"
    if tuple(getattr(product.motion_state, "visibility_tracks", ()) or ()):
        return False, "ABSTAIN_VISIBILITY_TRACK_SCHEMA_NOT_TYPED_FOR_RUNTIME_BAKE"
    order = _static_render_order(product)
    if any(not ids for ids in order.values()):
        return False, "ABSTAIN_RUNTIME_V2_CANNOT_REPRESENT_EMPTY_VISIBLE_VIEW"
    return True, "PASS_STATIC_SETUP_ORDER"


def measure_authored_motion_v1(product, *, policy=AuthoredMotionProbePolicyV1()) -> dict:
    policy.validate()
    skeleton = product.mechanical_state.skeleton
    order, by_id = joint_topological_order(skeleton)
    clips = tuple(sorted(product.motion_state.clips, key=lambda c: c.clip_id))
    if not clips: return _empty(policy)
    components = []
    for direction in sorted(product.directional_renderables.directions, key=lambda d: d.view_index):
        for component in sorted(direction.components, key=lambda c: c.component_id):
            runtime_id = f"V{int(direction.view_index)}:{component.component_id}"
            components.append((int(direction.view_index), component.component_id, runtime_id, *mesh_arrays(component.mesh, component.mesh_skin, skeleton)))
    if not components: raise ValueError("motion proof requires qualified directional components")

    bake_supported, bake_status = _runtime_bake_supported(product)
    static_order = _static_render_order(product) if bake_supported else {}
    runtime_bakes = []
    reports=[]; edges=[]; min_area=[]; max_area=[]; motion=[]; seams=[]
    nonfinite=degenerate=frames=effective=0
    for clip in clips:
        tracks = clip_tracks(product.motion_state, clip.clip_id)
        unknown = sorted(set(tracks) - set(by_id))
        if unknown: raise ValueError(f"motion track references unknown canonical joint:{unknown[:3]}")
        times = sample_times(clip, tracks, policy.uniform_sample_count)
        transforms = np.stack([skinning_transforms(skeleton, tracks, t, order, by_id) for t in times])
        comp_reports=[]; clip_motion=clip_seam=0.0; deformed_by_runtime_id={}
        for view, cid, runtime_id, rest, weights, faces in components:
            deformed = deform_component(rest, weights, transforms)
            deformed_by_runtime_id[runtime_id] = deformed
            m = component_measurements(rest, deformed, faces, geometry_epsilon=policy.geometry_epsilon)
            seam = float(np.linalg.norm(deformed[-1]-deformed[0], axis=-1).max()/bbox_diag(rest, policy.geometry_epsilon)) if clip.loop else 0.0
            m.update({"view_index":view,"component_id":cid,"runtime_mesh_id":runtime_id,"sample_count":len(times),"loop_seam_normalized":seam})
            comp_reports.append(m); nonfinite += int(m["nonfinite_value_count"]); degenerate += int(m["degenerate_triangle_instances"])
            edges.append(float(m["max_edge_relative_change"])); min_area.append(float(m["min_triangle_area_ratio"])); max_area.append(float(m["max_triangle_area_ratio"])); motion.append(float(m["max_motion_normalized"])); seams.append(seam)
            clip_motion=max(clip_motion,float(m["max_motion_normalized"])); clip_seam=max(clip_seam,seam); frames += len(times)
        is_effective = clip_motion > policy.min_effective_motion_normalized; effective += int(is_effective)
        reports.append({"clip_id":clip.clip_id,"clip_kind":clip.clip_kind,"loop":bool(clip.loop),"duration_sec":float(clip.duration_sec),"sample_times":list(times),"track_count":len(tracks),"effective_motion":bool(is_effective),"max_motion_normalized":clip_motion,"max_loop_seam_normalized":clip_seam,"components":comp_reports})
        if bake_supported:
            baked_frames=[]
            for frame_index, time_seconds in enumerate(times):
                mesh_vertices = {
                    runtime_id: [[float(x), float(y)] for x, y in deformed[frame_index, :, :2]]
                    for runtime_id, deformed in sorted(deformed_by_runtime_id.items())
                }
                render = {view_id:list(ids) for view_id,ids in static_order.items()}
                baked_frames.append({
                    "time_seconds": float(time_seconds),
                    "mesh_vertices_by_id": mesh_vertices,
                    "render_order_by_view": render,
                    "semantic_sha256": _frame_hash(float(time_seconds), mesh_vertices, render),
                })
            fps = max(1.0, float(len(times)-1) / max(float(clip.duration_sec), policy.geometry_epsilon))
            runtime_bakes.append(encode_runtime_deploy_bake(
                clip_id=str(clip.clip_id), duration_seconds=float(clip.duration_sec), fps=fps,
                loop=bool(clip.loop), sample_times_seconds=times, frames=baked_frames,
                evaluator_semantic_version=SERVICE_ID, sampling_policy=RUNTIME_BAKE_POLICY,
            ))

    max_edge=float(max(edges,default=0.0)); min_ar=float(min(min_area,default=1.0)); max_ar=float(max(max_area,default=1.0)); max_seam=float(max(seams,default=0.0))
    finite=np.asarray([x for x in edges if math.isfinite(x)],dtype=np.float64)
    area_factor=max(max_ar,1.0/max(min_ar,policy.geometry_epsilon))
    return {
        "schema":"RealSaS.AuthoredMotionMeasurement.v2","service_id":SERVICE_ID,"policy_hash":policy.policy_hash,
        "clip_count":len(clips),"effective_clip_count":effective,"component_probe_count":len(components)*len(clips),"sample_frame_count":frames,
        "nonfinite_value_count":nonfinite,"degenerate_triangle_instances":degenerate,
        "max_edge_relative_change":max_edge,"p95_edge_relative_change":float(np.percentile(finite,95)) if finite.size else (float("inf") if edges else 0.0),
        "min_triangle_area_ratio":min_ar,"max_triangle_area_ratio":max_ar,"max_motion_normalized":float(max(motion,default=0.0)),"max_loop_seam_normalized":max_seam,
        "max_edge_stretch_ratio":1.0+max_edge,"max_area_change_ratio":area_factor,"return_to_rest_error01":max_seam,
        "frozen_policy_thresholds":_threshold_aliases(policy),"clip_reports":reports,"policy":asdict(policy),
        "causal_owner_attribution":"NOT_PERFORMED","historical_semantic_origin":HISTORICAL_SEMANTIC_ORIGIN,
        "runtime_bake_status":bake_status,"runtime_deploy_bakes":runtime_bakes,
        "runtime_export_solver_replay":False,
    }


def authored_motion_measurement_passes_v1(m: dict) -> bool:
    p=dict(m.get("policy") or {})
    required=("max_edge_relative_change","min_triangle_area_ratio","max_triangle_area_ratio","max_loop_seam_normalized","min_effective_motion_normalized")
    if any(k not in p for k in required): raise ValueError("motion measurement missing bound policy")
    return bool(
        int(m.get("clip_count",0))>0 and int(m.get("effective_clip_count",0))>0 and int(m.get("component_probe_count",0))>0
        and int(m.get("nonfinite_value_count",0))==0 and int(m.get("degenerate_triangle_instances",0))==0
        and float(m.get("max_edge_relative_change",float("inf")))<=float(p["max_edge_relative_change"])
        and float(m.get("min_triangle_area_ratio",0.0))>=float(p["min_triangle_area_ratio"])
        and float(m.get("max_triangle_area_ratio",float("inf")))<=float(p["max_triangle_area_ratio"])
        and float(m.get("max_motion_normalized",0.0))>=float(p["min_effective_motion_normalized"])
        and float(m.get("max_loop_seam_normalized",float("inf")))<=float(p["max_loop_seam_normalized"])
    )
