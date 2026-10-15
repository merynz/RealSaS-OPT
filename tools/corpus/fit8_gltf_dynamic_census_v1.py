from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.corpus.fit8_dynamic_mechanics_metrics_v1 import measure_dynamic_mechanical_consequence_v1
from tools.corpus.fit8_dynamic_rendered_mechanical_census_contract_v1 import (
    DynamicPoseSampleV1,
    DynamicSubjectCensusV1,
)
from tools.corpus.fit8_gltf_dynamic_pose_v1 import (
    animation_time_range_v1,
    build_gltf_dynamic_source_v1,
    evaluate_authored_pose_v1,
)
from tools.corpus.fit8_static_draw_order_capacity_court_v2 import evaluate_subject_static_draw_order_capacity_v2
from tools.corpus.fit8_static_ray_census_v1 import StaticRayCensusPolicyV1
from tools.corpus.fit8_static_ray_census_v3 import build_static_ray_census_v3


SCHEMA_VERSION = "RealSaS.FIT8GLTFDynamicCensusProducer.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class GLTFDynamicSamplingPolicyV1:
    normalized_times: tuple[float, ...] = (0.125, 0.5, 0.875)
    presentation_resolution: int = 64
    frame_scale: float = 1.12
    require_all_native_clips: bool = True
    provenance: str = "FIT8_GLTF_DYNAMIC_CENSUS_V1"

    def validate(self) -> None:
        if not self.normalized_times:
            raise ValueError("FIT8_GLTF_DYNAMIC_POLICY_TIMES_EMPTY")
        if tuple(sorted(set(float(v) for v in self.normalized_times))) != tuple(float(v) for v in self.normalized_times):
            raise ValueError("FIT8_GLTF_DYNAMIC_POLICY_TIMES_NOT_STRICTLY_SORTED_UNIQUE")
        if any(not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0 for v in self.normalized_times):
            raise ValueError("FIT8_GLTF_DYNAMIC_POLICY_TIME_INVALID")
        if self.presentation_resolution < 32:
            raise ValueError("FIT8_GLTF_DYNAMIC_POLICY_RESOLUTION_TOO_SMALL")
        if not math.isfinite(float(self.frame_scale)) or self.frame_scale <= 1.0:
            raise ValueError("FIT8_GLTF_DYNAMIC_POLICY_FRAME_SCALE_INVALID")
        if not self.require_all_native_clips:
            raise ValueError("FIT8_GLTF_DYNAMIC_POLICY_ALL_CLIPS_REQUIRED")
        if not self.provenance:
            raise ValueError("FIT8_GLTF_DYNAMIC_POLICY_PROVENANCE_MISSING")

    def canonical_payload(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return _canonical_json_sha256(self.canonical_payload())


def _directed_edges(summary: dict, view_index: int) -> set[tuple[int, int]]:
    row = summary["pair_evidence"][str(int(view_index))]
    return {(int(values[0]), int(values[1])) for values in row.get("ordered_pairs", [])}


def _graph_delta(static_summary: dict, posed_summary: dict) -> tuple[int, int, int]:
    new_edges = 0
    reversed_edges = 0
    strict_pair_flips = 0
    for view_index in range(8):
        base = _directed_edges(static_summary, view_index)
        posed = _directed_edges(posed_summary, view_index)
        new_edges += len(posed - base)
        reversed_edges += sum(
            (b, a) in base and (a, b) not in base
            for a, b in posed
        )
        unordered_pairs = {tuple(sorted(edge)) for edge in base | posed}
        for a, b in unordered_pairs:
            before = ((a, b) in base, (b, a) in base)
            after = ((a, b) in posed, (b, a) in posed)
            if before in {(True, False), (False, True)} and after in {(True, False), (False, True)} and before != after:
                strict_pair_flips += 1
    return int(new_edges), int(reversed_edges), int(strict_pair_flips)


def build_gltf_dynamic_census_v1(
    *,
    source_path: Path,
    subject_key: str,
    policy: GLTFDynamicSamplingPolicyV1 = GLTFDynamicSamplingPolicyV1(),
) -> tuple[DynamicSubjectCensusV1, dict]:
    policy.validate()
    if not subject_key:
        raise ValueError("FIT8_GLTF_DYNAMIC_SUBJECT_KEY_MISSING")
    dynamic = build_gltf_dynamic_source_v1(Path(source_path))
    rest_vertices = np.asarray(dynamic.carrier_arrays["vertices"], dtype=np.float64)
    faces = np.asarray(dynamic.carrier_arrays["faces"], dtype=np.int64)
    unavailable = np.full(len(faces), -1, dtype=np.int32)
    ray_policy = StaticRayCensusPolicyV1(
        resolution=int(policy.presentation_resolution),
        frame_scale=float(policy.frame_scale),
        provenance=f"{policy.provenance}:POSED_PRESENTATION",
    )
    static_summary, _ = build_static_ray_census_v3(
        subject_key=subject_key,
        vertices=rest_vertices,
        faces=faces,
        face_material_id=unavailable,
        face_owner_id=unavailable,
        policy=ray_policy,
        owner_authority_state="UNSPECIFIED",
    )

    samples: list[DynamicPoseSampleV1] = []
    animation_count = len(dynamic.source.doc.get("animations", []))
    for clip_index in range(animation_count):
        lo, hi = animation_time_range_v1(dynamic, clip_index)
        if hi < lo:
            raise ValueError("FIT8_GLTF_DYNAMIC_CLIP_RANGE_INVALID")
        for normalized_time in policy.normalized_times:
            pose = evaluate_authored_pose_v1(
                dynamic,
                clip_index=clip_index,
                normalized_time=float(normalized_time),
            )
            mechanics = measure_dynamic_mechanical_consequence_v1(
                rest_vertices=rest_vertices,
                posed_vertices=pose.geometric_xyz,
                faces=faces,
            )
            posed_summary, _ = build_static_ray_census_v3(
                subject_key=subject_key,
                vertices=pose.geometric_xyz,
                faces=faces,
                face_material_id=unavailable,
                face_owner_id=unavailable,
                policy=ray_policy,
                owner_authority_state="UNSPECIFIED",
            )
            order = evaluate_subject_static_draw_order_capacity_v2(posed_summary)
            new_edges, reversed_edges, pair_flips = _graph_delta(static_summary, posed_summary)
            samples.append(
                DynamicPoseSampleV1(
                    clip_name=pose.clip_name,
                    clip_index=int(clip_index),
                    normalized_time=float(normalized_time),
                    authored_time=float(pose.authored_time),
                    quotient_worst_separation=float(pose.quotient_worst_separation),
                    edge_stretch_p99=float(mechanics.edge_stretch_p99),
                    face_condition_p99=float(mechanics.face_condition_p99),
                    face_area_ratio_p01=float(mechanics.face_area_ratio_p01),
                    face_area_ratio_p99=float(mechanics.face_area_ratio_p99),
                    face_area_ratio_min=float(mechanics.face_area_ratio_min),
                    face_area_ratio_max=float(mechanics.face_area_ratio_max),
                    directed_cycle_view_count=int(order.directed_cycle_view_count),
                    mixed_pair_view_count=int(order.mixed_pair_view_count),
                    mixed_pair_count_sum=int(order.mixed_pair_count_sum),
                    ordered_relation_count_sum=int(order.ordered_relation_count_sum),
                    maximum_distinct_component_stack=int(order.maximum_distinct_component_stack),
                    new_directed_edges_vs_static=new_edges,
                    reversed_directed_edges_vs_static=reversed_edges,
                    strict_pair_flips_vs_static=pair_flips,
                )
            )

    pose_module = Path(__file__).with_name("fit8_gltf_dynamic_pose_v1.py")
    ray_module = Path(__file__).with_name("fit8_static_ray_census_v3.py")
    result = DynamicSubjectCensusV1(
        subject_key=subject_key,
        source_sha256=str(dynamic.carrier_manifest["source_sha256"]),
        carrier_sha256=str(dynamic.carrier_manifest["source_carrier_sha256"]),
        animation_evaluator_sha256=_file_sha256(pose_module),
        static_ray_producer_sha256=_file_sha256(ray_module),
        sampling_policy_sha256=policy.sha256(),
        mechanics_authority_state="DIAGNOSTIC",
        presentation_authority_state="DIAGNOSTIC",
        samples=tuple(samples),
        all_native_clips_covered=bool(policy.require_all_native_clips),
    )
    result.validate()
    execution = {
        "schema": SCHEMA_VERSION,
        "subject_key": subject_key,
        "native_clip_count": int(animation_count),
        "sample_count": int(len(samples)),
        "sampling_policy": policy.canonical_payload(),
        "sampling_policy_sha256": policy.sha256(),
        "static_frame_sha256": static_summary["frame_sha256"],
        "static_ray_policy_sha256": static_summary["policy_sha256"],
        "authority": "DIAGNOSTIC_LOCAL_OR_RUNNER_EXECUTION__NOT_FIT8_PASS",
        "claims": {
            "all_native_clips_covered": bool(policy.require_all_native_clips),
            "quotient_motion_witness_present": True,
            "mechanical_tail_present": True,
            "posed_directed_order_graph_present": True,
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
        },
    }
    return result, execution


__all__ = [
    "GLTFDynamicSamplingPolicyV1",
    "SCHEMA_VERSION",
    "build_gltf_dynamic_census_v1",
]
