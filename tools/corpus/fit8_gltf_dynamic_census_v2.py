from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.corpus.fit8_dynamic_fixed_frame_presentation_v1 import (
    FRAME_SCOPE,
    compare_dynamic_presentation_to_rest_v1,
    evaluate_fixed_frame_presentation_v1,
    solve_subject_dynamic_frame_v1,
)
from tools.corpus.fit8_dynamic_mechanics_metrics_v1 import measure_dynamic_mechanical_consequence_v1
from tools.corpus.fit8_dynamic_rendered_mechanical_census_contract_v1 import DynamicPoseSampleV1
from tools.corpus.fit8_dynamic_rendered_mechanical_census_contract_v2 import DynamicSubjectCensusV2
from tools.corpus.fit8_gltf_dynamic_pose_v1 import (
    build_gltf_dynamic_source_v1,
    evaluate_authored_pose_v1,
)
from tools.corpus.fit8_static_ray_census_v1 import (
    StaticRayCensusPolicyV1,
    face_edge_components_v1,
)


SCHEMA_VERSION = "RealSaS.FIT8GLTFDynamicCensusProducer.v2"


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
class GLTFDynamicSamplingPolicyV2:
    normalized_times: tuple[float, ...] = (0.125, 0.5, 0.875)
    presentation_resolution: int = 64
    frame_scale: float = 1.12
    require_all_native_clips: bool = True
    provenance: str = "FIT8_GLTF_DYNAMIC_CENSUS_V2"

    def validate(self) -> None:
        values = tuple(float(v) for v in self.normalized_times)
        if not values or tuple(sorted(set(values))) != values:
            raise ValueError("FIT8_GLTF_DYNAMIC_V2_TIMES_NOT_STRICTLY_SORTED_UNIQUE")
        if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
            raise ValueError("FIT8_GLTF_DYNAMIC_V2_TIME_INVALID")
        if self.presentation_resolution < 32:
            raise ValueError("FIT8_GLTF_DYNAMIC_V2_RESOLUTION_TOO_SMALL")
        if not math.isfinite(float(self.frame_scale)) or self.frame_scale <= 1.0:
            raise ValueError("FIT8_GLTF_DYNAMIC_V2_FRAME_SCALE_INVALID")
        if not self.require_all_native_clips:
            raise ValueError("FIT8_GLTF_DYNAMIC_V2_ALL_CLIPS_REQUIRED")
        if not self.provenance:
            raise ValueError("FIT8_GLTF_DYNAMIC_V2_PROVENANCE_MISSING")

    def canonical_payload(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        return _canonical_json_sha256(self.canonical_payload())


def build_gltf_dynamic_census_v2(
    *,
    source_path: Path,
    subject_key: str,
    policy: GLTFDynamicSamplingPolicyV2 = GLTFDynamicSamplingPolicyV2(),
) -> tuple[DynamicSubjectCensusV2, dict]:
    policy.validate()
    if not subject_key:
        raise ValueError("FIT8_GLTF_DYNAMIC_V2_SUBJECT_KEY_MISSING")
    dynamic = build_gltf_dynamic_source_v1(Path(source_path))
    rest_vertices = np.asarray(dynamic.carrier_arrays["vertices"], dtype=np.float64)
    faces = np.asarray(dynamic.carrier_arrays["faces"], dtype=np.int64)
    component = face_edge_components_v1(faces)

    pose_rows = []
    animation_count = len(dynamic.source.doc.get("animations", []))
    for clip_index in range(animation_count):
        for normalized_time in policy.normalized_times:
            pose_rows.append(
                evaluate_authored_pose_v1(
                    dynamic,
                    clip_index=int(clip_index),
                    normalized_time=float(normalized_time),
                )
            )
    if not pose_rows:
        raise ValueError("FIT8_GLTF_DYNAMIC_V2_NO_AUTHORED_POSE_SAMPLES")

    frame = solve_subject_dynamic_frame_v1(
        rest_vertices=rest_vertices,
        posed_vertices=[row.geometric_xyz for row in pose_rows],
        frame_scale=float(policy.frame_scale),
    )
    ray_policy = StaticRayCensusPolicyV1(
        resolution=int(policy.presentation_resolution),
        frame_scale=float(policy.frame_scale),
        provenance=f"{policy.provenance}:FIXED_SUBJECT_DYNAMIC_FRAME",
    )
    rest_presentation = evaluate_fixed_frame_presentation_v1(
        vertices=rest_vertices,
        faces=faces,
        frame=frame,
        policy=ray_policy,
        face_component_id=component,
    )

    samples: list[DynamicPoseSampleV1] = []
    for pose in pose_rows:
        mechanics = measure_dynamic_mechanical_consequence_v1(
            rest_vertices=rest_vertices,
            posed_vertices=pose.geometric_xyz,
            faces=faces,
        )
        presentation = evaluate_fixed_frame_presentation_v1(
            vertices=pose.geometric_xyz,
            faces=faces,
            frame=frame,
            policy=ray_policy,
            face_component_id=component,
        )
        delta = compare_dynamic_presentation_to_rest_v1(rest_presentation, presentation)
        samples.append(
            DynamicPoseSampleV1(
                clip_name=pose.clip_name,
                clip_index=int(pose.clip_index),
                normalized_time=float(pose.normalized_time),
                authored_time=float(pose.authored_time),
                quotient_worst_separation=float(pose.quotient_worst_separation),
                edge_stretch_p99=float(mechanics.edge_stretch_p99),
                face_condition_p99=float(mechanics.face_condition_p99),
                face_area_ratio_p01=float(mechanics.face_area_ratio_p01),
                face_area_ratio_p99=float(mechanics.face_area_ratio_p99),
                face_area_ratio_min=float(mechanics.face_area_ratio_min),
                face_area_ratio_max=float(mechanics.face_area_ratio_max),
                directed_cycle_view_count=int(presentation.directed_cycle_view_count),
                mixed_pair_view_count=int(presentation.mixed_pair_view_count),
                mixed_pair_count_sum=int(presentation.mixed_pair_count_sum),
                ordered_relation_count_sum=int(presentation.ordered_relation_count_sum),
                maximum_distinct_component_stack=int(presentation.maximum_distinct_component_stack),
                new_directed_edges_vs_static=int(delta.new_directed_edges_vs_rest),
                reversed_directed_edges_vs_static=int(delta.reversed_directed_edges_vs_rest),
                strict_pair_flips_vs_static=int(delta.strict_pair_flips_vs_rest),
            )
        )

    pose_module = Path(__file__).with_name("fit8_gltf_dynamic_pose_v1.py")
    presentation_module = Path(__file__).with_name("fit8_dynamic_fixed_frame_presentation_v1.py")
    result = DynamicSubjectCensusV2(
        subject_key=subject_key,
        source_sha256=str(dynamic.carrier_manifest["source_sha256"]),
        carrier_sha256=str(dynamic.carrier_manifest["source_carrier_sha256"]),
        animation_evaluator_sha256=_file_sha256(pose_module),
        fixed_frame_presentation_producer_sha256=_file_sha256(presentation_module),
        sampling_policy_sha256=policy.sha256(),
        dynamic_frame_sha256=frame.sha256(),
        dynamic_frame_scope=FRAME_SCOPE,
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
        "dynamic_frame": frame.canonical_payload(),
        "dynamic_frame_sha256": frame.sha256(),
        "dynamic_frame_scope": FRAME_SCOPE,
        "per_pose_recenter_or_rescale": False,
        "authority": "DIAGNOSTIC_EXECUTION__NOT_FIT8_PASS",
        "claims": {
            "all_native_clips_covered": True,
            "one_common_frame_for_rest_and_all_samples": True,
            "quotient_motion_witness_present": True,
            "mechanical_tail_present": True,
            "posed_directed_order_graph_present": True,
            "fit8_pass_claimed": False,
            "product_pass_claimed": False,
        },
    }
    return result, execution


__all__ = [
    "GLTFDynamicSamplingPolicyV2",
    "SCHEMA_VERSION",
    "build_gltf_dynamic_census_v2",
]
