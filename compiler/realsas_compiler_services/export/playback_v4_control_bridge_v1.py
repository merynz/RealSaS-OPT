from __future__ import annotations

"""Consumer-first pose keyframes -> native eight-view Runtime V4.

The existing CanonicalMesh/Rig/Skin solver is the only mechanics authority.
Captured control tracks are retained for later editing; each keyframe's
baked 3D vertices are checked by re-executing the same deterministic solver.
The RuntimeV4 contract supplies independently source-qualified texture/UV/
draw-order/visibility. This bridge NEVER invents appearance or product PASS.
"""
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetBundleV1,
    ControllablePuppetSessionV1,
    PuppetRuntimeFrameV1,
    VIEW_IDS_V1,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
    deform_mesh_lbs_v1,
    solve_control_pose_v1,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    RuntimeV3FrameComposition,
    validate_frame_composition_v3,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    RuntimeV4Clip,
    RuntimeV4Frame,
    RuntimeV4PlaybackContract,
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError

SCHEMA = "RealSaS.ControllablePoseRuntimeV4Bridge.v1"


@dataclass(frozen=True)
class CapturedPuppetKeyframeV1:
    time_seconds: float
    frame: PuppetRuntimeFrameV1
    controls: tuple[ControlTransformIRV1, ...]
    composition_by_view: Mapping[str, RuntimeV3FrameComposition]
    schema_version: str = "RealSaS.CapturedPuppetKeyframe.v1"


@dataclass(frozen=True)
class PuppetControlClipBridgeReportV1:
    product_revision_id: str
    mesh_lineage_hash: str
    rig_lineage_hash: str
    skin_lineage_hash: str
    control_program_hash: str
    playback_contract_hash: str
    keyframe_times: tuple[float, ...]
    control_keyframes: tuple[tuple[ControlTransformIRV1, ...], ...]
    asset_ids: tuple[str, ...]
    runtime_qualified: bool
    authoring_hash: str
    bridge_hash: str
    schema_version: str = SCHEMA


def _copy_compositions(
    bundle: ControllablePuppetBundleV1,
    source: Mapping[str, RuntimeV3FrameComposition],
    *,
    expected_active: Mapping[str, str | None],
) -> dict[str, RuntimeV3FrameComposition]:
    if set(source) != set(VIEW_IDS_V1):
        raise QualificationError("PUPPET_V4_EIGHT_VIEW_COMPOSITION_REQUIRED")
    copies = {}
    for view in VIEW_IDS_V1:
        original = source[view]
        if original.view_id != view:
            raise QualificationError("PUPPET_V4_VIEW_COMPOSITION_ID_DRIFT")
        copy = RuntimeV3FrameComposition(
            view_id=view,
            draw_order_slot_ids=tuple(original.draw_order_slot_ids),
            active_attachment_by_slot=dict(original.active_attachment_by_slot),
            clip_intervals=tuple(original.clip_intervals),
        )
        validate_frame_composition_v3(bundle.playback, copy)
        # The studio may choose different ordering by angle, but it may not
        # fabricate a second active mechanical attachment for one view.
        if dict(copy.active_attachment_by_slot) != dict(expected_active):
            raise QualificationError("PUPPET_V4_ACTIVE_ATTACHMENT_POSE_DRIFT")
        copies[view] = copy
    return copies


def capture_puppet_pose_keyframe_v1(
    session: ControllablePuppetSessionV1,
    *,
    time_seconds: float,
    composition_by_view: Mapping[str, RuntimeV3FrameComposition],
) -> CapturedPuppetKeyframeV1:
    """Snapshot the current real FK/IK/LBS solution and editable controls."""
    if not math.isfinite(float(time_seconds)) or time_seconds < 0:
        raise QualificationError("PUPPET_V4_KEYFRAME_TIME_INVALID")
    frame = session.frame()
    controls = tuple(sorted(
        session.control_states.values(),
        key=lambda control: control.canonical_control_id,
    ))
    copies = _copy_compositions(
        session.bundle, composition_by_view,
        expected_active=frame.composition.active_attachment_by_slot,
    )
    if copies[frame.view_id].draw_order_slot_ids != frame.composition.draw_order_slot_ids:
        # A source frame can only certify its own draw order, not another
        # independently requested view. Other view orders remain explicit.
        raise QualificationError("PUPPET_V4_CAPTURED_VIEW_DRAW_ORDER_DRIFT")
    return CapturedPuppetKeyframeV1(float(time_seconds), frame, controls, copies)


def _validate_asset_bindings(
    bundle: ControllablePuppetBundleV1,
    contract: RuntimeV4PlaybackContract,
) -> dict[str, tuple[int, ...]]:
    by_attachment = {}
    expected = set(bundle.vertex_ids_by_attachment)
    if len(contract.assets) != len(expected):
        raise QualificationError("PUPPET_V4_ASSET_SET_DRIFT")
    canonical_index = {
        vertex.canonical_mesh_vertex_id: index
        for index, vertex in enumerate(bundle.mesh.vertices)
    }
    for asset in contract.assets:
        if asset.attachment_id in by_attachment or asset.attachment_id not in expected:
            raise QualificationError("PUPPET_V4_ASSET_ATTACHMENT_BINDING_DRIFT")
        ids = bundle.vertex_ids_by_attachment[asset.attachment_id]
        indices = tuple(canonical_index[value] for value in ids)
        rest = np.asarray([bundle.mesh.vertices[i].P for i in indices], dtype=np.float32)
        if rest.shape != asset.rest_xyz.shape or not np.allclose(
            rest, asset.rest_xyz, atol=1e-6, rtol=0,
        ):
            raise QualificationError("PUPPET_V4_REST_MESH_ADDRESS_DRIFT")
        variants = tuple(
            m for m in bundle.playback.meshes if m.attachment_id == asset.attachment_id
        )
        if {m.view_id for m in variants} != set(VIEW_IDS_V1):
            raise QualificationError("PUPPET_V4_MISSING_VIEW_VARIANTS")
        for variant in variants:
            if variant.slot_id != asset.slot_id or tuple(variant.triangles) != tuple(
                tuple(int(n) for n in row) for row in asset.triangles
            ):
                raise QualificationError("PUPPET_V4_CANONICAL_FACE_ADDRESS_DRIFT")
        by_attachment[asset.attachment_id] = indices
    if set(by_attachment) != expected:
        raise QualificationError("PUPPET_V4_ASSET_SET_DRIFT")
    if tuple(slot.slot_id for slot in contract.slots) != tuple(
        slot.slot_id for slot in bundle.playback.slots
    ):
        raise QualificationError("PUPPET_V4_SLOT_ORDER_DRIFT")
    return by_attachment


def build_runtime_v4_clip_from_puppet_keyframes_v1(
    *,
    bundle: ControllablePuppetBundleV1,
    contract: RuntimeV4PlaybackContract,
    captured: Sequence[CapturedPuppetKeyframeV1],
    clip_id: str,
    display_name: str,
    intent: str,
    duration_seconds: float,
    fps: float,
    loop: bool = False,
) -> tuple[RuntimeV4Clip, PuppetControlClipBridgeReportV1]:
    """Compile actual edited poses into the native renderer's single 3D pose.

    The returned clip is *unqualified* until the existing independent product
    court checks real source materials and publication authority.
    """
    bundle.validate()
    playback_hash = validate_playback_runtime_v4_contract(contract)
    attachment_indices = _validate_asset_bindings(bundle, contract)
    if not captured or len(captured) > 10000:
        raise QualificationError("PUPPET_V4_KEYFRAME_CARDINALITY")
    if not clip_id or not display_name or not intent:
        raise QualificationError("PUPPET_V4_CLIP_IDENTITY_REQUIRED")
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(float(fps)) or fps <= 0:
        raise QualificationError("PUPPET_V4_CLIP_TIMEBASE_INVALID")
    times = tuple(float(sample.time_seconds) for sample in captured)
    if not all(math.isfinite(t) for t in times) or abs(times[0]) > 1e-9 or abs(times[-1]-duration) > 1e-7:
        raise QualificationError("PUPPET_V4_KEYFRAME_ENDPOINTS_REQUIRED")
    if any(t1 <= t0 for t0,t1 in zip(times,times[1:])):
        raise QualificationError("PUPPET_V4_KEYFRAME_TIMES_NOT_STRICT")

    frames = []
    control_keys = []
    posed_hashes = []
    for sample in captured:
        if sample.schema_version != "RealSaS.CapturedPuppetKeyframe.v1":
            raise QualificationError("PUPPET_V4_CAPTURE_SCHEMA_DRIFT")
        if sample.frame.product_revision_id != bundle.product_revision_id:
            raise QualificationError("PUPPET_V4_PRODUCT_REVISION_DRIFT")
        expected_pose = solve_control_pose_v1(bundle.rig, sample.controls)
        expected_deform = deform_mesh_lbs_v1(
            bundle.mesh,bundle.rig,bundle.skin,expected_pose,
        )
        if (
            sample.frame.deformed.mesh_lineage_hash != bundle.mesh.mesh_lineage_hash
            or sample.frame.deformed.skin_lineage_hash != bundle.skin.skin_lineage_hash
            or sample.frame.deformed.rig_lineage_hash != bundle.rig.rig_lineage_hash
            or sample.frame.pose.rig_lineage_hash != bundle.rig.rig_lineage_hash
            or sample.frame.pose.control_program_hash != bundle.rig.control_program_hash
            or sample.frame.deformed.canonical_mesh_vertex_ids
            != expected_deform.canonical_mesh_vertex_ids
        ):
            raise QualificationError("PUPPET_V4_MECHANICS_LINEAGE_DRIFT")
        raw = np.asarray(sample.frame.deformed.positions,dtype=np.float64)
        proven = np.asarray(expected_deform.positions,dtype=np.float64)
        if raw.shape != proven.shape or not np.isfinite(raw).all() or not np.allclose(raw,proven,rtol=0,atol=1e-6):
            raise QualificationError("PUPPET_V4_CAPTURED_DEFORMATION_NOT_REPLAYABLE")
        composition = _copy_compositions(
            bundle,sample.composition_by_view,
            expected_active=sample.frame.composition.active_attachment_by_slot,
        )
        arrays = {}
        for asset in contract.assets:
            subset = np.ascontiguousarray(proven[attachment_indices[asset.attachment_id]],dtype=np.float32)
            arrays[asset.asset_id] = subset
            posed_hashes.append(content_sha256({
                "asset_id":asset.asset_id,"time":float(sample.time_seconds),
                "xyz":subset.tolist(),
            }))
        frames.append(RuntimeV4Frame(float(sample.time_seconds),arrays,composition))
        control_keys.append(tuple(sample.controls))
    clip = RuntimeV4Clip(
        clip_id=str(clip_id),display_name=str(display_name),intent=str(intent),
        duration_seconds=duration,fps=float(fps),loop=bool(loop),
        frames=tuple(frames),runtime_qualified=False,
    )
    validate_runtime_v4_clip(contract,clip)
    authoring_hash=content_sha256({
        "schema":"RealSaS.EditableControlKeyframes.v1",
        "product_revision":bundle.product_revision_id,
        "mesh":bundle.mesh.mesh_lineage_hash,
        "rig":bundle.rig.rig_lineage_hash,
        "skin":bundle.skin.skin_lineage_hash,
        "control_program":bundle.rig.control_program_hash,
        "times":list(times),
        "controls":[[asdict(ctrl) for ctrl in frame] for frame in control_keys],
    })
    report_hash=content_sha256({
        "schema":SCHEMA,"authoring":authoring_hash,
        "playback":playback_hash,
        "posed_frames":posed_hashes,"asset_ids":[a.asset_id for a in contract.assets],
        "runtime_qualified":False,
    })
    report=PuppetControlClipBridgeReportV1(
        product_revision_id=bundle.product_revision_id,
        mesh_lineage_hash=bundle.mesh.mesh_lineage_hash,
        rig_lineage_hash=bundle.rig.rig_lineage_hash,
        skin_lineage_hash=bundle.skin.skin_lineage_hash,
        control_program_hash=bundle.rig.control_program_hash,
        playback_contract_hash=playback_hash,
        keyframe_times=times,
        control_keyframes=tuple(control_keys),
        asset_ids=tuple(a.asset_id for a in contract.assets),
        runtime_qualified=False,
        authoring_hash=authoring_hash,
        bridge_hash=report_hash,
    )
    return clip,report


__all__=[
    "CapturedPuppetKeyframeV1","PuppetControlClipBridgeReportV1",
    "capture_puppet_pose_keyframe_v1",
    "build_runtime_v4_clip_from_puppet_keyframes_v1",
]
