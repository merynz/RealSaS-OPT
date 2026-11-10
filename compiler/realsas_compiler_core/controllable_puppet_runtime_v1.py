from __future__ import annotations

"""Consumer-first, 8-direction, 3D-mechanics controllable-puppet session.

Independent runtime design informed by (but does not copy) Spine 4.3's
bone/constraint update, slot/attachment state, clipping and hit-test separation.

This is a *geometry + interaction command consumer*. It does NOT render RGB
pixels, estimate a rig, invent attachment affinity, simulate secondary motion,
perform cloth collision or create source appearance. Absent capability evidence
raises QualificationError instead of quietly producing a convincing mock.
"""

from dataclasses import dataclass, field
import math
from typing import Callable, Mapping

import numpy as np

from .mechanics_control_solver_v1 import (
    ControlTransformIRV1,
    DeformedMeshIRV1,
    MechanicsPoseIRV1,
    deform_mesh_lbs_v1,
    solve_control_pose_v1,
)
from .canonical_mesh_ir_v1 import CanonicalMeshIR
from .playback_full_surface_v3 import (
    CameraProjectionV3,
    project_points_xyz_v3,
    qualify_camera_v3,
)
from .playback_runtime_v3 import (
    RuntimeV3FrameComposition,
    RuntimeV3Mesh,
    RuntimeV3PlaybackContract,
    RuntimeV3Vertex,
    validate_frame_composition_v3,
    validate_playback_runtime_v3_contract,
)
from .canonical_rig_ir_v2 import CanonicalRigIRV2
from .canonical_skin_ir_v2 import CanonicalSkinIRV2
from .types import QualificationError


VIEW_IDS_V1 = tuple(f"V{i}" for i in range(8))
AlphaSampleV1 = Callable[[str, str, float, float], float]


@dataclass(frozen=True)
class RuntimeSelectionBindingV1:
    """Persistent source-qualified address, independent of current view."""
    attachment_id: str
    canonical_component_id: str
    source_support_evidence_id: str
    control_id: str | None = None
    selectable: bool = True


@dataclass(frozen=True)
class ControllablePuppetBundleV1:
    mesh: CanonicalMeshIR
    rig: CanonicalRigIRV2
    skin: CanonicalSkinIRV2
    playback: RuntimeV3PlaybackContract
    cameras_by_view: Mapping[str, Mapping]
    vertex_ids_by_attachment: Mapping[str, tuple[str, ...]]
    selection_bindings: tuple[RuntimeSelectionBindingV1, ...]
    product_revision_id: str
    product_qualification_receipt: str
    schema_version: str = "RealSaS.ControllablePuppetBundle.v1"

    def validate(self) -> dict[str, CameraProjectionV3]:
        """Reject unaddressed topology and unqualified consumer input."""
        if self.schema_version != "RealSaS.ControllablePuppetBundle.v1":
            raise QualificationError("PUPPET_RUNTIME_SCHEMA_DRIFT")
        if not self.product_revision_id or not self.product_qualification_receipt:
            raise QualificationError("PUPPET_RUNTIME_PRODUCT_RECEIPT_REQUIRED")
        if self.rig.mesh_lineage_hash != self.mesh.mesh_lineage_hash:
            raise QualificationError("PUPPET_RUNTIME_RIG_MESH_LINEAGE_DRIFT")
        if self.skin.mesh_lineage_hash != self.mesh.mesh_lineage_hash:
            raise QualificationError("PUPPET_RUNTIME_SKIN_MESH_LINEAGE_DRIFT")
        if self.skin.deform_structure_hash != self.rig.deform_structure_hash:
            raise QualificationError("PUPPET_RUNTIME_SKIN_RIG_LINEAGE_DRIFT")
        validate_playback_runtime_v3_contract(
            self.playback, required_view_ids=VIEW_IDS_V1,
        )
        if set(self.cameras_by_view) != set(VIEW_IDS_V1):
            raise QualificationError("PUPPET_RUNTIME_EIGHT_CAMERAS_REQUIRED")
        cameras = {
            view_id: qualify_camera_v3(
                self.cameras_by_view[view_id],
                view_id=view_id,
                view_index=index,
            )
            for index, view_id in enumerate(VIEW_IDS_V1)
        }

        canonical_ids = tuple(v.canonical_mesh_vertex_id for v in self.mesh.vertices)
        all_vertex_ids = set(canonical_ids)
        if not all_vertex_ids or len(all_vertex_ids) != len(canonical_ids):
            raise QualificationError("PUPPET_RUNTIME_CANONICAL_VERTEX_IDS")
        canonical_faces = {
            frozenset(face) for face in self.mesh.faces
        }
        required_attachments = {mesh.attachment_id for mesh in self.playback.meshes}
        if set(self.vertex_ids_by_attachment) != required_attachments:
            raise QualificationError("PUPPET_RUNTIME_ATTACHMENT_VERTEX_ADDRESS_COVERAGE")
        selection = {b.attachment_id: b for b in self.selection_bindings}
        if len(selection) != len(self.selection_bindings) or set(selection) != required_attachments:
            raise QualificationError("PUPPET_RUNTIME_SELECTION_BINDING_COVERAGE")
        controls = {c.canonical_control_id for c in self.rig.controls}
        for bound in selection.values():
            if not bound.canonical_component_id or not bound.source_support_evidence_id:
                raise QualificationError("PUPPET_RUNTIME_SELECTION_SOURCE_BINDING_REQUIRED")
            if bound.control_id is not None and bound.control_id not in controls:
                raise QualificationError("PUPPET_RUNTIME_PICK_CONTROL_UNBOUND")

        # A view variant can use a subset of the canonical mesh but cannot
        # invent another mesh or reorder its vertex addressing without proof.
        for variant in self.playback.meshes:
            ids = self.vertex_ids_by_attachment[variant.attachment_id]
            if not ids or len(ids) != len(variant.vertices):
                raise QualificationError("PUPPET_RUNTIME_VIEW_VERTEX_ADDRESS_MISMATCH")
            if len(set(ids)) != len(ids) or not set(ids).issubset(all_vertex_ids):
                raise QualificationError("PUPPET_RUNTIME_VIEW_VERTEX_ID_UNKNOWN")
            for tri in variant.triangles:
                face = frozenset(ids[i] for i in tri)
                if face not in canonical_faces:
                    raise QualificationError("PUPPET_RUNTIME_VIEW_FACE_TOPOLOGY_DRIFT")
        return cameras


@dataclass(frozen=True)
class PuppetRuntimeFrameV1:
    """Projected draw-command geometry, NOT a rendered pixel/CAA image."""
    view_id: str
    meshes: tuple[RuntimeV3Mesh, ...]
    composition: RuntimeV3FrameComposition
    pose: MechanicsPoseIRV1
    deformed: DeformedMeshIRV1
    product_revision_id: str
    authority: str = "QUALIFIED_SOURCE_GEOMETRY_INTERACTION_CONSUMER"


@dataclass(frozen=True)
class PuppetPickV1:
    view_id: str
    attachment_id: str
    canonical_component_id: str
    source_support_evidence_id: str
    canonical_face_vertex_ids: tuple[str, str, str]
    barycentric: tuple[float, float, float]
    depth: float
    control_id: str | None


def _barycentric_2d(
    x: float, y: float, a: RuntimeV3Vertex, b: RuntimeV3Vertex, c: RuntimeV3Vertex,
) -> tuple[float, float, float] | None:
    denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y)
    if abs(denom) <= 1e-12:
        return None
    w0 = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / denom
    w1 = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / denom
    w2 = 1.0 - w0 - w1
    if min(w0, w1, w2) < -1e-8:
        return None
    return w0, w1, w2


class ControllablePuppetSessionV1:
    """Mutable controls and selection; frozen mesh/rig/skin/source addresses."""

    def __init__(self, bundle: ControllablePuppetBundleV1) -> None:
        self.bundle = bundle
        self.cameras = bundle.validate()
        self.view_index = 0
        self.control_states: dict[str, ControlTransformIRV1] = {}
        self.active_attachments = {
            slot.slot_id: slot.default_attachment_id
            for slot in bundle.playback.slots
        }
        self.selected: PuppetPickV1 | None = None
        self._controls = {c.canonical_control_id: c for c in bundle.rig.controls}
        self._binding = {x.attachment_id: x for x in bundle.selection_bindings}

    @property
    def view_id(self) -> str:
        return VIEW_IDS_V1[self.view_index]

    def set_view_index(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < 8:
            raise QualificationError("PUPPET_RUNTIME_VIEW_INDEX_OUT_OF_RANGE")
        self.view_index = index

    def set_control(self, transform: ControlTransformIRV1) -> None:
        if transform.canonical_control_id not in self._controls:
            raise QualificationError("PUPPET_RUNTIME_CONTROL_UNKNOWN")
        vals = tuple(transform.translation) + tuple(transform.rotation_quat_wxyz) + tuple(transform.scale)
        if not all(math.isfinite(float(v)) for v in vals):
            raise QualificationError("PUPPET_RUNTIME_CONTROL_NONFINITE")
        self.control_states[transform.canonical_control_id] = transform

    def drag_world(self, control_id: str, world_xyz: tuple[float, float, float]) -> None:
        if control_id not in self._controls:
            raise QualificationError("PUPPET_RUNTIME_CONTROL_UNKNOWN")
        if len(world_xyz) != 3 or not all(math.isfinite(float(v)) for v in world_xyz):
            raise QualificationError("PUPPET_RUNTIME_DRAG_WORLD_INVALID")
        c = self._controls[control_id]
        # The Compiler solver treats translation as a delta to declared rest.
        # Do not silently ignore parent-control transform hierarchies.
        parent = c.parent_canonical_control_id
        if parent is not None and parent in self.control_states:
            parent_state = self.control_states[parent]
            if (parent_state.translation != (0., 0., 0.)
                    or parent_state.rotation_quat_wxyz != (1., 0., 0., 0.)
                    or parent_state.scale != (1., 1., 1.)):
                raise QualificationError("PUPPET_RUNTIME_PARENT_SPACE_INVERSION_REQUIRED")
        delta = tuple(
            float(world_xyz[i]) - float(c.rest_P[i]) for i in range(3)
        )
        self.set_control(ControlTransformIRV1(control_id, translation=delta))

    def drag_screen(self, control_id: str, x: float, y: float) -> None:
        """Mouse drag in orthographic camera plane, preserving handle depth.

        This DOES NOT infer a hidden 3D target from one 2D cursor: depth is a
        deliberate 'preserve current control depth' interaction policy.
        """
        if control_id not in self._controls:
            raise QualificationError("PUPPET_RUNTIME_CONTROL_UNKNOWN")
        if not math.isfinite(x) or not math.isfinite(y):
            raise QualificationError("PUPPET_RUNTIME_SCREEN_CURSOR_NONFINITE")
        cam = self.cameras[self.view_id]
        c = self._controls[control_id]
        if c.parent_canonical_control_id in self.control_states:
            p = self.control_states[c.parent_canonical_control_id]
            if p.translation != (0., 0., 0.) or p.rotation_quat_wxyz != (1., 0., 0., 0.):
                raise QualificationError("PUPPET_RUNTIME_PARENT_SPACE_INVERSION_REQUIRED")
        old = self.control_states.get(control_id, ControlTransformIRV1(control_id))
        reference = np.asarray(c.rest_P, dtype=np.float64) + np.asarray(old.translation, dtype=np.float64)
        origin = np.asarray(cam.origin)
        right = np.asarray(cam.right)
        up = np.asarray(cam.screen_up)
        forward = np.asarray(cam.forward)
        d = float(np.dot(reference - origin, forward))
        gx = (2.0 * float(x) / cam.resolution - 1) * cam.half_extent
        gy = (1.0 - 2.0 * float(y) / cam.resolution) * cam.half_extent
        target = origin + right * gx + up * gy + forward * d
        self.drag_world(control_id, tuple(float(v) for v in target))

    def set_attachment(self, slot_id: str, attachment_id: str | None) -> None:
        slot_ids = {s.slot_id for s in self.bundle.playback.slots}
        if slot_id not in slot_ids:
            raise QualificationError("PUPPET_RUNTIME_SLOT_UNKNOWN")
        if attachment_id is not None:
            variants = [
                m for m in self.bundle.playback.meshes
                if m.attachment_id == attachment_id and m.slot_id == slot_id
            ]
            if {m.view_id for m in variants} != set(VIEW_IDS_V1):
                raise QualificationError("PUPPET_RUNTIME_ATTACHMENT_NOT_BOUND_IN_ALL_VIEWS")
        self.active_attachments[slot_id] = attachment_id

    def frame(self) -> PuppetRuntimeFrameV1:
        pose = solve_control_pose_v1(
            self.bundle.rig, tuple(self.control_states.values()),
        )
        deform = deform_mesh_lbs_v1(
            self.bundle.mesh, self.bundle.rig, self.bundle.skin, pose,
        )
        if tuple(deform.canonical_mesh_vertex_ids) != tuple(
            v.canonical_mesh_vertex_id for v in self.bundle.mesh.vertices
        ):
            raise QualificationError("PUPPET_RUNTIME_DEFORM_VERTEX_ID_DRIFT")
        frame = RuntimeV3FrameComposition(
            view_id=self.view_id,
            draw_order_slot_ids=tuple(
                s.slot_id for s in sorted(
                    self.bundle.playback.slots, key=lambda x: x.setup_order,
                )
            ),
            active_attachment_by_slot=dict(self.active_attachments),
        )
        validate_frame_composition_v3(self.bundle.playback, frame)
        camera = self.cameras[self.view_id]
        xyz = {
            vertex_id: np.asarray(pos, dtype=np.float64)
            for vertex_id, pos in zip(
                deform.canonical_mesh_vertex_ids, deform.positions,
            )
        }
        visible = []
        for variant in self.bundle.playback.meshes:
            if variant.view_id != self.view_id:
                continue
            if frame.active_attachment_by_slot[variant.slot_id] != variant.attachment_id:
                continue
            ids = self.bundle.vertex_ids_by_attachment[variant.attachment_id]
            projected = project_points_xyz_v3(
                np.stack([xyz[x] for x in ids]), camera,
            )
            verts = tuple(RuntimeV3Vertex(
                float(projected[i, 0]), float(projected[i, 1]),
                float(projected[i, 2]), original.u, original.v,
            ) for i, original in enumerate(variant.vertices))
            visible.append(RuntimeV3Mesh(
                view_id=variant.view_id,
                mesh_id=variant.mesh_id,
                slot_id=variant.slot_id,
                attachment_id=variant.attachment_id,
                attachment_kind=variant.attachment_kind,
                topology_class=variant.topology_class,
                vertices=verts,
                triangles=variant.triangles,
            ))
        return PuppetRuntimeFrameV1(
            view_id=self.view_id,
            meshes=tuple(visible),
            composition=frame,
            pose=pose,
            deformed=deform,
            product_revision_id=self.bundle.product_revision_id,
        )

    def pick(
        self, x: float, y: float, *,
        sample_alpha: AlphaSampleV1,
    ) -> PuppetPickV1 | None:
        """Source-aware click on posed 3D triangles at currently selected view.

        We require caller-provided source-qualified UV alpha sampling. No
        'everything opaque' fallback: translucent empty pixels must not grab
        control handles. Clipped pick geometry is not yet supported.
        """
        if not callable(sample_alpha) or not math.isfinite(x) or not math.isfinite(y):
            raise QualificationError("PUPPET_RUNTIME_PICK_INPUT_INVALID")
        f = self.frame()
        if f.composition.clip_intervals:
            raise QualificationError("PUPPET_RUNTIME_CLIPPED_HIT_TEST_NOT_IMPLEMENTED")
        order = {slot_id: i for i, slot_id in enumerate(f.composition.draw_order_slot_ids)}
        candidates: list[tuple[tuple[float, int, str], PuppetPickV1]] = []
        for mesh in f.meshes:
            binding = self._binding[mesh.attachment_id]
            if not binding.selectable:
                continue
            ids = self.bundle.vertex_ids_by_attachment[mesh.attachment_id]
            for tri in mesh.triangles:
                a, b, c = (mesh.vertices[k] for k in tri)
                weights = _barycentric_2d(float(x), float(y), a, b, c)
                if weights is None:
                    continue
                u = sum(w * v.u for w, v in zip(weights, (a, b, c)))
                v = sum(w * vertex.v for w, vertex in zip(weights, (a, b, c)))
                alpha = float(sample_alpha(self.view_id, mesh.attachment_id, u, v))
                if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                    raise QualificationError("PUPPET_RUNTIME_PICK_ALPHA_UNQUALIFIED")
                if alpha <= self.bundle.playback.visibility.alpha_cutout_threshold:
                    continue
                depth = sum(w * vertex.z for w, vertex in zip(weights, (a, b, c)))
                hit = PuppetPickV1(
                    view_id=self.view_id,
                    attachment_id=mesh.attachment_id,
                    canonical_component_id=binding.canonical_component_id,
                    source_support_evidence_id=binding.source_support_evidence_id,
                    canonical_face_vertex_ids=tuple(ids[k] for k in tri),
                    barycentric=weights, depth=depth,
                    control_id=binding.control_id,
                )
                candidates.append((
                    (depth, -order[mesh.slot_id], mesh.attachment_id), hit,
                ))
        self.selected = min(candidates, key=lambda pair: pair[0])[1] if candidates else None
        return self.selected


__all__ = [
    "ControllablePuppetBundleV1",
    "ControllablePuppetSessionV1",
    "PuppetRuntimeFrameV1",
    "PuppetPickV1",
    "RuntimeSelectionBindingV1",
    "VIEW_IDS_V1",
]
