from __future__ import annotations

"""Source-faithful 2D presentation optimization/proof over canonical 3D motion.

The optimizer is intentionally conservative. It does not mutate the canonical 3D
motion. It optimizes only source-view selection: target-view art is mandatory when
observed, otherwise the nearest observed one of the eight exact directions is used.
Physical visibility remains the canonical 3D z-buffer; exact-depth ties use the
qualified presentation policy. No pixels are generated or blended.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .camera_authority_v1 import qualified_camera_set_hash
from .hashing import content_sha256
from .motion_dynamic_proof_v1 import qualified_dynamic_motion_hash
from .product_appearance_v1 import (
    _face_donor_candidates,
    _select_donor_view,
    qualified_appearance_set_hash,
)
from .product_composition_v1 import composition_set_hash
from .product_authority_v1 import qualified_mesh_lineage_hash
from .types import QualificationError

Json=dict[str,Any]


@dataclass(frozen=True)
class MotionPresentationViewProofIR:
    view_index:int
    face_count:int
    direct_source_face_count:int
    cross_view_source_face_count:int
    maximum_cross_view_direction_steps:int
    mean_cross_view_direction_steps:float
    donor_histogram:tuple[tuple[int,int],...]
    view_proof_hash:str
    schema_version:str="RealSaS.MotionPresentationViewProofIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedMotionPresentationIR:
    dynamic_motion_binding_hash:str
    mesh_binding_hash:str
    appearance_binding_hash:str
    composition_binding_hash:str
    camera_set_binding_hash:str
    view_proofs:tuple[MotionPresentationViewProofIR,...]
    qualification_report:Json
    motion_presentation_hash:str
    schema_version:str="RealSaS.QualifiedMotionPresentationIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def _hash_without(value,field):
    payload=value.to_dict(); payload.pop(field,None)
    return content_sha256(payload)


def motion_presentation_view_hash(value):
    return _hash_without(value,"view_proof_hash")


def qualified_motion_presentation_hash(value):
    return _hash_without(value,"motion_presentation_hash")


def _circular_distance(a,b):
    raw=abs(int(a)-int(b))%8
    return min(raw,8-raw)


def build_qualified_motion_presentation_v1(
    *,
    dynamic,
    mesh,
    surface,
    appearance,
    composition,
    camera_set,
):
    if dynamic.dynamic_motion_hash!=qualified_dynamic_motion_hash(dynamic):
        raise QualificationError("MOTION_PRESENTATION_DYNAMIC_HASH_DRIFT")
    if mesh.mesh_lineage_hash!=qualified_mesh_lineage_hash(mesh):
        raise QualificationError("MOTION_PRESENTATION_MESH_HASH_DRIFT")
    if appearance.appearance_set_hash!=qualified_appearance_set_hash(appearance):
        raise QualificationError("MOTION_PRESENTATION_APPEARANCE_HASH_DRIFT")
    if composition.composition_set_hash!=composition_set_hash(composition):
        raise QualificationError("MOTION_PRESENTATION_COMPOSITION_HASH_DRIFT")
    if camera_set.camera_set_hash!=qualified_camera_set_hash(camera_set):
        raise QualificationError("MOTION_PRESENTATION_CAMERA_HASH_DRIFT")
    if dynamic.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("MOTION_PRESENTATION_DYNAMIC_MESH_DRIFT")
    if appearance.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("MOTION_PRESENTATION_APPEARANCE_MESH_DRIFT")
    if composition.mesh_binding_hash!=mesh.mesh_lineage_hash:
        raise QualificationError("MOTION_PRESENTATION_COMPOSITION_MESH_DRIFT")
    report=dict(dynamic.qualification_report or {})
    if not bool(report.get("dynamic_proof_passed",False)):
        raise QualificationError("MOTION_PRESENTATION_REQUIRES_BASELINE_DYNAMIC_PASS")
    if not bool(report.get("truly_unseen_dynamic_exposure_passed",False)):
        raise QualificationError("MOTION_PRESENTATION_REQUIRES_ZERO_UNSEEN_EXPOSURE")

    nodes={str(node.surface_id):node for node in surface.surface_nodes}
    vertices={str(vertex.canonical_mesh_vertex_id):vertex for vertex in mesh.vertices}
    bindings={int(row.target_view_index):row for row in appearance.bindings}
    composition_rows={int(row.view_index):row for row in composition.views}
    if set(bindings)!=set(range(8)) or set(composition_rows)!=set(range(8)):
        raise QualificationError("MOTION_PRESENTATION_REQUIRES_EXACT_8_VIEWS")

    proofs=[]
    total_cross=0
    total_faces=0
    for view_index in range(8):
        binding=bindings[view_index]
        corner={(int(c.face_index),int(c.corner_index)):c for c in binding.corner_bindings}
        direct=0; cross=0; distances=[]; histogram={}
        for face_index,face in enumerate(mesh.faces):
            donors={int(corner[(face_index,ci)].donor_view_index) for ci in range(3)}
            if len(donors)!=1:
                raise QualificationError("MOTION_PRESENTATION_FACE_DONOR_NOT_UNIFORM")
            donor=next(iter(donors))
            candidates=_face_donor_candidates(face,vertices,nodes)
            expected=_select_donor_view(view_index,candidates)
            if expected is None or donor!=int(expected):
                raise QualificationError("MOTION_PRESENTATION_DONOR_NOT_NEAREST_OBSERVED")
            histogram[donor]=histogram.get(donor,0)+1
            if donor==view_index:
                direct+=1
            else:
                cross+=1
                distances.append(_circular_distance(view_index,donor))
        comp=composition_rows[view_index]
        if comp.physical_occlusion_rule!="CANONICAL_Z_BUFFER_VISIBLE_OWNER_V1":
            raise QualificationError("MOTION_PRESENTATION_OCCLUSION_POLICY_DRIFT")
        if comp.equal_depth_tiebreak!="SOURCE_VISIBILITY_THEN_SLOT_ORDER_THEN_STABLE_FACE_KEY_V3":
            raise QualificationError("MOTION_PRESENTATION_EQUAL_DEPTH_POLICY_DRIFT")
        if comp.slot_order_role!="PRESENTATION_EQUAL_DEPTH_TIEBREAK_ONLY":
            raise QualificationError("MOTION_PRESENTATION_SLOT_ORDER_ROLE_DRIFT")
        proof=MotionPresentationViewProofIR(
            view_index=view_index,
            face_count=len(mesh.faces),
            direct_source_face_count=direct,
            cross_view_source_face_count=cross,
            maximum_cross_view_direction_steps=max(distances,default=0),
            mean_cross_view_direction_steps=(
                float(sum(distances))/float(len(distances)) if distances else 0.0
            ),
            donor_histogram=tuple(sorted((int(k),int(v)) for k,v in histogram.items())),
            view_proof_hash="",
            metadata={
                "donor_policy":"TARGET_VIEW_ELSE_NEAREST_OBSERVED_8_DIRECTION_V1",
                "cross_view_blending":False,
                "generated_pixels":False,
            },
        )
        proofs.append(replace(proof,view_proof_hash=motion_presentation_view_hash(proof)))
        total_cross+=cross; total_faces+=len(mesh.faces)

    value=QualifiedMotionPresentationIR(
        dynamic_motion_binding_hash=dynamic.dynamic_motion_hash,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        appearance_binding_hash=appearance.appearance_set_hash,
        composition_binding_hash=composition.composition_set_hash,
        camera_set_binding_hash=camera_set.camera_set_hash,
        view_proofs=tuple(proofs),
        qualification_report={
            "status":"PASS_SOURCE_FAITHFUL_2D_PRESENTATION",
            "baseline_dynamic_motion_passed_before_presentation":True,
            "canonical_motion_mutated":False,
            "per_view_motion_solution_used":False,
            "source_view_selection_optimized":True,
            "target_view_preferred_when_observed":True,
            "nearest_observed_direction_used_for_fallback":True,
            "cross_view_color_blending":False,
            "generated_appearance_used":False,
            "canonical_3d_zbuffer_physical_visibility":True,
            "equal_depth_source_then_slot_order":True,
            "truly_unseen_exposed_pixel_budget":0,
        },
        motion_presentation_hash="",
        metadata={
            "optimization_domain":"PRESENTATION_ONLY__NO_POSE_MUTATION",
            "motion_adjustment":"NONE_BASELINE_V1",
            "source_direction_count":8,
            "total_face_view_cells":total_faces*8 if total_faces else 0,
            "total_cross_view_face_cells":total_cross,
            "future_pose_exaggeration_policy":"SEPARATE_LEARNED_OR_ARTIST_EDIT_LAYER_ONLY",
        },
    )
    return replace(value,motion_presentation_hash=qualified_motion_presentation_hash(value))


def qualified_motion_presentation_from_dict(payload):
    if str(payload.get("schema_version") or payload.get("schema") or "")!="RealSaS.QualifiedMotionPresentationIR.v1":
        raise ValueError("MOTION_PRESENTATION_SCHEMA_MISMATCH")
    rows=[]
    for raw in payload.get("view_proofs") or ():
        row=MotionPresentationViewProofIR(
            int(raw["view_index"]),int(raw["face_count"]),int(raw["direct_source_face_count"]),
            int(raw["cross_view_source_face_count"]),int(raw["maximum_cross_view_direction_steps"]),
            float(raw["mean_cross_view_direction_steps"]),
            tuple((int(a),int(b)) for a,b in (raw.get("donor_histogram") or ())),
            str(raw["view_proof_hash"]),
            schema_version=str(raw.get("schema_version") or "RealSaS.MotionPresentationViewProofIR.v1"),
            metadata=dict(raw.get("metadata") or {}),
        )
        if row.view_proof_hash!=motion_presentation_view_hash(row):
            raise ValueError("MOTION_PRESENTATION_VIEW_HASH_MISMATCH")
        rows.append(row)
    value=QualifiedMotionPresentationIR(
        str(payload["dynamic_motion_binding_hash"]),str(payload["mesh_binding_hash"]),
        str(payload["appearance_binding_hash"]),str(payload["composition_binding_hash"]),
        str(payload["camera_set_binding_hash"]),tuple(rows),
        dict(payload.get("qualification_report") or {}),str(payload["motion_presentation_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedMotionPresentationIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.motion_presentation_hash!=qualified_motion_presentation_hash(value):
        raise ValueError("MOTION_PRESENTATION_HASH_MISMATCH")
    return value
