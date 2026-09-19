from __future__ import annotations

"""Role-free Spine-class presentation structure over qualified mechanics.

Structural component identity comes from MechanicalPartitionIR. Rigid ownership may be
verified from exact QualifiedSkinIR, but rigid skin never implies detachability or an
object category.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

from .hashing import content_sha256
from .product_authority_v1 import (
    ComponentCarrierPolicyIR,
    MechanicalPartitionIR,
    PresentationAttachmentIR,
    PresentationDecisionEvidenceIR,
    PresentationSlotIR,
    QualifiedMeshIR,
    validate_component_carrier_policy,
    validate_mechanical_partition,
)
from .types import QualificationError

Json=dict[str,Any]


@dataclass(frozen=True)
class QualifiedPresentationStructureIR:
    slots:tuple[PresentationSlotIR,...]
    attachments:tuple[PresentationAttachmentIR,...]
    decisions:tuple[PresentationDecisionEvidenceIR,...]
    surface_binding_hash:str
    skeleton_binding_hash:str
    skin_binding_hash:str
    mesh_binding_hash:str
    partition_binding_hash:str
    carrier_policy_binding_hash:str
    product_state_binding_hash:str
    structure_lineage_hash:str
    schema_version:str="RealSaS.QualifiedPresentationStructureIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def presentation_structure_lineage_hash(value:QualifiedPresentationStructureIR)->str:
    payload=value.to_dict()
    payload.pop("structure_lineage_hash",None)
    return content_sha256(payload)


def _presentation_face_groups(mesh, component_id:str):
    vertex_component={str(v.canonical_mesh_vertex_id):str(v.component_id) for v in mesh.vertices}
    face_rows=[]
    for fi,face in enumerate(mesh.faces):
        comps={vertex_component[str(vid)] for vid in face}
        if len(comps)!=1:
            raise QualificationError("PRESENTATION_FACE_CROSSES_MECHANICAL_COMPONENT")
        if next(iter(comps))==str(component_id):
            face_rows.append((int(fi),tuple(map(str,face))))
    if not face_rows:
        raise QualificationError("PRESENTATION_COMPONENT_HAS_NO_MESH_FACE")
    edge_to_faces={}
    for fi,face in face_rows:
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            edge=tuple(sorted((a,b)))
            edge_to_faces.setdefault(edge,[]).append(fi)
    neighbors={fi:set() for fi,_ in face_rows}
    for rows in edge_to_faces.values():
        if len(rows)>1:
            for a in rows:
                for b in rows:
                    if a!=b:
                        neighbors[a].add(b)
    unseen=set(neighbors)
    groups=[]
    while unseen:
        seed=min(unseen)
        stack=[seed]
        group=[]
        unseen.remove(seed)
        while stack:
            cur=stack.pop()
            group.append(cur)
            for nxt in sorted(neighbors[cur]):
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
        groups.append(tuple(sorted(group)))
    return tuple(sorted(groups,key=lambda g:(g[0],len(g),g)))


def _component_mechanical_class(component,*,skin,threshold:float,other_max:float):
    rows={str(row.surface_id):row for row in skin.rows}
    common_owner=None
    minimum_owner=1.0
    maximum_other=0.0
    for sid in component.surface_ids:
        row=rows.get(str(sid))
        if row is None or not row.influences:
            raise QualificationError("PRESENTATION_STRUCTURE_SKIN_ROW_MISSING")
        ranked=sorted(((str(j),float(w)) for j,w in row.influences),key=lambda x:(-x[1],x[0]))
        if any((not math.isfinite(w) or w<0.0) for _,w in ranked):
            raise QualificationError("PRESENTATION_STRUCTURE_SKIN_WEIGHT_INVALID")
        owner,owner_weight=ranked[0]
        other=float(sum(w for j,w in ranked if j!=owner))
        if common_owner is None:
            common_owner=owner
        elif owner!=common_owner:
            common_owner=""
        minimum_owner=min(minimum_owner,float(owner_weight))
        maximum_other=max(maximum_other,other)
    rigid=bool(common_owner) and minimum_owner>=float(threshold) and maximum_other<=float(other_max)
    return (
        "RIGID" if rigid else "DEFORMABLE",
        str(common_owner) if rigid else "",
        float(minimum_owner),
        float(maximum_other),
    )


def _expected_structure(
    *,
    surface,
    skeleton,
    skin,
    mesh:QualifiedMeshIR,
    partition:MechanicalPartitionIR,
    carrier_policy:ComponentCarrierPolicyIR,
    product_state,
    min_rigid_owner_weight:float,
    max_rigid_other_mass:float,
):
    joint_ids={str(j.canonical_joint_id) for j in skeleton.joints}
    if skeleton.root_id not in joint_ids:
        raise QualificationError("PRESENTATION_STRUCTURE_ROOT_INVALID")
    carrier_by_component={row.component_id:row.carrier_class for row in carrier_policy.decisions}
    slots=[]; attachments=[]; decisions=[]
    ordered=tuple(sorted(partition.components,key=lambda row:row.component_id))
    setup_order=0
    for component in ordered:
        carrier=carrier_by_component[component.component_id]
        if carrier!="MESH":
            raise QualificationError(
                f"PRESENTATION_PLANAR_PROXY_NOT_IMPLEMENTED:{component.component_id}:{carrier}"
            )
        mechanical_class,rigid_owner,min_owner,max_other=_component_mechanical_class(
            component,skin=skin,threshold=min_rigid_owner_weight,other_max=max_rigid_other_mass
        )
        bone_id=rigid_owner if mechanical_class=="RIGID" else str(skeleton.root_id)
        if bone_id not in joint_ids:
            raise QualificationError("PRESENTATION_STRUCTURE_OWNER_JOINT_UNKNOWN")
        face_groups=_presentation_face_groups(mesh,component.component_id)
        for group_index,face_indices in enumerate(face_groups):
            group_hash=content_sha256({
                "mesh":mesh.mesh_lineage_hash,
                "component":component.component_id,
                "face_indices":face_indices,
            })
            slot_id="SLOT:"+content_sha256({
                "product_state":product_state.product_state_hash,
                "component_id":component.component_id,
                "presentation_group_hash":group_hash,
            })[:24]
            attachment_id="ATT:"+content_sha256({
                "product_state":product_state.product_state_hash,
                "component_id":component.component_id,
                "presentation_group_hash":group_hash,
                "carrier":carrier,
            })[:24]
            slots.append(PresentationSlotIR(
                slot_id=slot_id,
                bone_id=bone_id,
                setup_order=int(setup_order),
                default_attachment_id=attachment_id,
                keyable_channels=("ATTACHMENT","TINT","ORDER","VISIBILITY"),
                metadata={
                    "mechanical_component_id":component.component_id,
                    "presentation_group_index":int(group_index),
                    "presentation_group_hash":group_hash,
                    "mesh_face_indices":face_indices,
                    "setup_order_role":"UI_SETUP_ONLY__NOT_PHYSICAL_OCCLUSION",
                    "categorical_identity":None,
                    "detachability_authority":"UNPROVEN",
                },
            ))
            attachments.append(PresentationAttachmentIR(
                attachment_id=attachment_id,
                slot_id=slot_id,
                mechanical_component_ids=(component.component_id,),
                mechanical_class=mechanical_class,
                carrier_class=carrier,
                carrier_binding_hash=mesh.mesh_lineage_hash,
                metadata={
                    "rigid_owner_joint_id":rigid_owner,
                    "minimum_owner_weight":min_owner,
                    "maximum_other_mass":max_other,
                    "presentation_group_hash":group_hash,
                    "mesh_face_indices":face_indices,
                    "detachability_authority":"UNPROVEN",
                    "replaceable_attachment_inferred":False,
                    "categorical_identity":None,
                },
            ))
            decisions.append(PresentationDecisionEvidenceIR(
                decision_id="DEC:"+content_sha256({
                    "kind":"PRESENTATION_GROUP_BINDING",
                    "component_id":component.component_id,
                    "presentation_group_hash":group_hash,
                    "product_state":product_state.product_state_hash,
                })[:24],
                decision_kind="SLOT_BINDING",
                authority_class="MECHANICAL",
                evidence_refs=(
                    partition.partition_lineage_hash,
                    skin.skin_lineage_hash,
                    mesh.mesh_lineage_hash,
                    product_state.product_state_hash,
                    group_hash,
                ),
                metadata={
                    "component_id":component.component_id,
                    "presentation_group_index":int(group_index),
                    "presentation_group_hash":group_hash,
                    "mesh_face_indices":face_indices,
                    "mechanical_class":mechanical_class,
                    "bone_id":bone_id,
                    "group_derivation":"CONNECTED_MESH_FACE_ISLAND_WITHIN_MECHANICAL_COMPONENT_V1",
                    "categorical_recognition_used":False,
                },
            ))
            setup_order+=1
    return tuple(slots),tuple(attachments),tuple(decisions)


def validate_presentation_structure(
    value:QualifiedPresentationStructureIR,
    *,
    surface,
    skeleton,
    skin,
    mesh,
    partition,
    carrier_policy,
    product_state,
    min_rigid_owner_weight:float=0.999,
    max_rigid_other_mass:float=0.001,
)->None:
    validate_mechanical_partition(partition,surface)
    validate_component_carrier_policy(carrier_policy,partition)
    expected_bindings={
        "surface_binding_hash":surface.geometry_lineage_hash,
        "skeleton_binding_hash":skeleton.skeleton_lineage_hash,
        "skin_binding_hash":skin.skin_lineage_hash,
        "mesh_binding_hash":mesh.mesh_lineage_hash,
        "partition_binding_hash":partition.partition_lineage_hash,
        "carrier_policy_binding_hash":carrier_policy.carrier_policy_lineage_hash,
        "product_state_binding_hash":product_state.product_state_hash,
    }
    for field_name,expected in expected_bindings.items():
        if getattr(value,field_name)!=expected:
            raise QualificationError(f"PRESENTATION_STRUCTURE_BINDING_DRIFT:{field_name}")
    expected=_expected_structure(
        surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,partition=partition,
        carrier_policy=carrier_policy,product_state=product_state,
        min_rigid_owner_weight=min_rigid_owner_weight,max_rigid_other_mass=max_rigid_other_mass,
    )
    if value.slots!=expected[0] or value.attachments!=expected[1] or value.decisions!=expected[2]:
        raise QualificationError("PRESENTATION_STRUCTURE_DERIVATION_DRIFT")
    if value.structure_lineage_hash!=presentation_structure_lineage_hash(value):
        raise QualificationError("PRESENTATION_STRUCTURE_HASH_MISMATCH")


def build_presentation_structure(
    *,
    surface,
    skeleton,
    skin,
    mesh,
    partition,
    carrier_policy,
    product_state,
    min_rigid_owner_weight:float=0.999,
    max_rigid_other_mass:float=0.001,
)->QualifiedPresentationStructureIR:
    slots,attachments,decisions=_expected_structure(
        surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,partition=partition,
        carrier_policy=carrier_policy,product_state=product_state,
        min_rigid_owner_weight=min_rigid_owner_weight,max_rigid_other_mass=max_rigid_other_mass,
    )
    value=QualifiedPresentationStructureIR(
        slots,attachments,decisions,
        surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,partition.partition_lineage_hash,
        carrier_policy.carrier_policy_lineage_hash,product_state.product_state_hash,"",
        metadata={
            "producer":"ROLE_FREE_PRESENTATION_GROUPS_V2",
            "min_rigid_owner_weight":float(min_rigid_owner_weight),
            "max_rigid_other_mass":float(max_rigid_other_mass),
            "detachability_inference_forbidden":True,
            "categorical_recognition_used":False,
            "planar_proxy_supported":False,
            "structural_partition_equals_presentation_segmentation":False,
            "presentation_group_derivation":"CONNECTED_MESH_FACE_ISLAND_WITHIN_MECHANICAL_COMPONENT_V1",
            "manual_split_merge_supported_by_authoring_bundle":True,
        },
    )
    value=replace(value,structure_lineage_hash=presentation_structure_lineage_hash(value))
    validate_presentation_structure(
        value,surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,partition=partition,
        carrier_policy=carrier_policy,product_state=product_state,
        min_rigid_owner_weight=min_rigid_owner_weight,max_rigid_other_mass=max_rigid_other_mass,
    )
    return value
