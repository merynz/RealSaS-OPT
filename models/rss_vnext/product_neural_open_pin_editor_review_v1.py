from __future__ import annotations

"""Unqualified model socket+OPEN pin -> exact canonical ID editor review.

This is NOT a Compiler admission, not a source/native pin receipt and not an
automatic XPBD bridge. It supplies reviewable stable canonical vertex IDs,
a host-chain hypothesis and exact full-M* topology checks. Runtime activation
must pass an independent, explicit Mechanics policy or native source court.
"""
from dataclasses import dataclass
import math

import torch

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,canonical_mesh_lineage_hash_v1,validate_canonical_mesh_domain_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalRigIRV2,canonical_rig_lineage_hash_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2,canonical_skin_lineage_hash_v2,
)
from .animation_ready_mesh import RSSvNextStructuralMeshCandidate
from .attachment_face_correspondence_v1 import (
    RSSvNextSocketFaceNeuralOutputV1,
    mesh_topology_digest_v1,propose_socket_face_pairs_v1,
)
from .product_open_pin_vertex_head_v1 import (
    OpenPinVertexNeuralOutputV1,propose_open_pin_vertex_indices_v1,
    open_hem_boundary_mask_v1,
)


@dataclass(frozen=True)
class OpenPinCanonicalEditorReviewV1:
    anchor_event_index:int
    source_mstar_topology_digest:str
    canonical_mesh_lineage_hash:str
    held_canonical_face:tuple[str,str,str]|None
    holder_canonical_face:tuple[str,str,str]|None
    candidate_pin_vertex_ids:tuple[str,...]
    candidate_component_id:str|None
    suggested_host_chain_id:str|None
    state:str
    authority:str="NEURAL_REVIEW_ONLY_NOT_SOURCE_OR_COMPILER_AUTHORITY"
    source_pin_qualified:bool=False
    compiler_xpbd_activation_qualified:bool=False
    inferred_physical_parameters:bool=False


def propose_open_pin_canonical_editor_review_v1(
    *,
    student:RSSvNextStructuralMeshCandidate,
    canonical:CanonicalMeshIR,rig:CanonicalRigIRV2,
    skin:CanonicalSkinIRV2,
    face_output:RSSvNextSocketFaceNeuralOutputV1,
    pin_output:OpenPinVertexNeuralOutputV1,
    min_pin_probability:float=.85,
    max_pinned_fraction:float=.5,
)->tuple[OpenPinCanonicalEditorReviewV1,...]:
    """Strict same-face-index handoff, no remesh and no source truth minting."""
    student.validate()
    validate_canonical_mesh_domain_v1(canonical)
    n=int(student.rest_vertices_iris.shape[1])
    f=int(student.faces.shape[0])
    if (
        canonical.mesh_lineage_hash!=canonical_mesh_lineage_hash_v1(canonical)
        or canonical.metadata.get("authority")!="COMPILER_CANONICAL_MESH"
        or rig.mesh_lineage_hash!=canonical.mesh_lineage_hash
        or rig.rig_lineage_hash!=canonical_rig_lineage_hash_v2(rig)
        or rig.metadata.get("authority")!="COMPILER_CANONICAL_RIG"
        or skin.skin_lineage_hash!=canonical_skin_lineage_hash_v2(skin)
        or skin.metadata.get("authority")!="COMPILER_CANONICAL_SKIN"
        or skin.mesh_lineage_hash!=canonical.mesh_lineage_hash
        or skin.deform_structure_hash!=rig.deform_structure_hash
        or face_output.mesh_topology_digest!=mesh_topology_digest_v1(student)
        or pin_output.mesh_topology_digest!=face_output.mesh_topology_digest
        or pin_output.axis_sequence_digest!=face_output.axis_sequence_digest
        or pin_output.anchor_event_indices!=face_output.anchor_event_indices
        or not torch.equal(pin_output.vertex_component_ids,
                           student.vertex_component_ids)
        or not torch.equal(face_output.face_component_ids,
                           student.face_component_ids)
        or not torch.equal(pin_output.open_boundary_mask,
                           open_hem_boundary_mask_v1(student))
        or type(max_pinned_fraction) not in (int,float)
        or not math.isfinite(max_pinned_fraction)
        or not 0<max_pinned_fraction<1.
        or len(canonical.vertices)!=n
        or len(canonical.faces)!=f
    ):
        raise ValueError("RSS_VNEXT_PIN_EDITOR_MSTAR_OR_PARENT_DRIFT")
    # Raw product canonical lowering has explicit RSSV:{i:08d} addresses.
    # A different vertex order, edited XYZ or altered face winding MUST fail.
    by_source={v.source_candidate_vertex_id:v for v in canonical.vertices}
    if len(by_source)!=n:
        raise ValueError("RSS_VNEXT_PIN_EDITOR_CANONICAL_VERTEX_DUPLICATE")
    ids=[]
    for i in range(n):
        source=f"RSSV:{i:08d}"
        v=by_source.get(source)
        if v is None:
            raise ValueError("RSS_VNEXT_PIN_EDITOR_SOURCE_INDEX_NOT_REPLAYED")
        xyz=student.rest_vertices_iris[0,i].detach().cpu().tolist()
        if any(abs(float(a)-float(b))>1e-6
               for a,b in zip(xyz,v.P)):
            raise ValueError("RSS_VNEXT_PIN_EDITOR_REST_MSTAR_REPLAY_DRIFT")
        ids.append(v.canonical_mesh_vertex_id)
    expected_faces=tuple(
        tuple(ids[int(i)] for i in face)
        for face in student.faces.detach().cpu().tolist()
    )
    if expected_faces!=canonical.faces:
        raise ValueError("RSS_VNEXT_PIN_EDITOR_ORIENTED_FACE_REPLAY_DRIFT")
    # Strict skin and joint ownership, but no implicit numerical validation
    # makes a neural pin prediction native source truth.
    skin_by_vid={row.canonical_mesh_vertex_id:dict(row.influences)
                 for row in skin.rows}
    if len(skin_by_vid)!=n or set(skin_by_vid)!=set(ids):
        raise ValueError("RSS_VNEXT_PIN_EDITOR_SKIN_VERTEX_ADDRESS_DRIFT")
    chains={chain.canonical_chain_id:set(chain.canonical_joint_ids)
            for chain in rig.chains}
    joints={joint.canonical_joint_id for joint in rig.joints}
    if not chains or any(not members or not members<=joints
                         for members in chains.values()):
        raise ValueError("RSS_VNEXT_PIN_EDITOR_HOST_RIG_NOT_CHAIN_QUALIFIED")
    face_props=propose_socket_face_pairs_v1(face_output)
    pin_props=propose_open_pin_vertex_indices_v1(
        pin_output,min_probability=min_pin_probability,
    )
    if len(face_props)!=len(pin_props):
        raise ValueError("RSS_VNEXT_PIN_EDITOR_ANCHOR_CARDINALITY")
    output=[]
    for fprop,pprop in zip(face_props,pin_props):
        held=fprop.held_face_index
        holder=fprop.holder_face_index
        reason="REVIEW_PIN_ANCHOR_UNQUALIFIED"
        selected=()
        component=None
        host=None
        held_face=canonical.faces[held] if held is not None else None
        holder_face=canonical.faces[holder] if holder is not None else None
        if held is None or holder is None:
            reason="ABSTAIN_NO_DIRECTED_FACE_PAIR"
        elif int(student.face_component_ids[held])==int(student.face_component_ids[holder]):
            reason="ABSTAIN_NOT_INDEPENDENT_BODY_AND_OPEN_COMPONENT"
        elif pprop.state!="UNQUALIFIED_REVIEW_PIN_VERTICES":
            reason="ABSTAIN_NO_CONFIDENT_OPEN_HEM_PINS"
        else:
            group=int(student.face_component_ids[held])
            local_faces=tuple(
                i for i in range(f)
                if int(student.face_component_ids[i])==group
            )
            local_vertices={
                int(vertex) for face_idx in local_faces
                for vertex in student.faces[face_idx].tolist()
            }
            pin_ids=tuple(pprop.candidate_pin_vertex_indices)
            if (
                len(pin_ids)<2 or len(pin_ids)>=len(local_vertices)
                or len(pin_ids)/len(local_vertices)>max_pinned_fraction
                or not set(pin_ids)<=local_vertices
                or any(
                    int(student.vertex_component_ids[i])!=group
                    for i in pin_ids
                )
                or len(local_vertices)>4096 or len(local_faces)>8192
            ):
                reason="ABSTAIN_CROSS_COMPONENT_OR_OVERPINNED"
            else:
                # MIRA's canonical SKIN gives a *hypothesis* for host chain:
                # score independently on the holder face's ACTUAL skin,
                # not from a bone-name/semantic nearest rule.
                scores={}
                for chain_id,members in chains.items():
                    score=sum(
                        sum(skin_by_vid[vid].get(j,0.) for j in members)
                        for vid in holder_face
                    )/3.
                    scores[chain_id]=score
                ranked=sorted(scores.items(),key=lambda kv:(-kv[1],kv[0]))
                top=ranked[0]
                runner=ranked[1][1] if len(ranked)>1 else 0.
                supported=all(
                    sum(skin_by_vid[ids[i]].get(j,0.)
                        for j in chains[top[0]])>.05
                    for i in pin_ids
                )
                if top[1]<.55 or top[1]-runner<.15 or not supported:
                    reason="ABSTAIN_HOST_CHAIN_SKIN_NOT_SEPARABLE"
                else:
                    reason="REVIEW_READY_NEURAL_PIN_AND_HOST_HYPOTHESIS"
                    component=next(
                        v.component_id for v in canonical.vertices
                        if v.canonical_mesh_vertex_id==ids[pin_ids[0]]
                    )
                    host=top[0]
                    selected=tuple(ids[i] for i in pin_ids)
        output.append(OpenPinCanonicalEditorReviewV1(
            anchor_event_index=fprop.anchor_event_index,
            source_mstar_topology_digest=face_output.mesh_topology_digest,
            canonical_mesh_lineage_hash=canonical.mesh_lineage_hash,
            held_canonical_face=held_face,
            holder_canonical_face=holder_face,
            candidate_pin_vertex_ids=selected,
            candidate_component_id=component,
            suggested_host_chain_id=host,
            state=reason,
        ))
    return tuple(output)


__all__=[
    "OpenPinCanonicalEditorReviewV1",
    "propose_open_pin_canonical_editor_review_v1",
]
