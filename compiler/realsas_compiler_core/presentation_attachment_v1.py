"""Explicit target-art attachments, independent of motion-source equipment."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .hashing import content_sha256
from .types import QualificationError


def qualify_target_attachments(contract, *, vertices, faces, carrier_basis_sha256,
                               skeleton_sha256, canonical_to_raw):
    if (contract.get("schema") != "RealSaS.TargetPresentationAttachments.v1"
            or contract.get("carrier_basis_sha256") != carrier_basis_sha256
            or contract.get("skeleton_sha256") != skeleton_sha256):
        raise QualificationError("PRESENTATION_ATTACHMENT_AUTHORITY_DRIFT")
    f = np.asarray(faces, dtype=np.int64)
    v = np.asarray(vertices, dtype=np.float64)
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    graph = coo_matrix((np.ones(2*len(edges)),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(v), len(v)))
    count, components = connected_components(graph, directed=False)
    owners = np.zeros(len(v), dtype=np.int32)
    rows, names = [], set()
    for ordinal, item in enumerate(contract.get("attachments", ()), 1):
        name = str(item["attachment_id"])
        selected = tuple(map(int, item["component_indices"]))
        joint = str(item["target_slot_canonical_joint_id"])
        if (not name or name in names or not selected or len(set(selected)) != len(selected)
                or any(c < 0 or c >= count for c in selected) or joint not in canonical_to_raw):
            raise QualificationError("PRESENTATION_ATTACHMENT_SELECTOR_INVALID")
        names.add(name)
        mask = np.isin(components, selected)
        if np.any(owners[mask] != 0):
            raise QualificationError("PRESENTATION_ATTACHMENT_COMPONENT_OWNERSHIP_CONFLICT")
        owners[mask] = ordinal
        rows.append({"attachment_id": name, "attachment_index": ordinal,
            "target_slot_canonical_joint_id": joint, "target_slot_raw_index_fit_only": int(canonical_to_raw[joint]),
            "component_indices": list(selected), "vertex_indices": np.flatnonzero(mask).tolist(),
            "rest_bounds_source_frame": [v[mask].min(axis=0).tolist(), v[mask].max(axis=0).tolist()],
            "local_placement_authority": "EXACT_TARGET_SOURCE_REST_FRAME",
            "source_motion_attachment_geometry_required": False})
    if np.any(owners[f] != owners[f[:, :1]]):
        raise QualificationError("PRESENTATION_ATTACHMENT_CROSSES_CANONICAL_FACE")
    result = {"schema": "RealSaS.QualifiedTargetPresentationAttachments.v1", "attachments": rows,
        "carrier_basis_sha256": carrier_basis_sha256, "skeleton_sha256": skeleton_sha256,
        "source_motion_equipment_presence_is_authority": False, "mechanical_skin_modified": False}
    result["attachment_hash"] = content_sha256(result)
    return owners, result


def visual_face_attachment_owners(binding, faces, mechanical_owners):
    anchors = np.asarray(binding["anchor_vertex"], dtype=np.int64)
    ancestry = np.asarray(binding["anchor_mechanical_vertices"], dtype=np.int64)
    domain = np.asarray(binding["domain_id"], dtype=np.int32)
    canonical = np.asarray(mechanical_owners, dtype=np.int32)[ancestry]
    if np.any(canonical != canonical[:, :1]):
        raise QualificationError("PRESENTATION_ATTACHMENT_ANCHOR_OWNERSHIP_CONFLICT")
    vertex_owner = np.full(len(domain), -1, dtype=np.int32)
    for chart in np.unique(domain):
        owners = np.unique(canonical[domain[anchors] == chart, 0])
        if len(owners) != 1:
            raise QualificationError("PRESENTATION_ATTACHMENT_DOMAIN_OWNERSHIP_AMBIGUOUS")
        vertex_owner[domain == chart] = owners[0]
    tri = np.asarray(faces, dtype=np.int64)
    if np.any(vertex_owner[tri] != vertex_owner[tri[:, :1]]):
        raise QualificationError("PRESENTATION_ATTACHMENT_VISUAL_FACE_MIXED")
    return vertex_owner[tri[:, 0]]


def qualify_body_motion_preset(contract, *, clip_hashes, target_attachments):
    if (contract.get("schema") != "RealSaS.BodyMotionPreset.v1"
            or contract.get("semantic_scope") != "BODY_KINEMATIC_DELTAS_ONLY"
            or contract.get("clip_sha256") != list(clip_hashes)
            or contract.get("target_attachment_geometry_required") is not False
            or contract.get("source_attachment_geometry_required") is not False):
        raise QualificationError("PRESENTATION_MOTION_PRESET_CONTRACT_INVALID")
    result = {**contract, "optional_target_attachments": [r["attachment_id"] for r in target_attachments["attachments"]],
              "attachment_local_placement_authority": "TARGET_SOURCE_REST_FRAME",
              "frame0_is_rest_pose": False, "sealed_retarget_semantics_unchanged": True}
    result["preset_hash"] = content_sha256(result)
    return result


def attachment_frame0_report(attachments, witness, clips):
    rows = []
    for attachment in attachments["attachments"]:
        selected = np.asarray(attachment["vertex_indices"], dtype=int)
        rest = witness["vertices"][selected]
        for clip in clips:
            prefix = clip["array_prefix"]
            posed = witness[f"{prefix}_canonical_xyz"][0, selected]
            rows.append({"attachment_id": attachment["attachment_id"], "clip_id": clip["clip_id"],
                "frame_index": 0, "time_seconds": float(witness[f"{prefix}_times"][0]),
                "rest_centroid_source_frame": rest.mean(axis=0).tolist(),
                "frame0_centroid_source_frame": posed.mean(axis=0).tolist(),
                "rest_to_frame0_centroid_displacement": float(np.linalg.norm(posed.mean(axis=0)-rest.mean(axis=0))),
                "target_slot_skin_matrix_source": witness[f"{prefix}_skin_matrices_source"][0,
                    attachment["target_slot_raw_index_fit_only"]].tolist(),
                "source_unarmed_hand_centroid_is_target_prop_position_authority": False})
    return {"schema": "RealSaS.TargetAttachmentFrame0Report.v1", "frames": rows,
            "scope": "EXACT_SEALED_MOTION_WITNESS__NO_RETARGET_OR_SKIN_CHANGE", "product_authority": False}
