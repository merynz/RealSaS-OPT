from __future__ import annotations

"""One-checkpoint / two-observation RSS.vNext inference evidence transport.

Never materializes canonical Mesh/Rig/Skin or ProductRevision authority.
Carries the original two-view pixels, original vision tokens, shared triplanes,
proposed indexed mesh, semantic/graphics/affordance evidence and learned
control events + exact-index skin to the Compiler in ONE immutable envelope.

The Compiler consumes candidate data only after independently verifying
release and source/artifact lineage. Unknown and absent facts stay absent.
"""

from dataclasses import dataclass
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from typing import Any, Mapping
import math

import numpy as np
import torch

from .unified_character_model import RSSvNextUnifiedCharacterModel
from models.axis.rig_program_model_v6 import RigTokenKindV6


SCHEMA = "RealSaS.RSSvNext.InferenceEvidenceBundle.v1"
AUTHORITY = "UNQUALIFIED_MODEL_EVIDENCE_ONLY"
_ALLOWED_STATUS = ("ABSTAIN", "MESH_CANDIDATE", "CONTROL_SKIN_CANDIDATE")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _is_sha(value: str) -> bool:
    return (len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def _numeric(name: str, value: torch.Tensor, *,
             max_elements: int = 25_000_000) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"RSS_VNEXT_CANDIDATE_EXPECTED_TENSOR:{name}")
    if value.numel() > max_elements or value.numel() == 0:
        raise ValueError(f"RSS_VNEXT_CANDIDATE_ARRAY_ELEMENT_BUDGET:{name}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"RSS_VNEXT_CANDIDATE_NONFINITE:{name}")
    if value.dtype in (torch.int32, torch.int64, torch.long):
        return np.asarray(value.detach().cpu().numpy(), dtype=np.int64)
    if value.dtype == torch.bool:
        return np.asarray(value.detach().cpu().numpy(), dtype=np.uint8)
    return np.asarray(value.detach().float().cpu().numpy(), dtype=np.float32)


@dataclass(frozen=True)
class RSSvNextInferenceEvidenceBundleV1:
    manifest: dict[str, Any]
    array_archive_npz: bytes

    def validate(self) -> None:
        m = self.manifest
        if m.get("schema_version") != SCHEMA or m.get("authority") != AUTHORITY:
            raise ValueError("RSS_VNEXT_CANDIDATE_AUTHORITY_DRIFT")
        if m.get("observed_view_count") != 2:
            raise ValueError("RSS_VNEXT_CANDIDATE_SOURCE_VIEWS_NOT_TWO")
        if (m.get("status") not in _ALLOWED_STATUS or
                m.get("product_revision_id") is not None):
            raise ValueError("RSS_VNEXT_CANDIDATE_PRODUCT_AUTHORITY_FORBIDDEN")
        if m.get("model_architecture_id") != RSSvNextUnifiedCharacterModel.architecture_id:
            raise ValueError("RSS_VNEXT_CANDIDATE_MODEL_ARCHITECTURE_DRIFT")
        src = m.get("source_image_sha256")
        if (not isinstance(src, list) or len(src) != 2
                or not all(isinstance(x, str) and _is_sha(x) for x in src)
                or src[0] == src[1]):
            raise ValueError("RSS_VNEXT_CANDIDATE_SOURCE_HASHES_INVALID")
        if not _is_sha(str(m.get("checkpoint_sha256", ""))):
            raise ValueError("RSS_VNEXT_CANDIDATE_CHECKPOINT_UNBOUND")
        if _sha(self.array_archive_npz) != m.get("array_archive_sha256"):
            raise ValueError("RSS_VNEXT_CANDIDATE_ARRAY_ARCHIVE_DRIFT")
        expected_specs = m.get("arrays")
        if not isinstance(expected_specs, dict) or not expected_specs:
            raise ValueError("RSS_VNEXT_CANDIDATE_ARRAY_SPECS_REQUIRED")
        with np.load(BytesIO(self.array_archive_npz), allow_pickle=False) as archive:
            if set(archive.files) != set(expected_specs):
                raise ValueError("RSS_VNEXT_CANDIDATE_ARRAY_SET_DRIFT")
            required = {"source_rgba", "source_camera_13d", "source_image_tokens",
                        "shared_scene_triplanes"}
            if not required.issubset(archive.files):
                raise ValueError("RSS_VNEXT_CANDIDATE_SHARED_SOURCE_LOST")
            if archive["source_rgba"].ndim != 5 or archive["source_rgba"].shape[1:3] != (2, 4):
                raise ValueError("RSS_VNEXT_CANDIDATE_FRONT_BACK_RGBA_DRIFT")
            if archive["source_camera_13d"].shape != (1, 2, 13):
                raise ValueError("RSS_VNEXT_CANDIDATE_FRONT_BACK_CAMERA_DRIFT")
            if archive["source_image_tokens"].shape[:2] != (1, 2):
                raise ValueError("RSS_VNEXT_CANDIDATE_VISION_TOKEN_DRIFT")
            if archive["shared_scene_triplanes"].shape[:2] != (1, 3):
                raise ValueError("RSS_VNEXT_CANDIDATE_SHARED_CARRIER_DRIFT")
            for key, spec in expected_specs.items():
                array = archive[key]
                if list(array.shape) != spec.get("shape") or str(array.dtype) != spec.get("dtype"):
                    raise ValueError(f"RSS_VNEXT_CANDIDATE_ARRAY_SHAPE_DTYPE:{key}")
                if array.dtype.kind not in ("f", "i", "u") or not np.isfinite(array).all():
                    raise ValueError(f"RSS_VNEXT_CANDIDATE_ARRAY_NONFINITE:{key}")
                if _sha(array.tobytes(order="C")) != spec.get("sha256"):
                    raise ValueError(f"RSS_VNEXT_CANDIDATE_ARRAY_BYTES_DRIFT:{key}")
            if m["status"] != "ABSTAIN":
                if not {"mesh_rest_xyz", "mesh_faces", "mesh_vertex_component",
                        "mesh_local_h", "mesh_normals", "mesh_role_logits",
                        "mesh_graphics_logits", "mesh_affordance_logits"}.issubset(archive.files):
                    raise ValueError("RSS_VNEXT_CANDIDATE_MESH_EVIDENCE_INCOMPLETE")
                mesh = archive["mesh_rest_xyz"]
                tri = archive["mesh_faces"]
                if (mesh.ndim != 3 or mesh.shape[0] != 1 or mesh.shape[-1] != 3
                        or tri.ndim != 2 or tri.shape[-1] != 3 or not len(tri)
                        or int(tri.min()) < 0 or int(tri.max()) >= mesh.shape[1]):
                    raise ValueError("RSS_VNEXT_CANDIDATE_INDEXED_MESH_INVALID")
                if archive["mesh_local_h"].shape[:2] != mesh.shape[:2]:
                    raise ValueError("RSS_VNEXT_CANDIDATE_MESH_H_IDENTITY")
                if archive["mesh_vertex_component"].shape != (mesh.shape[1],):
                    raise ValueError("RSS_VNEXT_CANDIDATE_COMPONENT_IDENTITY")
            if m["status"] == "CONTROL_SKIN_CANDIDATE":
                events = m.get("rig_events")
                if (not isinstance(events, list) or len(events) < 4
                        or events[0].get("kind") != int(RigTokenKindV6.BOS)
                        or events[-1].get("kind") != int(RigTokenKindV6.EOS)):
                    raise ValueError("RSS_VNEXT_CANDIDATE_RIG_EVENTS_REQUIRED")
                w = archive["skin_weights"]
                valid = archive["skin_joint_mask"]
                if w.ndim != 3 or w.shape[0] != 1 or w.shape[1] != archive["mesh_rest_xyz"].shape[1]:
                    raise ValueError("RSS_VNEXT_CANDIDATE_SKIN_MESH_ADDRESS")
                if valid.shape != (1, w.shape[2]) or not bool(valid.any()):
                    raise ValueError("RSS_VNEXT_CANDIDATE_SKIN_JOINT_MASK")
                n_joints = sum(e["kind"] == int(RigTokenKindV6.JOINT) for e in events)
                n_codes = sum(e["kind"] == int(RigTokenKindV6.SKIN_CODE) for e in events)
                if (n_joints != w.shape[-1] or n_joints == 0
                        or n_codes != n_joints * int(m.get("skin_tokens_per_joint") or 0)):
                    raise ValueError("RSS_VNEXT_CANDIDATE_SKIN_TOKEN_COUNT_DRIFT")
                if np.any(w < -1e-6) or np.max(np.abs(w.sum(axis=-1) - 1.)) > 1e-4:
                    raise ValueError("RSS_VNEXT_CANDIDATE_SKIN_SIMPLEX")
                if np.any(w[:, :, valid[0] == 0] > 1e-5):
                    raise ValueError("RSS_VNEXT_CANDIDATE_MASKED_JOINT_INFLUENCE")

    def to_manifest_json(self) -> bytes:
        self.validate()
        return (json.dumps(self.manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def build_rss_vnext_inference_evidence_v1(
    result: Mapping[str, Any],
    *,
    model_architecture_id: str,
    checkpoint_sha256: str,
    source_image_sha256: tuple[str, str],
) -> RSSvNextInferenceEvidenceBundleV1:
    """Extract lossless evidence, never coerce neural hypothesis to product IR."""
    state = result["state"]
    if model_architecture_id != RSSvNextUnifiedCharacterModel.architecture_id:
        raise ValueError("RSS_VNEXT_CANDIDATE_MODEL_FAMILY")
    if result.get("observed_view_count") != 2:
        raise ValueError("RSS_VNEXT_CANDIDATE_ONLY_TWO_REAL_INPUTS")
    if result.get("qualification_status") != "UNQUALIFIED_NEURAL_PROPOSAL":
        raise ValueError("RSS_VNEXT_CANDIDATE_NEURAL_AUTHORITY_ESCALATION")
    if len(source_image_sha256) != 2 or not all(_is_sha(x) for x in source_image_sha256):
        raise ValueError("RSS_VNEXT_CANDIDATE_SOURCE_BYTES_UNQUALIFIED")
    arrays = {
        "source_rgba": _numeric("source_rgba", state.observed_rgba),
        "source_camera_13d": _numeric("source_camera_13d", state.camera_features),
        "source_image_tokens": _numeric("source_image_tokens", state.observed_image_tokens),
        "shared_scene_triplanes": _numeric("shared_scene_triplanes", state.scene_triplanes),
    }
    mesh = result.get("mesh_candidate")
    if mesh is not None:
        mesh.validate()
        arrays.update({
            "mesh_rest_xyz": _numeric("mesh_rest_xyz", mesh.rest_vertices_iris),
            "mesh_faces": _numeric("mesh_faces", mesh.faces),
            "mesh_vertex_component": _numeric("mesh_vertex_component", mesh.vertex_component_ids),
            "mesh_face_component": _numeric("mesh_face_component", mesh.face_component_ids),
            "mesh_local_h": _numeric("mesh_local_h", mesh.local_features),
            "mesh_normals": _numeric("mesh_normals", mesh.normals),
            "mesh_role_logits": _numeric("mesh_role_logits", mesh.presentation_role_logits),
            "mesh_graphics_logits": _numeric("mesh_graphics_logits", mesh.graphics_logits),
            "mesh_affordance_logits": _numeric("mesh_affordance_logits", mesh.affordance_logits),
            "mesh_semantic_logits": _numeric("mesh_semantic_logits", mesh.semantic_logits),
        })
        if mesh.source_ray_evidence is not None:
            arrays.update({
                "mesh_source_ray_rgba": _numeric("mesh_source_ray_rgba", mesh.source_ray_evidence.rgba),
                "mesh_source_ray_valid": _numeric("mesh_source_ray_valid", mesh.source_ray_evidence.projection_valid),
            })
    if mesh is None and result.get("mesh_status") != "ABSTAIN":
        raise ValueError("RSS_VNEXT_CANDIDATE_MESH_STATUS_CONTRADICTS_DATA")
    events = result.get("control_event_proposal")
    weights = result.get("skin_weight_proposal")
    mask = result.get("joint_valid_mask")
    if events is not None and (mesh is None or weights is None or mask is None):
        raise ValueError("RSS_VNEXT_CANDIDATE_CONTROL_SKIN_INCOMPLETE")
    if weights is not None and events is None:
        raise ValueError("RSS_VNEXT_CANDIDATE_ORPHAN_SKIN_WEIGHTS")
    if events is not None:
        n_joints = sum(int(e.kind) == int(RigTokenKindV6.JOINT) for e in events)
        code_count = sum(int(e.kind) == int(RigTokenKindV6.SKIN_CODE) for e in events)
        if not n_joints or not code_count or code_count % n_joints:
            raise ValueError("RSS_VNEXT_CANDIDATE_RIG_SKIN_CODE_GRAMMAR")
        arrays.update({
            "skin_weights": _numeric("skin_weights", weights),
            "skin_joint_mask": _numeric("skin_joint_mask", mask),
        })
    status = (
        "ABSTAIN" if mesh is None else
        "CONTROL_SKIN_CANDIDATE" if events is not None else
        "MESH_CANDIDATE"
    )
    arch = {}
    for name, arr in sorted(arrays.items()):
        arch[name] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": _sha(arr.tobytes(order="C")),
        }
    sink = BytesIO()
    np.savez_compressed(sink, **arrays)
    rig_events = [asdict(e) for e in events] if events is not None else None
    # Enum values are IntEnum; JSON serializes their numerical identity
    # without promoting them to runtime control authority.
    raw = sink.getvalue()
    manifest = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "observed_view_count": 2,
        "model_architecture_id": model_architecture_id,
        "checkpoint_sha256": checkpoint_sha256,
        "source_image_sha256": list(source_image_sha256),
        "status": status,
        "mesh_failure": result.get("mesh_failure"),
        "product_revision_id": None,
        "rig_events": rig_events,
        "skin_tokens_per_joint": (
            code_count // n_joints if events is not None else 0
        ),
        "coordinate_domain": "IRIS_XYZ_MINUS1_TO_PLUS1",
        "mesh_identity_policy": "EXACT_TOKEN_ORDER_UNQUALIFIED",
        "appearance_authority": "SOURCE_RAY_ONLY_NOT_UNSEEN_TEXTURE",
        "arrays": arch,
        "array_archive_sha256": _sha(raw),
    }
    bundle = RSSvNextInferenceEvidenceBundleV1(manifest=manifest, array_archive_npz=raw)
    bundle.validate()
    return bundle


__all__ = [
    "RSSvNextInferenceEvidenceBundleV1",
    "build_rss_vnext_inference_evidence_v1",
]
