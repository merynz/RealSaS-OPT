from __future__ import annotations

"""Conditional source-observed static admission for RSS.vNext product.mesh.

No eight invented views, no preselection of latent 3D as source truth, and no
checkpoint load. The release must explicitly pin BOTH an independently frozen
two-observation acceptance policy and the existing G3 mechanical mesh policy.

A structurally valid TESSA candidate is NOT by itself a CanonicalMesh.
"""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mesh.product_policy_v1 import (
    load_mesh_policy_document,
)
from compiler.realsas_compiler_core.rss_vnext_neural_mesh_proposal_basis_v1 import (
    build_rss_vnext_neural_mesh_candidate_basis_v1,
)
from compiler.realsas_compiler_core.rss_vnext_two_view_mesh_qualification_v1 import (
    RSSvNextTwoViewStaticPolicyV1,
    qualify_rss_vnext_two_view_mesh_v1,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MESH_POLICY_PATH = (
    _REPO_ROOT / "canonical" / "QUALIFIED_MESH_PRODUCT_POLICY_V2_20260919.json"
)
_POLICY_FIELDS = frozenset(RSSvNextTwoViewStaticPolicyV1.__dataclass_fields__)


def maybe_admit_rss_vnext_static_mesh_v1(
    *,
    metadata: Mapping[str, Any],
    source,
    evidence,
    source_identity_court,
    diagnostic,
    computed_surface,
    canonical_bridge,
    model_config,
):
    """Return a typed PASS/FAIL court or None when release policy is unbound.

    This function can be called for a source candidate after the primary
    evidence-continuity record was sealed. A FAILED qualification MUST block
    CanonicalMesh emission; it cannot be silently replaced with nearest
    geometry, an average z-order, invented source depth or a low-quality mesh.
    """
    policy_keys = (
        "two_view_static_policy",
        "two_view_static_policy_sha256",
        "mesh_policy_document_sha256",
    )
    supplied = [name in metadata for name in policy_keys]
    if not any(supplied):
        return None
    if not all(supplied):
        raise RuntimeError("RSS_VNEXT_STATIC_RELEASE_POLICY_INCOMPLETE")
    raw = metadata["two_view_static_policy"]
    if type(raw) is not dict or set(raw) != _POLICY_FIELDS:
        raise RuntimeError("RSS_VNEXT_STATIC_RELEASE_POLICY_FIELD_SET_DRIFT")
    try:
        policy = RSSvNextTwoViewStaticPolicyV1(**raw)
        policy.validate()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("RSS_VNEXT_STATIC_RELEASE_POLICY_INVALID") from exc
    policy_sha = content_sha256(asdict(policy))
    if metadata["two_view_static_policy_sha256"] != policy_sha:
        raise RuntimeError("RSS_VNEXT_STATIC_RELEASE_POLICY_SHA_DRIFT")
    expected_mesh_policy = metadata["mesh_policy_document_sha256"]
    bytes_ = _MESH_POLICY_PATH.read_bytes()
    if (
        not isinstance(expected_mesh_policy, str)
        or expected_mesh_policy != sha256(bytes_).hexdigest()
    ):
        raise RuntimeError("RSS_VNEXT_STATIC_MESH_POLICY_FILE_SHA_DRIFT")
    mesh_policy = load_mesh_policy_document(_MESH_POLICY_PATH)

    if evidence.mesh_status != "UNQUALIFIED_MESH_CANDIDATE":
        return None
    if (
        canonical_bridge.state != "UNQUALIFIED_NEURAL_BASIS_CANONICAL_CANDIDATE"
        or canonical_bridge.candidate is None
        or canonical_bridge.canonical_mesh_authority
    ):
        raise RuntimeError("RSS_VNEXT_STATIC_NEURAL_CANDIDATE_BRIDGE_REQUIRED")
    independently_rebuilt_basis = build_rss_vnext_neural_mesh_candidate_basis_v1(
        evidence=evidence,
        computed_surface=computed_surface,
        model_config=model_config,
    )
    if (
        canonical_bridge.neural_basis_lineage_sha256 != independently_rebuilt_basis.basis_hash
        or canonical_bridge.candidate != independently_rebuilt_basis.candidate
    ):
        raise RuntimeError("RSS_VNEXT_STATIC_BRIDGE_NEURAL_BASIS_DRIFT")

    result = qualify_rss_vnext_two_view_mesh_v1(
        source=source,
        evidence=evidence,
        source_identity_court=source_identity_court,
        diagnostic=diagnostic,
        neural_basis=independently_rebuilt_basis,
        static_policy=policy,
        mesh_policy=mesh_policy,
    )
    if result.status == "PASS_OBSERVED_TWO_VIEW_STATIC_CARRIER":
        if result.canonical_mesh is None or result.static_qualification is None:
            raise RuntimeError("RSS_VNEXT_STATIC_PASS_WITHOUT_CANONICAL_MESH")
    elif result.canonical_mesh is not None or result.static_qualification is not None:
        raise RuntimeError("RSS_VNEXT_STATIC_FAIL_WITH_FORGED_MESH")
    return result


__all__ = ["maybe_admit_rss_vnext_static_mesh_v1"]
