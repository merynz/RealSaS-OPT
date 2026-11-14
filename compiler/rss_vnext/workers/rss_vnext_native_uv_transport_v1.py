from __future__ import annotations

"""Transport the *existing* CAS UV atlas into a native 2D compositor.

This is a byte-level adapter, not new canonical evidence or a second UV
authority. The source/learned color separation and uncertainty survive the
round trip. RuntimeGraphics remains the independent visibility/order owner.
"""

from hashlib import sha256
import struct

import numpy as np

from .rss_vnext_uv_texture_atlas_v1 import (
    reopen_rss_vnext_uv_texture_atlas_v1,
)

MAGIC = b"RSUVT001"
VERSION = 1
MAX_NATIVE_BYTES = 128 * 1024 * 1024
_ASSET_KIND = "RSS_VNEXT_UV_TEXTURE_CANDIDATE_NOT_SOURCE_SURFACE_TRUTH"


def _text(value: str) -> bytes:
    body = value.encode("ascii")
    if len(body) != 64 or any(byte not in b"0123456789abcdef" for byte in body):
        raise RuntimeError("RSS_NATIVE_UV_SHA256_FORMAT")
    return struct.pack("<I", len(body)) + body


def encode_native_rss_vnext_uv_v1(
    *, uv_atlas_cas: bytes,
    canonical_mesh_lineage_hash: str,
) -> bytes:
    arrays, receipt = reopen_rss_vnext_uv_texture_atlas_v1(uv_atlas_cas)
    if (
        receipt["canonical_mesh_lineage_hash"] != canonical_mesh_lineage_hash
        or receipt.get("source_3d_surface_ownership_verified") is not False
        or receipt.get("draw_order_is_not_texture_atlas_authority") is not True
        or receipt.get("neural_graphics_rgb_is_inferred_not_observed") is not True
        or receipt.get("neural_completion_never_overwrites_original_rgb_layers")
        is not True
        or receipt.get("product_revision_minted") is not False
    ):
        raise RuntimeError("RSS_NATIVE_UV_PARENT_OR_AUTHORITY_DRIFT")

    width, height = receipt["atlas_size"]
    faces = receipt["face_count"]
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIIII", VERSION, width, height, faces, 3))
    output.extend(_text(canonical_mesh_lineage_hash))
    output.extend(_text(sha256(uv_atlas_cas).hexdigest()))
    output.extend(_text(
        receipt["original_source_binding_sha256"]
    ))

    # Preserve the face-local indexed UV coordinates for every canonical
    # triangle. Vertex/frame motion never changes these texture addresses.
    raw_fields = (
        ("face_vertex_uv_f32", "<f4"),
        ("texture_rgba_u8", "u1"),
        ("texture_source_sample_mask_u8", "u1"),
        ("neural_rgb_candidate_rgba_u8", "u1"),
        ("neural_rgb_candidate_mask_u8", "u1"),
        ("source_view_conflict_u8", "u1"),
        ("neural_rgb_log_uncertainty_f16", "<f2"),
    )
    for key, dtype in raw_fields:
        field = np.ascontiguousarray(arrays[key], dtype=np.dtype(dtype))
        output.extend(field.tobytes(order="C"))
    if not len(output) <= MAX_NATIVE_BYTES:
        raise RuntimeError("RSS_NATIVE_UV_TRANSPORT_RESOURCE_BUDGET")
    return bytes(output)


__all__ = ["MAGIC", "VERSION", "encode_native_rss_vnext_uv_v1"]
