from __future__ import annotations

"""Capture *actual* post-Mechanics C++ Graphics and typed pixel sidecars.

A preview rendering is deliberately NOT an original-source observation.
This module executes a SHA-pinned native C++ player on a live RSPOSE01/02
canonical frame and independently reopened, immutable Graphics/UV assets.

It returns raw RGBA/face-owner/color-authority/unknown-occlusion bytes in the
same typed NPZ ABI as the independently witnessed Graphics replay court.
The capture alone is NOT a trusted source witness, camera-registration
certificate, original RGB teacher or ProductRevision PASS.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.rss_vnext_source_pose_graphics_court_v1 import (
    _array_sha,
)
from .rss_vnext_live_mechanics_native_pose_v1 import (
    bind_rss_vnext_live_pose_to_native_preview_v1,
)
from .rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)
from .rss_vnext_source_pose_graphics_evidence_replay_v1 import (
    NATIVE_BUNDLE_SCHEMA, _NATIVE_KEYS, _reopen_bounded_npz,
)

SCHEMA = "RealSaS.RSSvNext.NativeCPPGraphicsCandidateCapture.v1"
AUTHORITY = "CPP_EXECUTED_CANONICAL_GRAPHICS_CANDIDATE_NOT_SOURCE_TRUTH"
_MAX_RENDERER_BYTES = 32 * 1024 * 1024


def _sha256_file_bounded(path: Path, *, budget: int) -> str:
    if not path.is_file() or path.stat().st_size not in range(1, budget + 1):
        raise QualificationError("RSS_CPP_RASTER_EXECUTABLE_OR_ARCHIVE_BUDGET")
    h = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class NativeCPPGraphicsCandidateCaptureV1:
    renderer_executable_sha256: str
    source_template_rss_sha256: str
    native_pose_bytes_sha256: str
    graphics_bytes_sha256: str
    texture_bytes_sha256: str
    canonical_mesh_lineage_hash: str
    canonical_rig_lineage_hash: str
    canonical_skin_lineage_hash: str
    camera_policy_sha256: str
    frame_index: int
    view_index: int
    source_view: int
    frame_count: int
    native_face_count: int
    resolution: int
    native_rgba_sha256: str
    native_front_face_sha256: str
    native_color_authority_sha256: str
    native_unknown_occlusion_sha256: str
    native_raster_bundle_sha256: str
    native_raster_bundle_schema: str = NATIVE_BUNDLE_SCHEMA
    native_executable_was_run: bool = True
    independent_source_camera_registered: bool = False
    source_color_or_visibility_qualified: bool = False
    product_release_admission: bool = False
    authority: str = AUTHORITY
    schema_version: str = SCHEMA
    receipt_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def native_cpp_capture_receipt_hash_v1(
    value: NativeCPPGraphicsCandidateCaptureV1,
) -> str:
    payload = value.to_dict()
    payload.pop("receipt_hash", None)
    return content_sha256(payload)


def _npz_bundle(data: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{
        key: np.ascontiguousarray(data[key])
        for key in sorted(_NATIVE_KEYS)
    })
    return buffer.getvalue()


def reopen_native_cpp_candidate_capture_v1(
    *,
    native_raster_bundle: bytes,
    receipt: Mapping[str, Any],
    expected_renderer_executable_sha256: str,
    expected_template_rss_sha256: str,
    expected_native_pose_bytes_sha256: str,
) -> tuple[dict[str, np.ndarray], NativeCPPGraphicsCandidateCaptureV1]:
    """Recheck bytes and parent authority, never promote camera/source status."""
    if not isinstance(receipt, Mapping):
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_RECEIPT_REQUIRED")
    try:
        claim = NativeCPPGraphicsCandidateCaptureV1(**dict(receipt))
    except (TypeError, ValueError) as exc:
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_RECEIPT_INVALID") from exc
    if (
        claim.schema_version != SCHEMA
        or claim.authority != AUTHORITY
        or claim.native_executable_was_run is not True
        or claim.independent_source_camera_registered is not False
        or claim.source_color_or_visibility_qualified is not False
        or claim.product_release_admission is not False
        or claim.source_view != -1
        or claim.native_raster_bundle_schema != NATIVE_BUNDLE_SCHEMA
        or claim.renderer_executable_sha256 != expected_renderer_executable_sha256
        or claim.source_template_rss_sha256 != expected_template_rss_sha256
        or claim.native_pose_bytes_sha256 != expected_native_pose_bytes_sha256
        or claim.native_raster_bundle_sha256
            != sha256(native_raster_bundle).hexdigest()
        or claim.receipt_hash != native_cpp_capture_receipt_hash_v1(claim)
        or type(claim.resolution) is not int or not 16 <= claim.resolution <= 2048
        or type(claim.frame_index) is not int or claim.frame_index < 0
        or type(claim.view_index) is not int or claim.view_index < 0
        or type(claim.native_face_count) is not int or claim.native_face_count < 1
    ):
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_RECEIPT_DRIFT")
    images = _reopen_bounded_npz(native_raster_bundle, _NATIVE_KEYS)
    h = claim.resolution
    rgba, owner, authority, unknown = (
        images["native_rgba_u8"],
        images["native_front_face_i32"],
        images["native_color_authority_u8"],
        images["native_unknown_occlusion_u8"],
    )
    if (
        rgba.shape != (h,h,4) or rgba.dtype != np.uint8
        or owner.shape != (h,h) or owner.dtype != np.int32
        or authority.shape != (h,h) or authority.dtype != np.uint8
        or unknown.shape != (h,h) or unknown.dtype != np.uint8
        or not np.isin(authority,[0,1,2]).all()
        or not np.isin(unknown,[0,1]).all()
        or np.any(owner < -1) or np.any(owner >= claim.native_face_count)
        or _array_sha(rgba) != claim.native_rgba_sha256
        or _array_sha(owner) != claim.native_front_face_sha256
        or _array_sha(authority) != claim.native_color_authority_sha256
        or _array_sha(unknown) != claim.native_unknown_occlusion_sha256
    ):
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_TYPED_PIXELS_DRIFT")
    return images, claim


def execute_native_cpp_candidate_raster_capture_v1(
    *,
    renderer_executable: Path,
    expected_renderer_executable_sha256: str,
    template_rss_path: Path,
    expected_template_rss_sha256: str,
    live_pose_bytes: bytes,
    live_pose_receipt: Mapping[str, Any],
    frame_index: int,
    view_index: int = 0,
) -> tuple[bytes, NativeCPPGraphicsCandidateCaptureV1]:
    """Execute actual raw-mode C++ on post-Mechanics pose, NOT a model image.

    Raw template always uses a virtual camera unregistered to source, hence
    native player source_view=-1. In order to qualify source view 0/1, a
    separate *source-registered* camera/texture pipeline and independent
    source registration witness must run. This utility cannot fabricate it.
    """
    exe = Path(renderer_executable)
    template = Path(template_rss_path)
    if sys.byteorder != "little":
        raise QualificationError("RSS_CPP_RASTER_NATIVE_ENDIAN_UNSUPPORTED")
    if (
        type(frame_index) is not int or type(view_index) is not int
        or type(expected_renderer_executable_sha256) is not str
        or type(expected_template_rss_sha256) is not str
        or not 0 <= frame_index <= 511 or not 0 <= view_index <= 7
        or not isinstance(live_pose_receipt, Mapping)
        or live_pose_receipt.get("product_release_admission") is not False
        or live_pose_receipt.get("source_color_or_visibility_qualified") is not False
    ):
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_INPUT_POLICY")
    actual_exe = _sha256_file_bounded(exe, budget=_MAX_RENDERER_BYTES)
    actual_archive = _sha256_file_bounded(template, budget=160*1024*1024)
    if (
        actual_exe != expected_renderer_executable_sha256
        or actual_archive != expected_template_rss_sha256
    ):
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_PINNED_EXECUTABLE_OR_TEMPLATE")
    state = reopen_rss_vnext_native_preview_package_v1(template)
    manifest = state["manifest"]
    if (
        manifest.get("virtual_preview_camera_registered_to_graphics_view") != "0"
        or manifest.get("source_rgb_surface_ownership_qualified") != "0"
        or not 0 <= frame_index < live_pose_receipt.get("frame_count",0)
        or not 0 <= view_index < live_pose_receipt.get("view_count",0)
        or live_pose_receipt.get("resolution") not in range(16,2049)
        or len(state["mesh"].faces) < 1
    ):
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_UNREGISTERED_CAMERA_REQUIRED")
    assets, transport = bind_rss_vnext_live_pose_to_native_preview_v1(
        template_rss_path=template,
        live_pose_bytes=live_pose_bytes,
        live_pose_receipt=live_pose_receipt,
    )
    side = int(live_pose_receipt["resolution"])
    if side*side > 2048*2048:
        raise QualificationError("RSS_CPP_RASTER_CAPTURE_PIXEL_BUDGET")
    with TemporaryDirectory(prefix="rss-native-render-") as dirname:
        work = Path(dirname)
        for name,data in assets.items():
            (work/name).write_bytes(data)
        args = [
            str(exe.resolve()),
            str(work/"graphics.bin"),str(work/"texture.bin"),
            str(work/"pose.bin"),str(frame_index),str(view_index),
            "-1",str(work/"rgba.bin"),str(work/"owner.i32"),
            str(work/"color_authority.u8"),
            "--unknown-mask="+str(work/"unknown_occlusion.u8"),
        ]
        try:
            result = subprocess.run(
                args, cwd=work, capture_output=True, text=True,
                check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise QualificationError("RSS_CPP_RASTER_NATIVE_EXECUTION_FAILED") from exc
        if (
            result.returncode != 0
            or "RSS_UV_NATIVE_POSED_2D_RENDER_PASS" not in result.stdout
            or "rss_archive=0" not in result.stdout
            or "source_layer=-1" not in result.stdout
            or "semantic_role_is_source_truth=0" not in result.stdout
        ):
            raise QualificationError("RSS_CPP_RASTER_NATIVE_EXECUTION_INVALID")
        n = side*side
        expected_sizes = {
            "rgba.bin":4*n,
            "owner.i32":4*n,
            "color_authority.u8":n,
            "unknown_occlusion.u8":n,
        }
        for name,size in expected_sizes.items():
            path = work/name
            if not path.is_file() or path.stat().st_size != size:
                raise QualificationError("RSS_CPP_RASTER_NATIVE_SIDECAR_BYTE_COUNT")
        images={
            "native_rgba_u8":
                np.frombuffer((work/"rgba.bin").read_bytes(),dtype=np.uint8)
                  .reshape(side,side,4).copy(),
            "native_front_face_i32":
                np.frombuffer((work/"owner.i32").read_bytes(),dtype="<i4")
                  .reshape(side,side).copy(),
            "native_color_authority_u8":
                np.frombuffer((work/"color_authority.u8").read_bytes(),dtype=np.uint8)
                  .reshape(side,side).copy(),
            "native_unknown_occlusion_u8":
                np.frombuffer((work/"unknown_occlusion.u8").read_bytes(),dtype=np.uint8)
                  .reshape(side,side).copy(),
        }
    bundle=_npz_bundle(images)
    receipt=NativeCPPGraphicsCandidateCaptureV1(
        renderer_executable_sha256=actual_exe,
        source_template_rss_sha256=actual_archive,
        native_pose_bytes_sha256=sha256(live_pose_bytes).hexdigest(),
        graphics_bytes_sha256=transport["graphics_sha256"],
        texture_bytes_sha256=transport["texture_sha256"],
        canonical_mesh_lineage_hash=state["mesh"].mesh_lineage_hash,
        canonical_rig_lineage_hash=state["rig"].rig_lineage_hash,
        canonical_skin_lineage_hash=state["skin"].skin_lineage_hash,
        camera_policy_sha256=live_pose_receipt["camera_policy_sha256"],
        frame_index=frame_index,view_index=view_index,
        source_view=-1,frame_count=live_pose_receipt["frame_count"],
        native_face_count=len(state["mesh"].faces),resolution=side,
        native_rgba_sha256=_array_sha(images["native_rgba_u8"]),
        native_front_face_sha256=_array_sha(images["native_front_face_i32"]),
        native_color_authority_sha256=_array_sha(
            images["native_color_authority_u8"],
        ),
        native_unknown_occlusion_sha256=_array_sha(
            images["native_unknown_occlusion_u8"],
        ),
        native_raster_bundle_sha256=sha256(bundle).hexdigest(),
    )
    receipt=replace(receipt,receipt_hash=native_cpp_capture_receipt_hash_v1(receipt))
    reopen_native_cpp_candidate_capture_v1(
        native_raster_bundle=bundle,receipt=receipt.to_dict(),
        expected_renderer_executable_sha256=actual_exe,
        expected_template_rss_sha256=actual_archive,
        expected_native_pose_bytes_sha256=sha256(live_pose_bytes).hexdigest(),
    )
    return bundle,receipt


__all__=[
    "SCHEMA","AUTHORITY","NativeCPPGraphicsCandidateCaptureV1",
    "native_cpp_capture_receipt_hash_v1",
    "reopen_native_cpp_candidate_capture_v1",
    "execute_native_cpp_candidate_raster_capture_v1",
]
