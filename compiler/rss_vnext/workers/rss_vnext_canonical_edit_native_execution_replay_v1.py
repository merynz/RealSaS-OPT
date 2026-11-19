from __future__ import annotations

"""Independent C++ runtime replay for an immutable .rss -> .rsedit authoring edit.

Only native EXECUTION and user-edit consequence are evidenced here. Original
source RGB/camera, artist animation, hidden appearance, and ProductRevision
are never inferred from a successful render or a rehashed receipt.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import math
import os
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from .rss_vnext_canonical_authoring_revision_v1 import (
    reopen_rss_vnext_canonical_user_edit_revision_v1,
)
from .rss_vnext_edited_graphics_preview_v1 import (
    reopen_rss_vnext_edited_graphics_preview_v1,
)
from .rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)

SCHEMA = "RealSaS.RSSvNext.CanonicalUserEditNativeExecutionCourt.v1"
AUTHORITY = "REEXECUTED_USER_EDIT_CPP_PREVIEW_NOT_SOURCE_OR_PRODUCT_PROOF"
MAGIC = b"RSEDPX01"
MAX_BUNDLE_BYTES = 112 * 1024 * 1024
MAX_RENDERER_BYTES = 64 * 1024 * 1024
MAX_RSS_BYTES = 160 * 1024 * 1024
MAX_EDIT_BYTES = 64 * 1024 * 1024
ARRAY_ORDER = (
    "original_native_rgba_u8",
    "original_front_face_i32",
    "original_color_authority_u8",
    "original_unknown_occlusion_u8",
    "edited_native_rgba_u8",
    "edited_front_face_i32",
    "edited_color_authority_u8",
    "edited_unknown_occlusion_u8",
)


def _file_sha(path: Path, maximum: int) -> str:
    if not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise QualificationError("RSS_EDIT_CPP_PINNED_FILE_BUDGET")
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _dimensions(state: Mapping, view: int, frame: int) -> int:
    frames = state["control_frames"]
    camera = state["virtual_camera"]
    if not 0 <= frame < len(frames):
        raise QualificationError("RSS_EDIT_CPP_FRAME_RANGE")
    if isinstance(camera, Mapping):
        if view != 0:
            raise QualificationError("RSS_EDIT_CPP_SINGLE_CAMERA_RANGE")
        camera_row = camera
    elif isinstance(camera, (tuple, list)) and len(camera) == 8:
        if not 0 <= view < 8:
            raise QualificationError("RSS_EDIT_CPP_EIGHT_CAMERA_RANGE")
        camera_row = camera[view]
    else:
        raise QualificationError("RSS_EDIT_CPP_CAMERA_FORMAT")
    side = camera_row.get("resolution")
    if type(side) is not int or not 16 <= side <= 2048:
        raise QualificationError("RSS_EDIT_CPP_RESOLUTION_BUDGET")
    return side


def _raster_cpp(
    executable: Path, archive: Path, *, frame: int, view: int,
    side: int, face_count: int,
) -> dict[str, np.ndarray]:
    if sys.byteorder != "little":
        raise QualificationError("RSS_EDIT_CPP_ENDIAN_UNSUPPORTED")
    n = side * side
    with TemporaryDirectory(prefix="rss-edited-raster-") as td:
        dest = Path(td)
        files = {
            "rgba":dest/"image.rgba",
            "owners":dest/"front.i32",
            "authority":dest/"color_authority.u8",
            "unknown":dest/"unknown.u8",
        }
        args = [
            str(executable.resolve()), "--rss", str(archive.resolve()),
            str(frame),str(view),"-1",
            str(files["rgba"]),str(files["owners"]),
            str(files["authority"]),
            "--unknown-mask="+str(files["unknown"]),
        ]
        try:
            result = subprocess.run(
                args,cwd=dest,capture_output=True,text=True,
                timeout=30,check=False,
            )
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise QualificationError("RSS_EDIT_CPP_NATIVE_EXECUTION_FAILED") from exc
        if (
            result.returncode != 0
            or "RSS_UV_NATIVE_POSED_2D_RENDER_PASS" not in result.stdout
            or "rss_archive=1" not in result.stdout
            or "source_layer=-1" not in result.stdout
            or "semantic_role_is_source_truth=0" not in result.stdout
        ):
            raise QualificationError("RSS_EDIT_CPP_NATIVE_EXECUTION_INVALID")
        expected = {"rgba":4*n,"owners":4*n,"authority":n,"unknown":n}
        for key, length in expected.items():
            if (
                not files[key].is_file()
                or files[key].stat().st_size != length
            ):
                raise QualificationError("RSS_EDIT_CPP_SIDECAR_BYTE_SIZE")
        rgba = np.frombuffer(files["rgba"].read_bytes(),dtype=np.uint8).reshape(side,side,4).copy()
        owners = np.frombuffer(files["owners"].read_bytes(),dtype="<i4").reshape(side,side).copy()
        authority = np.frombuffer(files["authority"].read_bytes(),dtype=np.uint8).reshape(side,side).copy()
        unknown = np.frombuffer(files["unknown"].read_bytes(),dtype=np.uint8).reshape(side,side).copy()
    if (
        np.any(owners < -1) or np.any(owners >= face_count)
        or not np.isin(authority,[0,2]).all()
        or not np.isin(unknown,[0,1]).all()
        or np.any((unknown != 0) & (authority != 0))
    ):
        raise QualificationError("RSS_EDIT_CPP_SIDECAR_DOMAIN_OR_SOURCE_AUTHORITY")
    return {
        "rgba":rgba,"owners":owners,
        "authority":authority,"unknown":unknown,
    }


def _pack(original: Mapping, edited: Mapping, side: int) -> bytes:
    payloads = (
        original["rgba"].tobytes(),original["owners"].astype("<i4").tobytes(),
        original["authority"].tobytes(),original["unknown"].tobytes(),
        edited["rgba"].tobytes(),edited["owners"].astype("<i4").tobytes(),
        edited["authority"].tobytes(),edited["unknown"].tobytes(),
    )
    out = bytearray(MAGIC)
    out.extend(struct.pack("<I",side))
    for item in payloads:
        out.extend(struct.pack("<I",len(item)))
        out.extend(item)
    if len(out) > MAX_BUNDLE_BYTES:
        raise QualificationError("RSS_EDIT_CPP_RASTER_BUNDLE_BUDGET")
    return bytes(out)


def reopen_native_edited_puppet_rasters_v1(
    binary: bytes, *, resolution: int, face_count: int,
) -> dict[str, np.ndarray]:
    if (
        type(binary) is not bytes or len(binary)>MAX_BUNDLE_BYTES
        or type(resolution) is not int or not 16 <= resolution <= 2048
        or type(face_count) is not int or not 1 <= face_count <= 4096
        or not binary.startswith(MAGIC)
    ):
        raise QualificationError("RSS_EDIT_CPP_BUNDLE_MAGIC_OR_BUDGET")
    pos = len(MAGIC)
    if len(binary) < pos + 4 or struct.unpack_from("<I",binary,pos)[0] != resolution:
        raise QualificationError("RSS_EDIT_CPP_BUNDLE_RESOLUTION_MISMATCH")
    pos += 4
    n = resolution*resolution
    rows={}
    for name in ARRAY_ORDER:
        if len(binary) < pos+4:
            raise QualificationError("RSS_EDIT_CPP_BUNDLE_TRUNCATED")
        length=struct.unpack_from("<I",binary,pos)[0]
        pos += 4
        dtype=np.dtype("<i4") if name.endswith("face_i32") else np.dtype("u1")
        channels=4 if "rgba" in name else 1
        if length != n*channels*dtype.itemsize or len(binary) < pos+length:
            raise QualificationError("RSS_EDIT_CPP_BUNDLE_SHAPE_OR_LENGTH")
        arr=np.frombuffer(binary,dtype=dtype,count=n*channels,offset=pos).copy()
        rows[name]=arr.reshape(
            (resolution,resolution,4) if channels==4 else (resolution,resolution)
        )
        pos += length
    if pos != len(binary):
        raise QualificationError("RSS_EDIT_CPP_BUNDLE_TRAILING")
    for prefix in ("original","edited"):
        owners=rows[prefix+"_front_face_i32"]
        authority=rows[prefix+"_color_authority_u8"]
        unknown=rows[prefix+"_unknown_occlusion_u8"]
        if (
            np.any(owners < -1) or np.any(owners >= face_count)
            or not np.isin(authority,[0,2]).all()
            or not np.isin(unknown,[0,1]).all()
            or np.any((unknown!=0)&(authority!=0))
        ):
            raise QualificationError("RSS_EDIT_CPP_BUNDLE_FALSE_SOURCE_OR_FACE_OWNER")
    return rows


@dataclass(frozen=True)
class CanonicalUserEditNativeExecutionCourtV1:
    original_rss_sha256: str
    rsedit_sha256: str
    edited_preview_rss_sha256: str
    renderer_executable_sha256: str
    original_mesh_lineage_hash: str
    edited_mesh_lineage_hash: str
    edited_rig_lineage_hash: str
    edited_skin_lineage_hash: str
    authoring_revision_id: str
    original_native_rgba_sha256: str
    edited_native_rgba_sha256: str
    edited_native_face_owner_sha256: str
    edited_native_color_authority_sha256: str
    edited_native_unknown_occlusion_sha256: str
    native_raster_bundle_sha256: str
    frame_index: int
    view_index: int
    resolution: int
    visible_rgba_changed_pixel_count: int
    visible_owner_changed_pixel_count: int
    source_view_index: int = -1
    native_edit_replay_executed: bool = True
    edited_source_rgb_qualified: bool = False
    edited_source_camera_observed: bool = False
    continuous_animation_proven: bool = False
    product_release_admission: bool = False
    schema: str = SCHEMA
    authority: str = AUTHORITY
    court_hash: str = ""

    def to_dict(self):
        return asdict(self)


def native_user_edit_court_hash_v1(court: CanonicalUserEditNativeExecutionCourtV1) -> str:
    body=court.to_dict()
    body.pop("court_hash")
    return content_sha256(body)


def execute_canonical_user_edit_native_cpp_capture_v1(
    *, source_rss_path: Path, rsedit_path: Path,
    edited_preview_rss_path: Path, renderer_executable: Path,
    expected_renderer_executable_sha256: str,
    frame_index: int, view_index: int = 0,
) -> tuple[bytes, CanonicalUserEditNativeExecutionCourtV1]:
    if type(frame_index) is not int or type(view_index) is not int:
        raise QualificationError("RSS_EDIT_CPP_FRAME_OR_VIEW_TYPE")
    source_path=Path(source_rss_path)
    edit_path=Path(rsedit_path)
    preview_path=Path(edited_preview_rss_path)
    executable=Path(renderer_executable)
    source_sha=_file_sha(source_path,MAX_RSS_BYTES)
    edit_sha=_file_sha(edit_path,MAX_EDIT_BYTES)
    preview_sha=_file_sha(preview_path,MAX_RSS_BYTES)
    exe_sha=_file_sha(executable,MAX_RENDERER_BYTES)
    if exe_sha!=expected_renderer_executable_sha256:
        raise QualificationError("RSS_EDIT_CPP_RENDERER_BINARY_PIN_DRIFT")
    edit=reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=edit_path,source_rss_path=source_path,
    )
    reopened=reopen_rss_vnext_edited_graphics_preview_v1(
        preview_path=preview_path,source_rss_path=source_path,
        revision_path=edit_path,
    )
    base=reopen_rss_vnext_native_preview_package_v1(source_path)
    revised=reopened["preview"]
    side=_dimensions(base,view_index,frame_index)
    if _dimensions(revised,view_index,frame_index)!=side:
        raise QualificationError("RSS_EDIT_CPP_ORIGINAL_EDITED_CAMERA_RESOLUTION_DRIFT")
    if (
        edit["mesh"].mesh_lineage_hash!=revised["mesh"].mesh_lineage_hash
        or edit["rig"].rig_lineage_hash!=revised["rig"].rig_lineage_hash
        or edit["skin"].skin_lineage_hash!=revised["skin"].skin_lineage_hash
        or edit["pose_bytes"]!=revised["pose_bytes"]
        or base["mesh"].faces != revised["mesh"].faces
        or edit["receipt"]["base_rss_sha256"]!=source_sha
    ):
        raise QualificationError("RSS_EDIT_CPP_CANONICAL_PARENT_OR_POSE_DRIFT")
    face_count=len(base["mesh"].faces)
    original=_raster_cpp(
        executable,source_path,frame=frame_index,view=view_index,
        side=side,face_count=face_count,
    )
    edited=_raster_cpp(
        executable,preview_path,frame=frame_index,view=view_index,
        side=side,face_count=face_count,
    )
    binary=_pack(original,edited,side)
    pixels=int(np.count_nonzero(
        np.any(original["rgba"]!=edited["rgba"],axis=2)
    ))
    owners=int(np.count_nonzero(original["owners"]!=edited["owners"]))
    court=CanonicalUserEditNativeExecutionCourtV1(
        original_rss_sha256=source_sha,
        rsedit_sha256=edit_sha,
        edited_preview_rss_sha256=preview_sha,
        renderer_executable_sha256=exe_sha,
        original_mesh_lineage_hash=base["mesh"].mesh_lineage_hash,
        edited_mesh_lineage_hash=edit["mesh"].mesh_lineage_hash,
        edited_rig_lineage_hash=edit["rig"].rig_lineage_hash,
        edited_skin_lineage_hash=edit["skin"].skin_lineage_hash,
        authoring_revision_id=edit["authoring_revision_id"],
        original_native_rgba_sha256=_hash_array(original["rgba"]),
        edited_native_rgba_sha256=_hash_array(edited["rgba"]),
        edited_native_face_owner_sha256=_hash_array(edited["owners"]),
        edited_native_color_authority_sha256=_hash_array(edited["authority"]),
        edited_native_unknown_occlusion_sha256=_hash_array(edited["unknown"]),
        native_raster_bundle_sha256=sha256(binary).hexdigest(),
        frame_index=frame_index,view_index=view_index,resolution=side,
        visible_rgba_changed_pixel_count=pixels,
        visible_owner_changed_pixel_count=owners,
    )
    court=replace(court,court_hash=native_user_edit_court_hash_v1(court))
    return binary,court


def independently_replay_canonical_user_edit_native_cpp_v1(
    *, source_rss_path: Path,rsedit_path: Path,
    edited_preview_rss_path: Path,renderer_executable: Path,
    raster_bundle: bytes, claimed_court: Mapping,
) -> CanonicalUserEditNativeExecutionCourtV1:
    if not isinstance(claimed_court,Mapping):
        raise QualificationError("RSS_EDIT_CPP_COURT_REQUIRED")
    try:
        recorded=CanonicalUserEditNativeExecutionCourtV1(**dict(claimed_court))
    except (TypeError,ValueError) as exc:
        raise QualificationError("RSS_EDIT_CPP_COURT_WIRE_INVALID") from exc
    if (
        recorded.schema!=SCHEMA or recorded.authority!=AUTHORITY
        or recorded.source_view_index != -1
        or recorded.native_edit_replay_executed is not True
        or any(getattr(recorded,name) is not False for name in (
            "edited_source_rgb_qualified","edited_source_camera_observed",
            "continuous_animation_proven","product_release_admission",
        ))
        or recorded.court_hash!=native_user_edit_court_hash_v1(recorded)
        or type(recorded.frame_index) is not int
        or type(recorded.view_index) is not int
    ):
        raise QualificationError("RSS_EDIT_CPP_COURT_FALSE_AUTHORITY")
    rows=reopen_native_edited_puppet_rasters_v1(
        raster_bundle,resolution=recorded.resolution,
        face_count=len(
            reopen_rss_vnext_native_preview_package_v1(
                Path(source_rss_path)
            )["mesh"].faces
        ),
    )
    if (
        sha256(raster_bundle).hexdigest()!=recorded.native_raster_bundle_sha256
        or _hash_array(rows["original_native_rgba_u8"])!=recorded.original_native_rgba_sha256
        or _hash_array(rows["edited_native_rgba_u8"])!=recorded.edited_native_rgba_sha256
        or _hash_array(rows["edited_front_face_i32"])!=recorded.edited_native_face_owner_sha256
        or _hash_array(rows["edited_color_authority_u8"])!=recorded.edited_native_color_authority_sha256
        or _hash_array(rows["edited_unknown_occlusion_u8"])!=recorded.edited_native_unknown_occlusion_sha256
    ):
        raise QualificationError("RSS_EDIT_CPP_COURT_BUNDLE_HASH_DRIFT")
    reproduced,independent=execute_canonical_user_edit_native_cpp_capture_v1(
        source_rss_path=source_rss_path,rsedit_path=rsedit_path,
        edited_preview_rss_path=edited_preview_rss_path,
        renderer_executable=renderer_executable,
        expected_renderer_executable_sha256=recorded.renderer_executable_sha256,
        frame_index=recorded.frame_index,view_index=recorded.view_index,
    )
    if reproduced!=raster_bundle or independent!=recorded:
        raise QualificationError("RSS_EDIT_CPP_INDEPENDENT_NATIVE_RASTER_REPLAY_DRIFT")
    return independent


__all__=[
    "SCHEMA","AUTHORITY","CanonicalUserEditNativeExecutionCourtV1",
    "execute_canonical_user_edit_native_cpp_capture_v1",
    "independently_replay_canonical_user_edit_native_cpp_v1",
    "reopen_native_edited_puppet_rasters_v1",
]
