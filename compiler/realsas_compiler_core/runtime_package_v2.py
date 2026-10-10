from __future__ import annotations

"""Compact dependency-free RealSaS V2 runtime package writer/reader."""

from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image

from .runtime_authority_v2 import RuntimeProjectionV2IR
from .runtime_visual_authority_v1 import SourceOwnedVisualRuntimeProjectionV1IR
from .types import QualificationError
from .visual_completion_evidence_v1 import validate_source_visual_boundary_evidence
from .visual_material_v1 import (
    MATERIAL_CONTRACT, SAMPLING_CONTRACT, DEPTH_CONTRACT,
    DEPTH_TIE_EPSILON, MAXIMUM_FRAGMENT_LAYERS, validate_visual_material,
)

MAGIC = b"RSASV2R1"


def _entry_bytes(entries: OrderedDict[str, bytes]) -> bytes:
    out = io.BytesIO()
    out.write(MAGIC)
    out.write(struct.pack("<I", len(entries)))
    for name, payload in entries.items():
        encoded = name.encode("utf-8")
        if not encoded or len(encoded) > 65535:
            raise QualificationError("RSS_V2_ENTRY_NAME_INVALID")
        out.write(struct.pack("<H", len(encoded)))
        out.write(encoded)
        out.write(struct.pack("<Q", len(payload)))
        out.write(payload)
    return out.getvalue()


def write_rss_v2(path: Path, entries: OrderedDict[str, bytes]) -> dict:
    raw = _entry_bytes(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_bytes": len(raw),
        "entry_names": tuple(entries.keys()),
        "package_format": "REALSAS_RSS_V2_UNCOMPRESSED_CONTAINER",
    }


def read_rss_v2(path: Path) -> OrderedDict[str, bytes]:
    raw = path.read_bytes()
    offset = 0
    if raw[: len(MAGIC)] != MAGIC:
        raise QualificationError("RSS_V2_MAGIC_INVALID")
    offset += len(MAGIC)
    if offset + 4 > len(raw):
        raise QualificationError("RSS_V2_TRUNCATED")
    count = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    entries: OrderedDict[str, bytes] = OrderedDict()
    for _ in range(count):
        if offset + 2 > len(raw):
            raise QualificationError("RSS_V2_TRUNCATED_NAME")
        name_len = struct.unpack_from("<H", raw, offset)[0]
        offset += 2
        if offset + name_len + 8 > len(raw):
            raise QualificationError("RSS_V2_TRUNCATED_ENTRY_HEADER")
        name = raw[offset : offset + name_len].decode("utf-8")
        offset += name_len
        size = struct.unpack_from("<Q", raw, offset)[0]
        offset += 8
        if offset + size > len(raw):
            raise QualificationError("RSS_V2_TRUNCATED_ENTRY_DATA")
        if name in entries:
            raise QualificationError("RSS_V2_DUPLICATE_ENTRY")
        entries[name] = raw[offset : offset + size]
        offset += size
    if offset != len(raw):
        raise QualificationError("RSS_V2_TRAILING_BYTES")
    return entries


def _mesh_payload(arrays: dict) -> bytes:
    vertices = np.asarray(arrays["vertices"], dtype="<f8")
    faces = np.asarray(arrays["faces"], dtype="<u4")
    uv = np.asarray(arrays["face_uv"], dtype="<f8")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise QualificationError("RSS_V2_VERTEX_SHAPE_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise QualificationError("RSS_V2_FACE_SHAPE_INVALID")
    if uv.shape != (len(faces), 3, 2):
        raise QualificationError("RSS_V2_UV_SHAPE_INVALID")
    return (
        struct.pack("<II", len(vertices), len(faces))
        + vertices.tobytes(order="C")
        + faces.tobytes(order="C")
        + uv.tobytes(order="C")
    )


def _camera_payload(projection: RuntimeProjectionV2IR) -> bytes:
    out = io.BytesIO()
    out.write(struct.pack("<I", len(projection.views)))
    for view in sorted(projection.views, key=lambda row: row.view_index):
        camera = dict(view.camera)
        values = (
            tuple(map(float, camera["origin"]))
            + tuple(map(float, camera["right"]))
            + tuple(map(float, camera["screen_up"]))
            + tuple(map(float, camera["forward"]))
            + (float(camera["half_extent"]),)
        )
        if len(values) != 13:
            raise QualificationError("RSS_V2_CAMERA_VECTOR_INVALID")
        out.write(struct.pack("<13dI", *values, int(camera["resolution"])))
    return out.getvalue()



def _runtime_page_rows(view) -> tuple[dict, ...]:
    metadata = dict(view.metadata or {})
    raw = tuple(metadata.get("texture_pages") or ())
    if not raw:
        return (
            {
                "page_index": 0,
                "path": str(view.texture_path),
                "sha256": str(view.texture_sha256),
            },
        )
    rows = tuple(sorted((dict(row) for row in raw), key=lambda row: int(row["page_index"])))
    if tuple(int(row["page_index"]) for row in rows) != tuple(range(len(rows))):
        raise QualificationError("RSS_V2_TEXTURE_PAGE_INDEX_DRIFT")
    if (
        str(rows[0]["path"]) != str(view.texture_path)
        or str(rows[0]["sha256"]) != str(view.texture_sha256)
    ):
        raise QualificationError("RSS_V2_PRIMARY_TEXTURE_PAGE_DRIFT")
    return rows


def _is_paged_projection(projection: RuntimeProjectionV2IR) -> bool:
    flags = [bool(dict(view.metadata or {}).get("paged_atlas")) for view in projection.views]
    if any(flags) and not all(flags):
        raise QualificationError("RSS_V2_MIXED_PAGED_TEXTURE_MODE")
    return bool(flags and all(flags))


def _paged_texture_entries(projection: RuntimeProjectionV2IR):
    entries = OrderedDict()
    rows_by_view = []
    page_count = None
    page_shape = None
    for view in sorted(projection.views, key=lambda row: row.view_index):
        rows = _runtime_page_rows(view)
        if page_count is None:
            page_count = len(rows)
        if len(rows) != page_count:
            raise QualificationError("RSS_V2_TEXTURE_PAGE_COUNT_DRIFT")
        current = []
        for row in rows:
            path = Path(str(row["path"]))
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != str(row["sha256"]):
                raise QualificationError("RSS_V2_TEXTURE_PAGE_BYTES_DRIFT")
            with Image.open(path) as image:
                image = image.convert("RGBA")
                shape = (int(image.height), int(image.width))
            if page_shape is None:
                page_shape = shape
            if shape != page_shape:
                raise QualificationError("RSS_V2_TEXTURE_PAGE_DIMENSION_DRIFT")
            page_index = int(row["page_index"])
            name = f"texture_v{int(view.view_index)}_p{page_index}.png"
            entries[name] = payload
            current.append((page_index, name, digest))
        rows_by_view.append((int(view.view_index), tuple(current)))
    if page_count is None or page_count <= 0 or page_shape is None:
        raise QualificationError("RSS_V2_TEXTURE_PAGE_SET_EMPTY")
    return entries, tuple(rows_by_view), int(page_count), page_shape


def _validated_provenance_pages(projection: RuntimeProjectionV2IR) -> tuple[np.ndarray, np.ndarray]:
    path = Path(projection.provenance_npz_path)
    with np.load(path, allow_pickle=False) as data:
        required = {"provenance", "source_view"}
        if not required.issubset(data.files):
            raise QualificationError("RSS_V2_PROVENANCE_LINEAGE_ARRAY_MISSING")
        value = np.asarray(data["provenance"], dtype=np.uint8)
        source_view = np.asarray(data["source_view"], dtype="<i2")
    if value.ndim == 3:
        value = value[:, None, :, :]
        source_view = source_view[:, None, :, :]
    if value.ndim != 4 or value.shape[0] != 8:
        raise QualificationError("RSS_V2_PROVENANCE_SHAPE_INVALID")
    if source_view.shape != value.shape:
        raise QualificationError("RSS_V2_SOURCE_VIEW_SHAPE_INVALID")
    padding = np.iinfo(np.int16).min
    target_view = np.broadcast_to(
        np.arange(8, dtype=np.int16)[:, None, None, None],
        source_view.shape,
    )
    direct = value == 0
    other = value == 1
    harmonic = value == 2
    unsupported = value == 3
    canonical_global = value == 4
    padding_mask = value == 255
    valid_provenance = (
        direct | other | harmonic | unsupported | canonical_global | padding_mask
    )
    if not np.all(valid_provenance):
        raise QualificationError("RSS_V2_PROVENANCE_CLASS_INVALID")
    if np.any(
        (direct | other)
        & ~((source_view >= 0) & (source_view < 8))
    ):
        raise QualificationError("RSS_V2_SOURCE_VIEW_VALUE_INVALID")
    if np.any(harmonic & (source_view != -2)):
        raise QualificationError("RSS_V2_HARMONIC_SOURCE_VIEW_IDENTITY_DRIFT")
    if np.any(canonical_global & (source_view != -3)):
        raise QualificationError(
            "RSS_V2_CANONICAL_GLOBAL_SOURCE_VIEW_IDENTITY_DRIFT"
        )
    if np.any(unsupported & (source_view != -4)):
        raise QualificationError("RSS_V2_UNSUPPORTED_SOURCE_VIEW_IDENTITY_DRIFT")
    if np.any(padding_mask & (source_view != padding)):
        raise QualificationError("RSS_V2_SOURCE_VIEW_PADDING_DRIFT")
    if np.any(direct & (source_view != target_view)):
        raise QualificationError("RSS_V2_DIRECT_SOURCE_VIEW_IDENTITY_DRIFT")
    if np.any(
        other
        & (
            (source_view < 0)
            | (source_view >= 8)
            | (source_view == target_view)
        )
    ):
        raise QualificationError("RSS_V2_OTHER_VIEW_IDENTITY_DRIFT")
    return value, source_view


def _paged_provenance_payload(projection: RuntimeProjectionV2IR) -> bytes:
    value, source_view = _validated_provenance_pages(projection)
    raw = value.tobytes(order="C") + source_view.astype("<i2", copy=False).tobytes(order="C")
    compressed = zlib.compress(raw, level=9)
    return (
        struct.pack(
            "<IIIIQ",
            int(value.shape[0]),
            int(value.shape[1]),
            int(value.shape[2]),
            int(value.shape[3]),
            len(raw),
        )
        + compressed
    )


def _face_page_payload(arrays: dict, *, page_count: int) -> bytes:
    faces = np.asarray(arrays["faces"], dtype=np.uint32)
    page = np.asarray(
        arrays.get("face_page_index", np.zeros((len(faces),), dtype=np.int32)),
        dtype=np.int64,
    )
    if page.shape != (len(faces),) or np.any(page < 0) or np.any(page >= int(page_count)):
        raise QualificationError("RSS_V2_FACE_PAGE_INDEX_INVALID")
    value = page.astype("<u4", copy=False)
    return struct.pack("<I", len(value)) + value.tobytes(order="C")

def _texture_payload(projection: RuntimeProjectionV2IR) -> bytes:
    images = []
    shape = None
    for view in sorted(projection.views, key=lambda row: row.view_index):
        path = Path(view.texture_path)
        image = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
        if shape is None:
            shape = image.shape
        if image.shape != shape:
            raise QualificationError("RSS_V2_TEXTURE_DIMENSION_DRIFT")
        images.append(image)
    stacked = np.stack(images, axis=0)
    return (
        struct.pack("<III", stacked.shape[0], stacked.shape[1], stacked.shape[2])
        + stacked.tobytes(order="C")
    )


def _provenance_payload(projection: RuntimeProjectionV2IR) -> bytes:
    value, source_view = _validated_provenance_pages(projection)
    value = value[:, 0, :, :]
    source_view = source_view[:, 0, :, :]
    return (
        struct.pack("<III", value.shape[0], value.shape[1], value.shape[2])
        + value.tobytes(order="C")
        + source_view.astype("<i2", copy=False).tobytes(order="C")
    )

def _clip_payload(times: np.ndarray, positions: np.ndarray) -> bytes:
    times = np.asarray(times, dtype="<f8")
    positions = np.asarray(positions, dtype="<f8")
    if times.ndim != 1 or positions.ndim != 3 or positions.shape[0] != len(times) or positions.shape[2] != 3:
        raise QualificationError("RSS_V2_CLIP_ARRAY_SHAPE_INVALID")
    return (
        struct.pack("<II", len(times), positions.shape[1])
        + times.tobytes(order="C")
        + positions.tobytes(order="C")
    )


def build_rss_v2_entries(projection: RuntimeProjectionV2IR) -> OrderedDict[str, bytes]:
    with np.load(projection.projection_npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    required = {"vertices", "faces", "face_uv"}
    if not required.issubset(arrays):
        raise QualificationError("RSS_V2_PROJECTION_ARRAYS_MISSING")

    entries: OrderedDict[str, bytes] = OrderedDict()
    paged = _is_paged_projection(projection)
    manifest = [
        "schema=RealSaS.RuntimePackage.v2",
        f"projection_hash={projection.projection_hash}",
        f"mesh_hash={projection.mesh_binding_hash}",
        f"appearance_asset_hash={projection.appearance_asset_binding_hash}",
        f"dynamic_motion_hash={projection.dynamic_motion_binding_hash}",
        f"visibility_contract_hash={projection.visibility_contract_hash}",
        "playback_sampling_contract=SEALED_FRAME_INDEX_ONLY",
        "view_selection_contract=SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        "cross_direction_blending_authorized=0",
        "host_interpolation_authorized=0",
        "presentation_state_execution_authorized=0",
        "clipping_authorized=0",
        "tint_order_visibility_authorized=0",
        "texture_sampling_contract=BASE_LEVEL_BILINEAR_LINEAR_PM_ONLY",
        "mip_generation_authorized=0",
        "pixel_coverage_contract=FIXED_2X2_QUARTER_SUBSAMPLES__LINEAR_PM_AVERAGE",
        "pixel_coverage_sample_count=4",
        "depth_buffer_contract=IEEE754_FLOAT64_SOFTWARE_SORT",
        "depth_equivalence_epsilon_camera_z=1e-12",
        "source_view_identity_contract=PER_TEXEL_INT16_PRESERVED__DIAGNOSTIC_ONLY",
        "source_view_identity_render_authority=0",
        "source_view_identity_compiled_harmonic_code=-2",
        "source_view_identity_unsupported_abstain_code=-4",
        "source_view_identity_physical_padding_code=INT16_MIN",
        "source_view_identity_mixed_sample_code=-3",
        "geometry_uv_position_precision=IEEE754_FLOAT64",
        f"clip_count={len(projection.clips)}",
        f"view_count={len(projection.views)}",
    ]
    entries["mesh.bin"] = _mesh_payload(arrays)
    entries["cameras.bin"] = _camera_payload(projection)

    if paged:
        texture_entries, rows_by_view, page_count, page_shape = _paged_texture_entries(
            projection
        )
        provenance_pages, _ = _validated_provenance_pages(projection)
        if int(provenance_pages.shape[1]) != page_count:
            raise QualificationError("RSS_V2_TEXTURE_PROVENANCE_PAGE_COUNT_DRIFT")
        if tuple(map(int, provenance_pages.shape[2:])) != tuple(map(int, page_shape)):
            raise QualificationError("RSS_V2_TEXTURE_PROVENANCE_PAGE_SHAPE_DRIFT")
        entries["face_pages.bin"] = _face_page_payload(arrays, page_count=page_count)
        entries["provenance_paged.zlib"] = _paged_provenance_payload(projection)
        for name, payload in texture_entries.items():
            entries[name] = payload
        manifest.extend(
            [
                "atlas_paging_contract=FACE_INDEX_TO_FIXED_PHYSICAL_PAGE_V1",
                f"atlas_page_count={page_count}",
                f"atlas_page_height={int(page_shape[0])}",
                f"atlas_page_width={int(page_shape[1])}",
                "paged_texture_transport=PNG_RGBA8",
                "paged_provenance_transport=ZLIB_U8_I16_V1",
                "face_page_entry=face_pages.bin",
                "provenance_entry=provenance_paged.zlib",
            ]
        )
        for view_index, rows in rows_by_view:
            for page_index, name, digest in rows:
                manifest.extend(
                    [
                        f"texture.{view_index}.{page_index}.entry={name}",
                        f"texture.{view_index}.{page_index}.sha256={digest}",
                    ]
                )
    else:
        entries["textures.bin"] = _texture_payload(projection)
        entries["provenance.bin"] = _provenance_payload(projection)
        manifest.extend(
            [
                "atlas_paging_contract=LEGACY_SINGLE_PAGE_V1",
                "atlas_page_count=1",
            ]
        )

    for index, clip in enumerate(projection.clips):
        times_key = f"{clip.array_prefix}_times"
        positions_key = f"{clip.array_prefix}_positions"
        if times_key not in arrays or positions_key not in arrays:
            raise QualificationError("RSS_V2_CLIP_ARRAY_MISSING")
        entry_name = f"clip_{index}.bin"
        entries[entry_name] = _clip_payload(
            arrays[times_key], arrays[positions_key]
        )
        manifest.extend(
            [
                f"clip.{index}.id={clip.clip_id}",
                f"clip.{index}.entry={entry_name}",
                f"clip.{index}.frame_count={clip.frame_count}",
                f"clip.{index}.duration_seconds={clip.duration_seconds:.17g}",
                f"clip.{index}.loop={1 if clip.loop else 0}",
            ]
        )
    entries["manifest.txt"] = ("\n".join(manifest) + "\n").encode("utf-8")
    return entries


def _source_owned_visual_mesh_payload(
    projection: SourceOwnedVisualRuntimeProjectionV1IR,
    arrays: dict,
    *,
    view_index: int,
) -> bytes:
    view = next(
        row for row in projection.views if int(row.view_index) == int(view_index)
    )
    uv = np.asarray(arrays[f"view_{view_index}_uv"], dtype="<f8")
    faces = np.asarray(arrays[f"view_{view_index}_faces"], dtype="<u4")
    if uv.shape != (int(view.visual_vertex_count), 2):
        raise QualificationError("RSS_V2_VISUAL_UV_SHAPE_INVALID")
    if faces.shape != (int(view.visual_face_count), 3):
        raise QualificationError("RSS_V2_VISUAL_FACE_SHAPE_INVALID")
    return (
        b"RSVM1\0\0\0"
        + struct.pack(
            "<IIII",
            int(view.source_width),
            int(view.source_height),
            int(view.visual_vertex_count),
            int(view.visual_face_count),
        )
        + uv.tobytes(order="C")
        + faces.tobytes(order="C")
    )


def _source_owned_visual_positions_payload(
    positions: np.ndarray,
) -> bytes:
    frames = np.asarray(positions, dtype="<f8")
    if frames.ndim != 3 or frames.shape[2] != 2 or not np.isfinite(frames).all():
        raise QualificationError("RSS_V2_VISUAL_POSITION_SHAPE_INVALID")
    return (
        b"RSVP1\0\0\0"
        + struct.pack("<II", int(frames.shape[0]), int(frames.shape[1]))
        + frames.tobytes(order="C")
    )


def build_source_owned_visual_rss_v2_entries(
    projection: SourceOwnedVisualRuntimeProjectionV1IR,
) -> OrderedDict[str, bytes]:
    """Pack the qualified source-owned 2D presentation into RSS v2.

    This is the canonical transport form of the 1941-trunk RSVM1/RSVP1
    visual-render path. Runtime consumes already-compiled visual positions;
    it never rebuilds bindings, runs ARAP, searches donors, or renders the
    mechanical relation mesh.
    """
    if hashlib.sha256(Path(projection.projection_npz_path).read_bytes()).hexdigest() != projection.projection_npz_sha256:
        raise QualificationError("RSS_V2_VISUAL_PROJECTION_BYTES_DRIFT")
    with np.load(projection.projection_npz_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}

    views = tuple(sorted(projection.views, key=lambda row: int(row.view_index)))
    if (
        len(views) != 8
        or tuple(int(row.view_index) for row in views) != tuple(range(8))
    ):
        raise QualificationError("RSS_V2_VISUAL_REQUIRES_EXACT_V0_V7")

    entries: OrderedDict[str, bytes] = OrderedDict()
    material = projection.metadata.get("visual_material_contract")
    if material is not None and material != MATERIAL_CONTRACT:
        raise QualificationError("RSS_V2_VISUAL_MATERIAL_CONTRACT_INVALID")
    if material and projection.metadata.get("depth_ownership_contract") != DEPTH_CONTRACT:
        raise QualificationError("RSS_V2_VISUAL_DEPTH_CONTRACT_REQUIRED")
    manifest = [
        "schema=RealSaS.RuntimePackage.v2",
        f"projection_hash={projection.projection_hash}",
        "presentation_geometry_mode=SOURCE_OWNED_VISUAL_PRESENTATION_V1",
        "mechanical_mesh_render_authority=0",
        "runtime_visual_mesh_rebuild=0",
        "runtime_binding_solve=0",
        "runtime_generation=0",
        "donor_search_at_runtime=0",
        "playback_sampling_contract=SEALED_FRAME_INDEX_ONLY",
        "view_selection_contract=SEALED_DISCRETE_DIRECTION_INDEX_ONLY",
        "cross_direction_blending_authorized=0",
        "host_interpolation_authorized=0",
        "presentation_state_execution_authorized=0",
        "clipping_authorized=0",
        "tint_order_visibility_authorized=0",
        "texture_sampling_contract=SOURCE_RGBA8_BILINEAR_STRAIGHT_TO_PM_V1",
        "mip_generation_authorized=0",
        f"clip_count={len(projection.clips)}",
        f"view_count={len(views)}",
        f"qualified_visual_presentation_hash={projection.qualified_visual_presentation_binding_hash}",
        f"mechanical_mesh_hash={projection.mechanical_mesh_binding_hash}",
        f"appearance_asset_hash={projection.appearance_asset_binding_hash}",
        f"dynamic_motion_hash={projection.dynamic_motion_binding_hash}",
        f"visual_deformation_operator_id={projection.visual_deformation_operator_id}",
        f"visual_deformation_policy_hash={projection.visual_deformation_policy_hash}",
    ]

    if material:
        manifest = [row for row in manifest if not row.startswith("texture_sampling_contract=")]
        manifest.extend([
            f"texture_sampling_contract={SAMPLING_CONTRACT}",
            f"visual_material_contract={MATERIAL_CONTRACT}",
            f"appearance_qualification_hash={projection.appearance_qualification_binding_hash}",
            f"depth_ownership_contract={DEPTH_CONTRACT}",
            f"depth_tie_epsilon={DEPTH_TIE_EPSILON:.17g}",
            f"maximum_fragment_layers={MAXIMUM_FRAGMENT_LAYERS}",
        ])

    boundary = projection.metadata.get("visual_boundary_evidence")
    if boundary is not None:
        validate_source_visual_boundary_evidence(boundary)
        entries["authority/source_visual_boundary_evidence_v1.json"] = json.dumps(
            boundary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        manifest.append("source_visual_boundary_evidence_hash=" + boundary["evidence_hash"])
        manifest.append("full_visual_acceptance_passed=0")

    for view in views:
        vi = int(view.view_index)
        mesh_entry = f"visual_mesh_v{vi}.bin"
        texture_entry = f"visual_texture_v{vi}.png"
        entries[mesh_entry] = _source_owned_visual_mesh_payload(
            projection,
            arrays,
            view_index=vi,
        )
        texture_path = Path(str(view.texture_path))
        texture_bytes = texture_path.read_bytes()
        if hashlib.sha256(texture_bytes).hexdigest() != str(view.texture_sha256):
            raise QualificationError("RSS_V2_VISUAL_TEXTURE_BYTES_DRIFT")
        with Image.open(texture_path) as image:
            width, height = image.size
        if (
            int(width) != int(view.source_width)
            or int(height) != int(view.source_height)
        ):
            raise QualificationError("RSS_V2_VISUAL_TEXTURE_DIMENSION_DRIFT")
        entries[texture_entry] = texture_bytes
        if material:
            with Image.open(texture_path) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            p = np.asarray(arrays.get(f"view_{vi}_material_provenance"))
            s = np.asarray(arrays.get(f"view_{vi}_material_source_view"))
            validate_visual_material(p, s, view_index=vi, rgba=rgba)
            material_entry = f"visual_material_v{vi}.bin"
            entries[material_entry] = (b"RSVA1\0\0\0" + struct.pack("<II", width, height)
                + p.tobytes(order="C") + s.astype("<i2").tobytes(order="C"))
            manifest.append(f"view.{vi}.material_entry={material_entry}")
        resolution = int(dict(view.camera).get("resolution") or 0)
        if resolution <= 0:
            raise QualificationError("RSS_V2_VISUAL_OUTPUT_RESOLUTION_INVALID")
        manifest.extend(
            [
                f"view.{vi}.id={view.view_id}",
                f"view.{vi}.mesh_entry={mesh_entry}",
                f"view.{vi}.texture_entry={texture_entry}",
                f"view.{vi}.resolution={resolution}",
                f"view.{vi}.visual_mesh_hash={view.visual_mesh_hash}",
                f"view.{vi}.texture_sha256={view.texture_sha256}",
            ]
        )

    for clip_index, clip in enumerate(projection.clips):
        times_key = f"{clip.array_prefix}_times"
        if times_key not in arrays:
            raise QualificationError("RSS_V2_VISUAL_CLIP_TIMES_MISSING")
        times = np.asarray(arrays[times_key], dtype=np.float64)
        if times.shape != (int(clip.frame_count),):
            raise QualificationError("RSS_V2_VISUAL_CLIP_TIME_SHAPE_INVALID")
        manifest.extend(
            [
                f"clip.{clip_index}.id={clip.clip_id}",
                f"clip.{clip_index}.frame_count={int(clip.frame_count)}",
                f"clip.{clip_index}.duration_seconds={float(clip.duration_seconds):.17g}",
                f"clip.{clip_index}.loop={1 if clip.loop else 0}",
            ]
        )
        for view in views:
            vi = int(view.view_index)
            key = f"{clip.array_prefix}_view_{vi}_positions"
            if key not in arrays:
                raise QualificationError("RSS_V2_VISUAL_CLIP_POSITION_MISSING")
            positions = np.asarray(arrays[key], dtype=np.float64)
            expected = (
                int(clip.frame_count),
                int(view.visual_vertex_count),
                2,
            )
            if positions.shape != expected:
                raise QualificationError("RSS_V2_VISUAL_CLIP_POSITION_SHAPE_INVALID")
            entry_name = f"clip_{clip_index}_v{vi}.positions.bin"
            entries[entry_name] = _source_owned_visual_positions_payload(
                positions
            )
            if material:
                depth = np.asarray(arrays.get(f"{clip.array_prefix}_view_{vi}_depths"), dtype="<f8")
                if depth.shape != expected[:2] or not np.isfinite(depth).all() or np.any(depth <= 0):
                    raise QualificationError("RSS_V2_VISUAL_DEPTH_ARRAY_INVALID")
                depth_entry = f"clip_{clip_index}_v{vi}.depths.bin"
                entries[depth_entry] = (b"RSVD1\0\0\0" + struct.pack("<II", *depth.shape)
                    + depth.tobytes(order="C"))
                manifest.append(f"clip.{clip_index}.view.{vi}.depths_entry={depth_entry}")
            manifest.append(
                f"clip.{clip_index}.view.{vi}.positions_entry={entry_name}"
            )

    entries["manifest.txt"] = ("\n".join(manifest) + "\n").encode("utf-8")
    return entries
