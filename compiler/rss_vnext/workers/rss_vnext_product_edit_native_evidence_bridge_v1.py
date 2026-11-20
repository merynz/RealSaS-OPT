from __future__ import annotations

"""CAS -> host-only independent native edit execution evidence bridge.

No CAS receipt authorizes a product. A sample counts as independently replayed
only when the operator injects a trusted callable which actually reopens the
same immutable original/edited .rss files, .rsedit and SHA-pinned C++ binary.
"""

from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Mapping

import numpy as np

from compiler.realsas_compiler_core.types import QualificationError
from .rss_vnext_canonical_edit_native_execution_replay_v1 import (
    AUTHORITY,
    SCHEMA,
    MAX_BUNDLE_BYTES,
    MAX_EDIT_BYTES,
    MAX_RSS_BYTES,
    MAX_RENDERER_BYTES,
    CanonicalUserEditNativeExecutionCourtV1,
    independently_replay_canonical_user_edit_native_cpp_v1,
    native_user_edit_court_hash_v1,
    reopen_native_edited_puppet_rasters_v1,
)
from .rss_vnext_canonical_authoring_revision_v1 import (
    reopen_rss_vnext_canonical_user_edit_revision_v1,
)
from .rss_vnext_edited_graphics_preview_v1 import (
    reopen_rss_vnext_edited_graphics_preview_v1,
)
from .rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)

COURT_TYPE = "RealSaS.RSSvNextCanonicalUserEditNativeExecutionCourt"
RASTER_TYPE = "RealSaS.RSSvNextCanonicalUserEditNativeRasterBundle"
EDIT_TYPE = "RealSaS.RSSvNextCanonicalUserEditRevision"
PREVIEW_TYPE = "RealSaS.RSSvNextEditedGraphicsPreview"
ORIGINAL_TYPE = "RealSaS.RSSvNextNativeInteractivePuppetPreview"
MAX_COURT_BYTES = 64 * 1024

def _read_one(request: Mapping, read_object: Callable, kind: str, limit: int) -> bytes:
    rows = [
        row for row in request.get("input_artifacts") or ()
        if type(row) is dict and row.get("artifact_type") == kind
    ]
    if len(rows) != 1 or rows[0].get("schema_version") != "v1":
        raise QualificationError("PRODUCT_PROOF_EDIT_CAS_CARDINALITY_OR_SCHEMA:" + kind)
    ref = rows[0]
    if (
        type(ref.get("size_bytes")) is not int
        or not 0 < ref["size_bytes"] <= limit
        or type(ref.get("content_sha256")) is not str
        or type(ref.get("storage_key")) is not str
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_CAS_REF_INVALID:" + kind)
    raw = read_object(ref)
    if (
        type(raw) is not bytes
        or len(raw) != ref["size_bytes"]
        or sha256(raw).hexdigest() != ref["content_sha256"]
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_CAS_BYTES_DRIFT:" + kind)
    return raw


def _court(raw: bytes) -> CanonicalUserEditNativeExecutionCourtV1:
    import json

    try:
        payload = json.loads(raw)
        if type(payload) is not dict:
            raise ValueError("court must be a JSON object")
        value = CanonicalUserEditNativeExecutionCourtV1(**payload)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise QualificationError("PRODUCT_PROOF_EDIT_COURT_WIRE_INVALID") from exc

    # Type-check every persisted value; bool must not count as int.
    string_fields = (
        "original_rss_sha256","rsedit_sha256","edited_preview_rss_sha256",
        "renderer_executable_sha256","original_mesh_lineage_hash",
        "edited_mesh_lineage_hash","edited_rig_lineage_hash",
        "edited_skin_lineage_hash","authoring_revision_id",
        "original_native_rgba_sha256","edited_native_rgba_sha256",
        "edited_native_face_owner_sha256","edited_native_color_authority_sha256",
        "edited_native_unknown_occlusion_sha256",
        "native_raster_bundle_sha256","court_hash",
    )
    if (
        value.schema != SCHEMA or value.authority != AUTHORITY
        or any(type(getattr(value, name)) is not str or not getattr(value,name)
               for name in string_fields)
        or any(type(getattr(value, name)) is not int for name in (
            "frame_index","view_index","resolution",
            "visible_rgba_changed_pixel_count",
            "visible_owner_changed_pixel_count","source_view_index",
        ))
        or value.source_view_index != -1
        or value.native_edit_replay_executed is not True
        or any(getattr(value,name) is not False for name in (
            "edited_source_rgb_qualified","edited_source_camera_observed",
            "continuous_animation_proven","product_release_admission",
        ))
        or value.court_hash != native_user_edit_court_hash_v1(value)
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_COURT_FALSE_OR_INVALID_AUTHORITY")
    return value


def make_trusted_native_edit_replayer_v1(
    *, renderer_executable: Path, expected_renderer_executable_sha256: str,
) -> Callable:
    """Operator-only factory; never deserialize an executable path from CAS.

    The caller must pin and provision a trusted C++ binary. This factory
    invokes the actual native binary when supplied with a sealed sample;
    even a fully rehashed tampered sample cannot be accepted by comparing
    receipt fields alone.
    """
    executable = Path(renderer_executable)
    if (
        type(expected_renderer_executable_sha256) is not str
        or len(expected_renderer_executable_sha256) != 64
        or any(c not in "0123456789abcdef"
               for c in expected_renderer_executable_sha256)
        or not executable.is_file()
        or not 0 < executable.stat().st_size <= MAX_RENDERER_BYTES
        or sha256(executable.read_bytes()).hexdigest()
            != expected_renderer_executable_sha256
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_HOST_RENDERER_BINARY_NOT_PINNED")

    def _replay(
        *, original_rss_bytes: bytes, rsedit_bytes: bytes,
        edited_preview_rss_bytes: bytes, raster_bundle: bytes,
        claimed_court: CanonicalUserEditNativeExecutionCourtV1,
    ) -> CanonicalUserEditNativeExecutionCourtV1:
        if type(claimed_court) is not CanonicalUserEditNativeExecutionCourtV1:
            raise QualificationError("PRODUCT_PROOF_EDIT_HOST_COURT_TYPE")
        if claimed_court.renderer_executable_sha256 != expected_renderer_executable_sha256:
            raise QualificationError("PRODUCT_PROOF_EDIT_HOST_RENDERER_PARENT_MISMATCH")
        with TemporaryDirectory(prefix="proof-edit-native-host-") as work:
            root=Path(work)
            original=root/"original.rss"
            revision=root/"revision.rsedit"
            preview=root/"edited.rss"
            original.write_bytes(original_rss_bytes)
            revision.write_bytes(rsedit_bytes)
            preview.write_bytes(edited_preview_rss_bytes)
            return independently_replay_canonical_user_edit_native_cpp_v1(
                source_rss_path=original,rsedit_path=revision,
                edited_preview_rss_path=preview,
                renderer_executable=executable,raster_bundle=raster_bundle,
                claimed_court=claimed_court.to_dict(),
            )
    return _replay


def reopen_product_proof_user_edit_native_evidence_v1(
    request: Mapping, read_object: Callable, *,
    canonical_mesh,canonical_rig,canonical_skin,
    expected_original_native_preview_sha256: str | None,
    trusted_native_edit_replayer: Callable | None = None,
) -> dict:
    """Strict CAS source comparison, then optional REAL host-rerun admission.

    With no host replayer, a local candidate court is counted, but no
    independently executed sample exists. One frame never qualifies the
    entire edit-consequence release gate.
    """
    if trusted_native_edit_replayer is not None and not callable(
        trusted_native_edit_replayer
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_TRUSTED_HOST_REPLAYER_CALLABLE")
    refs=request.get("input_artifacts") or ()
    types={COURT_TYPE,RASTER_TYPE,EDIT_TYPE,PREVIEW_TYPE}
    provided=[
        row for row in refs if type(row) is dict
        and row.get("artifact_type") in types
    ]
    if not provided:
        return {
            "typed_user_edit_court_reopened": False,
            "candidate_edit_native_execution_claim_count": 0,
            "independently_native_replayed_edit_frame_count": 0,
            "native_edit_source_and_product_release_qualified": False,
            "native_edit_product_revision_minted": False,
            "native_edit_sample": None,
        }
    if (
        len(provided)!=4
        or {row["artifact_type"] for row in provided}!=types
        or type(expected_original_native_preview_sha256) is not str
        or len(expected_original_native_preview_sha256)!=64
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_INCOMPLETE_OR_UNBOUND_CAS")
    raw_court=_read_one(request,read_object,COURT_TYPE,MAX_COURT_BYTES)
    raster=_read_one(request,read_object,RASTER_TYPE,MAX_BUNDLE_BYTES)
    original=_read_one(request,read_object,ORIGINAL_TYPE,MAX_RSS_BYTES)
    revision=_read_one(request,read_object,EDIT_TYPE,MAX_EDIT_BYTES)
    preview=_read_one(request,read_object,PREVIEW_TYPE,MAX_RSS_BYTES)
    court=_court(raw_court)
    if (
        sha256(original).hexdigest()!=expected_original_native_preview_sha256
        or sha256(original).hexdigest()!=court.original_rss_sha256
        or sha256(revision).hexdigest()!=court.rsedit_sha256
        or sha256(preview).hexdigest()!=court.edited_preview_rss_sha256
        or sha256(raster).hexdigest()!=court.native_raster_bundle_sha256
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_ORIGINAL_CAS_OR_COURT_PARENT_DRIFT")
    with TemporaryDirectory(prefix="proof-edit-cas-") as work:
        root=Path(work)
        original_path=root/"original.rss"
        edit_path=root/"edit.rsedit"
        preview_path=root/"edited.rss"
        original_path.write_bytes(original)
        edit_path.write_bytes(revision)
        preview_path.write_bytes(preview)
        base=reopen_rss_vnext_native_preview_package_v1(original_path)
        edited=reopen_rss_vnext_canonical_user_edit_revision_v1(
            revision_path=edit_path,source_rss_path=original_path,
        )
        regenerated=reopen_rss_vnext_edited_graphics_preview_v1(
            preview_path=preview_path,source_rss_path=original_path,
            revision_path=edit_path,
        )
    if (
        base["mesh"].mesh_lineage_hash!=canonical_mesh.mesh_lineage_hash
        or base["rig"].rig_lineage_hash!=canonical_rig.rig_lineage_hash
        or base["skin"].skin_lineage_hash!=canonical_skin.skin_lineage_hash
        or court.original_mesh_lineage_hash!=canonical_mesh.mesh_lineage_hash
        or court.edited_mesh_lineage_hash!=edited["mesh"].mesh_lineage_hash
        or court.edited_rig_lineage_hash!=edited["rig"].rig_lineage_hash
        or court.edited_skin_lineage_hash!=edited["skin"].skin_lineage_hash
        or court.authoring_revision_id!=edited["authoring_revision_id"]
        or regenerated["preview"]["mesh"].mesh_lineage_hash
            !=edited["mesh"].mesh_lineage_hash
        or court.frame_index<0
        or court.view_index<0
        or court.visible_rgba_changed_pixel_count<0
        or court.visible_owner_changed_pixel_count<0
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_CANONICAL_PARENT_OR_SAMPLE_DRIFT")
    faces=len(canonical_mesh.faces)
    arrays=reopen_native_edited_puppet_rasters_v1(
        raster,resolution=court.resolution,face_count=faces,
    )
    changed_rgba=int(np.count_nonzero(np.any(
        arrays["original_native_rgba_u8"]!=arrays["edited_native_rgba_u8"],
        axis=2,
    )))
    changed_owner=int(np.count_nonzero(
        arrays["original_front_face_i32"]!=arrays["edited_front_face_i32"]
    ))
    def _sha(arr):
        return sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    if (
        changed_rgba!=court.visible_rgba_changed_pixel_count
        or changed_owner!=court.visible_owner_changed_pixel_count
        or _sha(arrays["original_native_rgba_u8"])
            !=court.original_native_rgba_sha256
        or _sha(arrays["edited_native_rgba_u8"])
            !=court.edited_native_rgba_sha256
        or _sha(arrays["edited_front_face_i32"])
            !=court.edited_native_face_owner_sha256
        or _sha(arrays["edited_color_authority_u8"])
            !=court.edited_native_color_authority_sha256
        or _sha(arrays["edited_unknown_occlusion_u8"])
            !=court.edited_native_unknown_occlusion_sha256
    ):
        raise QualificationError("PRODUCT_PROOF_EDIT_RASTER_SAMPLES_OR_HASH_DRIFT")

    host_verified=False
    if trusted_native_edit_replayer is not None:
        replayed=trusted_native_edit_replayer(
            original_rss_bytes=original,rsedit_bytes=revision,
            edited_preview_rss_bytes=preview,raster_bundle=raster,
            claimed_court=court,
        )
        if (
            type(replayed) is not CanonicalUserEditNativeExecutionCourtV1
            or replayed!=court
        ):
            raise QualificationError("PRODUCT_PROOF_EDIT_INDEPENDENT_HOST_REPLAY_DRIFT")
        host_verified=True

    return {
        "typed_user_edit_court_reopened":True,
        "candidate_edit_native_execution_claim_count":1,
        "independently_native_replayed_edit_frame_count":int(host_verified),
        "native_edit_source_and_product_release_qualified":False,
        "native_edit_product_revision_minted":False,
        "native_edit_sample":{
            "original_rss_sha256":court.original_rss_sha256,
            "edited_preview_rss_sha256":court.edited_preview_rss_sha256,
            "authoring_revision_id":court.authoring_revision_id,
            "renderer_executable_sha256":court.renderer_executable_sha256,
            "frame_index":court.frame_index,
            "view_index":court.view_index,
            "resolution":court.resolution,
            "rgba_changed_pixels":court.visible_rgba_changed_pixel_count,
            "face_owner_changed_pixels":court.visible_owner_changed_pixel_count,
            "original_source_camera_or_rgb_qualified":False,
            "full_temporal_edit_coverage_qualified":False,
            "independent_host_replay_passed":host_verified,
        },
    }


__all__=[
    "COURT_TYPE","RASTER_TYPE","EDIT_TYPE","PREVIEW_TYPE",
    "make_trusted_native_edit_replayer_v1",
    "reopen_product_proof_user_edit_native_evidence_v1",
]
