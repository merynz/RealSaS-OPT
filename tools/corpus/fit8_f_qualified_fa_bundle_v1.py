from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import FlexiCubesDenseGridV1
from tools.corpus.fit8_f_reference_dense_grid_authority_v1 import (
    CUBE_ORDERING,
    REGULAR_LATTICE,
    SCALAR_SIGN_CONVENTION,
    FReferenceDenseGridAuthorityV1,
)
from tools.corpus.fit8_f_reference_input_bundle_contract_v1 import (
    FReferenceInputBundleManifestV1,
    QUALIFIED_FIT8,
    ZERO_SHA256,
)
from tools.corpus.fit8_mesh_field_teacher_v1 import (
    MeshFieldTeacherPolicyV1,
    qualify_single_component_global_sign_v1,
)
from tools.corpus.fit8_mesh_field_teacher_v2 import build_mesh_field_teacher_v2
from tools.corpus.fit8_static_self_intersection_qualification_v1 import (
    StaticSelfIntersectionPolicyV1,
)
from tools.corpus.fit8_static_self_intersection_qualification_v2 import (
    qualify_static_self_intersection_v2,
)
from tools.corpus.fit8_surface_topology_invariants_v1 import (
    compute_surface_topology_invariants_v1,
)

SCHEMA_VERSION = "RealSaS.FQualifiedFABundleBuilder.v1"
NPZ_FORMAT = "DETERMINISTIC_NPZ_V1_FIXED_ZIP_METADATA_SORTED_ARRAY_NAMES"
BACKEND_SCALAR_DTYPE = "float32"
TEACHER_SCALAR_DTYPE = "float64"


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_nonzero_sha(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"F_QUALIFIED_FA_SHA_INVALID:{name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"F_QUALIFIED_FA_SHA_NONHEX:{name}") from exc
    if value == ZERO_SHA256:
        raise ValueError(f"F_QUALIFIED_FA_ZERO_SHA_FORBIDDEN:{name}")


@dataclass(frozen=True)
class QualifiedFAAuthorityRefsV1:
    source_sha256: str
    teacher_bundle_sha256: str
    extraction_truth_sha256: str
    canonical_camera_contract_sha256: str
    extraction_scale_manifest_sha256: str
    measurement_protocol_sha256: str
    source_exact_bytes_validation: str
    source_geometry_frame_validation: str
    teacher_bundle_validation: str
    extraction_truth_validation: str
    canonical_camera_contract_validation: str
    extraction_scale_manifest_validation: str
    measurement_protocol_preregistration: str
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_QUALIFIED_FA_REFS_SCHEMA_INVALID")
        for name in (
            "source_sha256",
            "teacher_bundle_sha256",
            "extraction_truth_sha256",
            "canonical_camera_contract_sha256",
            "extraction_scale_manifest_sha256",
            "measurement_protocol_sha256",
        ):
            _validate_nonzero_sha(name, getattr(self, name))
        for name in (
            "source_exact_bytes_validation",
            "source_geometry_frame_validation",
            "teacher_bundle_validation",
            "extraction_truth_validation",
            "canonical_camera_contract_validation",
            "extraction_scale_manifest_validation",
            "measurement_protocol_preregistration",
            "provenance",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"F_QUALIFIED_FA_VALIDATION_PROVENANCE_MISSING:{name}")


@dataclass(frozen=True)
class FQualifiedFABuildReportV1:
    subject_key: str
    bundle_npz_sha256: str
    input_manifest_sha256: str
    dense_grid_authority_sha256: str
    mesh_field_policy_sha256: str
    topology_component_count: int
    topology_boundary_edge_count: int
    topology_nonmanifold_edge_count: int
    static_self_intersection_free: bool
    orientation_valid_count: int
    lattice_vertex_count: int
    cube_count: int
    provenance: str
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("F_QUALIFIED_FA_REPORT_SCHEMA_INVALID")
        if not self.subject_key or not self.provenance:
            raise ValueError("F_QUALIFIED_FA_REPORT_IDENTITY_MISSING")
        for name in (
            "bundle_npz_sha256",
            "input_manifest_sha256",
            "dense_grid_authority_sha256",
            "mesh_field_policy_sha256",
        ):
            _validate_nonzero_sha(name, getattr(self, name))
        if self.topology_component_count < 1:
            raise ValueError("F_QUALIFIED_FA_REPORT_COMPONENT_COUNT_INVALID")
        if self.topology_boundary_edge_count < 0 or self.topology_nonmanifold_edge_count < 0:
            raise ValueError("F_QUALIFIED_FA_REPORT_TOPOLOGY_COUNT_INVALID")
        if self.orientation_valid_count != self.lattice_vertex_count:
            raise ValueError("F_QUALIFIED_FA_REPORT_ORIENTATION_NOT_CLOSED")
        if self.cube_count < 1:
            raise ValueError("F_QUALIFIED_FA_REPORT_CUBE_COUNT_INVALID")


def deterministic_npz_bytes_v1(arrays: dict[str, np.ndarray]) -> bytes:
    """Write stable NPZ bytes; standard np.savez* ZIP timestamps are not authority-safe."""
    if not arrays:
        raise ValueError("F_QUALIFIED_FA_NPZ_EMPTY")
    if len(arrays) != len(set(arrays)):
        raise ValueError("F_QUALIFIED_FA_NPZ_DUPLICATE_NAME")
    name_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    output = BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=False,
    ) as archive:
        for name in sorted(arrays):
            if not name_re.fullmatch(name):
                raise ValueError(f"F_QUALIFIED_FA_NPZ_NAME_INVALID:{name}")
            value = np.asarray(arrays[name])
            if value.dtype.hasobject:
                raise ValueError(f"F_QUALIFIED_FA_NPZ_OBJECT_DTYPE_FORBIDDEN:{name}")
            if not value.flags.c_contiguous:
                value = np.ascontiguousarray(value)
            payload = BytesIO()
            np.save(payload, value, allow_pickle=False)
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def write_deterministic_npz_v1(path: Path, arrays: dict[str, np.ndarray]) -> str:
    blob = deterministic_npz_bytes_v1(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return _sha256_bytes(blob)


def _containment_semantics_qualified(topology, *, self_intersection_free: bool) -> bool:
    if topology.component_count != 1 or not self_intersection_free:
        return False
    component = topology.components[0]
    return bool(
        component.boundary_edge_count == 0
        and component.nonmanifold_edge_count == 0
        and component.orientability_valid
        and component.orientable is True
    )


def build_qualified_fa_bundle_v1(
    *,
    subject_key: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: FlexiCubesDenseGridV1,
    authority_refs: QualifiedFAAuthorityRefsV1,
    mesh_field_policy: MeshFieldTeacherPolicyV1,
    self_intersection_policy: StaticSelfIntersectionPolicyV1,
    output_npz: Path,
    output_manifest: Path,
    output_grid_authority: Path,
    provenance: str,
) -> FQualifiedFABuildReportV1:
    """Build one QUALIFIED_FIT8 vanilla FlexiCubes F_A input bundle.

    This builder only admits the globally signed, single-component closed court.
    Open, multi-component-composition, non-manifold, self-intersecting or sign-
    ambiguous subjects must use chart/support-aware F-B/F-R paths instead of
    fabricating a vanilla global SDF.
    """
    if not subject_key or subject_key.startswith("SYNTHETIC_"):
        raise ValueError("F_QUALIFIED_FA_SUBJECT_INVALID")
    if not provenance:
        raise ValueError("F_QUALIFIED_FA_PROVENANCE_MISSING")
    authority_refs.validate()
    grid.validate()
    mesh_field_policy.validate()
    self_intersection_policy.validate()

    xyz = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("F_QUALIFIED_FA_VERTICES_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or not len(tri) or np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise ValueError("F_QUALIFIED_FA_FACES_INVALID")

    topology = compute_surface_topology_invariants_v1(tri.tolist())
    self_intersection = qualify_static_self_intersection_v2(
        vertices=xyz,
        faces=tri,
        policy=self_intersection_policy,
    )
    containment = _containment_semantics_qualified(
        topology,
        self_intersection_free=self_intersection.self_intersection_free,
    )
    sign_admission = qualify_single_component_global_sign_v1(
        topology,
        self_intersection_free=self_intersection.self_intersection_free,
        containment_semantics_qualified=containment,
        provenance=f"{provenance};GLOBAL_SIGN_ADMISSION",
    )
    if not sign_admission.admitted:
        raise ValueError(
            "F_QUALIFIED_FA_GLOBAL_SIGN_NOT_ADMITTED:"
            + ",".join(sign_admission.reasons)
        )

    query_xyz = np.asarray(grid.vertex_xyz, dtype=np.float64)
    field = build_mesh_field_teacher_v2(
        vertices=xyz,
        faces=tri,
        query_xyz=query_xyz,
        sign_admission=sign_admission,
        policy=mesh_field_policy,
        provenance=f"{provenance};MESH_FIELD_TEACHER_V2",
    )
    valid = np.asarray(field.orientation_valid, dtype=bool)
    if not np.all(valid):
        invalid = int(np.count_nonzero(~valid))
        raise ValueError(f"F_QUALIFIED_FA_FIELD_ORIENTATION_ABSTENTION:{invalid}")
    oriented = np.asarray(field.oriented_level_set, dtype=np.float64)
    if not np.isfinite(oriented).all():
        raise ValueError("F_QUALIFIED_FA_FIELD_NONFINITE")

    arrays = {
        "x_nx3": np.asarray(grid.vertex_xyz, dtype=np.float32),
        "s_n": oriented.astype(np.float32),
        "cube_fx8": np.asarray(grid.cube_fx8, dtype=np.int64),
        "orientation_valid": valid.astype(np.bool_),
        "teacher_unsigned_distance_f64": np.asarray(field.unsigned_distance, dtype=np.float64),
        "teacher_oriented_level_set_f64": oriented,
        "teacher_nearest_triangle_id": np.asarray(field.nearest_triangle_id, dtype=np.int32),
        "teacher_nearest_barycentric_f64": np.asarray(field.nearest_barycentric, dtype=np.float64),
        "teacher_nearest_address_tie_count": np.asarray(field.nearest_address_tie_count, dtype=np.int32),
        "teacher_inside": np.asarray(field.inside, dtype=np.bool_),
        "teacher_nonambiguous_vote_count": np.asarray(field.nonambiguous_vote_count, dtype=np.int16),
        "teacher_ambiguous_ray_count": np.asarray(field.ambiguous_ray_count, dtype=np.int16),
    }
    bundle_sha = write_deterministic_npz_v1(output_npz, arrays)

    grid_authority = FReferenceDenseGridAuthorityV1(
        cell_min_ijk=grid.cell_min_ijk,
        cell_max_exclusive_ijk=grid.cell_max_exclusive_ijk,
        lattice_origin_xyz=grid.lattice_origin_xyz,
        extraction_pitch=grid.extraction_pitch,
        cube_ordering=CUBE_ORDERING,
        scalar_sign_convention=SCALAR_SIGN_CONVENTION,
        vertex_position_mode=REGULAR_LATTICE,
        bundle_npz_sha256=bundle_sha,
        provenance=f"{provenance};DENSE_GRID_AUTHORITY",
        metadata={
            "qualified_fa_builder_schema": SCHEMA_VERSION,
            "source_geometry_frame_validation": authority_refs.source_geometry_frame_validation,
            "extraction_scale_manifest_sha256": authority_refs.extraction_scale_manifest_sha256,
        },
    )
    grid_authority.validate()
    grid_authority_payload = grid_authority.canonical_payload()
    output_grid_authority.parent.mkdir(parents=True, exist_ok=True)
    output_grid_authority.write_text(
        json.dumps(grid_authority_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = FReferenceInputBundleManifestV1(
        subject_key=subject_key,
        arm="F_A",
        authority_state=QUALIFIED_FIT8,
        res=grid.res,
        global_sign_authority=True,
        support_clipping_required=False,
        source_sha256=authority_refs.source_sha256,
        teacher_bundle_sha256=authority_refs.teacher_bundle_sha256,
        extraction_truth_sha256=authority_refs.extraction_truth_sha256,
        canonical_camera_contract_sha256=authority_refs.canonical_camera_contract_sha256,
        extraction_scale_manifest_sha256=authority_refs.extraction_scale_manifest_sha256,
        measurement_protocol_sha256=authority_refs.measurement_protocol_sha256,
        bundle_npz_sha256=bundle_sha,
        provenance=f"{provenance};QUALIFIED_FIT8_F_A_BUNDLE",
        metadata={
            "source_exact_bytes_verified": True,
            "teacher_bundle_validated": True,
            "extraction_truth_validated": True,
            "measurement_protocol_preregistered": True,
            "canonical_camera_contract_validated": True,
            "extraction_scale_manifest_validated": True,
            "source_geometry_frame_validated": True,
            "source_exact_bytes_validation": authority_refs.source_exact_bytes_validation,
            "source_geometry_frame_validation": authority_refs.source_geometry_frame_validation,
            "teacher_bundle_validation": authority_refs.teacher_bundle_validation,
            "extraction_truth_validation": authority_refs.extraction_truth_validation,
            "canonical_camera_contract_validation": authority_refs.canonical_camera_contract_validation,
            "extraction_scale_manifest_validation": authority_refs.extraction_scale_manifest_validation,
            "measurement_protocol_preregistration": authority_refs.measurement_protocol_preregistration,
            "authority_refs_provenance": authority_refs.provenance,
            "mesh_field_teacher_schema": field.schema_version,
            "mesh_field_teacher_policy_sha256": field.policy_sha256,
            "mesh_field_distance_authority": field.distance_authority,
            "mesh_field_sign_authority": field.sign_authority,
            "static_self_intersection_provenance": self_intersection.provenance,
            "dense_grid_authority_sha256": grid_authority.sha256(),
            "npz_format": NPZ_FORMAT,
            "backend_scalar_dtype": BACKEND_SCALAR_DTYPE,
            "teacher_scalar_dtype": TEACHER_SCALAR_DTYPE,
            "field_quantization": "FLOAT64_TEACHER_TO_FLOAT32_FLEXICUBES_INPUT",
            "orientation_valid_fraction": 1.0,
            "topology_component_count": topology.component_count,
            "topology_boundary_edge_count": topology.total_boundary_edges,
            "topology_nonmanifold_edge_count": topology.total_nonmanifold_edges,
        },
    )
    manifest.validate()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest.canonical_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if _sha256_file(output_npz) != bundle_sha:
        raise RuntimeError("F_QUALIFIED_FA_BUNDLE_POSTWRITE_SHA_DRIFT")
    report = FQualifiedFABuildReportV1(
        subject_key=subject_key,
        bundle_npz_sha256=bundle_sha,
        input_manifest_sha256=manifest.manifest_sha256(),
        dense_grid_authority_sha256=grid_authority.sha256(),
        mesh_field_policy_sha256=field.policy_sha256,
        topology_component_count=topology.component_count,
        topology_boundary_edge_count=topology.total_boundary_edges,
        topology_nonmanifold_edge_count=topology.total_nonmanifold_edges,
        static_self_intersection_free=self_intersection.self_intersection_free,
        orientation_valid_count=int(np.count_nonzero(valid)),
        lattice_vertex_count=len(grid.vertex_xyz),
        cube_count=len(grid.cube_fx8),
        provenance=provenance,
    )
    report.validate()
    return report


__all__ = [
    "BACKEND_SCALAR_DTYPE",
    "FQualifiedFABuildReportV1",
    "NPZ_FORMAT",
    "QualifiedFAAuthorityRefsV1",
    "SCHEMA_VERSION",
    "TEACHER_SCALAR_DTYPE",
    "build_qualified_fa_bundle_v1",
    "deterministic_npz_bytes_v1",
    "write_deterministic_npz_v1",
]
