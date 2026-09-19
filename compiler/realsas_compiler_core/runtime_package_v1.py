from __future__ import annotations

"""Stage-37 exact Runtime-v4 package seal."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .hashing import content_sha256

Json=dict[str,Any]


@dataclass(frozen=True)
class RuntimeV4PackageSealIR:
    projection_binding_hash:str
    product_state_binding_hash:str
    dynamic_motion_binding_hash:str
    archive_path:str
    archive_sha256:str
    runtime_binary_sha256:str
    source_binding_sha256:str
    playback_contract_hash:str
    reference_raster_contract_hash:str
    package_total_bytes:int
    runtime_binary_bytes:int
    view_count:int
    clip_count:int
    frame_count_total:int
    qualification_report:Json
    package_seal_hash:str
    schema_version:str="RealSaS.RuntimeV4PackageSealIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def runtime_v4_package_seal_hash(value:RuntimeV4PackageSealIR)->str:
    payload=value.to_dict(); payload.pop("package_seal_hash",None)
    return content_sha256(payload)


def build_runtime_v4_package_seal(*,projection,result:Mapping[str,Any])->RuntimeV4PackageSealIR:
    value=RuntimeV4PackageSealIR(
        projection_binding_hash=projection.projection_hash,
        product_state_binding_hash=projection.product_state_binding_hash,
        dynamic_motion_binding_hash=projection.dynamic_motion_binding_hash,
        archive_path=str(result["archive_path"]),
        archive_sha256=str(result["archive_sha256"]),
        runtime_binary_sha256=str(result["runtime_binary_sha256"]),
        source_binding_sha256=str(result["source_binding_sha256"]),
        playback_contract_hash=str(result["playback_contract_hash"]),
        reference_raster_contract_hash=str(result["reference_raster_contract_hash"]),
        package_total_bytes=int(result["package_total_bytes"]),
        runtime_binary_bytes=int(result["runtime_binary_bytes"]),
        view_count=int(result["view_count"]),
        clip_count=int(result["clip_count"]),
        frame_count_total=int(result["frame_count_total"]),
        qualification_report={
            "status":"PASS_RUNTIME_V4_PACKAGE_MATERIALIZED",
            "solver_replay_performed":False,
            "stage36_exact_frames_written":True,
            "shared_canonical_xyz_representation":True,
            "full_product_proof_claimed":False,
            "native_playback_required":True,
        },
        package_seal_hash="",
        metadata={
            "source_authority_kind":"RUNTIME_V4_ADMISSION_CERTIFICATE",
            "zip_compression":"DEFLATED_LEVEL_1",
            "runtime_binary_write_seconds":float(result.get("runtime_binary_write_seconds",0.0)),
            "archive_write_seconds":float(result.get("archive_write_seconds",0.0)),
            "wall_seconds":float(result.get("wall_seconds",0.0)),
        },
    )
    return replace(value,package_seal_hash=runtime_v4_package_seal_hash(value))


def runtime_v4_package_seal_from_dict(payload:Mapping[str,Any])->RuntimeV4PackageSealIR:
    if str(payload.get("schema_version") or payload.get("schema") or "")!="RealSaS.RuntimeV4PackageSealIR.v1":
        raise ValueError("RUNTIME_PACKAGE_SEAL_SCHEMA_MISMATCH")
    value=RuntimeV4PackageSealIR(
        str(payload["projection_binding_hash"]),str(payload["product_state_binding_hash"]),
        str(payload["dynamic_motion_binding_hash"]),str(payload["archive_path"]),
        str(payload["archive_sha256"]),str(payload["runtime_binary_sha256"]),
        str(payload["source_binding_sha256"]),str(payload["playback_contract_hash"]),
        str(payload["reference_raster_contract_hash"]),int(payload["package_total_bytes"]),
        int(payload["runtime_binary_bytes"]),int(payload["view_count"]),int(payload["clip_count"]),
        int(payload["frame_count_total"]),dict(payload.get("qualification_report") or {}),
        str(payload["package_seal_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.RuntimeV4PackageSealIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.package_seal_hash!=runtime_v4_package_seal_hash(value):
        raise ValueError("RUNTIME_PACKAGE_SEAL_HASH_MISMATCH")
    return value
