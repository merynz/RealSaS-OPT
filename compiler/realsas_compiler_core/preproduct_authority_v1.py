from __future__ import annotations

"""Typed authority artifacts for mainline stages 03-23.

Heavy learned fitting may execute on an external GPU worker, but mainline never treats
an unbound filename or a model-authored canonical decision as product authority.
Preregistration, execution receipts, exact checkpoint/result/proposal bytes, and
Compiler-side qualification remain separate hash-bound artifacts.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .hashing import content_sha256

Json=dict[str,Any]
Vec3=tuple[float,float,float]


def _hash_without(value,field_name:str)->str:
    payload=value.to_dict(); payload.pop(field_name,None)
    return content_sha256(payload)


@dataclass(frozen=True)
class SourceMechanicalAuditViewIR:
    view_index:int
    source_raster_sha256:str
    foreground_mask_sha256:str
    width:int
    height:int
    foreground_pixel_count:int
    border_margin_px:int
    touches_border:bool
    schema_version:str="RealSaS.SourceMechanicalAuditViewIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class SourceMechanicalAuditIR:
    source_byte_seal_sha256:str
    views:tuple[SourceMechanicalAuditViewIR,...]
    minimum_required_border_margin_px:int
    audit_report:Json
    audit_hash:str
    schema_version:str="RealSaS.SourceMechanicalAuditIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def source_mechanical_audit_hash(value:SourceMechanicalAuditIR)->str:
    return _hash_without(value,"audit_hash")


@dataclass(frozen=True)
class FullSubjectAdmissionIR:
    source_mechanical_audit_binding_hash:str
    admitted_view_indices:tuple[int,...]
    admission_mode:str
    admission_report:Json
    admission_hash:str
    schema_version:str="RealSaS.FullSubjectAdmissionIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def full_subject_admission_hash(value:FullSubjectAdmissionIR)->str:
    return _hash_without(value,"admission_hash")


@dataclass(frozen=True)
class ObservationRenderViewIR:
    view_index:int
    source_raster_path:str
    source_raster_sha256:str
    width:int
    height:int
    schema_version:str="RealSaS.ObservationRenderViewIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class ObservationRenderSetIR:
    admission_binding_hash:str
    camera_set_binding_hash:str
    views:tuple[ObservationRenderViewIR,...]
    render_set_hash:str
    schema_version:str="RealSaS.ObservationRenderSetIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def observation_render_set_hash(value:ObservationRenderSetIR)->str:
    return _hash_without(value,"render_set_hash")


@dataclass(frozen=True)
class NormalizationDomainIR:
    observation_set_binding_hash:str
    camera_set_binding_hash:str
    center_xyz:Vec3
    half_extent:float
    normalized_bounds_min:Vec3
    normalized_bounds_max:Vec3
    coordinate_frame:str
    normalization_hash:str
    schema_version:str="RealSaS.NormalizationDomainIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def normalization_domain_hash(value:NormalizationDomainIR)->str:
    return _hash_without(value,"normalization_hash")


@dataclass(frozen=True)
class ModelFitPreregistrationIR:
    lane:str
    architecture_id:str
    executor_kind:str
    random_seed:int
    model_source_path:str
    model_source_sha256:str
    preregistration_document_path:str
    preregistration_document_sha256:str
    upstream_bindings:tuple[tuple[str,str],...]
    expected_output_contract:tuple[str,...]
    qualification_policy:Json
    preregistration_hash:str
    schema_version:str="RealSaS.ModelFitPreregistrationIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def model_fit_preregistration_hash(value:ModelFitPreregistrationIR)->str:
    return _hash_without(value,"preregistration_hash")


@dataclass(frozen=True)
class ModelFitExecutionIR:
    lane:str
    architecture_id:str
    preregistration_binding_hash:str
    model_source_sha256:str
    upstream_bindings:tuple[tuple[str,str],...]
    execution_receipt_path:str
    execution_receipt_sha256:str
    checkpoint_path:str
    checkpoint_sha256:str
    result_path:str
    result_sha256:str
    proposal_path:str|None
    proposal_sha256:str|None
    qualification_policy:Json
    execution_report:Json
    execution_hash:str
    schema_version:str="RealSaS.ModelFitExecutionIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def model_fit_execution_hash(value:ModelFitExecutionIR)->str:
    return _hash_without(value,"execution_hash")


@dataclass(frozen=True)
class ModelCheckpointSealIR:
    lane:str
    execution_binding_hash:str
    checkpoint_path:str
    checkpoint_sha256:str
    result_sha256:str
    qualified_output_binding_hash:str|None
    checkpoint_seal_hash:str
    schema_version:str="RealSaS.ModelCheckpointSealIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def model_checkpoint_seal_hash(value:ModelCheckpointSealIR)->str:
    return _hash_without(value,"checkpoint_seal_hash")


@dataclass(frozen=True)
class SignedZeroSurfaceSealIR:
    checkpoint_seal_binding_hash:str
    observation_set_binding_hash:str
    normalization_binding_hash:str
    npz_path:str
    npz_sha256:str
    metadata_path:str
    metadata_sha256:str
    vertex_count:int
    face_count:int
    vertices_sha256:str
    faces_sha256:str
    implicit_normals_sha256:str
    zero_surface_hash:str
    schema_version:str="RealSaS.SignedZeroSurfaceSealIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def signed_zero_surface_hash(value:SignedZeroSurfaceSealIR)->str:
    return _hash_without(value,"zero_surface_hash")


@dataclass(frozen=True)
class RestReprojectionGeometryViewIR:
    view_index:int
    recall:float
    precision:float
    largest_coherent_hole_fraction:float
    interior_uncovered_fraction:float
    source_foreground_pixel_count:int
    predicted_pixel_count:int
    passed:bool
    schema_version:str="RealSaS.RestReprojectionGeometryViewIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RestReprojectionGeometryGateIR:
    zero_surface_binding_hash:str
    observation_set_binding_hash:str
    camera_set_binding_hash:str
    threshold_policy:Json
    views:tuple[RestReprojectionGeometryViewIR,...]
    qualification_report:Json
    geometry_gate_hash:str
    schema_version:str="RealSaS.RestReprojectionGeometryGateIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def rest_reprojection_geometry_gate_hash(value:RestReprojectionGeometryGateIR)->str:
    return _hash_without(value,"geometry_gate_hash")


@dataclass(frozen=True)
class RiggingSurfaceQualificationIR:
    surface_binding_hash:str
    tensorization_hash:str
    certificate_hash:str
    node_count:int
    edge_count:int
    observed_node_count:int
    completed_node_count:int
    qualification_report:Json
    qualification_hash:str
    schema_version:str="RealSaS.RiggingSurfaceQualificationIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def rigging_surface_qualification_hash(value:RiggingSurfaceQualificationIR)->str:
    return _hash_without(value,"qualification_hash")


def source_mechanical_audit_from_dict(p:Mapping[str,Any])->SourceMechanicalAuditIR:
    rows=tuple(SourceMechanicalAuditViewIR(
        int(r["view_index"]),str(r["source_raster_sha256"]),str(r["foreground_mask_sha256"]),
        int(r["width"]),int(r["height"]),int(r["foreground_pixel_count"]),int(r["border_margin_px"]),
        bool(r["touches_border"]),
        schema_version=str(r.get("schema_version") or "RealSaS.SourceMechanicalAuditViewIR.v1"),
        metadata=dict(r.get("metadata") or {}),
    ) for r in p.get("views") or ())
    v=SourceMechanicalAuditIR(str(p["source_byte_seal_sha256"]),rows,int(p["minimum_required_border_margin_px"]),
        dict(p.get("audit_report") or {}),str(p["audit_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.SourceMechanicalAuditIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.audit_hash!=source_mechanical_audit_hash(v): raise ValueError("SOURCE_MECHANICAL_AUDIT_HASH_MISMATCH")
    return v


def full_subject_admission_from_dict(p:Mapping[str,Any])->FullSubjectAdmissionIR:
    v=FullSubjectAdmissionIR(str(p["source_mechanical_audit_binding_hash"]),tuple(map(int,p.get("admitted_view_indices") or ())),
        str(p["admission_mode"]),dict(p.get("admission_report") or {}),str(p["admission_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.FullSubjectAdmissionIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.admission_hash!=full_subject_admission_hash(v): raise ValueError("FULL_SUBJECT_ADMISSION_HASH_MISMATCH")
    return v


def observation_render_set_from_dict(p:Mapping[str,Any])->ObservationRenderSetIR:
    rows=tuple(ObservationRenderViewIR(
        int(r["view_index"]),str(r["source_raster_path"]),str(r["source_raster_sha256"]),int(r["width"]),int(r["height"]),
        schema_version=str(r.get("schema_version") or "RealSaS.ObservationRenderViewIR.v1"),metadata=dict(r.get("metadata") or {})
    ) for r in p.get("views") or ())
    v=ObservationRenderSetIR(str(p["admission_binding_hash"]),str(p["camera_set_binding_hash"]),rows,str(p["render_set_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.ObservationRenderSetIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.render_set_hash!=observation_render_set_hash(v): raise ValueError("OBSERVATION_RENDER_SET_HASH_MISMATCH")
    return v


def normalization_domain_from_dict(p:Mapping[str,Any])->NormalizationDomainIR:
    v=NormalizationDomainIR(
        str(p["observation_set_binding_hash"]),str(p["camera_set_binding_hash"]),tuple(map(float,p["center_xyz"])),
        float(p["half_extent"]),tuple(map(float,p["normalized_bounds_min"])),tuple(map(float,p["normalized_bounds_max"])),
        str(p["coordinate_frame"]),str(p["normalization_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.NormalizationDomainIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.normalization_hash!=normalization_domain_hash(v): raise ValueError("NORMALIZATION_DOMAIN_HASH_MISMATCH")
    return v


def model_fit_preregistration_from_dict(p:Mapping[str,Any])->ModelFitPreregistrationIR:
    v=ModelFitPreregistrationIR(
        str(p["lane"]),str(p["architecture_id"]),str(p["executor_kind"]),int(p["random_seed"]),
        str(p["model_source_path"]),str(p["model_source_sha256"]),str(p["preregistration_document_path"]),
        str(p["preregistration_document_sha256"]),tuple((str(a),str(b)) for a,b in p.get("upstream_bindings") or ()),
        tuple(map(str,p.get("expected_output_contract") or ())),dict(p.get("qualification_policy") or {}),
        str(p["preregistration_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.ModelFitPreregistrationIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.preregistration_hash!=model_fit_preregistration_hash(v): raise ValueError("MODEL_FIT_PREREG_HASH_MISMATCH")
    return v


def model_fit_execution_from_dict(p:Mapping[str,Any])->ModelFitExecutionIR:
    v=ModelFitExecutionIR(
        str(p["lane"]),str(p["architecture_id"]),str(p["preregistration_binding_hash"]),str(p["model_source_sha256"]),
        tuple((str(a),str(b)) for a,b in p.get("upstream_bindings") or ()),str(p["execution_receipt_path"]),
        str(p["execution_receipt_sha256"]),str(p["checkpoint_path"]),str(p["checkpoint_sha256"]),
        str(p["result_path"]),str(p["result_sha256"]),
        None if p.get("proposal_path") is None else str(p["proposal_path"]),
        None if p.get("proposal_sha256") is None else str(p["proposal_sha256"]),
        dict(p.get("qualification_policy") or {}),dict(p.get("execution_report") or {}),str(p["execution_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.ModelFitExecutionIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.execution_hash!=model_fit_execution_hash(v): raise ValueError("MODEL_FIT_EXECUTION_HASH_MISMATCH")
    return v


def model_checkpoint_seal_from_dict(p:Mapping[str,Any])->ModelCheckpointSealIR:
    v=ModelCheckpointSealIR(
        str(p["lane"]),str(p["execution_binding_hash"]),str(p["checkpoint_path"]),str(p["checkpoint_sha256"]),
        str(p["result_sha256"]),None if p.get("qualified_output_binding_hash") is None else str(p["qualified_output_binding_hash"]),
        str(p["checkpoint_seal_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.ModelCheckpointSealIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.checkpoint_seal_hash!=model_checkpoint_seal_hash(v): raise ValueError("MODEL_CHECKPOINT_SEAL_HASH_MISMATCH")
    return v


def signed_zero_surface_from_dict(p:Mapping[str,Any])->SignedZeroSurfaceSealIR:
    v=SignedZeroSurfaceSealIR(
        str(p["checkpoint_seal_binding_hash"]),str(p["observation_set_binding_hash"]),str(p["normalization_binding_hash"]),
        str(p["npz_path"]),str(p["npz_sha256"]),str(p["metadata_path"]),str(p["metadata_sha256"]),
        int(p["vertex_count"]),int(p["face_count"]),str(p["vertices_sha256"]),str(p["faces_sha256"]),
        str(p["implicit_normals_sha256"]),str(p["zero_surface_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.SignedZeroSurfaceSealIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.zero_surface_hash!=signed_zero_surface_hash(v): raise ValueError("SIGNED_ZERO_SURFACE_HASH_MISMATCH")
    return v


def rest_reprojection_geometry_gate_from_dict(p:Mapping[str,Any])->RestReprojectionGeometryGateIR:
    rows=tuple(RestReprojectionGeometryViewIR(
        int(r["view_index"]),float(r["recall"]),float(r["precision"]),float(r["largest_coherent_hole_fraction"]),
        float(r["interior_uncovered_fraction"]),int(r["source_foreground_pixel_count"]),int(r["predicted_pixel_count"]),
        bool(r["passed"]),schema_version=str(r.get("schema_version") or "RealSaS.RestReprojectionGeometryViewIR.v1"),
        metadata=dict(r.get("metadata") or {})
    ) for r in p.get("views") or ())
    v=RestReprojectionGeometryGateIR(str(p["zero_surface_binding_hash"]),str(p["observation_set_binding_hash"]),
        str(p["camera_set_binding_hash"]),dict(p.get("threshold_policy") or {}),rows,
        dict(p.get("qualification_report") or {}),str(p["geometry_gate_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.RestReprojectionGeometryGateIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.geometry_gate_hash!=rest_reprojection_geometry_gate_hash(v): raise ValueError("REST_REPROJECTION_GATE_HASH_MISMATCH")
    return v


def rigging_surface_qualification_from_dict(p:Mapping[str,Any])->RiggingSurfaceQualificationIR:
    v=RiggingSurfaceQualificationIR(str(p["surface_binding_hash"]),str(p["tensorization_hash"]),str(p["certificate_hash"]),
        int(p["node_count"]),int(p["edge_count"]),int(p["observed_node_count"]),int(p["completed_node_count"]),
        dict(p.get("qualification_report") or {}),str(p["qualification_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.RiggingSurfaceQualificationIR.v1"),metadata=dict(p.get("metadata") or {}))
    if v.qualification_hash!=rigging_surface_qualification_hash(v): raise ValueError("RIGGING_SURFACE_QUALIFICATION_HASH_MISMATCH")
    return v
