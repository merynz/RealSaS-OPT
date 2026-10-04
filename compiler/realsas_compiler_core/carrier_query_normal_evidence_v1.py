from __future__ import annotations

"""Carrier-bound signed normal evidence for learned mechanical queries.

Mechanical carrier topology/XYZ and learned-model signed normal evidence are
separate authorities.

Why this exists:
- Stage18 candidate face tuples are canonicalized for deterministic topology and
  do not preserve oriented winding.
- A cross product of those tuples therefore cannot own a signed normal field.
- GSA/RiggingSurfaceIR already carries scene-first signed normals.
- Those signed normals may be transported onto the exact carrier only through
  the carrier vertex's *geometry* SurfaceSupportBinding.

This artifact is model-query evidence. It is not a claim that GSA normals own
mechanical topology and it deliberately never consumes seam skin-support
coefficients.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]
SCHEMA = "RealSaS.CarrierSignedQueryNormalEvidenceIR.v1"


@dataclass(frozen=True)
class CarrierSignedQueryNormalEvidenceIR:
    candidate_mesh_binding_hash: str
    mechanical_carrier_evidence_binding_hash: str
    source_surface_binding_hash: str
    ordered_vertex_ids: tuple[str, ...]
    signed_normals: tuple[Vec3, ...]
    normal_valid: tuple[bool, ...]
    geometry_support_hash: str
    query_normal_evidence_hash: str
    schema_version: str = SCHEMA
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_normal(value) -> Vec3:
    raw=tuple(float(x) for x in value)
    if len(raw)!=3 or not all(math.isfinite(x) for x in raw):
        raise QualificationError("CARRIER_QUERY_NORMAL_VEC3_INVALID")
    length=math.sqrt(sum(x*x for x in raw))
    if length<=1e-12:
        raise QualificationError("CARRIER_QUERY_NORMAL_ZERO_SOURCE")
    return tuple(x/length for x in raw)  # type: ignore[return-value]


def carrier_signed_query_normal_evidence_hash(
    value: CarrierSignedQueryNormalEvidenceIR,
) -> str:
    payload=value.to_dict()
    payload.pop("query_normal_evidence_hash",None)
    return content_sha256(payload)


def _geometry_support(vertex) -> tuple[tuple[str,float],...]:
    binding=vertex.support_binding
    rows=tuple((str(sid),float(weight)) for sid,weight in binding.coefficients)
    if not rows:
        raise QualificationError("CARRIER_QUERY_NORMAL_GEOMETRY_SUPPORT_EMPTY")
    seen=set()
    total=0.0
    out=[]
    for sid,weight in rows:
        if (
            not sid
            or sid in seen
            or not math.isfinite(weight)
            or weight<0.0
        ):
            raise QualificationError("CARRIER_QUERY_NORMAL_GEOMETRY_SUPPORT_INVALID")
        seen.add(sid)
        total+=weight
        out.append((sid,weight))
    if abs(total-1.0)>1e-8:
        raise QualificationError("CARRIER_QUERY_NORMAL_GEOMETRY_SUPPORT_SIMPLEX")
    return tuple(out)


def build_carrier_signed_query_normal_evidence_v1(
    candidate,
    *,
    mechanical_carrier_evidence,
    surface,
) -> CarrierSignedQueryNormalEvidenceIR:
    if (
        str(candidate.candidate_lineage_hash)
        != str(mechanical_carrier_evidence.candidate_mesh_binding_hash)
    ):
        raise QualificationError("CARRIER_QUERY_NORMAL_CANDIDATE_BINDING_DRIFT")
    if (
        str(candidate.surface_binding_hash)
        != str(surface.geometry_lineage_hash)
        or str(mechanical_carrier_evidence.source_surface_binding_hash)
        != str(surface.geometry_lineage_hash)
    ):
        raise QualificationError("CARRIER_QUERY_NORMAL_SURFACE_BINDING_DRIFT")

    ordered=tuple(
        sorted(candidate.vertices,key=lambda row:str(row.candidate_vertex_id))
    )
    ids=tuple(str(v.candidate_vertex_id) for v in ordered)
    if ids!=tuple(mechanical_carrier_evidence.ordered_vertex_ids):
        raise QualificationError("CARRIER_QUERY_NORMAL_VERTEX_ORDER_DRIFT")

    source={str(n.surface_id):n for n in surface.surface_nodes}
    if len(source)!=len(surface.surface_nodes):
        raise QualificationError("CARRIER_QUERY_NORMAL_SOURCE_ID_DUPLICATE")

    support_payload=[]
    normals=[]
    valid=[]
    invalid_missing=0
    invalid_cancellation=0

    for vertex in ordered:
        rows=_geometry_support(vertex)
        support_payload.append({
            "candidate_vertex_id":str(vertex.candidate_vertex_id),
            "mode":str(vertex.support_binding.mode),
            "geometry_support":rows,
        })
        accum=[0.0,0.0,0.0]
        complete=True
        for sid,coeff in rows:
            node=source.get(sid)
            if node is None:
                raise QualificationError("CARRIER_QUERY_NORMAL_SUPPORT_OUTSIDE_SURFACE")
            if node.derived_normal is None:
                complete=False
                continue
            n=_finite_normal(node.derived_normal)
            accum[0]+=coeff*n[0]
            accum[1]+=coeff*n[1]
            accum[2]+=coeff*n[2]

        if not complete:
            normals.append((0.0,0.0,0.0))
            valid.append(False)
            invalid_missing+=1
            continue

        length=math.sqrt(sum(x*x for x in accum))
        if not math.isfinite(length) or length<=1e-12:
            normals.append((0.0,0.0,0.0))
            valid.append(False)
            invalid_cancellation+=1
            continue

        normals.append(tuple(float(x/length) for x in accum))
        valid.append(True)

    support_hash=content_sha256({
        "schema":"RealSaS.CarrierQueryNormalGeometrySupport.v1",
        "candidate_mesh_binding_hash":str(candidate.candidate_lineage_hash),
        "source_surface_binding_hash":str(surface.geometry_lineage_hash),
        "rows":tuple(support_payload),
    })

    value=CarrierSignedQueryNormalEvidenceIR(
        candidate_mesh_binding_hash=str(candidate.candidate_lineage_hash),
        mechanical_carrier_evidence_binding_hash=str(
            mechanical_carrier_evidence.carrier_evidence_hash
        ),
        source_surface_binding_hash=str(surface.geometry_lineage_hash),
        ordered_vertex_ids=ids,
        signed_normals=tuple(normals),
        normal_valid=tuple(valid),
        geometry_support_hash=support_hash,
        query_normal_evidence_hash="",
        metadata={
            "authority":"CARRIER_BOUND_SIGNED_MODEL_QUERY_EVIDENCE",
            "signed_normal_source":"RIGGING_SURFACE_DERIVED_NORMAL",
            "transport":"EXACT_GEOMETRY_SURFACE_SUPPORT_BINDING",
            "mechanical_topology_authority":False,
            "skin_support_consumed":False,
            "geometry_support_consumed":True,
            "canonical_face_winding_used":False,
            "invalid_missing_source_normal_count":int(invalid_missing),
            "invalid_normal_cancellation_count":int(invalid_cancellation),
            "valid_fraction":float(sum(valid)/max(len(valid),1)),
        },
    )
    return replace(
        value,
        query_normal_evidence_hash=carrier_signed_query_normal_evidence_hash(value),
    )


def carrier_signed_query_normal_evidence_from_dict(
    payload: Mapping[str,Any],
) -> CarrierSignedQueryNormalEvidenceIR:
    schema=str(payload.get("schema_version") or payload.get("schema") or "")
    if schema!=SCHEMA:
        raise QualificationError("CARRIER_QUERY_NORMAL_SCHEMA_DRIFT")
    value=CarrierSignedQueryNormalEvidenceIR(
        candidate_mesh_binding_hash=str(payload["candidate_mesh_binding_hash"]),
        mechanical_carrier_evidence_binding_hash=str(
            payload["mechanical_carrier_evidence_binding_hash"]
        ),
        source_surface_binding_hash=str(payload["source_surface_binding_hash"]),
        ordered_vertex_ids=tuple(map(str,payload.get("ordered_vertex_ids") or ())),
        signed_normals=tuple(
            tuple(float(x) for x in row)
            for row in payload.get("signed_normals") or ()
        ),
        normal_valid=tuple(bool(x) for x in payload.get("normal_valid") or ()),
        geometry_support_hash=str(payload["geometry_support_hash"]),
        query_normal_evidence_hash=str(payload["query_normal_evidence_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    n=len(value.ordered_vertex_ids)
    if (
        n<1
        or len(value.signed_normals)!=n
        or len(value.normal_valid)!=n
        or len(set(value.ordered_vertex_ids))!=n
    ):
        raise QualificationError("CARRIER_QUERY_NORMAL_SHAPE_DRIFT")
    for row,flag in zip(value.signed_normals,value.normal_valid):
        if len(row)!=3 or not all(math.isfinite(float(x)) for x in row):
            raise QualificationError("CARRIER_QUERY_NORMAL_VEC3_DRIFT")
        length=math.sqrt(sum(float(x)*float(x) for x in row))
        if flag and abs(length-1.0)>1e-5:
            raise QualificationError("CARRIER_QUERY_NORMAL_UNIT_DRIFT")
        if not flag and length>1e-12:
            raise QualificationError("CARRIER_QUERY_NORMAL_INVALID_ROW_NONZERO")
    if (
        value.query_normal_evidence_hash
        != carrier_signed_query_normal_evidence_hash(value)
    ):
        raise QualificationError("CARRIER_QUERY_NORMAL_HASH_MISMATCH")
    return value


__all__=[
    "CarrierSignedQueryNormalEvidenceIR",
    "build_carrier_signed_query_normal_evidence_v1",
    "carrier_signed_query_normal_evidence_from_dict",
    "carrier_signed_query_normal_evidence_hash",
]
