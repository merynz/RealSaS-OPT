from __future__ import annotations

"""Stage-38/39 native package playback and visual-motion evidence IRs."""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256

Json=dict[str,Any]


@dataclass(frozen=True)
class NativePlaybackProbeIR:
    clip_id:str
    view_id:str
    time_seconds:float
    rendered_png_path:str
    rendered_png_sha256:str
    rendered_rgba_sha256:str
    visible_alpha_pixel_count:int
    stdout_sha256:str
    probe_hash:str
    schema_version:str="RealSaS.NativePlaybackProbeIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedNativePlaybackIR:
    package_seal_binding_hash:str
    projection_binding_hash:str
    native_player_sha256:str
    probes:tuple[NativePlaybackProbeIR,...]
    qualification_report:Json
    native_playback_hash:str
    schema_version:str="RealSaS.QualifiedNativePlaybackIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class VisualMotionViewEvidenceIR:
    clip_id:str
    view_id:str
    frame_count:int
    unique_rgba_frame_count:int
    visible_alpha_all_frames:bool
    rgba_sequence_hash:str
    changed_over_time:bool
    evidence_hash:str
    schema_version:str="RealSaS.VisualMotionViewEvidenceIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedVisualMotionEvidenceIR:
    package_seal_binding_hash:str
    projection_binding_hash:str
    native_playback_binding_hash:str
    native_player_sha256:str
    views:tuple[VisualMotionViewEvidenceIR,...]
    visible_motion_clip_ids:tuple[str,...]
    professional_visible_motion_clip_ids:tuple[str,...]
    qualification_report:Json
    visual_motion_hash:str
    schema_version:str="RealSaS.QualifiedVisualMotionEvidenceIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def _hash_without(value,name):
    payload=value.to_dict(); payload.pop(name,None)
    return content_sha256(payload)

def native_probe_hash(v): return _hash_without(v,"probe_hash")
def native_playback_hash(v): return _hash_without(v,"native_playback_hash")
def visual_view_hash(v): return _hash_without(v,"evidence_hash")
def visual_motion_hash(v): return _hash_without(v,"visual_motion_hash")


def seal_native_playback(*,package_seal,projection,native_player_sha256:str,probes):
    rows=tuple(sorted(probes,key=lambda x:(x.clip_id,x.view_id)))
    expected={(c.clip_id,v.view_id) for c in projection.clips for v in projection.views}
    actual={(p.clip_id,p.view_id) for p in rows}
    if actual!=expected or len(rows)!=len(expected):
        raise ValueError("NATIVE_PLAYBACK_PROBE_MATRIX_INCOMPLETE")
    if any(p.visible_alpha_pixel_count<=0 for p in rows):
        raise ValueError("NATIVE_PLAYBACK_TRANSPARENT_OUTPUT")
    value=QualifiedNativePlaybackIR(
        package_seal.package_seal_hash,projection.projection_hash,str(native_player_sha256),rows,
        {
            "status":"PASS_NATIVE_PACKAGE_OPEN_PLAYBACK",
            "all_clip_view_probes_rendered":True,
            "all_probes_visible_alpha":True,
            "renderer_contract":"RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE",
            "motion_quality_claimed":False,
        },"",
        metadata={"probe_count":len(rows),"native_execution_required":True},
    )
    return replace(value,native_playback_hash=native_playback_hash(value))


def seal_visual_motion(*,package_seal,projection,native_playback,native_player_sha256:str,views):
    rows=tuple(sorted(views,key=lambda x:(x.clip_id,x.view_id)))
    expected={(c.clip_id,v.view_id) for c in projection.clips for v in projection.views}
    actual={(p.clip_id,p.view_id) for p in rows}
    if actual!=expected or len(rows)!=len(expected):
        raise ValueError("VISUAL_MOTION_VIEW_MATRIX_INCOMPLETE")
    if any(not row.visible_alpha_all_frames for row in rows):
        raise ValueError("VISUAL_MOTION_ALPHA_VISIBILITY_FAIL")
    visible=tuple(sorted({row.clip_id for row in rows if row.changed_over_time}))
    professional={c.clip_id for c in projection.clips if bool(c.metadata.get("professional_motion_evidence",False))}
    professional_visible=tuple(sorted(professional.intersection(visible)))
    value=QualifiedVisualMotionEvidenceIR(
        package_seal.package_seal_hash,projection.projection_hash,native_playback.native_playback_hash,
        str(native_player_sha256),rows,visible,professional_visible,
        {
            "status":"PASS_NATIVE_VISUAL_BAKE",
            "all_frames_native_rendered":True,
            "visible_motion_clip_count":len(visible),
            "professional_visible_motion_clip_count":len(professional_visible),
            "founder_visual_pass_claimed":False,
            "programmatic_visual_motion_evidence_only":True,
        },"",
        metadata={"professional_clip_ids":tuple(sorted(professional))},
    )
    return replace(value,visual_motion_hash=visual_motion_hash(value))


def native_playback_from_dict(payload:Mapping[str,Any])->QualifiedNativePlaybackIR:
    probes=tuple(NativePlaybackProbeIR(
        str(p["clip_id"]),str(p["view_id"]),float(p["time_seconds"]),str(p["rendered_png_path"]),
        str(p["rendered_png_sha256"]),str(p["rendered_rgba_sha256"]),int(p["visible_alpha_pixel_count"]),
        str(p["stdout_sha256"]),str(p["probe_hash"]),
        schema_version=str(p.get("schema_version") or "RealSaS.NativePlaybackProbeIR.v1"),
        metadata=dict(p.get("metadata") or {}),
    ) for p in payload.get("probes") or ())
    value=QualifiedNativePlaybackIR(
        str(payload["package_seal_binding_hash"]),str(payload["projection_binding_hash"]),
        str(payload["native_player_sha256"]),probes,dict(payload.get("qualification_report") or {}),
        str(payload["native_playback_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedNativePlaybackIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.native_playback_hash!=native_playback_hash(value): raise ValueError("NATIVE_PLAYBACK_HASH_MISMATCH")
    for p in probes:
        if p.probe_hash!=native_probe_hash(p): raise ValueError("NATIVE_PLAYBACK_PROBE_HASH_MISMATCH")
    return value


def visual_motion_from_dict(payload:Mapping[str,Any])->QualifiedVisualMotionEvidenceIR:
    rows=tuple(VisualMotionViewEvidenceIR(
        str(v["clip_id"]),str(v["view_id"]),int(v["frame_count"]),int(v["unique_rgba_frame_count"]),
        bool(v["visible_alpha_all_frames"]),str(v["rgba_sequence_hash"]),bool(v["changed_over_time"]),
        str(v["evidence_hash"]),
        schema_version=str(v.get("schema_version") or "RealSaS.VisualMotionViewEvidenceIR.v1"),
        metadata=dict(v.get("metadata") or {}),
    ) for v in payload.get("views") or ())
    value=QualifiedVisualMotionEvidenceIR(
        str(payload["package_seal_binding_hash"]),str(payload["projection_binding_hash"]),
        str(payload["native_playback_binding_hash"]),str(payload["native_player_sha256"]),rows,
        tuple(map(str,payload.get("visible_motion_clip_ids") or ())),
        tuple(map(str,payload.get("professional_visible_motion_clip_ids") or ())),
        dict(payload.get("qualification_report") or {}),str(payload["visual_motion_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.QualifiedVisualMotionEvidenceIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.visual_motion_hash!=visual_motion_hash(value): raise ValueError("VISUAL_MOTION_HASH_MISMATCH")
    for row in rows:
        if row.evidence_hash!=visual_view_hash(row): raise ValueError("VISUAL_MOTION_VIEW_HASH_MISMATCH")
    return value
