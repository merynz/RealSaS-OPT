from __future__ import annotations

"""Native Runtime-v4 execution helpers shared by stages 38 and 39."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.runtime_native_proof_v1 import (
    NativePlaybackProbeIR, VisualMotionViewEvidenceIR, native_probe_hash,
    seal_native_playback, seal_visual_motion, visual_view_hash,
)
from compiler.realsas_compiler_core.runtime_package_v1 import runtime_v4_package_seal_from_dict
from compiler.realsas_compiler_core.runtime_projection_v1 import runtime_v4_projection_from_dict
from compiler.realsas_compiler_core.runtime_native_proof_v1 import native_playback_from_dict
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.product_mesh_v1 import (
    _resolved_path,_sha256,_stage_output_payload,_write_ir,
)


def _player(ctx):
    cfg=dict(ctx["run_manifest"].get("runtime") or {})
    ref=dict(cfg.get("native_player") or {})
    path=_resolved_path(str(ref.get("path") or ""))
    expected=str(ref.get("sha256") or "")
    if not path.is_file() or len(expected)!=64 or _sha256(path)!=expected:
        raise QualificationError("RUNTIME_NATIVE_PLAYER_REF_INVALID")
    return path,expected


def _rgba(path:Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"),dtype=np.uint8)


def _run_render(player:Path,package:Path,clip_id:str,view_id:str,t:float,out:Path):
    out.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.run(
        [str(player),str(package),"--clip",clip_id,"--view",view_id,"--time",str(float(t)),"--out",str(out)],
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=False,
    )
    if proc.returncode!=0:
        raise QualificationError(f"RUNTIME_NATIVE_PLAYER_FAILED:{clip_id}:{view_id}:{proc.stdout[-500:]}")
    if "renderer=RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE" not in proc.stdout:
        raise QualificationError("RUNTIME_NATIVE_RENDERER_CONTRACT_DRIFT")
    if not out.is_file():
        raise QualificationError("RUNTIME_NATIVE_RENDER_OUTPUT_MISSING")
    rgba=_rgba(out)
    if rgba.ndim!=3 or rgba.shape[2]!=4:
        raise QualificationError("RUNTIME_NATIVE_RENDER_RGBA_INVALID")
    rgba_hash=sha256(np.ascontiguousarray(rgba).tobytes()).hexdigest()
    alpha=int(np.count_nonzero(rgba[:,:,3]))
    return proc.stdout,rgba,rgba_hash,alpha


def native_package_open_playback_stage(ctx:dict)->dict:
    projection=runtime_v4_projection_from_dict(
        _stage_output_payload(ctx,"36_RUNTIME_V4_PROJECT","RealSaS.RuntimeV4ProjectionIR.v1")
    )
    package=runtime_v4_package_seal_from_dict(
        _stage_output_payload(ctx,"37_RSS_MATERIALIZE_COMPACT","RealSaS.RuntimeV4PackageSealIR.v1")
    )
    if package.projection_binding_hash!=projection.projection_hash:
        raise QualificationError("RUNTIME_STAGE38_PACKAGE_PROJECTION_DRIFT")
    archive=Path(package.archive_path).resolve()
    if not archive.is_file() or _sha256(archive)!=package.archive_sha256:
        raise QualificationError("RUNTIME_STAGE38_PACKAGE_BYTES_DRIFT")
    player,player_sha=_player(ctx)
    root=ctx["run_root"]/"artifacts"/"38_NATIVE_PACKAGE_OPEN_PLAYBACK"
    probes=[]; outputs=[]
    for clip in projection.clips:
        t=0.5*float(clip.duration_seconds)
        for view in projection.views:
            out=root/"frames"/clip.clip_id/f"{view.view_id}.png"
            stdout,rgba,rgba_hash,alpha=_run_render(player,archive,clip.clip_id,view.view_id,t,out)
            if rgba.shape[:2]!=(int(view.camera["resolution"]),int(view.camera["resolution"])):
                raise QualificationError("RUNTIME_STAGE38_FRAMEBUFFER_DIMENSION_DRIFT")
            probe=NativePlaybackProbeIR(
                clip.clip_id,view.view_id,t,str(out),_sha256(out),rgba_hash,alpha,
                sha256(stdout.encode("utf-8")).hexdigest(),"",
                metadata={"native_player_sha256":player_sha},
            )
            probe=replace(probe,probe_hash=native_probe_hash(probe)); probes.append(probe)
            outputs.append({"path":str(out),"sha256":probe.rendered_png_sha256,"authority_class":"NATIVE_PLAYBACK_PROBE_PNG","schema":"image/png"})
    evidence=seal_native_playback(package_seal=package,projection=projection,native_player_sha256=player_sha,probes=tuple(probes))
    outputs.insert(0,_write_ir(root/"qualified_native_playback.json",evidence,authority_class="QUALIFIED_NATIVE_PLAYBACK"))
    return {"status":"PASS","outputs":outputs,"diagnostics":{
        "native_playback_hash":evidence.native_playback_hash,"probe_count":len(probes),
        "native_player_sha256":player_sha,
    }}


def visual_motion_render_bake_stage(ctx:dict)->dict:
    projection=runtime_v4_projection_from_dict(
        _stage_output_payload(ctx,"36_RUNTIME_V4_PROJECT","RealSaS.RuntimeV4ProjectionIR.v1")
    )
    package=runtime_v4_package_seal_from_dict(
        _stage_output_payload(ctx,"37_RSS_MATERIALIZE_COMPACT","RealSaS.RuntimeV4PackageSealIR.v1")
    )
    playback=native_playback_from_dict(
        _stage_output_payload(ctx,"38_NATIVE_PACKAGE_OPEN_PLAYBACK","RealSaS.QualifiedNativePlaybackIR.v1")
    )
    if playback.package_seal_binding_hash!=package.package_seal_hash or playback.projection_binding_hash!=projection.projection_hash:
        raise QualificationError("RUNTIME_STAGE39_UPSTREAM_BINDING_DRIFT")
    archive=Path(package.archive_path).resolve()
    if not archive.is_file() or _sha256(archive)!=package.archive_sha256:
        raise QualificationError("RUNTIME_STAGE39_PACKAGE_BYTES_DRIFT")
    player,player_sha=_player(ctx)
    if playback.native_player_sha256!=player_sha:
        raise QualificationError("RUNTIME_STAGE39_NATIVE_PLAYER_DRIFT")
    root=ctx["run_root"]/"artifacts"/"39_VISUAL_MOTION_RENDER_BAKE"
    rows=[]; outputs=[]
    for clip in projection.clips:
        times=tuple(float(f.time_seconds) for f in clip.frames)
        for view in projection.views:
            hashes=[]; all_visible=True
            for index,t in enumerate(times):
                out=root/"frames"/clip.clip_id/view.view_id/f"{index:04d}.png"
                _,rgba,rgba_hash,alpha=_run_render(player,archive,clip.clip_id,view.view_id,t,out)
                if rgba.shape[:2]!=(int(view.camera["resolution"]),int(view.camera["resolution"])):
                    raise QualificationError("RUNTIME_STAGE39_FRAMEBUFFER_DIMENSION_DRIFT")
                hashes.append(rgba_hash); all_visible=all_visible and alpha>0
                outputs.append({"path":str(out),"sha256":_sha256(out),"authority_class":"NATIVE_VISUAL_MOTION_FRAME","schema":"image/png"})
            unique=len(set(hashes))
            row=VisualMotionViewEvidenceIR(
                clip.clip_id,view.view_id,len(times),unique,all_visible,
                content_sha256({"clip_id":clip.clip_id,"view_id":view.view_id,"rgba_sha256":tuple(hashes)}),
                unique>1,"",
                metadata={"sample_times":times,"native_player_sha256":player_sha},
            )
            rows.append(replace(row,evidence_hash=visual_view_hash(row)))
    evidence=seal_visual_motion(
        package_seal=package,projection=projection,native_playback=playback,
        native_player_sha256=player_sha,views=tuple(rows),
    )
    outputs.insert(0,_write_ir(root/"qualified_visual_motion.json",evidence,authority_class="QUALIFIED_NATIVE_VISUAL_MOTION"))
    return {"status":"PASS","outputs":outputs,"diagnostics":{
        "visual_motion_hash":evidence.visual_motion_hash,
        "visible_motion_clip_count":len(evidence.visible_motion_clip_ids),
        "professional_visible_motion_clip_count":len(evidence.professional_visible_motion_clip_ids),
        "native_frame_count":sum(row.frame_count for row in rows),
        "founder_visual_pass_claimed":False,
    }}
