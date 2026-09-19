from __future__ import annotations

"""Native Runtime-v4 execution helpers shared by stages 38 and 39.

All multi-node native rendering uses the proven batch transport. The runtime package
is opened once per batch; exact content-addressed PNG cache hits are restored without
native execution. Scalar per-frame subprocess loops are intentionally forbidden here.
"""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

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
from compiler.realsas_compiler_services.cache.reference_render import NativeReferenceRenderCache
from compiler.realsas_compiler_services.cache.reference_render_batch import render_many_v4
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


def _cache(ctx,archive:Path,player:Path)->NativeReferenceRenderCache:
    return NativeReferenceRenderCache(
        runtime_package=archive,
        runtime_demo=player,
        cache_root=ctx["authority_root"]/"cache",
    )


def _batch_render(ctx,*,archive:Path,player:Path,requests):
    started=perf_counter()
    rows=render_many_v4(_cache(ctx,archive,player),requests)
    total_seconds=perf_counter()-started
    cache_hits=sum(bool(row.get("render_cache_hit",False)) for row in rows)
    native_batches={float(row["native_batch_seconds"]) for row in rows if row.get("native_batch_seconds") is not None}
    return rows,{
        "request_count":len(rows),
        "cache_hit_count":cache_hits,
        "cache_miss_count":len(rows)-cache_hits,
        "native_batch_process_count":1 if native_batches else 0,
        "native_batch_seconds":max(native_batches) if native_batches else 0.0,
        "batch_wall_seconds":total_seconds,
        "scalar_subprocess_loop_used":False,
    }


def _validate_render_result(row,*,resolution:int):
    path=Path(str(row["path"])).resolve()
    if not path.is_file() or _sha256(path)!=str(row["sha256"]):
        raise QualificationError("RUNTIME_NATIVE_BATCH_OUTPUT_IDENTITY_DRIFT")
    rgba=_rgba(path)
    if rgba.ndim!=3 or rgba.shape[2]!=4:
        raise QualificationError("RUNTIME_NATIVE_RENDER_RGBA_INVALID")
    if rgba.shape[:2]!=(int(resolution),int(resolution)):
        raise QualificationError("RUNTIME_NATIVE_FRAMEBUFFER_DIMENSION_DRIFT")
    rgba_hash=sha256(np.ascontiguousarray(rgba).tobytes()).hexdigest()
    alpha=int(np.count_nonzero(rgba[:,:,3]))
    stdout=str(row.get("stdout") or "")
    if "renderer=RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE" not in stdout:
        raise QualificationError("RUNTIME_NATIVE_RENDERER_CONTRACT_DRIFT")
    return path,rgba_hash,alpha,stdout


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

    requests=[]
    index={}
    for clip in projection.clips:
        t=0.5*float(clip.duration_seconds)
        for view in projection.views:
            out=root/"frames"/clip.clip_id/f"{view.view_id}.png"
            index[(clip.clip_id,view.view_id)]=(t,int(view.camera["resolution"]))
            requests.append({"clip":clip.clip_id,"view":view.view_id,"time_seconds":t,"out_png":out})

    rendered,perf=_batch_render(ctx,archive=archive,player=player,requests=requests)
    probes=[]; outputs=[]
    for row in rendered:
        clip_id=str(next(req["clip"] for req in requests if str(req["out_png"])==str(row["path"])))
        view_id=str(row["view"])
        t,resolution=index[(clip_id,view_id)]
        path,rgba_hash,alpha,stdout=_validate_render_result(row,resolution=resolution)
        probe=NativePlaybackProbeIR(
            clip_id,view_id,t,str(path),str(row["sha256"]),rgba_hash,alpha,
            sha256(stdout.encode("utf-8")).hexdigest(),"",
            metadata={
                "native_player_sha256":player_sha,
                "render_cache_hit":bool(row.get("render_cache_hit",False)),
                "render_cache_key":str(row.get("render_cache_key") or ""),
                "native_batch":bool(row.get("native_batch",False)),
            },
        )
        probe=replace(probe,probe_hash=native_probe_hash(probe)); probes.append(probe)
        outputs.append({"path":str(path),"sha256":probe.rendered_png_sha256,"authority_class":"NATIVE_PLAYBACK_PROBE_PNG","schema":"image/png"})

    evidence=seal_native_playback(package_seal=package,projection=projection,native_player_sha256=player_sha,probes=tuple(probes))
    outputs.insert(0,_write_ir(root/"qualified_native_playback.json",evidence,authority_class="QUALIFIED_NATIVE_PLAYBACK"))
    return {"status":"PASS","outputs":outputs,"diagnostics":{
        "native_playback_hash":evidence.native_playback_hash,"probe_count":len(probes),
        "native_player_sha256":player_sha,
        "transport":"CONTENT_ADDRESSED_BATCH_NATIVE_V4",
        **perf,
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

    requests=[]; resolution_by_view={v.view_id:int(v.camera["resolution"]) for v in projection.views}
    for clip in projection.clips:
        for view in projection.views:
            for index,frame in enumerate(clip.frames):
                requests.append({
                    "clip":clip.clip_id,"view":view.view_id,"time_seconds":float(frame.time_seconds),
                    "out_png":root/"frames"/clip.clip_id/view.view_id/f"{index:04d}.png",
                })

    rendered,perf=_batch_render(ctx,archive=archive,player=player,requests=requests)
    grouped={}
    outputs=[]
    for req,row in zip(requests,rendered):
        path,rgba_hash,alpha,_stdout=_validate_render_result(row,resolution=resolution_by_view[str(req["view"])])
        key=(str(req["clip"]),str(req["view"]))
        grouped.setdefault(key,[]).append((float(req["time_seconds"]),rgba_hash,alpha,bool(row.get("render_cache_hit",False))))
        outputs.append({"path":str(path),"sha256":str(row["sha256"]),"authority_class":"NATIVE_VISUAL_MOTION_FRAME","schema":"image/png"})

    rows=[]
    clip_by={c.clip_id:c for c in projection.clips}
    for clip in projection.clips:
        for view in projection.views:
            evidence_rows=grouped[(clip.clip_id,view.view_id)]
            times=tuple(x[0] for x in evidence_rows)
            hashes=tuple(x[1] for x in evidence_rows)
            all_visible=all(x[2]>0 for x in evidence_rows)
            unique=len(set(hashes))
            row=VisualMotionViewEvidenceIR(
                clip.clip_id,view.view_id,len(times),unique,all_visible,
                content_sha256({"clip_id":clip.clip_id,"view_id":view.view_id,"rgba_sha256":hashes}),
                unique>1,"",
                metadata={
                    "sample_times":times,"native_player_sha256":player_sha,
                    "cache_hit_count":sum(x[3] for x in evidence_rows),
                    "transport":"CONTENT_ADDRESSED_BATCH_NATIVE_V4",
                },
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
        "transport":"CONTENT_ADDRESSED_BATCH_NATIVE_V4",
        **perf,
    }}
